"""
Module `preparation.py`

Ce module contient des fonctions pour la préparation et le nettoyage des données, ainsi que pour l'analyse des corrélations entre variables dans un dataset.

Fonctions princaples :
-----------
- `import_data`: Importation et nettoyage initial d'un fichier de données.
- `compute_correlation`: Calcul et visualisation de la matrice de corrélation pour les variables numériques.
- `variables_selection`: Sélection des variables avec des corrélations inférieures à un seuil donné.
- `encode': One Hot Encoding des variables catégorielles.
- `normalise`: normalise les variables numériques.
- `do_the_job`: Pipeline complet pour préparer les données pour un modèle.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.preprocessing as preproc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.impute import KNNImputer

pd.options.mode.use_inf_as_na = True # '' ou numpy.inf considérées comme des NA

def import_data(
        path: str = 'models/data/data.csv'
    ) -> pd.DataFrame:
    """
    Importe un fichier CSV et applique un nettoyage.

    Paramètres :
    ------------
    - `path` (str, optionnel) : Chemin vers le fichier CSV à importer. Par défaut : `'models/data/data.csv'`.

    Retour :
    --------
    - `pd.DataFrame` : Le dataframe importé et nettoyé.
    """

    data = pd.read_csv(path, sep=";")
    data = data.drop(columns='duration')

    for column in data:
        if data[column].dtype == 'O': 
            data[column] = data[column].astype(str)  # Convertit en chaîne de caractères

    # Clean les NAs, '' ou np.inf
    #if True in np.unique(data.isna()):
    #    data.dropna(how='any')
    #    print('Quelques NAs, "" ou np.inf - on les drop')

    return data

def get_dtypes(data:pd.DataFrame) -> dict:
    """
    Retourne un dictionnaire liant chaque type de données (`dtype`) à la liste des colonnes ayant ce type.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe dont les types des colonnes doivent être analysés.

    Retour :
    --------
    - `dict` : Un dictionnaire où chaque clé est un type de données (`dtype`), et chaque valeur est une liste des colonnes de ce type.

    Exemple :
    ---------
    Pour un dataframe avec des colonnes de types int64, float64, et object :
    {
        np.dtype('int64'): ['col1', 'col2'],
        np.dtype('float64'): ['col3'],
        np.dtype('O'): ['col4']
    }
    """
    column_to_dtype = data.dtypes.to_dict()

    column_dtype = [y for x,y in column_to_dtype.items()]
    column_dtype = list(set(column_dtype)) # prend les élements uniques de la liste

    dtype_to_column = {
        type:[] for type in column_dtype
    }
    for key in column_to_dtype.keys():
        dtype_to_column[column_to_dtype[key]].append(key)
    
    return dtype_to_column

def compute_correlation(data:pd.DataFrame, method:str = 'pearson', plot_bool:bool = True):
    """
    Calcule, affiche sous forme de heatmap, et retourne la matrice de corrélation pour les variables numériques.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données.
    - `method` (str, optionnel) : Méthode de calcul de la corrélation (`'pearson'`, `'kendall'`, ou `'spearman'`). Par défaut : `'pearson'`.

    Retour :
    --------
    - `pd.DataFrame` : La matrice de corrélation absolue.

    Description :
    -------------
    - Extrait les colonnes de types numériques (`int64` et `float64`).
    - Calcule la matrice de corrélation.
    - Affiche une heatmap des corrélations avec les valeurs au-dessus de la diagonale masquées.
    """

    # Calcul de la corrélation
    dtype_to_column = get_dtypes(data)
    labels_numériques = dtype_to_column[np.dtype('float64')] + dtype_to_column[np.dtype('int64')]
    vars_numeriques = data[labels_numériques]
    correlation = np.round(vars_numeriques.corr(method=method).abs(),2)

    if plot_bool:
        # Heatmap des corrélations
        upper_indices = np.triu_indices_from(correlation, k=1)  # Indices au-dessus de la diagonale
        correlation_to_be_showed = correlation.copy()
        correlation_to_be_showed.values[upper_indices] = None  # Remplacer par None au dessus de la diagonale
        
        sns.heatmap(correlation_to_be_showed, 
                    center=0.5, 
                    cmap=sns.diverging_palette(220, 20, as_cmap=True), 
                    annot=True, 
                    annot_kws={'alpha':0.5, "color": 'white'}
                )       

    return correlation

def variables_selection(
        data:pd.DataFrame,
        correlation_threshold:float = 0.95,
        plot_bool:bool = False
    ) -> pd.DataFrame:
    """
    Sélectionne les colonnes numériques ayant une corrélation inférieure à un seuil donné.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données.
    - `correlation_threshold` (float, optionnel) : Seuil de corrélation au-delà duquel une colonne sera supprimée. Par défaut : `0.95`.

    Retour :
    --------
    - `pd.DataFrame` : Le dataframe avec les colonnes non corrélées sélectionnées.

    Description :
    -------------
    - Calcule la matrice de corrélation pour les colonnes numériques.
    - Identifie les colonnes ayant des corrélations supérieures au seuil avec d'autres colonnes.
    - Supprime ces colonnes du dataframe.
    """

    correlation = compute_correlation(data=data, plot_bool=plot_bool)

    # Séléctionne la matrice triangulaire supérieure
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))

    # Trouve les features avec une corrélation > correlation_threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    data = data.drop(to_drop, axis=1)

    return data

def cramers_v_table(df, plot_bool: bool = True):
    """
    Calcule et affiche la table du V de Cramér pour toutes les paires de variables catégorielles dans un DataFrame.
    
    Paramètres :
        df (pd.DataFrame) : Le DataFrame contenant les variables catégorielles.
        plot_bool (bool) : Indique si une heatmap doit être affichée. Par défaut : True.
        
    Retourne :
        pd.DataFrame : Une matrice carrée contenant les valeurs du V de Cramér pour chaque paire de colonnes.
    """
    # Sélection des colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['O']).columns
    
    # Initialisation de la matrice de résultats
    cramers_v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
    
    # Calcul du V de Cramér pour chaque paire
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0  # Corrélation parfaite avec soi-même
            else:
                # Table de contingence pour les deux colonnes
                contingency_table = pd.crosstab(df[col1], df[col2])
                
                # Calcul du Chi-carré
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()  # Taille totale des données
                k, r = contingency_table.shape
                # Calcul du V de Cramér
                cramers_v = np.sqrt(chi2 / (n * (min(k - 1, r - 1))))
                cramers_v_matrix.loc[col1, col2] = cramers_v

    # Affiche une heatmap si demandé
    if plot_bool:
        # Masquer les valeurs au-dessus de la diagonale
        upper_indices = np.triu_indices_from(cramers_v_matrix, k=1)
        cramers_v_to_show = cramers_v_matrix.copy()
        cramers_v_to_show.values[upper_indices] = None

        # Heatmap
        sns.heatmap(
            cramers_v_to_show, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True), 
            annot=True, 
            center=0.5, 
            annot_kws={'alpha': 0.5, "color": 'white'}
        )
    
    return cramers_v_matrix


def categorical_variables_selection(data: pd.DataFrame, testcramerV_threshold: float = 0.95, plot_bool: bool = False) -> pd.DataFrame:
    """
    Sélectionne les colonnes catégorielles ayant une corrélation inférieure à un seuil donné.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données.
    - `testcramerV_threshold` (float, optionnel) : Seuil de corrélation au-delà duquel une colonne sera supprimée. Par défaut : `0.95`.
    - `plot_bool` (bool, optionnel) : Indique si une heatmap doit être affichée. Par défaut : False.

    Retour :
    --------
    - `pd.DataFrame` : Le dataframe avec les colonnes non corrélées sélectionnées.
    """
    # Calcul de la matrice des V de Cramér
    correlation = cramers_v_table(data, plot_bool=plot_bool)
    
    # Sélectionne la matrice triangulaire supérieure
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    
    # Trouve les colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > testcramerV_threshold)]
    
    # Retourne les colonnes catégorielles non corrélées
    return data.drop(columns=to_drop)


# def encode(data:pd.DataFrame) -> pd.DataFrame:
#     """
#     Effectue un encodage One-Hot des colonnes catégorielles d'un dataframe.

#     Paramètres :
#     ------------
#     - `data` (pd.DataFrame) : Le dataframe contenant les données à encoder.

#     Retour :
#     --------
#     - `pd.DataFrame` : Un nouveau dataframe contenant les colonnes encodées.
#     """

#     labels_categorielles = get_dtypes(data)[np. dtype('O')]
#     vars_categorielles = data[labels_categorielles].copy()
#     data = data.drop(columns=labels_categorielles)

#     #One hot encoding des variables catégorielles
#     preproc_ohe = preproc.OneHotEncoder(handle_unknown='ignore')
#     preproc_ohe = preproc.OneHotEncoder(drop='first', sparse_output = False).fit(vars_categorielles) 

#     variables_categorielles_ohe = preproc_ohe.transform(vars_categorielles)
#     variables_categorielles_ohe = pd.DataFrame(variables_categorielles_ohe, 
#                                             columns = preproc_ohe.get_feature_names_out(vars_categorielles.columns))
#     return pd.concat([data, variables_categorielles_ohe], axis=1)d

def encode(data: pd.DataFrame, method: str = "ordinal") -> pd.DataFrame:
    """
    Encode les variables catégorielles d'un DataFrame en numérique.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le DataFrame contenant les variables à encoder.
    - `method` (str, optionnel) : Méthode d'encodage à utiliser. Options :
        - `"ordinal"` (par défaut) : Encode les catégories avec des entiers (LabelEncoder).
        - `"onehot"` : Encode avec des vecteurs binaires (One-Hot Encoding).

    Retour :
    --------
    - `pd.DataFrame` : Le DataFrame avec les colonnes catégorielles encodées.

    Description :
    -------------
    - Identifie les colonnes catégorielles dans le DataFrame.
    - Applique l'encodage choisi à ces colonnes.
    - Retourne un nouveau DataFrame avec les colonnes encodées.
    """
    # Copie du DataFrame pour éviter les modifications sur l'original
    encoded_data = data.copy()

    # Identifier les colonnes catégorielles
    categorical_cols = encoded_data.select_dtypes(include=['O']).columns

    # Appliquer l'encodage
    if method == "ordinal":
        # Utilise LabelEncoder pour chaque colonne catégorielle
        for col in categorical_cols:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col])
    elif method == "onehot":
        # Utilise pd.get_dummies pour One-Hot Encoding
        encoded_data = pd.get_dummies(encoded_data, columns=categorical_cols, drop_first=True)
    else:
        raise ValueError("Méthode d'encodage invalide. Utilisez 'ordinal' ou 'onehot'.")
    
    return encoded_data

def normalize(data: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    """
    Normalise uniquement les colonnes spécifiées dans une liste en utilisant une standardisation 
    (moyenne à 0 et écart-type à 1).

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données à normaliser.
    - `columns_to_scale` (list) : La liste des noms de colonnes à scaler.

    Retour :
    --------
    - `pd.DataFrame` : Un dataframe avec les colonnes spécifiées normalisées.
    """

    # Vérifie si les colonnes spécifiées existent dans le dataframe
    for col in columns_to_scale:
        if col not in data.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")
    
    # Scale uniquement les colonnes spécifiées
    preproc_scale = preproc.StandardScaler(with_mean=True, with_std=True)
    preproc_scale.fit(data[columns_to_scale])
    scaled_columns = preproc_scale.transform(data[columns_to_scale])
    
    # Remplace les colonnes dans le DataFrame d'origine
    data[columns_to_scale] = scaled_columns
    
    return data


def knnImpute(data: pd.DataFrame, columns: list, n_neighbors: int) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes (NaN) dans les colonnes spécifiées par la classe majoritaire des n_neighbors
    plus proches voisins en utilisant l'algorithme KNN.

    Paramètres :
    -------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données à imputer.
    - `columns` (list) : Liste des noms des colonnes où les valeurs manquantes doivent être imputées.
    - `n_neighbors` (int) : Le nombre de voisins à prendre en compte pour l'imputation.

    Retour :
    --------
    - `pd.DataFrame` : Le dataframe avec les valeurs manquantes imputées.
    """
    
    # Sélectionner uniquement les colonnes spécifiées
    data_to_impute = data[columns]
    
    # Appliquer KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
    
    # Imputation des NaN (pour les colonnes spécifiées)
    data_imputed = knn_imputer.fit_transform(data_to_impute)
    
    # Remplacer les colonnes dans le dataframe original avec les nouvelles valeurs imputées
    data[columns] = data_imputed
    
    return data


def do_the_job(
        path: str = 'models/data/data.csv',
        correlation_threshold:float = 0.95,
        cramers_vthreshold:float = 0.95,
        plot_bool:bool = False,
    ) -> pd.DataFrame:
    """
    Pipeline complet pour préparer les données pour un modèle.

    Paramètres :
    ------------
    - `path` (str) : Chemin vers le fichier CSV à importer.
    - `correlation_threshold` (float, optionnel) : Seuil de corrélation pour la sélection des variables. Par défaut : `0.95`.

    Retour :
    --------
    - `dict` : Un dictionnaire contenant deux dataframes nettoyés :
        - `'input'` : Le dataframe des variables cibles (`y`).
        - `'output'` : Le dataframe des variables explicatives (toutes les colonnes sauf `y`).

    Description :
    -------------
    - Importe et nettoie le fichier CSV via `import_data`.
    - Applique la sélection des variables via `variables_selection`.
    - Sépare les données en deux dataframes : input (`y`) et output (le reste des colonnes).
    """

    data = import_data(path)
    data = variables_selection(data, correlation_threshold, plot_bool)

    #AJOUT IDRISS: POUR REMPLACER LES UNKNOWN PAR NAN
    data = data.replace('unknown', np.nan)

    # Sélectionne ouput et input
    output = data['y'].copy()
    input = data.drop(columns='y').copy()

    # Normalise variables numériques
    input = normalize(input)

    # One hot encoding variables catégorielles
    input = encode(input)

    # Encoder les labels
    label_encoder = LabelEncoder()
    output = label_encoder.fit_transform(output)

    return {'input':input, 'output':output}



def plot_all_histograms(df, bins=10, figsize=(15, 8)):
    """
    Trace des histogrammes pour toutes les colonnes d'un DataFrame pandas, y compris les colonnes numériques et catégoriques.
    
    Colonnes numériques : Histogramme avec des intervalles (bins).
    Colonnes catégoriques : Graphique en barres des fréquences des classes.

    Paramètres :
        df (pd.DataFrame) : Le DataFrame contenant les données.
        bins (int) : Nombre d'intervalles pour les histogrammes numériques. Par défaut, 10.
        figsize (tuple) : Taille de la figure par ligne. Par défaut, (15, 8).

    Retourne :
        None
    """
    # Séparer les colonnes numériques et catégoriques
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Combiner toutes les colonnes à tracer
    all_columns = list(numeric_cols) + list(categorical_cols)

    num_cols = len(all_columns)
    if num_cols == 0:
        print("Aucune colonne à tracer.")
        return

    # Définir le nombre d'histogrammes par ligne
    histograms_per_row = 4
    n_rows = (num_cols + histograms_per_row - 1) // histograms_per_row  # Calculer le nombre de lignes nécessaires

    fig, axes = plt.subplots(nrows=n_rows, ncols=histograms_per_row, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten()  # Aplatir les axes pour une itération facile

    # Tracer chaque colonne
    for i, col in enumerate(all_columns):
        ax = axes[i]
        if col in numeric_cols:
            # Histogramme pour les colonnes numériques
            ax.hist(df[col].dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f"Histogramme de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Fréquence")
        else:
            # Graphique en barres pour les colonnes catégoriques
            value_counts = df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values, color='orange', edgecolor='black', alpha=0.7)
            ax.set_title(f"Graphique en barres de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Fréquence")
            ax.tick_params(axis='x', rotation=45)  # Tourner les étiquettes de l'axe x pour une meilleure lisibilité

    # Masquer les sous-graphiques inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
