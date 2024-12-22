"""
Module `preparation.py`

Ce module contient des fonctions pour la préparation et le nettoyage des données, ainsi que pour l'analyse des corrélations entre variables dans un dataset.

Fonctions princaples :
-----------
- `import_data`: Importation et formatage des types des variables d'un fichier de données.
- `compute_correlation`: Calcul et visualisation de la matrice de corrélation pour les variables numériques.
- `variables_selection`: Sélection des variables avec des corrélations inférieures à un seuil donné.
- `encode': Ordinal Encoding des variables catégorielles.
- `impute_with_random_forest`: Fonction qui remplace les valeurs manquantes par les valeurs des individus les plus "proches" par l'algo Random Forest
- `normalise`: normalise les variables numériques.
- `do_the_job`: Pipeline complet pour préparer les données pour un modèle.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.preprocessing as preproc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

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


def encode(data: pd.DataFrame, method: str = "ordinal") -> pd.DataFrame:
    """
    Encode les variables catégorielles d'un DataFrame en numérique.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le DataFrame contenant les variables à encoder.
    - `method` (str, optionnel) : Méthode d'encodage à utiliser. Options :
        - `"ordinal"` (par défaut) : Encode les catégories avec des entiers (OrdinalEncoder).
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
    encoded_data = data.iloc[:,:-1].copy()
    y=data.iloc[:,-1]
    # Identifier les colonnes catégorielles
    categorical_cols = encoded_data.select_dtypes(include=['O']).columns

    if method == "ordinal":
        # Appliquer OrdinalEncoder sur toutes les colonnes catégorielles
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        encoded_data[categorical_cols] = oe.fit_transform(encoded_data[categorical_cols])

    elif method == "onehot":
        # Appliquer OneHotEncoder et concaténer avec les colonnes restantes
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        onehot_encoded = pd.DataFrame(
            ohe.fit_transform(encoded_data[categorical_cols]),
            columns=ohe.get_feature_names_out(categorical_cols),
            index=encoded_data.index
        )
        # Supprimer les colonnes catégorielles et ajouter les colonnes one-hot
        encoded_data = pd.concat([encoded_data.drop(columns=categorical_cols), onehot_encoded], axis=1)

    else:
        raise ValueError("Méthode d'encodage invalide. Utilisez 'ordinal' ou 'onehot'.")
    
    #Encodage des target spécifiquement pour que 0=non et 1=oui

    le = LabelEncoder()
    le.fit(['no', 'yes'])
    y_encoded = le.transform(y).astype('int64')
    # Convertir y_encoded en DataFrame et renommer la colonne
    y_encoded_df = pd.DataFrame(y_encoded, columns=['y'], index=encoded_data.index)

    # Ajouter la colonne 'y' encodée au DataFrame final
    encoded_data = pd.concat([encoded_data, y_encoded_df], axis=1)
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


def impute_with_random_forest(df, target_column, param_grid=None):
    """
    Impute les valeurs manquantes d'une colonne cible à l'aide d'un modèle de forêt aléatoire.
    Utilise GridSearchCV pour optimiser les hyperparamètres du modèle et retourne le DataFrame nettoyé.

    Args:
        df: Le DataFrame contenant les données.
        target_column: Le nom de la colonne à imputer (chaîne de caractères).
        param_grid: La grille de recherche d'hyperparamètres pour GridSearchCV (par défaut, None).

    Returns:
        Un DataFrame avec les valeurs manquantes imputées par le modèle optimisé.
    """

    # Séparer les features et la cible
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    # Identifier les indices des valeurs manquantes dans la colonne cible
    missing_idx = y.isna()
    
    # Filtrer les données où la cible n'est pas manquante
    X_train = X[~missing_idx]
    y_train = y[~missing_idx]

    # Créer le modèle de forêt aléatoire
    model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=10, random_state=42)

    # Si param_grid est fourni, utiliser GridSearchCV pour la recherche des meilleurs hyperparamètres
    if param_grid:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Mettre à jour le modèle avec les meilleurs paramètres trouvés
        model = grid_search.best_estimator_
        print(f"Meilleurs hyperparamètres trouvés : {grid_search.best_params_}")
    else:
        # Si aucune grille n'est fournie, utiliser le modèle par défaut
        model.fit(X_train, y_train)

    # Prédire les valeurs manquantes pour X[missing_idx]
    X_missing = X[missing_idx]
    predicted_values = model.predict(X_missing)

    # Remplacer les valeurs manquantes par les valeurs prédites
    y[missing_idx] = predicted_values

    # Combiner les features et la cible imputée
    df_imputed = pd.concat([X, y], axis=1)

    #Pour avoir la target à la fin 
    # Inverser l'ordre de "A" et "B"
    cols = list(df_imputed)
    cols[-1], cols[-2] = cols[-2], cols[-1]

    return df_imputed


def do_the_job(
        path: str = 'models/data/data.csv',
        cramers_vthreshold:float = 0.75,
        gridSearch:bool = False,
    ) -> pd.DataFrame:
    """
    Pipeline complet pour préparer les données pour un modèle.

    Paramètres :
    ------------
    - `path` (str) : Chemin vers le fichier CSV à importer.
    - `cramers_vthreshold` (float, optionnel) : Seuil de 'corrélation' pour la sélection des variables catégorielles. Par défaut : `0.75`.
    - `gridSearch` (bool,optionnel): Booléen pour effectuer un grid search cv lors de l'utilisation du RandomForestClassifier pour 'education' (par défaut, les valeurs choisies sont les optimales)

    Retour :
    --------
    - `dict` : Un dictionnaire contenant deux dataframes nettoyés :
        - `'input'` : Le dataframe des variables cibles (`y`).
        - `'output'` : Le dataframe des variables explicatives (toutes les colonnes sauf `y`).

    """

    data = import_data(path)

    #On drop 'default'
    data=data.drop(columns=['default'])

    #Remplacement des 'unknown' par np.nan
    data_copy = data.replace('unknown', np.nan)

    #Suppression des individus avec les na dans les colonnes correspondantes
    data_copy=data_copy.dropna(subset=[col for col in data_copy.columns if col != 'education'])

    #Encodage des données
    data_copy_encoded=encode(data=data_copy,method="ordinal")

    #RandomForest sur 'education'
    if gridSearch:
        data_copy_encoded=impute_with_random_forest(df=data_copy_encoded,target_column='education',param_grid={'n_estimators': [100,150,200],'max_depth': [5,10,20],'min_samples_split': [2,5,10] })
    else:
        data_copy_encoded=impute_with_random_forest(df=data_copy_encoded,target_column='education')

    #drop de 'emp_var_rate' car très corrélé
    data_copy=data_copy.drop(columns=['emp.var.rate'])
    data_copy_encoded.drop(columns=['emp.var.rate'])

    #Sélection des vars catégorielles
    data_copy=categorical_variables_selection(data_copy,testcramerV_threshold=cramers_vthreshold)
    data_copy_encoded=data_copy_encoded[data_copy.columns]

    #On utilise un StandardScaler ici et on scale uniquement les colonnes dont la valeur max en valeur absolue est supétrieure à 1
    data_copy_encoded=normalize(data=data_copy_encoded,columns_to_scale=[col for col in data_copy_encoded.columns if data_copy_encoded[col].abs().max() > 1 ]) 

    # Sélectionne ouput et input
    output = data_copy_encoded['y'].copy()
    input = data_copy_encoded.drop(columns='y').copy()

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
