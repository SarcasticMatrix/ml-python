"""
Module `preparation.py`

Ce module contient des fonctions pour la préparation et le nettoyage des données, ainsi que pour l'analyse des corrélations entre variables dans un dataset.

Fonctions princaples :
-----------
- `import_data`: Importation et nettoyage initial d'un fichier de données.
- `compute_correlation`: Calcul et visualisation de la matrice de corrélation pour les variables numériques.
- `variables_selection`: Sélection des variables avec des corrélations inférieures à un seuil donné.
- `do_the_job`: Pipeline complet pour préparer les données pour un modèle.

"""

import pandas as pd
import numpy as np
import seaborn as sns
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
    if True in np.unique(data.isna()):
        data.dropna(how='any')
        print('Quelques NAs, "" ou np.inf - on les drop')

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

def compute_correlation(data:pd.DataFrame, method:str = 'pearson'):
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
    vars_numériques = data[labels_numériques]
    correlation = np.round(vars_numériques.corr(method=method).abs(),2)

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

    correlation = compute_correlation(data=data)

    # Séléctionne la matrice triangulaire supérieure
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))

    # Trouve les features avec une corrélation > correlation_threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    data = data.drop(to_drop, axis=1)

    return data


def do_the_job(
        path:str,
        correlation_threshold:float = 0.95,
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
    data = variables_selection(data, correlation_threshold)
    input = data['y'].copy()
    output = data.drop(columns='y').copy()

    return {'input':input, 'output':output}