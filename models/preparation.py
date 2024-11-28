"""
Fonctions utilisées dans la préparation et nettoyage des données
"""

import pandas as pd
import numpy as np
import seaborn as sns

def import_data(
        path: str = 'models/data/data.csv'
    ) -> pd.DataFrame:
    """
    Import le fichier data.csv avec les bons paramètrages.
    """

    data = pd.read_csv(path, sep=";")
    data = data.drop(columns='duration')

    for column in data:
        if data[column].dtype == 'O':  # Vérifie si le type est objet
            data[column] = data[column].astype(str)  # Convertit en chaîne de caractères

    return data

def get_dtypes(data:pd.DataFrame) -> dict:
    """
    Return le dictionnary qui lie un dtype à la liste des colonnes dont les éléments sont de ce dtype,
    {
        dtype1: [column_i1, column_i2, etc.],
        dtype2: [column_j1, column_j2, etc.]
        etc.
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
    Compute, plot et return la matrice de corrélation des valeurs numériques d'un dataframe 
    """

    # Calcul de la corrélation
    dtype_to_column = get_dtypes(data)
    labels_numériques = dtype_to_column[np.dtype('float64')] + dtype_to_column[np.dtype('int64')]
    vars_numériques = data[labels_numériques]
    correlation = np.round(np.abs(vars_numériques.corr(method=method)),2)

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


def do_the_job():
    """
    -> TO DO <-
    Prépare deux dataframe nettoyés et préparés: input, output.
    Objectif: juste utiliser cette fonction pour préparer les données pour chaque modèle
    """