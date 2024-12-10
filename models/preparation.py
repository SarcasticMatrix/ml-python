"""
Module `preparation.py`

Ce module contient des fonctions pour la préparation et le nettoyage des données, ainsi que pour l'analyse des corrélations entre variables dans un dataset.

Fonctions princaples :
-----------
- `import_data`: Importation et nettoyage initial d'un fichier de données.
- `compute_correlation`: Calcul et visualisation de la matrice de corrélation pour les variables numériques.
- `point_biserial_correlation`: Calcul et visualisation de la matrice de corrélation pour tout les variables.
- `variables_selection`: Sélection des variables avec des corrélations inférieures à un seuil donné.
- `encode': One Hot Encoding des variables catégorielles.
- `normalise`: normalise les variables numériques.
- `do_the_job`: Pipeline complet pour préparer les données pour un modèle.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.preprocessing as preproc
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA





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

def point_biserial_correlation(data,plot=False):
    """
    Calculer la corrélation point-bisériale pour toutes les paires de colonnes dans le dataframe.

    Arguments :
        data (pd.DataFrame) : Le dataframe d'entrée contenant les données.
        plot (bool, optionnel) : Si True, affiche une carte thermique des corrélations. Par défaut False.
    """
    all_columns = data.columns

    point_biserial_corr = {
    (col1, col2): pointbiserialr(
        data[col1].cat.codes if data[col1].dtype.name == 'category' else data[col1],
        data[col2].cat.codes if data[col2].dtype.name == 'category' else data[col2],
    )
    for col1 in all_columns for col2 in all_columns if col1 != col2
    }

    point_biserial_corr=pd.DataFrame(point_biserial_corr)
    def star_significance(pval):
        if pval.iloc[0] < 0.05:
            return '*'
        else:
            return ''
    point_biserial_corr=point_biserial_corr.apply(lambda x: (round(x[[0]],4)).astype("str")+star_significance(x[[1]]))

    point_biserial_corr=point_biserial_corr.stack().loc[0].fillna("1")
    if plot:
        point_biserial_corr_numeric = point_biserial_corr.applymap(lambda x: (x.rstrip('*'))).astype(float)
        plt.figure(figsize=(20, 20))
        sns.heatmap(point_biserial_corr_numeric, annot=point_biserial_corr, fmt='', cmap='coolwarm', center=0, annot_kws={"size": 10})

    else:
        return point_biserial_corr

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
                    annot_kws={'alpha':0.5, "color": 'black'}
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

def encode(data:pd.DataFrame) -> pd.DataFrame:
    """
    Effectue un encodage One-Hot des colonnes catégorielles d'un dataframe.

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données à encoder.

    Retour :
    --------
    - `pd.DataFrame` : Un nouveau dataframe contenant les colonnes encodées.
    """

    labels_categorielles = get_dtypes(data)[np. dtype('O')]
    vars_categorielles = data[labels_categorielles].copy()
    data = data.drop(columns=labels_categorielles)

    #One hot encoding des variables catégorielles
    preproc_ohe = preproc.OneHotEncoder(handle_unknown='ignore')
    preproc_ohe = preproc.OneHotEncoder(drop='first', sparse_output = False).fit(vars_categorielles) 

    variables_categorielles_ohe = preproc_ohe.transform(vars_categorielles)
    variables_categorielles_ohe = pd.DataFrame(variables_categorielles_ohe, 
                                            columns = preproc_ohe.get_feature_names_out(vars_categorielles.columns))
    return pd.concat([data, variables_categorielles_ohe], axis=1)


def normalise(data:pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes numériques d'un dataframe en utilisant une standardisation (moyenne à 0 et écart-type à 1).

    Paramètres :
    ------------
    - `data` (pd.DataFrame) : Le dataframe contenant les données à normaliser.

    Retour :
    --------
    - `pd.DataFrame` : Un dataframe contenant uniquement les colonnes numériques, normalisées.
    """

    # Liste des variables numériques
    dtype_to_column = get_dtypes(data)
    labels_numériques = dtype_to_column[np.dtype('float64')]
    vars_numeriques = data[labels_numériques]
    
    # Scale les variables numériques
    preproc_scale = preproc.StandardScaler(with_mean=True, with_std=True)
    preproc_scale.fit(vars_numeriques)
    vars_numeriques_scaled = preproc_scale.transform(vars_numeriques)
    vars_numeriques_scaled = pd.DataFrame(vars_numeriques_scaled, 
                                columns = vars_numeriques.columns)
    data[labels_numériques] = vars_numeriques_scaled[labels_numériques]
    return data

def do_the_job(
        path: str = 'models/data/data.csv',
        correlation_threshold:float = 0.95,
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

    # Sélectionne ouput et input
    output = data['y'].copy()
    input = data.drop(columns='y').copy()

    # Normalise variables numériques
    input = normalise(input)

    # One hot encoding variables catégorielles
    input = encode(input)

    # Encoder les labels
    label_encoder = LabelEncoder()
    output = label_encoder.fit_transform(output)

    return {'input':input, 'output':output}
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    A custom transformer for feature engineering in a machine learning pipeline.
    Methods
    -------
    __init__():
        Initializes the transformer.
    fit(X, y=None):
        Fits the transformer to the data. This transformer does not learn from the data, so this method returns self.
    transform(X):
        Transforms the input DataFrame by performing various feature engineering steps:
        - Drops the `duration` column.
        - Replaces 999 in `pdays` column with -1.
        - Converts object type columns to categorical type.
        - Creates a new `prev_contact` column based on `pdays` with specified bins and labels.
        - Creates a new `age_cat` column based on `age` with specified bins and labels.
        - Creates a new `campaign_efficiency` column based on `previous` and `campaign` columns.
        - Adds the first principal component applied to economic factors as `EconStabSentPCA`.
        - Drops specified columns and returns the transformed DataFrame.
    
    -------
    pandas.DataFrame
        The transformed DataFrame with new features and selected columns.
        
    Example
    -------

    pipeline = Pipeline([
        ('Feature_Engineering', FeatureEngineering()),
    ])

    data_path = r'models\data\data.csv'
    raw_data = pd.read_csv(data_path, delimiter=';')

    cleaned_data = pipeline.fit_transform(raw_data)
    cleaned_data.head()
    
    """
    def fit(self, X, y=None):return self
    
    def transform(self, X):
        X = X.drop(columns='duration').copy()
        X['pdays'] = X['pdays'].replace(999, -1)
        
        for column in X.select_dtypes(include='object').columns:
            X[column] = X[column].astype('category')
        
        X['prev_contact'] = pd.cut(X['pdays'], 
                                    bins=[-1, 0, 8, 16, 28],
                                    right=False,
                                    labels=['Never', '[0-7]', '[7-15]', '15<'])
        
        X['age_cat'] = pd.cut(X['age'], 
                                bins=[0, 23-0.01, 30-0.01, 40-0.01, 60-0.01, 100], 
                                labels=['[0-23[', '[23-30[', '[30-40[', '[40-60[', '60<'])
        
        X['campaign_efficiency'] = X.apply(lambda row: row['previous'] / row['campaign'] if row['poutcome'] == 'success' else 0, axis=1)
        
        economic_factors = X[['cons.price.idx', 'cons.conf.idx', 'emp.var.rate', 'euribor3m', 'nr.employed']]
        scaler = preproc.StandardScaler()
        economic_factors_scaled = scaler.fit_transform(economic_factors)
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(economic_factors_scaled)
        principal_components = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
        
        X['EconStabSentPCA'] = principal_components["PC1"]
        
        cols_to_drop = ['pdays', 'age', 'previous', 'poutcome'] + economic_factors.columns.drop('cons.conf.idx').tolist()
        data4analysis = X
        # data4analysis = X.drop(columns=cols_to_drop)
        
        return data4analysis


class Preprocessing(BaseEstimator, TransformerMixin):
    """
    A stramlined preprocessing transformer that:
    - Encodes categorical features using OneHotEncoder.
    - Scales numerical features using StandardScaler.
    - Encodes target variable (y) using LabelEncoder.

    Attributes
    ----------
    cat_encoder_ : OneHotEncoder
        Encoder for categorical variables.

    num_scaler_ : StandardScaler
        Scaler for numerical variables.

    label_encoder_ : LabelEncoder
        Encoder for the target variable (y).

    Parameters
    ----------
    scale : bool, default=True
        Whether to scale numerical features.

    encode : bool, default=True
        Whether to encode categorical features.
    """
    def __init__(self, scale=True, encode=True):
        self.scale = scale
        self.encode = encode

    def fit(self, X, y=None):
        """
        Fit encoders and scalers to the data.
        """
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.encode and self.cat_cols_:
            self.cat_encoder_ = preproc.OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X[self.cat_cols_])

        if self.scale and self.num_cols_:
            self.num_scaler_ = preproc.StandardScaler().fit(X[self.num_cols_])

        if y is not None:
            self.label_encoder_ = LabelEncoder().fit(y)

        return self

    def transform(self, X, y=None):
        """
        Transform the data by encoding and scaling.
        """
        X_transformed = X.copy()

        if self.encode and self.cat_cols_:
            encoded = self.cat_encoder_.transform(X_transformed[self.cat_cols_])
            encoded_df = pd.DataFrame(encoded, columns=self.cat_encoder_.get_feature_names_out(self.cat_cols_))
            X_transformed = pd.concat([X_transformed[self.num_cols_].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            X_transformed.columns = [col.replace('[', '').replace(']', '').replace('<', 'lt') for col in X_transformed.columns]

        if self.scale and self.num_cols_:
            X_transformed[self.num_cols_] = self.num_scaler_.transform(X_transformed[self.num_cols_])

        y_transformed = None
        if y is not None and hasattr(self, "label_encoder_"):
            y_transformed = self.label_encoder_.transform(y)
            
        return X_transformed, y_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in a single step.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform_y(self, y):
        """
        Inverse transform target variable (y) to its original labels.
        """
        if hasattr(self, "label_encoder_"):
            return self.label_encoder_.inverse_transform(y)
        raise ValueError("Label encoder has not been fitted.")


if __name__=="__main__":
    pipeline = Pipeline([
        ('Feature_Engineering', FeatureEngineering()),
        ('preprocessing', Preprocessing()),
        ])
    
    data_path = r'models\data\data.csv'
    raw_data = pd.read_csv(data_path, delimiter=';')

    X,y = pipeline.fit_transform(raw_data.drop(columns='y'), raw_data['y'])
    print(X,y)