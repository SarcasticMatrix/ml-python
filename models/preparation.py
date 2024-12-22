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
