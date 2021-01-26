import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

def fix_label_type(y):
    splitted = y['TestResultsCode'].str.rsplit('_', n=2,expand=True)
    y = y.copy()
    y.loc[:,'Virus'] = splitted[0]
    y.loc[:,'Spreader'] = splitted[1]
    y.loc[:,'AtRisk'] = splitted[2]
    y = y.drop(labels=['TestResultsCode'], axis=1)
    return y

def handle_age_group(X):
    return pd.get_dummies(X, columns=["AgeGroup"], prefix=["AgeGroup"])

def handle_syndrome_class(X):
    return pd.get_dummies(X, columns=["SyndromeClass"], prefix=["SyndromeClass"])


# Transformers
syndrome_class_transformer = FunctionTransformer(handle_syndrome_class)
age_group_transformer = FunctionTransformer(handle_age_group)

features_data_types_pipeline = Pipeline([
    ('handle_syndrome_class', syndrome_class_transformer),
    ('handle_age_group', age_group_transformer),
])

label_transformer = FunctionTransformer(fix_label_type)

normal_features = ['DisciplineScore', 'TimeOnSocialActivities', 'AgeGroup', 'StepsPerYear', 'pcrResult4', 'pcrResult1',
                   'pcrResult12', 'pcrResult5', 'pcrResult16']

class Imputer:
    def __init__(self):
        self.iterative_imputer = IterativeImputer(initial_strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)

    def fit(self, X, y, **kargs):
        self.iterative_imputer.fit(X[normal_features])
        self.knn_imputer.fit(X)
        # print(X.shape)
        return self

    def transform(self, X):
        X[normal_features] = self.iterative_imputer.transform(X[normal_features])
        res = self.knn_imputer.transform(X)
        X = pd.DataFrame(res, columns=X.columns)
        return X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

class DataPreprocessor:

    def __init__(self):
        self.imputer = Imputer()
        self.scaler = RobustScaler(quantile_range=(0.15, 0.85), unit_variance=True)

    def fit(self, X, y, **kwargs):
        tmp_X = handle_syndrome_class(X)
        # tmp_X = handle_age_group(tmp_X)
        self.encoded_columns = tmp_X.columns
        self.imputer = self.imputer.fit(tmp_X, y)
        tmp_X = self.imputer.transform(tmp_X)
        self.scaler = self.scaler.fit(tmp_X, y)

        return self
    
    def transform(self, X, **kwargs):
        X_transformed = handle_syndrome_class(X)
        # X_transformed = handle_age_group(X)
        X_transformed = X_transformed.reindex(columns = self.encoded_columns, fill_value=0)
        X_transformed = self.imputer.transform(X_transformed)
        X_transformed = pd.DataFrame(self.scaler.transform(X_transformed), columns=X_transformed.columns)

        return X_transformed

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)
