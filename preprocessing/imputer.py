import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from preprocessing.constants import numeric_features, positive_scaled_features, negative_scaled_features


class Imputer:
    def __init__(self):
        self.iterative_imputer = IterativeImputer(initial_strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)

    def fit(self, X, y, **kargs):
        self.iterative_imputer.fit(X[numeric_features])
        self.knn_imputer.fit(X)
        # print(X.shape)
        return self

    def transform(self, X):
        X[numeric_features] = self.iterative_imputer.transform(X[numeric_features])
        res = self.knn_imputer.transform(X)
        X = pd.DataFrame(res, columns=X.columns)
        return X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)