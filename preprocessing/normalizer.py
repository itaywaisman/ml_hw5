import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from preprocessing.constants import numeric_features, positive_scaled_features, negative_scaled_features

class Normalizer:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        self.max_abs_scaler = MaxAbsScaler()

    def fit(self, X, y, **kargs):
        self.min_max_scaler.fit(X[positive_scaled_features])
        self.max_abs_scaler.fit(X[negative_scaled_features])

        return self

    def transform(self, X, **kargs):
        X[positive_scaled_features] = self.min_max_scaler.transform(X[positive_scaled_features])
        X[negative_scaled_features] = self.max_abs_scaler.transform(X[negative_scaled_features])

        return X