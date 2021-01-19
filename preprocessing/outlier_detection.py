import numpy as np
import pandas as pd

from preprocessing.constants import numeric_features, positive_scaled_features, negative_scaled_features

class OutlierClipper:
    def __init__(self, features):
        self._features = features
        self._feature_map = {}

    def fit(self, X, y, **kwargs):
        df = X[self._features]
        features = list(df.columns)
        for feature in features:
            f_q1 = df[feature].quantile(0.25)
            f_q3 = df[feature].quantile(0.75)
            f_iqr = f_q3 - f_q1
            self._feature_map[feature] = (f_q1 - (1.5 * f_iqr), f_q3 + (1.5 * f_iqr))
        return self

    def transform(self, data):
        data_copy = data.copy()
        for feature in self._feature_map.keys():
            data_copy[feature] = data_copy[feature].clip(lower=self._feature_map[feature][0],
                                                         upper=self._feature_map[feature][1])
        return data_copy

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)