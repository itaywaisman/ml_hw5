import pandas as pd
from preprocessing.transformations import handle_syndrome_class
from preprocessing.imputer import Imputer
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:

    def __init__(self):
        self.imputer = Imputer()
        self.scaler = RobustScaler(quantile_range=(0.15, 0.85), unit_variance=True)

    def fit(self, X, y, **kwargs):
        tmp_X = handle_syndrome_class(X)
        self.imputer = self.imputer.fit(tmp_X, y)
        tmp_X = self.imputer.transform(tmp_X)
        self.scaler = self.scaler.fit(tmp_X, y)

        return self
    
    def transform(self, X, **kwargs):
        X_transformed = handle_syndrome_class(X)
        X_transformed = self.imputer.transform(X_transformed)
        X_transformed = pd.DataFrame(self.scaler.transform(X_transformed), columns=X_transformed.columns)

        return X_transformed

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)
