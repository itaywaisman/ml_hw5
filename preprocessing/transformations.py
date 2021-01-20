import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def fix_label_type(y):
    splitted = y['TestResultsCode'].str.rsplit('_', n=2,expand=True)
    y['Virus'] = splitted[0]
    y['Spreader'] = splitted[1]
    y['AtRisk'] = splitted[2]
    y = y.drop(labels=['TestResultsCode'], axis=1)
    return y


def handle_syndrome_class(X):
    return pd.get_dummies(X, columns=["SyndromeClass"], prefix=["SyndromeClass"])


# Transformers
syndrome_class_transformer = FunctionTransformer(handle_syndrome_class)
features_data_types_pipeline = Pipeline([
    ('handle_syndrome_class', syndrome_class_transformer),
])

label_transformer = FunctionTransformer(fix_label_type)