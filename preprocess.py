import pandas as pd
import numpy as np
from preprocessing.preprocess import split
from utils.save_to_csv import save_data_to_csv
from preprocessing.preprocess import DataPreprocessor 
from preprocessing.transformations import fix_label_type 
import pickle


## Loading Data And preprocessing
def preprocess():
    df = pd.read_csv('data/virus_hw2.csv')

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    save_data_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, suffix='before')

    preprocess = DataPreprocessor()

    X_train = preprocess.fit_transform(X_train, y_train)
    X_val = preprocess.transform(X_val)
    X_test = preprocess.transform(X_test)

    save_data_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, suffix='after')

    pickle.dump(preprocess, open('dumps/preprocessor.pkl', 'wb'))
    return preprocess


if __name__ == '__main__':
    preprocess()