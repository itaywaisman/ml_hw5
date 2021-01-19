import pandas as pd
import os


def save_data_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, suffix='before'):
    train_dataset = pd.concat([X_train, y_train], axis=1)
    val_dataset = pd.concat([X_val, y_val], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    train_dataset.to_csv(f'results/train_{suffix}.csv')
    val_dataset.to_csv(f'results/val_{suffix}.csv')
    test_dataset.to_csv(f'results/test_{suffix}.csv')