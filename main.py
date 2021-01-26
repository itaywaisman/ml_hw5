import pandas as pd
import numpy as np
from preprocess import DataPreprocessor, fix_label_type
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score



##################################################
## Load train dataset

df_train = pd.read_csv('Data/virus_hw5.csv')
X_ids = df_train['PatientID']
X_train, y_train = df_train.drop(labels=['TestResultsCode'], axis=1), df_train[['TestResultsCode']]
y_train = fix_label_type(y_train)

X_train = X_train[['DisciplineScore', 'TimeOnSocialActivities', 'AgeGroup', 'StepsPerYear', 'pcrResult4', 'pcrResult1',
                   'pcrResult12', 'pcrResult5', 'pcrResult16', 'pcrResult14', 'SyndromeClass']]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

preprocessor = DataPreprocessor().fit(X_train, y_train)
X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)



##################################################
## Training various models

# ## Each target column has a different wanted score (with it's params)
# column_score_map = {'Virus' : (precision_score, {'average':'weighted'}), 
#     'Spreader': (recall_score, {'average': "binary", 'pos_label': "Spreader"}), 
#     'AtRisk': (recall_score, {'average': "binary", 'pos_label': "atRisk"})}

## Each target column has a different wanted score (with it's params)
column_score_map = {'Virus' : (accuracy_score, dict()), 
    'Spreader': (accuracy_score, dict()), 
    'AtRisk': (accuracy_score, dict())}

## We use grid search to optimize our parameters
def grid_search(model, X, y, param_grid, score_fn):
    clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score_fn,
                    n_jobs=-1)
    clf.fit(X, y)
    return clf 

svc = (
        'SVC', 
        svm.SVC(), 
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100, 1000]
        }
    )

knn = (
        'KNN', 
        KNeighborsClassifier(), 
        {
            'n_neighbors': np.linspace(2, 10, 9, dtype=int)
        }
    )

lgr = (
    'LogisticRegression', 
    LogisticRegression(), 
    {
        'max_iter': [1000],
        'C':[0.1, 1, 5, 7, 10]
    }
)

random_forest = (
                'RandomForest', 
                RandomForestClassifier(), 
                {
                    'max_depth': np.floor(np.linspace(5, 100, 20)).astype(int),
                    'min_samples_split': [2, 4, 8, 16],
                    'min_samples_leaf': [1, 2, 4, 8, 16]
                }
            )

ada_boost = (
                'AdaBoost', 
                AdaBoostClassifier(), 
                {
                    'learning_rate': [0.5],
                    'n_estimators': np.floor(np.linspace(1, 20, 10)).astype(int)
                }
            )

gradient_boost = (
                'GradientBoosting', 
                GradientBoostingClassifier(), 
                {
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "max_depth":[3],
                    'n_estimators': np.floor(np.linspace(10, 100, 40)).astype(int)
                }
            )

xgboost = (
    'XGBoost',
    XGBClassifier(),
    {
        'max_depth': np.floor(np.linspace(1, 10, 10)).astype(int),
        'n_estimators': np.floor(np.linspace(50, 1000, 50)).astype(int), 
        'learning_rate': np.logspace(-10, 1, 10)
    }
)
# models = [ svc, knn, lgr, random_forest, ada_boost, gradient_boost ]
models = [xgboost]


print('### Training ###')

models_per_column = dict()

for column in column_score_map:
    print(f'# Fitting for column {column}\n')
    fitted_models = []
    for name, model, param_grid in models:
        print(f'## Fitting model {name}')

        score_fn, params = column_score_map[column]

        clf = grid_search(model, X_train, y_train[column], param_grid, make_scorer(score_fn, **params))

        print(f'Train accuracy: {score_fn(y_train[column], clf.predict(X_train), **params)}')

        fitted_models.append((name, clf))

    best_model, best_score = None, -1
    for name, model in fitted_models:
        print(f'## Checking model {name}')
        y_hat = model.predict(X_val)
        score_function, params = column_score_map[column]
        score = score_function(y_val[column], y_hat, **params)
        if score > best_score:
            best_model, best_score = model, score
    
    print('\n\n Best model: ')
    print((best_model.best_estimator_, best_model.best_params_, best_score))

    models_per_column[column] = fitted_models


# ## Each target column has a different wanted score (with it's params)
# column_score_map = {'Virus' : (precision_score, {'average':'weighted'}), 
#     'Spreader': (recall_score, {'average': "binary", 'pos_label': "Spreader"}), 
#     'AtRisk': (recall_score, {'average': "binary", 'pos_label': "atRisk"})}


##################################################
## Choose the best model for each task

best_models = dict()

for column in ['Virus', 'Spreader', 'AtRisk']:
    print(f'# Checking column {column}')
    fitted_models = models_per_column[column]
    best_model, best_score = None, -1
    for name, model in fitted_models:
        print(f'## Checking model {name}')
        y_hat = model.predict(X_val)
        score_function, params = column_score_map[column]
        score = score_function(y_val[column], y_hat, **params)
        if score > best_score:
            best_model, best_score = model, score
    
    best_models[column] = (best_model.best_estimator_, best_model.best_params_, best_score)

print(best_models)
    

##################################################
## Predict on test data using chosen models

df_test = pd.read_csv('Data/virus_hw5_test.csv')
X_ret = pd.DataFrame(df_test['PatientID'], columns=['PatientID'])
X_test = df_test.drop(labels=['PatientID'], axis=1)
X_test = X_test[['DisciplineScore', 'TimeOnSocialActivities', 'AgeGroup', 'StepsPerYear', 'pcrResult4', 'pcrResult1',
                   'pcrResult12', 'pcrResult5', 'pcrResult16', 'pcrResult14', 'SyndromeClass']]
X_test = preprocessor.transform(X_test)
X_ret['Virus'] = best_models['Virus'][0].predict(X_test)
X_ret['Spreader'] = best_models['Spreader'][0].predict(X_test)
X_ret['Risk'] = best_models['AtRisk'][0].predict(X_test)

X_ret.to_csv('results/predicted.csv', index=False)


