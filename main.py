import pandas as pd
import numpy as np
from preprocess import DataPreprocessor
from preprocessing.transformations import fix_label_type
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


##################################################
## Load train dataset

df_train = pd.read_csv('Data/virus_hw5.csv')
X_ids = df_train['PatientID']
X_train, y_train = df_train.drop(labels=['TestResultsCode'], axis=1), df_train[['TestResultsCode']]
y_train = fix_label_type(y_train)

X_train = X_train[['DisciplineScore', 'TimeOnSocialActivities', 'AgeGroup', 'StepsPerYear', 'pcrResult4', 'pcrResult1',
                   'pcrResult12', 'pcrResult5', 'pcrResult16', 'pcrResult14', 'SyndromeClass']]
preprocessor = DataPreprocessor().fit(X_train, y_train)
X_train = preprocessor.transform(X_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



##################################################
## Training various models

## We use grid search to optimize our parameters
def grid_search(model, X, y, param_grid):
    clf = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, verbose=4)
    clf.fit(X, y)
    return clf 

print('### Training ###')
models = [('SVC_rbf', svm.SVC(kernel='rbf'), dict(C=[0.1, 1, 10, 100, 1000], gamma=[0.0001, 0.001, 0.01, 0.1, 1])),
          # ('SVC_poly', svm.SVC(kernel='poly'), dict(C=[0.001, 0.1, 1, 10], gamma=[0.0001, 0.001, 0.01, 0.1, 1])),
          ('KNN', KNeighborsClassifier(), dict(n_neighbors=np.linspace(2, 10, 9, dtype=int))),
          ('RandomForest', RandomForestClassifier(), dict(max_depth=np.linspace(40, 50, 11))),
          ('LogisticRegression', LogisticRegression(max_iter=1000), dict(C=[0.1, 1, 5, 7, 10])),
          ('PolynomialLogisticRegression',  Pipeline([('poly', PolynomialFeatures(degree=2)),
                                                      ('linear', LogisticRegression())]), dict()),
          ('AdaBoost', AdaBoostClassifier(), dict(n_estimators=[35, 36, 37, 38, 39], learning_rate=[0.01, 0.1, 0.5,
                                                                                                    0.7, 1]))]

models_per_column = dict()

for column in y_train.columns:
    print(f'# Fitting for column {column}')
    fitted_models = []
    for name, model, param_grid in models:
        print(f'## Fitting model {name}')
        clf = grid_search(model, X_train, y_train[column], param_grid)
        fitted_models.append((name, clf))
    models_per_column[column] = fitted_models


## Each target column has a different wanted score (with it's params)
column_score_map = {'Virus' : (precision_score, {'average':'weighted'}), 
    'Spreader': (recall_score, {'average': "binary", 'pos_label': "Spreader"}), 
    'AtRisk': (recall_score, {'average': "binary", 'pos_label': "atRisk"})}


##################################################
## Choose the best model for each task

best_models = dict()

for column in ['Virus', 'Spreader', 'AtRisk']:
    fitted_models = models_per_column[column]
    best_model, best_score = None, -1
    for name, model in fitted_models:
        y_hat = model.predict(X_val)
        score_function, params = column_score_map[column]
        score = score_function(y_val[column], y_hat, **params)
        if score > best_score:
            best_model, best_score = model, score
    
    best_models[column] = (best_model, best_score)
    

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


