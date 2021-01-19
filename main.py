import pandas as pd
import numpy as np
import pickle

from preprocessing.preprocess import split
from utils.save_to_csv import save_data_to_csv
from preprocess import DataPreprocessor
from preprocessing.transformations import fix_label_type 

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

#preprocess = pickle.load(open('dumps/preprocessor.pkl', 'rb'))


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


##################################################
## Training various models

## We use grid search to optimize our parameters
def grid_search(model, X, y, param_grid):
    clf = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1)
    clf.fit(X, y)
    return clf 

print('### Training ###')
models = [('SVC', svm.SVC(), dict(C= np.logspace(-10, 10, 10), kernel=['rbf','poly'])),
          ('KNN', KNeighborsClassifier(), dict(n_neighbors=np.linspace(2,10,9, dtype=int))),
          ('RandomForest', RandomForestClassifier(max_depth=10, random_state=0), dict(max_depth=np.linspace(2,16,15))),
          ('LogisticRegression', LogisticRegression(random_state=0), dict()),
          ('PolynomialLogisticRegression',  Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LogisticRegression())]), dict())]

models_per_column = dict()

for column in y_train.columns:
    print(f'# Fitting for column {column}')
    fitted_models = []
    for name, model, param_grid in models:
        print(f'## Fitting model {name}')
        clf = grid_search(model, X_train, y_train[column], param_grid)
        fitted_models.append((name, clf))
    models_per_column[column] = fitted_models

##################################################
## Choosing best model using validation dataset


df_val = pd.read_csv('results/val_after.csv')
X_val, y_val = df_val.drop(labels=['TestResultsCode'], axis=1), df_val[['TestResultsCode']]
y_val = fix_label_type(y_val)
X_val = X_val.drop(labels=['PatientID'], axis=1)


## Each target column has a different wanted score (with it's params)
column_score_map = {'Virus' : (precision_score, {'average':'weighted'}), 
    'Spreader': (recall_score, {'average': "binary", 'pos_label': "Spreader"}), 
    'AtRisk': (recall_score, {'average': "binary", 'pos_label': "atRisk"})}

best_models = dict()

for column in ['Virus', 'Spreader', 'AtRisk']:
    fitted_models = models_per_column[column]
    best_model, best_score = None, -1
    for name, model in fitted_models:
        y_hat = model.predict(X_val)
        score_function, params =  column_score_map[column]
        score = score_function(y_val[column], y_hat, **params)
        if score > best_score:
            best_model, best_score = model, score
    
    best_models[column] = (best_model, best_score)
    

##################################################
## Testing our model on the dataset

df_test = pd.read_csv('results/test_after.csv')
X_test, y_test = df_test.drop(labels=['TestResultsCode'], axis=1), df_test[['TestResultsCode']]
y_test = fix_label_type(y_test)
X_test = X_test.drop(labels=['PatientID'], axis=1)

print('### Testing Scores ###')

print('#########################################')
print(f'# checking for column Virus')
column = 'Virus'
model, _  = best_models['Virus']
y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test[column], y_hat)
print(f'accuracy: {accuracy}')
percision = precision_score(y_test[column],y_hat, average='weighted')
print(f'percision: {percision}')
recall = recall_score(y_test[column],y_hat, average='macro')
print(f'recall score: {recall}')
f1 = f1_score(y_test[column],y_hat, average='macro')
print(f'f1 score: {f1}')

print('#########################################')
print(f'# checking for column Spreader')
column = 'Spreader'
model,_ = best_models['Spreader']
y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test[column], y_hat)
print(f'accuracy: {accuracy}')
percision = precision_score(y_test[column],y_hat, average="binary", pos_label="Spreader")
print(f'percision: {percision}')
recall = recall_score(y_test[column],y_hat, average="binary", pos_label="Spreader")
print(f'recall score: {recall} (PREFERED)')
f1 = f1_score(y_test[column],y_hat, average="binary", pos_label="Spreader")
print(f'f1 score: {f1}')


print('#########################################')
print(f'# checking for column AtRisk')
column = 'AtRisk'
model, _ = best_models['AtRisk']

y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test[column], y_hat)
print(f'accuracy: {accuracy}')
percision = precision_score(y_test[column],y_hat , average="binary", pos_label="atRisk")
print(f'percision: {percision}')
recall = recall_score(y_test[column],y_hat, average="binary", pos_label="atRisk")
print(f'recall score: {recall} (PREFERED)')
f1 = f1_score(y_test[column],y_hat, average="binary", pos_label="atRisk")
print(f'f1 score: {f1}')


##################################################
## Predicting unkown values

df_unknown = pd.read_csv('data/virus_hw3_unlabeled.csv')

X_unknown = df_unknown.drop(labels=['TestResultsCode'], axis=1) 
X_unknown = preprocess.transform(X_unknown)


patientIds = X_unknown['PatientID']
X_unknown = X_unknown.drop(labels=['PatientID'], axis=1)


y_pred = pd.DataFrame()
y_pred['Virus'] = pd.Series(best_models['Virus'][0].predict(X_unknown))
y_pred['Spreader'] = pd.Series(best_models['Spreader'][0].predict(X_unknown))
y_pred['AtRisk'] = pd.Series(best_models['AtRisk'][0].predict(X_unknown))

y_pred['TestResultsCode'] = y_pred[['Virus', 'Spreader', 'AtRisk']].agg('_'.join, axis=1)

Result =  pd.concat([patientIds, y_pred['TestResultsCode']], axis=1)

Result.to_csv('results/predicted.csv', index=False)


