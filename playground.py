import pandas as pd
import numpy as np
from preprocessing.preprocess import DataPreprocessor
from preprocessing.transformations import fix_label_type
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

##################################################
## Load train dataset

df_train = pd.read_csv('Data/virus_hw5.csv')
X_ids = df_train['PatientID']
X, y = df_train.drop(labels=['TestResultsCode'], axis=1), df_train[['TestResultsCode']]
y = fix_label_type(y)
y = y['Virus']
X = X[['DisciplineScore', 'TimeOnSocialActivities', 'AgeGroup', 'StepsPerYear', 'pcrResult4', 'pcrResult1',
                   'pcrResult12', 'pcrResult5', 'pcrResult16', 'pcrResult14', 'SyndromeClass']]

preprocessor = DataPreprocessor().fit(X, y)
X = preprocessor.transform(X)

##################################################
## Training various models

seed = 7
logreg = LogisticRegression(penalty='l1', solver='liblinear',multi_class='auto')
lr_param = {
    'penalty':['l1'], 
    'C': [0.5, 1, 5, 10], 
    'max_iter':[100, 200, 500, 1000]
}

rfc = RandomForestClassifier(n_estimators=500)
rfc_param = {
    "max_depth": np.floor(np.linspace(10, 100, 40)).astype(int)),
    "max_features": ["auto"],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "bootstrap": [False],
    "criterion": ["entropy", "gini"]
}


mlp = MLPClassifier(random_state=seed)
mlp_param = {
    'hidden_layer_sizes': [(50,50,50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'max_iter': [10000],
    'alpha': [0.0001],
    'learning_rate': ['constant']
}

ada = AdaBoostClassifier()
ada_param = {
    'n_estimators': np.floor(np.linspace(10, 100, 40)).astype(int))
    'base_estimator': DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=3)
}

gbm = GradientBoostingClassifier()
gbm_param = {"loss":["deviance"],
    "learning_rate": [0.001],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_depth":[3],
    "max_features":["auto"],
    "criterion": ["friedman_mse"],
    "n_estimators":[50]
    }

svm = SVC(gamma="scale")
tuned_parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75)}

inner_cv = KFold(n_splits=10, shuffle=True, random_state=seed)

outer_cv = KFold(n_splits=10, shuffle=True, random_state=seed)


models = []

models.append(('GBM', GridSearchCV(gbm, param, cv=inner_cv, n_jobs=-1, verbose=4)))
models.append(('RFC', GridSearchCV(rfc, param_grid, cv=inner_cv, n_jobs=-1, verbose=4)))
models.append(('LR', GridSearchCV(logreg, LR_par, cv=inner_cv, n_jobs=-1, verbose=4)))
models.append(('SVM', GridSearchCV(svm, tuned_parameters, cv=inner_cv, n_jobs=-1, verbose=4)))
models.append(('MLP', GridSearchCV(mlp, parameter_space, cv=inner_cv, n_jobs=-1, verbose=4)))

results = []
names = []
scoring = 'accuracy'
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


for name, model in models:
    nested_cv_results = cross_val_score(model, X, y, cv=outer_cv, scoring=scoring)
    results.append(nested_cv_results)
    names.append(name)
    msg = "Nested CV Accuracy %s: %f (+/- %f )" % (name, nested_cv_results.mean()*100, nested_cv_results.std()*100)
    print(msg)
    model.fit(X_train, Y_train)
    print('Test set accuracy: {:.2f}'.format(model.score(X_test, Y_test)*100),  '%')