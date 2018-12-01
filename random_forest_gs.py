#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
bank_data_final = pd.read_csv(filepath_or_buffer="parallel_logistic/bank-additional-full-final.csv", delimiter=',')
# bank_data = pd.read_csv(filepath_or_buffer="bank-additional.csv", delimiter=';')
bank_data_final.drop([])

#%%
# Feature Engineering-----------------
X = bank_data_final.iloc[:, 1:43].values
y = bank_data_final.iloc[:, 43].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling for numerical attributes only
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# X_train[:, np.r_[31, 33:41]] = sc.fit_transform(X_train[:, np.r_[31, 33:41]])
# X_test[:, np.r_[31, 33:41]] = sc.transform(X_test[:, np.r_[31, 33:41]])
X_train[:, 31:42] = sc.fit_transform(X_train[:, 31:42])
X_test[:,31:42] = sc.transform(X_test[:, 31:42])

# Solving imbalance output problem(accuracy paradox) by oversampling
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

classifier = []
from sklearn.ensemble import RandomForestClassifier
classifier.append(RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0))
classifier[0].fit(X_train, y_train)

y_pred = []
y_pred.append(classifier[0].predict_proba(X_test)[:, 1] > 0.3)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = []
for x in range(0,len(y_pred)):
    cm.append(confusion_matrix(y_test, y_pred[x]))

# Making classification_report
from sklearn.metrics import classification_report
for x in range(0,len(y_pred)):
    print(x)
    print(classification_report(y_test, y_pred[x]))

# Grid Search CV Sequential
import time
from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
tStart_seq = time.time()
parameters = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
             "min_samples_split": [0.5, 0.9],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}
gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=parameters)
gs.fit(X_train, y_train.astype('int'))
tEnd_seq = time.time()
print('Time required to do Sequential Grid Search CV: {} seconds'.format(tEnd_seq - tStart_seq))

# Grid Search CV Parallel
import time
from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
import spark_sklearn
from spark_sklearn import GridSearchCV

from spark_sklearn.util import createLocalSparkSession
sc = createLocalSparkSession().sparkContext

tStart_par = time.time()
parameters = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
             "min_samples_split": [0.5, 0.9],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}
gs = spark_sklearn.GridSearchCV(sc,estimator=RandomForestClassifier(), param_grid=parameters)
gs.fit(X_train, y_train.astype('int'))
tEnd_par = time.time()
print('Time required to do Parallel Grid Search CV: {} seconds'.format(tEnd_par - tStart_par))