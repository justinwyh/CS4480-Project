from sklearn.datasets import make_blobs

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time

#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

#%%

bank_data = pd.read_csv(filepath_or_buffer="bank-additional-full.csv", delimiter=';')
# bank_data = pd.read_csv(filepath_or_buffer="bank-additional.csv", delimiter=';')

#%%
# Drop rows if these 5 columns contains unknown job	marital	education
# default(has credit in default?)	housing(has housing loan?)	loan(has personal loan?)
bank_data = bank_data[(bank_data['job'] != 'unknown')
                      & (bank_data['marital'] != 'unknown')
                      & (bank_data['education'] != 'unknown')
                      & (bank_data['default'] != 'unknown')
                      & (bank_data['housing'] != 'unknown')
                      & (bank_data['loan'] != 'unknown')]

# bank_data.to_csv('bank-additional-full(without_Unknown).csv')


# visualization (understanding the data by parts)

#%%
# Data Preprocessing(categorical, binary, ordinal encoding)-----------------


# Ordinal encoding-- education
edu_mapping = {label:idx for idx, label in enumerate(['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
    'high.school',  'professional.course', 'university.degree'])}
print(edu_mapping)
bank_data['education']  = bank_data['education'].map(edu_mapping)

# Label encoding pdays
bank_data['pdays'] = (bank_data['pdays'] >998).astype(int)


# Label encoding y(independent variable)
bank_data['y'].replace(('yes', 'no'), (1, 0), inplace=True)

#%%
# One hot encoding and filling in missing values if missing values is present
cat_si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,handle_unknown='ignore'))
cat_steps = [cat_si_step, cat_ohe_step]
cat_pipe = Pipeline(cat_steps)
cat_cols = [1,2,4,5,6,7,8,9,14] #removed education
cat_transformers = [('cat', cat_pipe, cat_cols)]

# remainder should be passthrough so that the numerical columns also included in the result
ct = ColumnTransformer(transformers=cat_transformers,remainder='passthrough')

X_cat_transformed = ct.fit_transform(bank_data.iloc[:,:])
X_cat_transformed.shape

pl = ct.named_transformers_['cat']
ohe = pl.named_steps['ohe']
# showing the columns name after encoding
a = ohe.get_feature_names()
cat_col_names = a.tolist()


ncol_name = cat_col_names + ["age","education","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]
bank_data_final = pd.DataFrame(data=X_cat_transformed[:,:],columns=ncol_name)

#%%
# drop columns to prevent dummy variable trap
# we need to drop 'duration' as suggested in the readme.txt
bank_data_final.drop(['x0_unemployed','x1_single','x2_no','x3_no','x4_no','x5_telephone','x6_sep','x7_wed','x8_success'],axis = 1,inplace = True)

#%%
# visualize correlations between columns and ready for machine learning(without feature scaling)
# bank_data_final_corr = bank_data_final.corr()
# bank_data_final_corr['y']
#
# sns.heatmap(bank_data_final_corr, cmap='coolwarm', linecolor='white', linewidths=1)
# ax = plt.gca()
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
# ax.set_title('Correlation of the all features and output')

#%%
# Feature Engineering-----------------
X = bank_data_final.iloc[:, 0:42].values
y = bank_data_final.iloc[:, 42].values

# drop duration
X = np.delete(X, [33], axis=1)
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling for numerical attributes only
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, np.r_[31, 33:41]] = sc.fit_transform(X_train[:, np.r_[31, 33:41]])
X_test[:, np.r_[31, 33:41]] = sc.transform(X_test[:, np.r_[31, 33:41]])

#%%
# Solving imbalance output problem(accuracy paradox) by oversampling
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0)))


#%%
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# #%%
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_

#%%
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit_i(self,X,y,i):
        z = np.dot(X, self.theta)
        h = self.__sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        self.theta -= self.lr * gradient

        z = np.dot(X, self.theta)
        h = self.__sigmoid(z)
        loss = self.__loss(h, y)

        if (self.verbose == True and i % 10000 == 0):
            print(f'loss: {loss} \t')

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

            # weights initialization
            self.theta = np.zeros(X.shape[1])
            Parallel(n_jobs=1, require='sharedmem')(delayed(self.fit_i)(X, y, i) for i in range(self.num_iter))

        #for i in range(self.num_iter):
         #   self.fit_i(X,y,i)


    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round() #threshold = 0.5


    def score(self):
        preds = model.predict(X_test)
        return (preds == y_test).mean()

model = LogisticRegression(lr=0.1, num_iter=300000)
tStart = time.time()
model.fit(X_train,y_train)
tEnd = time.time()
print ("It costs " + str(tEnd - tStart) + "sec")
#%%
preds = model.predict(X_test)
print("The accuracy is " + str((preds == y_test).mean()))
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds)
model.theta

#%%
# Making classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

