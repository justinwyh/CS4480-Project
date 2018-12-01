
from sklearn.datasets import make_blobs

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import math
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
# # Applying PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_

#%%
class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=5):
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        print(self.n_features, "sha: ", x.shape[1])
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf = x, y, sample_sz, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_sz)), depth=self.depth, min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)


def std_agg(cnt, s1, s2): return math.sqrt((s2 / cnt) - (s1 / cnt) ** 2)


class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        print(f_idxs)
        #         print(self.depth)
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)
        #Parallel(n_jobs=3,require='sharedmem')(delayed(self.find_better_split)(i) for i in self.f_idxs)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1;
            rhs_cnt -= 1
            lhs_sum += yi;
            rhs_sum -= yi
            lhs_sum2 += yi ** 2;
            rhs_sum2 -= yi ** 2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self):
        return self.x.columns[self.var_idx]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)



#%%
tStart = time.time()
model = RandomForest(pd.DataFrame(X_train),y_train,50,'sqrt',40000)
tEnd = time.time()

#%%
preds = model.predict(X_test).round()
print("The accuracy is " + str((preds == y_test).mean()))
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds)

#%%
# Making classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))
