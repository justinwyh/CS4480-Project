import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic


bank_data = pd.read_csv(filepath_or_buffer="bank-additional-full.csv", delimiter=';')
# bank_data = pd.read_csv(filepath_or_buffer="bank-additional.csv", delimiter=';')



# Drop rows if these 5 columns contains unknown job	marital	education
# default(has credit in default?)	housing(has housing loan?)	loan(has personal loan?)
bank_data = bank_data[(bank_data['job'] != 'unknown') 
                      & (bank_data['marital'] != 'unknown') 
                      & (bank_data['education'] != 'unknown')
                      & (bank_data['default'] != 'unknown')
                      & (bank_data['housing'] != 'unknown')
                      & (bank_data['loan'] != 'unknown')] 

#bank_data.to_csv('bank-additional-full(without_Unknown).csv')


#visualization (understanding the data by parts)

#
plt.interactive(False)
g = sns.FacetGrid(bank_data, col='marital' ,row='y')
g = g.map(sns.distplot, 'age', bins=30)
plt.subplots_adjust(top=0.4)
# g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Histogram of Age by Y and marital')
plt.show()

#
sns.boxplot(x='marital', y='age', hue='y', data=bank_data, palette='coolwarm',fliersize=0.2)
ax = plt.gca()
ax.set_title('Boxplot of Age by Y and marital')
ax.legend(loc = 2)
ax.get_ylim()

#
mosaic(bank_data, ['job','y'], gap=0.001, label_rotation=30)
ax = plt.gca()
ax.set_title('Mosaic plot of job by y')


#
mosaic(bank_data, ['housing','loan'], gap=0.001, title='Mosaic plot of housing(x) and loan(y)')
ax = plt.gca()
ax.set_title('Mosaic plot of housing(x) and loan(y)')


#
sns.countplot(x='education', data=bank_data, hue='y')
ax = plt.gca()
ax.set_title('Mosaic plot of housing(x) and loan(y)')

#
sns.countplot(x='marital', data=bank_data, hue='y')
ax = plt.gca()
ax.set_title('Count plot of marital and y')

#
sns.countplot(x='job', data=bank_data, hue='y')
ax = plt.gca()
ax.set_title('Count plot of job and y')


#
sns.violinplot(x="contact", y="age", data=bank_data ,hue='y',split=True,palette='Set1')
ax = plt.gca()
ax.set_title('violin plot of contact by age and y')



#
mosaic(bank_data, ['month','day_of_week','y'], gap=0.001, label_rotation=30, title='Mosaic plot of month(x) and day_of_week(y) by y')
ax = plt.gca()
ax.set_title('Mosaic plot of month(x) and day_of_week(y) by y')


#
sns.lmplot(x='duration',y='age',data=bank_data ,hue='y')
ax = plt.gca()
ax.set_title('Regression plot of age against duration by y')



#
sns.catplot(x="previous", col="y",hue='poutcome' ,data=bank_data, kind="count")
ax = plt.gca()
ax.set_title('Count plot of previous of different poutcome and separated by y (count limited to 3000)\n y=yes')
ax.set_ylim(0,3000)


#
sns.heatmap(bank_data.iloc[:,15:20].corr(),cmap='coolwarm',annot=True)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_title('HeatMap of emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed')
plt.show()

#
g = sns.pairplot(bank_data.iloc[:,15:20].corr())
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Pair plot of emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed')





#Data Preprocessing


#Label encoding Ordinal feature -- education
edu_mapping = {label:idx for idx, label in enumerate(['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
    'high.school',  'professional.course', 'university.degree'])}
print(edu_mapping)
bank_data['education']  = bank_data['education'].map(edu_mapping)

#Label encoding pdays
bank_data['pdays'] = (bank_data['pdays'] >998).astype(int)

#Label encoding y(independent variable)
bank_data['y'].replace(('yes', 'no'), (1, 0), inplace=True)



#One hot encoding and filling in missing values if missing values is present
cat_si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,handle_unknown='ignore'))
cat_steps = [cat_si_step, cat_ohe_step]
cat_pipe = Pipeline(cat_steps)
cat_cols = [1,2,4,5,6,7,8,9,14] #removed education
cat_transformers = [('cat', cat_pipe, cat_cols)]

#remainder should be passthrough so that the numerical columns also included in the result
ct = ColumnTransformer(transformers=cat_transformers,remainder='passthrough')

X_cat_transformed = ct.fit_transform(bank_data.iloc[:,:])
X_cat_transformed.shape

pl = ct.named_transformers_['cat']
ohe = pl.named_steps['ohe']
a = ohe.get_feature_names()
cat_col_names = a.tolist()




#one_hot_encoded_col = ['x0_admin.','x0_blue-collar','x0_entrepreneur','x0_housemaid','x0_management','x0_retired','x0_self-employed','x0_services','x0_student',
# 'x0_technician','x0_unemployed','x0_unknown','x1_divorced','x1_married','x1_single','x1_unknown','x2_basic.4y','x2_basic.6y','x2_basic.9y',
# 'x2_high.school','x2_illiterate','x2_professional.course','x2_university.degree','x2_unknown','x3_no','x3_unknown','x3_yes',
# 'x4_no','x4_unknown','x4_yes','x5_no','x5_unknown','x5_yes','x6_cellular','x6_telephone','x7_apr','x7_aug','x7_dec','x7_jul',
# 'x7_jun','x7_mar','x7_may','x7_nov','x7_oct','x7_sep','x8_fri','x8_mon','x8_thu','x8_tue','x8_wed','x9_failure','x9_nonexistent','x9_success']
ncol_name = cat_col_names + ["age","education","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"] 
bank_data_final = pd.DataFrame(data=X_cat_transformed[:,:],columns=ncol_name)




#drop columns to prevent dummy variable trap
#we need to drop 'duration' as suggested in the readme.txt
bank_data_final.drop(['x0_unemployed','x1_single','x2_no','x3_no','x4_no','x5_telephone','x6_sep','x7_wed','x8_success'],axis = 1,inplace = True)




#visualize correlations between columns and ready for machine learning
bank_data_final_corr = bank_data_final.corr()
bank_data_final_corr['y']

sns.heatmap(bank_data_final_corr, cmap='coolwarm', linecolor='white', linewidths=1)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_title('Correlation of the all features and output')

















