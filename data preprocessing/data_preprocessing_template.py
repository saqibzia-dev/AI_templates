"""# %%"""

#checking missing values
# handling categorical features(label encoding,onehot encoding)
# checking for imbalanced dataset
# checking for outliers using IQR
# Standardizing the data
#splitting the dataset into train and test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import impute

df = pd.read_csv('Data.csv')
print(df.head())  
print("describing the dataset")
print(df.describe())
# Outlier detection using IQR (do it after handling missing values )
# IQR = Q3-Q1
# upper_limit(max) = Q3+1.5*IQR
# lower_limit(min) = Q1-1.5*IQR
# for Age
q1_age = df['Age'].quantile(0.25)
q3_age = df['Age'].quantile(0.75)
IQR_age = q3_age - q1_age
max_age = q3_age + (1.5*IQR_age)
min_age = q1_age - (1.5*IQR_age)
#for Salary
q1_salary = df['Salary'].quantile(0.25)
q3_salary = df['Salary'].quantile(0.75)
IQR_salary = q3_salary - q1_salary
max_salary = q3_salary + (1.5*IQR_salary)
min_salary = q1_salary - (1.5*IQR_salary)
# printing outlier
print("Here are the outliers")
df_outliers = df[(df.Age<min_age) | 
                (df.Age>max_age) | (df.Salary<min_salary) | (df.Salary>max_salary) ]
print(df_outliers) # no outliers detected
# remove outliers if needed
df_no_outliers = df[(df.Age>=min_age)&(df.Age<=max_age)&
                    (df.Salary>=min_salary) & (df.Salary<= max_salary)
                    ]
import sys
sys.exit()
# checking for imbalanced dataset
print(df['Purchased'].value_counts())

# if not balanced then we can use these three approaches
# https://github.com/saqibzia-dev/deep-learning-keras-tf-tutorial/blob/master/14_imbalanced/handling_imbalanced_data.ipynb

class_0_count,class_1_count = df['Purchased'].value_counts()
#---------------down sampling
# down sample majority class(here dataset is balanced)
df_class_0 = df[df['Purchased'] == 'No']
df_class_1 = df[df['Purchased']  == 'Yes']
# suppose our class 0(no) is in minority then we will down sample majority class 1(yes)
df_class_1_down_sampled = df_class_1.sample(class_0_count)
df_balanced = pd.concat([df_class_0,df_class_1_down_sampled],axis = 0)
print('Random down-sampling:')
print(df_balanced.Purchased.value_counts()) 

########### Up Sampling  ######################
# up sampling minority class
class_0_count,class_1_count = df['Purchased'].value_counts()
# suppose class_0 is minority class then we will upsample it
df_class_0 = df[df['Purchased'] == 'No']
df_class_1 = df[df['Purchased'] == 'Yes']

df_class_0_upsampled = df.sample(class_1_count,replace = True)
df_balanced = pd.concat([df_class_1,df_class_0_upsampled])
print(df_balanced['Purchased'].value_counts())

##############SMOTE sampling ##############################
# To install imbalanced-learn library use pip install imbalanced-learn command
X = df.drop('Purchased',axis = 'columns')
y= df['Purchased']
##################
#checking missing values
print(df.isnull().sum())
print(df['Age'])

"""Dividing Dataset into independent and dependent variables"""
# print(dataset.loc[(dataset['Age']==np.nan)])
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy = "median")
#python data_preprocessing_template.py
X[:,1:3] = imputer.fit_transform(X[:,1:3])
#handling categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
y = LabelEncoder().fit_transform(y)
from sklearn.compose import ColumnTransformer
col_transformer = ColumnTransformer([("encoder",OneHotEncoder(),[0])],
                                    remainder = "passthrough")
X = col_transformer.fit_transform(X)
print(type(X)) 
###############
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy = "minority")
X_smote,y_smote = smote.fit_resample(X,y)
# print(y_smote.value_counts())


####################################

"""Train test split"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y,
                                                random_state = 15)

"""Standardizing the data"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
"""Dividing Dataset into independent and dependent variables"""
# X = df.iloc[:,:-1].values
# Y = df.iloc[:,-1].values

#checking missing values
# print(df.isnull().sum())
# print(df['Age'])
"""Dividing Dataset into independent and dependent variables"""
# print(dataset.loc[(dataset['Age']==np.nan)])
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.nan,strategy = "median")
# #python data_preprocessing_template.py
# X[:,1:3] = imputer.fit_transform(X[:,1:3])
# #handling categorical features
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# y = LabelEncoder().fit_transform(Y)
# from sklearn.compose import ColumnTransformer
# col_transformer = ColumnTransformer([("encoder",OneHotEncoder(),[0])],
#                                     remainder = "passthrough")
# X = col_transformer.fit_transform(X)
# print(type(X)) 





