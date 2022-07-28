from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle

url='https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
df=pd.read_csv(url, sep=';')

df2 = df.drop_duplicates(keep='first')
categorical=(['job', 'marital', 'education', 'default', 'housing', 'loan'])
for i in categorical:
    df[i] = df[i].replace(['unknown'],df[i].mode())
df.isin(['unknown']).any()

df.loc[df['y']=='yes','y'] = 1
df.loc[df['y']=='no','y'] = 0

df['Age'] = pd.cut(x=df['age'], bins=[10,20,30,40,50,60,70,80,90,100])

df= df[(df['age']>=25) & (df['age']<=47)]

encoder = LabelEncoder()

df['Age'] = encoder.fit_transform(df['Age'])
df['education'] = encoder.fit_transform(df['education'])

df = pd.get_dummies(df, columns = ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])

month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
df['month']= df['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
df['day_of_week']= df['day_of_week'].map(day_dict) 

scale= StandardScaler()
 
# separate the independent and dependent variables
edu = df[['education']]
Age=df[['Age']]
# standardization of dependent variables
education_st = scale.fit_transform(edu) 
df['education_st']=education_st
Age_st = scale.fit_transform(Age) 
df['Age_st']=Age_st
df

df['y']=df['y'].astype('int')

X = df[['Age_st','job_unemployed','marital_married','default_yes','housing_yes','loan_yes','education_st']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=423)

optimized_model = LogisticRegression(C= 100, penalty='l2', solver= 'newton-cg')

optimized_model.fit(X_train, y_train)

y_pred_ = optimized_model.predict(X_test)
y_pred_

accuracy_score(y_pred_, y_test)
confusion_matrix(y_pred, y_test)

pickle.dump(optimized_model, open('../models/best_model_logistic.pickle', 'wb'))
