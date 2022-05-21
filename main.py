# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:01:20 2022 GMT-3

@author: Jony
"""

import pandas as pd
import requests
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data1 = pd.read_csv('train1.csv', index_col=0) 
data2 = pd.read_csv('train2.csv', index_col=0, sep=';')
data3dict = requests.get('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first')\
    .json()
data4dict = requests.get('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second')\
    .json()
data5dict = requests.get('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third')\
    .json()
    
data3 = pd.DataFrame.from_dict(data3dict)
data4 = pd.DataFrame.from_dict(data4dict)
data5 = pd.DataFrame.from_dict(data5dict)

# VECTORIZAR ANNEXIMAINACTIVITYLABEL
features = ['eprtrSectorName', 'reportingYear','MONTH', 'avg_wind_speed', 'avg_temp', 'DAY WITH FOGS', 'CITY ID', 'pollutant']

print(data1.eprtrSectorName.unique())
print(data1.targetRelease.unique())

df1 = data1[features]
df2 = data2[features]
df3 = data3[features]
df4 = data4[features]
df5 = data5[features]
dfs = [df1, df2, df3, df4, df5]

df = pd.concat(dfs, ignore_index=True, axis=0)

#To comply with target encoding requirements
lb_make = LabelEncoder()
df["pollutant_enc"] = lb_make.fit_transform(df["pollutant"])
df['pollutant_enc'] = df['pollutant_enc'].replace([0,1,2],[1,2,0])

#Encode Sector Names
dummiesSector = pd.get_dummies(df['eprtrSectorName'])
df_enc = pd.concat([df, dummiesSector], axis=1)
df_enc = df_enc.drop(['eprtrSectorName', 'pollutant', 'CITY ID'], axis=1)
stats = df_enc.describe()
df_enc.isnull().sum() # Checks for missing values


y = df_enc['pollutant_enc']
X = df_enc.drop('pollutant_enc', axis=1)
     
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 10)}
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=35)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

test_x = pd.read_csv('test_x.csv', index_col=0)
test_x = test_x[['eprtrSectorName', 'reportingYear','MONTH', 'avg_wind_speed', 'avg_temp', 'DAY WITH FOGS', 'CITY ID']]
dummies = pd.get_dummies(test_x['eprtrSectorName'])
test_x = pd.concat([test_x, dummies], axis=1)
test_x = test_x.drop(['eprtrSectorName', 'CITY ID'], axis=1)

prediction = cv.predict(test_x)
predictions = pd.DataFrame(prediction, columns=['pollutant']).to_csv('predictions.csv')
predictionsjson = pd.DataFrame(predictions, columns=['pollutant']).to_json('predictions.json')
