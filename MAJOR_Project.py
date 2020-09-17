# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:41:22 2020

@author: dell-pc
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

train = pd.read_csv("Data_Train_Final.csv")
test = pd.read_csv("Data_Test_Final.csv")

#description
description_train = train.describe()
description_test = test.describe()

#null values
null_train = train.isnull().sum()
null_test = test.isnull().sum()

#preprocessing - Categorical Values

#extracting the Brand and the Model  
make_train = train["Name"].str.split(" ", expand = True)
make_test = test["Name"].str.split(" ", expand = True)

train["Brand"] = make_train[0]
test["Brand"] = make_test[0]

train["Model"] = make_train[1]
test["Model"] = make_test[1]

#droping full name of car
train.drop(['Name'], axis=1, inplace = True)
test.drop(['Name'], axis=1, inplace = True)

#inserting on the 1 and 2 position
train.insert(0,'Brand', train.pop("Brand"))
test.insert(0,'Brand', test.pop("Brand"))
train.insert(1,'Model', train.pop("Model"))
test.insert(1,'Model', test.pop("Model"))

#Calculating the car age
train["Age"] = 2020 - train["Year"]
test["Age"] = 2020 - test["Year"]

#droping the years column
train.drop(['Year'], axis=1, inplace = True)
test.drop(['Year'], axis=1, inplace = True)

#inserting year column to 4 
train.insert(3, "Age", train.pop("Age"))
test.insert(3, "Age", test.pop("Age"))

#preprocessing - Numerical Values
#outliers is present in the Kilometers_Driven column


train.drop(train[train.Kilometers_Driven>100000].index,axis=0,inplace=True)

#extracting the mileage value out of the mileage 
train_mileage = train["Mileage"].str.split(" ", expand = True)
test_mileage = test["Mileage"].str.split(" ", expand = True) 

train["Mileage"] = train_mileage[0]
test["Mileage"] = test_mileage[0]

#converting that mileage from string to float
train["Mileage"] = pd.to_numeric(train["Mileage"], downcast="float")
test["Mileage"] = pd.to_numeric(test["Mileage"], downcast="float")


#dealing with the missing values
train["Mileage"].fillna(train["Mileage"].astype("float64").median(), inplace = True)
test["Mileage"].fillna(test["Mileage"].astype("float64").median(), inplace = True)

#checking for outliers and eliminating the outliers
sns.boxplot("Mileage", data = train)
train.drop(train[train['Mileage']==0.0].index, axis=0 , inplace=True)
sns.boxplot("Mileage", data = train, color = "green")

#applying the above steps on Engine and Power 
train['Engine'] = train['Engine'].apply(lambda x : str(x).split(" ")[0]).astype("float64")
test['Engine'] = test['Engine'].apply(lambda x : str(x).split(" ")[0]).astype("float64")
train["Engine"] = pd.to_numeric(train['Engine'], errors = 'coerce')
test["Engine"] = pd.to_numeric(test['Engine'], errors = 'coerce')
train["Engine"].fillna(train["Engine"].astype("float64").mean(), inplace = True)
test["Engine"].fillna(test["Engine"].astype("float64").mean(), inplace = True)

train['Power'] = train['Power'].replace('null bhp','0 bhp')
test['Power'] = test['Power'].replace('null bhp','0 bhp')
train['Power'] = train['Power'].apply(lambda x : str(x).split(" ")[0]).astype("float64")
test['Power'] = test['Power'].apply(lambda x : str(x).split(" ")[0]).astype("float64")
train["Power"] = pd.to_numeric(train['Power'], errors = 'coerce')
test["Power"] = pd.to_numeric(test['Power'], errors = 'coerce')
train["Power"].fillna(train["Power"].astype("float64").mean(), inplace = True)
test["Power"].fillna(test["Power"].astype("float64").mean(), inplace = True)
train["Power"].replace(0,train["Power"].mean(),inplace=True)
test["Power"].replace(0,test["Power"].mean(),inplace=True)

#seats(Null Values)
train["Seats"].fillna(train["Seats"].astype("float64").mean(), inplace = True)
test["Seats"].fillna(test["Seats"].astype("float64").mean(), inplace = True)

#outliers in the Price column
train.drop(train[train['Price']>100].index, axis=0 , inplace=True)



 
                   #MODEL
train['Brand'] = train['Brand'].astype('category')
train['Fuel_Type'] = train['Fuel_Type'].astype('category')
train['Transmission'] = train['Transmission'].astype('category')
train['Owner_Type'] = train['Owner_Type'].astype('category')
train['Model'] = train['Model'].astype('category')
train['Location'] = train['Location'].astype('category')

test['Brand'] = test['Brand'].astype('category')
test['Fuel_Type'] = test['Fuel_Type'].astype('category')
test['Transmission'] = test['Transmission'].astype('category')
test['Owner_Type'] = test['Owner_Type'].astype('category')
test['Model'] = test['Model'].astype('category')
test['Location'] = test['Location'].astype('category')

X = train.drop(['Price'], axis = 1)
Y = train.Price

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['Brand'] = label_encoder.fit_transform(X['Brand'])
X['Fuel_Type'] = label_encoder.fit_transform(X['Fuel_Type'])
X['Transmission'] = label_encoder.fit_transform(X['Transmission'])
X['Mileage'] = label_encoder.fit_transform(X['Mileage'])
X['Owner_Type'] = label_encoder.fit_transform(X['Owner_Type'])
X['Model'] = label_encoder.fit_transform(X['Model'])
X['Location'] = label_encoder.fit_transform(X['Location'])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

#applying the random forest Algorithm
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
Regressor.fit(X_train, Y_train)

#Preictions
y_pred = Regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score = r2_score(Y_test, y_pred)

#______________________________________________________________________________________

#testset

testX = test.copy(deep=True)
#label_encoder = LabelEncoder()
testX['Brand'] = label_encoder.fit_transform(testX['Brand'])
testX['Model'] = label_encoder.fit_transform(testX['Model'])
testX['Location'] = label_encoder.fit_transform(testX['Location'])
testX['Fuel_Type'] = label_encoder.fit_transform(testX['Fuel_Type'])
testX['Transmission'] = label_encoder.fit_transform(testX['Transmission'])
testX['Mileage'] = label_encoder.fit_transform(testX['Mileage'])
testX['Owner_Type'] = label_encoder.fit_transform(testX['Owner_Type'])

pred_Y = Regressor.predict(testX)

test['Predicted Price'] = pred_Y




