# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 02:22:19 2023

@author: tamer
"""

#1.libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.data preprocessing
#2.1.data loading
data = pd.read_csv('iris.csv')
#pd.read_csv("veriler.csv")


x = data.iloc[:,1:4].values #independent variables
y = data.iloc[:,4:].values #dependent variables

#splitting data for training and testing
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#scaling of data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# classification algorithms

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)


#confusion matrix
print("LOGISTIC")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# 2. KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#confusion matrix
print("KNN")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3. SVC
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(x_train,y_train)
          
y_pred = svc.predict(x_test)

#confusion matrix
print("SVC")
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

#confusion matrix
print("GNB")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 5. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)

#confusion matrix
print("DTC")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100,criterion="entropy")
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

#confusion matrix
print("RFC")
cm = confusion_matrix(y_test, y_pred)
print(cm)
