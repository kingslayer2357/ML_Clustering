# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:27:49 2020

@author: kingslayer
"""

#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Clustering\LosingSleep.csv")
X=dataset.iloc[:,2:4].values


#Elbow Method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

#Applying Kmeans
kmeans=KMeans(n_clusters=2)
y_pred=kmeans.fit_predict(X)

#plot
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],c="red")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],c="yellow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="blue")
plt.show()