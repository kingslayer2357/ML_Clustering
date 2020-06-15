# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:30:42 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Clustering\Anscombe.csv")
X=dataset.iloc[:,1:5].values

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X=pca.fit_transform(X)
var=pca.explained_variance_ratio_


#using elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.show()


#Applying KMeans
kmeans=KMeans(n_clusters=3)
y_pred=kmeans.fit_predict(X)


#plot
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],c="red")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],c="yellow")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],c="green")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="blue")
plt.show()