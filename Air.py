# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:02:10 2019

@author: Shriyash Shende

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

a = pd.read_csv("C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Cluster\\EastWestAirlines.csv")
a_new = a.drop(["ID#"],axis=1)
a_new.info()
a_new.describe(include = 'all')
a_new.plot(kind='bar',stacked=True)

def Norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
a_norm = Norm_fun(a_new)

k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(a_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(a_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,a_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))                    
    
# Scree plot 
plt.plot(k,TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.xticks(k)
    
model = KMeans(n_clusters=5)
model.fit(a_norm)
model.labels_
md = pd.Series(model.labels_)
a['Membership'] = md
a.head()
a = a.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
a.groupby(a.Membership).mean()
a.to_csv("C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Cluster\\airlinesK_with_labels.csv",index=False)

