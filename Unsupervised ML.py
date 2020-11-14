#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#load the iris data and use head function
df= pd.read_csv("D:/datasets/Iris.csv")
df.head(10)
#to know number of rows and columns
df.shape
#to find if any null value is present
df.isnull().sum()
#to see summary statistics
df.describe().T
#process data into attributes by selecting Iris dataset features into variable x
X= df.iloc[:, [0,1,2,3]].values
#implement kmeans clustering using k=5 i.e trail and error method to check the clustering
KMeans5= KMeans(n_clusters=5)
y_KMeans5= KMeans5.fit_predict(X)
print(y_KMeans5)

KMeans5.cluster_centers_
#To find the optimal number of clusters in the dataset using Elbow method
from sklearn.cluster import KMeans
cluster_range = range(1,20)
cluster_errors = []

for num_cluster in cluster_range:
    clusters = KMeans(num_cluster, n_init = 10)
    clusters.fit(X.data)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)
    
clusters_df = pd.DataFrame({'num_cluster': cluster_range, 'cluster_errors': cluster_errors})
clusters_df[0:20]
#plotting elbow curve to find the number of cluster
plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_cluster, clusters_df.cluster_errors, marker = 'o') 
plt.xlabel('Values of K') 
plt.ylabel('Error') 
plt.title('The Elbow Method using Distortion') 
plt.show()
#implement kmeans using k=3
KMeans3=KMeans(n_clusters=3)
y_KMeans3=KMeans3.fit_predict(X.data)
print(y_KMeans3)
KMeans3.cluster_centers_
#visualizing clustering
plt.scatter(X[:,0], X[:, 1], c=y_KMeans3,cmap='rainbow')
