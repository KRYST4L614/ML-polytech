import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster
from pandas import read_csv

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

files = ["clustering_1.csv", "clustering_2.csv", "clustering_3.csv"]

def visualization(data, cluster, file_name, type_cluster):
    pca = PCA(2)
    df = pca.fit_transform(data)
    labels = cluster.labels_
    u_labels = np.unique(labels)

    for i in u_labels:
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
    plt.title("{0} {1}".format(file_name, type_cluster))
    plt.legend()
    plt.savefig("{0}_{1}_task2.png".format(file_name, type_cluster))
    plt.close()


print("""
####################################################
########## Without standartization
####################################################
""")

data = read_csv(files[0], delimiter='\t')
agg = AgglomerativeClustering(2).fit(data)
kmeans = KMeans(2).fit(data)
dbscan = DBSCAN().fit(data)
visualization(data, agg, files[0], "Иерархическая стандартизация")
visualization(data, kmeans, files[0], "kMeans")
visualization(data, dbscan, files[0], "DBSCAN")

data = read_csv(files[1], delimiter='\t')
agg = AgglomerativeClustering(3).fit(data)
kmeans = KMeans(3).fit(data)
dbscan = DBSCAN().fit(data)
visualization(data, agg, files[1], "Иерархическая стандартизация")
visualization(data, kmeans, files[1], "kMeans")
visualization(data, dbscan, files[1], "DBSCAN")

data = read_csv(files[2], delimiter='\t')
agg = AgglomerativeClustering(2).fit(data)
kmeans = KMeans(2).fit(data)
dbscan = DBSCAN().fit(data)
visualization(data, agg, files[2], "Иерархическая стандартизация")
visualization(data, kmeans, files[2], "kMeans")
visualization(data, dbscan, files[2], "DBSCAN")
