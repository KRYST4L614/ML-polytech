import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


data = read_csv("pluton.csv")


def visualization(data, kmeans, std):
    pca = PCA(2)
    df = pca.fit_transform(data)
    labels = kmeans.labels_
    u_labels = np.unique(labels)
    print("Davies-Bouldin: ", davies_bouldin_score(data, labels))
    print("Silhouette score: ", silhouette_score(data, labels, metric='euclidean'))

    for i in u_labels:
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
    plt.title("KMeans, max_iter = {0}, std = {1}".format(kmeans.max_iter, std))
    plt.legend()
    plt.savefig("KMeans_max_iter={0}_std={1}.png".format(kmeans.max_iter, std))
    plt.close()


print("""
####################################################
########## Without standartization
####################################################
""")

for i in [1, 100, 1000]:
    kmeans = KMeans(n_clusters=3, max_iter=i)
    kmeans.fit(data)
    visualization(data, kmeans, False)
    print("#############################################")

scaler = StandardScaler()
scaler.fit(data)
scaled_features = scaler.transform(data)
scaled_data = pd.DataFrame(scaled_features, columns = data.columns)

print("""
####################################################
########## With standartization
####################################################
""")

for i in [1, 100, 1000]:
    kmeans = KMeans(n_clusters=3, max_iter=i)
    kmeans.fit(scaled_data)
    visualization(scaled_data, kmeans, True)
    print("#############################################")

