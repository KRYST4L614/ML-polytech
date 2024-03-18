import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from first import make_graph

GLASS_DATA = "glass.csv"

if __name__ == "__main__":
    data = pd.read_csv(GLASS_DATA)
    data.drop(columns=["Id"], inplace=True)

    X = data.drop(columns=["Type"])
    y = data["Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    errors = []
    neighbors = np.arange(3, 100)

    for neighbor in neighbors:
        kn_clf = KNeighborsClassifier(n_neighbors=neighbor)
        kn_clf.fit(X_train, y_train)

        y_pred = kn_clf.predict(X_test)

        error = 1 - accuracy_score(y_test, y_pred)
        errors.append(error)

    make_graph(neighbors, errors, "Number of Neighbors", "Classification Error",
               "Dependence of Classification Error on Number of Neighbors").show()

    distance_metrics = ["euclidean", "chebyshev", "manhattan"]
    accuracy_results = {}

    for metric in distance_metrics:
        knn_model = KNeighborsClassifier(n_neighbors=6, metric=metric)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[metric] = accuracy

    for metric, accuracy in accuracy_results.items():
        print(f"{metric}: {accuracy}")

    data_frame = pd.DataFrame([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]], columns=X.columns)
    predicted_types = []

    for k in neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric="chebyshev")
        knn_model.fit(X, y)
        predicted_type = knn_model.predict(data_frame)
        predicted_types.append(predicted_type[0])

    make_graph(neighbors, predicted_types, "Number of Neighbors", "Predicted Glass Type",
               "Predicted Glass Type vs Number of Neighbors").show()