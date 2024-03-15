from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from first import make_graph


if __name__ == "__main__":
    data = pd.read_csv("spam7.csv")

    X = data.drop(columns=["yesno"])
    y = data["yesno"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(X_train, y_train)
    tree.plot_tree(clf, fontsize=5)
    print(accuracy_score(y_test, clf.predict(X_test)))
    plt.show()