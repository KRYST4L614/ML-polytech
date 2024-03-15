from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from first import make_graph

if __name__ == "__main__":
    data_train = pd.read_csv("bank_scoring_train.csv")

    X_train = data_train.drop(columns=["SeriousDlqin2yrs"])
    y_train = data_train["SeriousDlqin2yrs"]

    data_test = pd.read_csv("bank_scoring_test.csv")

    X_test = data_train.drop(columns=["SeriousDlqin2yrs"])
    y_test = data_train["SeriousDlqin2yrs"]

    max_accuracy = -1
    max_depth = 0
    for i in range(1, 21):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train, y_train)
        if max_accuracy < accuracy_score(y_test, clf.predict(X_test)):
            max_depth = i
            max_accuracy = accuracy_score(y_test, clf.predict(X_test))
        print(f"{i} {accuracy_score(y_test, clf.predict(X_test))}")
    # tree.plot_tree(clf, fontsize=8)
    # plt.show()