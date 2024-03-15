from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from first import make_graph

if __name__ == "__main__":
    data_train = pd.read_csv("bank_scoring_train.csv", delimiter='\t')

    X_train = data_train.drop(columns=["SeriousDlqin2yrs"])
    y_train = data_train["SeriousDlqin2yrs"]

    data_test = pd.read_csv("bank_scoring_test.csv", delimiter='\t')

    X_test = data_test.drop(columns=["SeriousDlqin2yrs"])
    y_test = data_test["SeriousDlqin2yrs"]

    clf = GaussianNB()
    clf = clf.fit(X_train, y_train)
    print(f"{accuracy_score(y_test, clf.predict(X_test))}")

    kn_clf = KNeighborsClassifier(n_neighbors=round(pow(X_test.size, 0.5)))
    kn_clf.fit(X_train, y_train)
    print(f"{accuracy_score(y_test, kn_clf.predict(X_test))}")

    # max_accuracy = -1
    # max_depth = 0
    # for i in range(1, 21):
    #     clf = CategoricalNB()
    #     clf = clf.fit(X_train, y_train)
    #     if max_accuracy < accuracy_score(y_test, clf.predict(X_test)):
    #         max_depth = i
    #         max_accuracy = accuracy_score(y_test, clf.predict(X_test))
    #     print(f"{i} {accuracy_score(y_test, clf.predict(X_test))}")
    # tree.plot_tree(clf, fontsize=8)
    # plt.show()