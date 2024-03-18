import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

GLASS_DATA = "glass.csv"

if __name__ == "__main__":
    data = pd.read_csv(GLASS_DATA)
    data.drop(columns=["Id"], inplace=True)

    X = data.drop(columns=["Type"])
    y = data["Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    max_accuracy = -1
    max_depth = 0
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    tree.plot_tree(clf, fontsize=4)
    plt.show()
    print(clf.get_depth())
    for i in range(1, 21):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train, y_train)
        if max_accuracy < accuracy_score(y_test, clf.predict(X_test)):
            max_depth = i
            max_accuracy = accuracy_score(y_test, clf.predict(X_test))
        print(f"{i} {accuracy_score(y_test, clf.predict(X_test))}")
    print(f"{max_depth} {max_accuracy}")
    for i in range(2, 21):
        clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=i)
        clf = clf.fit(X_train, y_train)
        if max_accuracy < accuracy_score(y_test, clf.predict(X_test)):
            max_depth = i
            max_accuracy = accuracy_score(y_test, clf.predict(X_test))
        print(f"{i} {accuracy_score(y_test, clf.predict(X_test))}")
    print(f"{max_depth} {max_accuracy}")