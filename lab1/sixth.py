import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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
    cm = confusion_matrix(y_test, clf.predict(X_test))

    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp_cm.plot()
    plt.show()

    kn_clf = KNeighborsClassifier(n_neighbors=round(pow(X_test.size, 0.5)))
    kn_clf.fit(X_train, y_train)
    print(f"{accuracy_score(y_test, kn_clf.predict(X_test))}")
    cm = confusion_matrix(y_test, kn_clf.predict(X_test))

    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp_cm.plot()
    plt.show()
