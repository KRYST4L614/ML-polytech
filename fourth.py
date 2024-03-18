import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

if __name__ == "__main__":
    svmdata = pd.read_csv("svmdata_a.txt", delimiter='\t')
    dumies_svmdata = pd.get_dummies(svmdata)
    X = dumies_svmdata.iloc[:, 0:-2].values
    y = dumies_svmdata.iloc[:, -2]

    C = 1.0  # SVM regularization parameter
    model = svm.LinearSVC(C=C)
    clf = model.fit(X, y)

    fig, sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = X[:, 0], X[:, 1]
    ax = sub
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel="X1",
        ylabel="X2",
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("SVC with linear kernel")
    plt.show()

    svmdata_test = pd.read_csv("svmdata_a_test.txt", delimiter='\t')
    dumies_svmdata_test = pd.get_dummies(svmdata)
    X_test = dumies_svmdata_test.iloc[:, 0:-2].values
    y_test = dumies_svmdata_test.iloc[:, -2]

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    cm = confusion_matrix(y_test, y_pred)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp_cm.plot()
    plt.show()

