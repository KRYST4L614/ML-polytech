import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from first import make_graph
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

if __name__ == "__main__":
    svmdata = pd.read_csv("svmdata_d.txt", delimiter='\t')
    dumies_svmdata = pd.get_dummies(svmdata)
    X = dumies_svmdata.iloc[:, 0:-2].values
    y = dumies_svmdata.iloc[:, -2]

    C = 1.0  # SVM regularization parameter
    models = (
        svm.SVC(kernel="sigmoid", gamma="auto", C=C),
        svm.SVC(kernel="rbf", gamma="auto", C=C),
        svm.SVC(kernel="poly", degree=1, gamma="auto", C=C),
        svm.SVC(kernel="poly", degree=2, gamma="auto", C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
        svm.SVC(kernel="poly", degree=4, gamma="auto", C=C),
        svm.SVC(kernel="poly", degree=5, gamma="auto", C=C),
    )
    for model in models:
        for gamma in np.arange(1, 102, 50):
            model.gamma = gamma
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
            ax.set_title(f"{model.kernel} {model.degree} {gamma}")
            plt.show()
