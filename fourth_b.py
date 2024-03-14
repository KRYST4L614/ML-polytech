import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from first import make_graph
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

svmdata = pd.read_csv("svmdata_b.txt", delimiter='\t')
dumies_svmdata = pd.get_dummies(svmdata)
X = dumies_svmdata.iloc[:, 0:-2].values
y = dumies_svmdata.iloc[:, -2]

svmdata_test = pd.read_csv("svmdata_b_test.txt", delimiter='\t')
dumies_svmdata_test = pd.get_dummies(svmdata)
X_test = dumies_svmdata_test.iloc[:, 0:-2].values
y_test = dumies_svmdata_test.iloc[:, -2]

accuracy_train = [0]
for i in np.arange(1, 500.0, 1):
    clf = svm.LinearSVC(C=i, dual="auto", max_iter=10000000)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    accuracy_train.append(accuracy_score(y, y_pred))

make_graph(np.arange(0, 500.0, 1), accuracy_train, "a", "b", "a").xlim(-100, 500 + 0.1)
plt.xlim(-100, 500 + 0.1)
plt.ylim(0, max(accuracy_train) + 0.1)
plt.show()

accuracy_test = [0]
for i in np.arange(1, 500.0, 1):
    clf = svm.LinearSVC(C=i, dual="auto", max_iter=10000000)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    accuracy_test.append(accuracy_score(y_test, y_pred))

make_graph(np.arange(0, 500.0, 1), accuracy_test, "a", "b", "a").xlim(-100, 500 + 0.1)
plt.xlim(-100, 500 + 0.1)
plt.ylim(0, max(accuracy_test) + 0.1)
plt.show()