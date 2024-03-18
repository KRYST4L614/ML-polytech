import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

numpy.random.seed(0)
class Data:
    def __init__(self, mat_expect, cov, size):
        self.mat_expect = mat_expect
        self.cov = cov
        self.size = size
        self.dataset = numpy.random.multivariate_normal(mat_expect, cov, size)


if __name__ == "__main__":
    class1 = Data([18, 15], [[9, 0], [0, 9]], 20)
    class2 = Data([17, 16], [[1, 0], [0, 1]], 80)
    X = numpy.vstack((class1.dataset, class2.dataset))
    y = numpy.hstack((numpy.full(class1.size, -1), numpy.full(class2.size, 1)))

    plt.figure(figsize=(10, 6))
    plt.scatter(class1.dataset[:, 0], class1.dataset[:, 1], c='r', label='Class -1')
    plt.scatter(class2.dataset[:, 0], class2.dataset[:, 1], c='b', label='Class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    cm = confusion_matrix(y_test, y_pred)

    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
    disp_cm.plot()
    plt.show()

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    disp_roc_auc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='Naive Bayes')
    disp_roc_auc.plot()
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.show()

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)

    disp_precision_recall = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp_precision_recall.plot()
    plt.show()
