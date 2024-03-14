import numpy
import pandas
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

TIC_TAC_TOE = "tic_tac_toe.txt"
SPAM = "spam.csv"
PLT_X_LABEL = "Train proportion"
PLT_Y_LABEL = "Accuracy"


def calculate_accuracy(X, y, model):
    accuracy = []
    ratios = numpy.arange(0.1, 1.0, 0.1)
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=10)
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        accuracy.append(test_acc)
    return ratios, accuracy


def make_graph(x, y, x_label, y_label, title):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0, max(x) + 0.1)
    plt.ylim(0, max(y) + 0.1)
    plt.plot(x, y)
    plt.grid(True)
    return plt


if __name__ == '__main__':
    data_tic_tac = pandas.read_csv(TIC_TAC_TOE)
    dummies_tic_tac = pandas.get_dummies(data_tic_tac.iloc[:, :-1])
    X_tic_tac = dummies_tic_tac.values
    y_tic_tac = data_tic_tac.iloc[:, -1]
    tic_tac_toe_ratios, tic_tac_toe_test_accuracy = calculate_accuracy(X_tic_tac, y_tic_tac, CategoricalNB())
    make_graph(tic_tac_toe_ratios, tic_tac_toe_test_accuracy, PLT_X_LABEL, PLT_Y_LABEL, TIC_TAC_TOE).show()

    data_spam = pandas.read_csv(SPAM)
    X_spam = data_spam.iloc[:, :-1]
    y_spam = data_spam.iloc[:, -1]
    spam_ratios, spam_accuracy = calculate_accuracy(X_spam, y_spam, GaussianNB())
    make_graph(spam_ratios, spam_accuracy, PLT_X_LABEL, PLT_Y_LABEL, SPAM).show()
