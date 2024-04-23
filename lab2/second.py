import multiprocessing
from multiprocessing.pool import Pool

import numpy
import numpy as np
import pandas
import tensorflow as tf
from keras import Input
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split
from tensorflow import keras

from lab2.utils import prepare_data

optimizers = ["adagrad", "adadelta", "adam", "adamax", "nadam", "sgd"]


def calculate_accuracy(title, opt, activ):
    x_train, x_test, y_train, y_test = prepare_data(title)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=(2,), activation=activ)])
    model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer=opt)
    accs = []
    for i in range(10, 101, 10):
        model.fit(x_train, y_train, epochs=i, verbose=0)
        _, acc = model.evaluate(x_test, y_test)
        accs.append(acc)
    return accs

def calculate_accuracy_nn1(opt, activ1, activ2, activ3):
    x_train, x_test, y_train, y_test = prepare_data("nn_1.csv")
    model = tf.keras.Sequential([keras.layers.Dense(units=64, input_shape=(2,), activation=activ1),
                                keras.layers.Dense(units=32, input_shape=(2,), activation=activ2),
                                keras.layers.Dense(units=2, input_shape=(2,), activation=activ3)])
    model.compile(metrics=['accuracy'], loss='sparse_categorical_crossentropy', optimizer=opt)
    accs = []
    for i in range(10, 101, 10):
        model.fit(x_train, y_train, epochs=i, verbose=0)
        _, acc = model.evaluate(x_test, y_test)
        accs.append(acc)
    return accs

def work(opt):
    import matplotlib.pyplot as plt
    activators = ["elu", "softmax", 'selu', "softplus", "softsign", "relu", "sigmoid", "tanh"]
    plt.figure(figsize=(15, 9))
    for activ in activators:
        plt.plot(range(10, 101, 10), calculate_accuracy("nn_1.csv", opt, activ),
                 label="activ = {0}".format(activ))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(fontsize=16)
    plt.title("Opt={0}".format(opt))
    plt.savefig('bin_nn1_{0}_2'.format(opt))
    plt.close()
    return


#
#
if __name__ == '__main__':
    with multiprocessing.Pool(6) as pool:
        pool.map(work, optimizers)
    #
    # activ1 = ""
    # activ2 = ""
    # activ3 = ""
    # activators = ['selu', "relu", "sigmoid", "tanh"]
    # crnt = [0]
    # for i in activators:
    #     for k in activators:
    #         for j in activators:
    #             tmp = calculate_accuracy_nn1("adagrad", i, j, k)
    #             if tmp[-1] > crnt[-1]:
    #                 crnt = tmp
    #                 activ1 = i
    #                 activ2 = j
    #                 activ3 = k
    #
    # plt.plot(range(10, 101, 10), crnt, label="nn_1")
    # print(activ1)
    # print(activ2)
    # print(activ3)
    #
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(fontsize=16)
    # plt.show()


# for activ in activators:
#     plt.figure(figsize=(15, 9))
#     for opt in optimizers:
#         model = tf.keras.Sequential([keras.layers.Dense(units=64, input_shape=(2,), activation=activ),
#                                      keras.layers.Dense(units=64, input_shape=(2,), activation=activ),
#                                      keras.layers.Dense(units=1, input_shape=(2,), activation=activ)])
#         model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=opt)
#         df = pandas.read_csv("nn_1.csv")
#         x = df.iloc[:, :-1]
#         y = df.iloc[:, -1]
#         history = model.fit(x, y, validation_split=0.3, epochs=300)
#
#         xnew = numpy.linspace(1, len(history.history['val_accuracy']) + 1, 5)
#
#         # define spline
#         spl = make_interp_spline(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], k=3)
#         y_smooth = spl(xnew)
#
#         plt.plot(xnew, y_smooth,
#                  label="nn_0, opt = {0}".format(opt))
#         plt.xlabel("epoch")
#         plt.ylabel("accuracy")
#         plt.legend(fontsize=20)
#     plt.title("activ={0}".format(activ))
#     plt.savefig('{0}31_gladkoe'.format(activ))

# for activ in activators:
#     plt.figure(figsize=(15, 9))
#     for opt in optimizers:
#         model = tf.keras.Sequential([keras.layers.Dense(units=2, input_shape=(2,), activation=activ)])
#         model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer="adam")
#         df = pandas.read_csv("nn_1.csv")
#         x = df.iloc[:, :-1]
#         y = df.iloc[:, -1]
#         history = model.fit(x, y, epochs=300, validation_split=0.3)
#
#         plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label="nn_0, opt = {0}".format(opt))
#         plt.xlabel("epoch")
#         plt.ylabel("accuracy")
#         plt.legend(fontsize=20)
#     plt.title("activ={0}".format(activ))
#     plt.savefig('{0}experiment.png'.format(activ))
# df2 = pandas.read_csv("nn_1.csv")
# x2 = df2.iloc[:, :-1]
# y2 = df2.iloc[:, -1]
#
# model2 = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
# model2.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='sgd')
# history = model.fit(x, y, epochs=300)
# history2 = model2.fit(x2, y2, epochs=300)
#
# plt.plot(range(1, 301), history.history['accuracy'], label="nn_0")
# plt.plot(range(1, 301), history2.history['accuracy'], label="nn_1")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.legend(fontsize=20)
# plt.show()
