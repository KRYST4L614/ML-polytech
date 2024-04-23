import matplotlib.pyplot as plt
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

from lab2.utils import prepare_data


def calculate_accuracy(title):
    x_train, x_test, y_train, y_test = prepare_data(title)
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[2])])
    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer="SGD")
    accs = []
    for i in range(10, 101, 10):
        model.fit(x_train, y_train, epochs=i, verbose=0)
        _, acc = model.evaluate(x_test, y_test)
        accs.append(acc)
    return accs

plt.plot(range(10, 101, 10), calculate_accuracy("nn_0.csv"), label="nn_0")

plt.plot(range(10, 101, 10), calculate_accuracy("nn_1.csv"), label="nn_1")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(fontsize=16)
plt.show()

