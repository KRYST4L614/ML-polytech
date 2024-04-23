from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense, Flatten
from keras.src.utils.module_utils import tensorflow
from matplotlib import pyplot as plt

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / X_train.max()
    X_test = X_test / X_test.max()
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(784, activation='relu'),
        Dense(10, activation='softmax')
    ])
    # Выведем полученную модель на экран
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=500, verbose=1,
                        epochs=20, validation_split=0.2)
    plt.plot(range(1, 21), history.history['val_accuracy'])
    plt.show()

