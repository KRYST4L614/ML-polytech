import pandas
from sklearn.model_selection import train_test_split


def prepare_data(title):
    np_array = pandas.read_csv(title).to_numpy()
    y = np_array[:, -1]
    y[y == -1] = 0
    y[y == 1] = 1
    y = y.astype('int')
    x = np_array[:, :-1]
    x = x.astype('float')
    return train_test_split(x, y, test_size=0.30)