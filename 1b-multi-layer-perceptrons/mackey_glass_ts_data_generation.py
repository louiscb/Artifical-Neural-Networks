import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def create_time_series(stop=1500, beta=0.2, gamma=0.1, n=10, tau=25):
    ts = np.zeros(stop, dtype=float)
    ts[0] = 1.5
    # tau needs to be greater than 17 apparently
    # stop needs to be greater than tau
    for t in range(stop - 1):
        ts[t + 1] = ts[t] + (beta * ts[t - tau]) / (1 + ts[t - tau] ** n) - gamma * ts[t]
    return ts


def add_noise(ts, variance):
    noise_rv = norm(loc=0, scale=variance**2)
    ts += noise_rv.rvs(size=len(ts))


def plot_time_series(ts):
    plt.plot(list(range(len(ts))), ts)
    plt.show()


def create_data_sets(time_series, train_percentage, train_start=300):
    data_set = np.zeros((len(time_series) - 5 - train_start, 5))
    labels = np.zeros((len(data_set), 1))
    for i in range(len(data_set)):
        index = i + train_start
        data, label = get_input_and_output(time_series, index)
        data_set[i] = data
        labels[i] = label
    train_val_data, test_data = np.split(data_set, [len(data_set) - 200], axis=0)
    train_val_labels, test_labels = np.split(labels, [len(labels) - 200], axis=0)

    split_index = round(len(train_val_data) * train_percentage)
    train_data, val_data = np.split(train_val_data, [split_index], axis=0)
    train_labels, val_labels = np.split(train_val_labels, [split_index], axis=0)
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_input_and_output(time_series, index):
    data = np.zeros(5)
    for i, offset in enumerate(range(20, -5, -5)):
        data[i] = time_series[index - offset]
    label = time_series[index + 5]
    return data, label
