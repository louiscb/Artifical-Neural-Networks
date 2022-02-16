import numpy as np


def create_time_series(stop=1500, beta=0.2, gamma=0.1, n=10, tau=25):
    ts = np.zeros(stop, dtype=float)
    ts[0] = 1.5
    # tau needs to be greater than 17 apparently
    # stop needs to be greater than tau
    for t in range(stop - 1):
        ts[t + 1] = ts[t] + (beta * ts[t - tau]) / (1 + ts[t - tau] ** n) - gamma * ts[t]
    return ts


def create_data_sets(time_series, train_start):
    data_set = np.zeros((len(time_series) - 5 - train_start, 5))
    labels = np.zeros(len(data_set))
    for i in range(len(data_set)):
        index = i + train_start
        data, label = get_input_and_output(time_series, index)
        data_set[i] = data
        labels[i] = label


def get_input_and_output(time_series, index):
    data = np.zeros(5)
    for i, offset in enumerate(range(20, -5, -5)):
        data[i] = time_series[index - offset]
    label = time_series[index + 5]
    return data, label