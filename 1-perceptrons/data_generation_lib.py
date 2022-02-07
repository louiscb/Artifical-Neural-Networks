import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data_points(param1, param2, n, class1_removal=None, class2_removal=None):
    distr1 = multivariate_normal(param1[0], param1[1])
    distr2 = multivariate_normal(param2[0], param2[1])
    data1 = np.zeros((n, 2))
    data2 = np.zeros((n, 2))
    for i in range(n):
            data1[i] = distr1.rvs()
            data2[i] = distr2.rvs()
    if class1_removal is not None:
        class1_adjusted_len = n - round(n * class1_removal)
        data1 = data1[:class1_adjusted_len]
    if class2_removal is not None:
        class2_adjusted_len = n - round(n * class2_removal)
        data2 = data2[:class2_adjusted_len]
    return data1, data2

def remove_from_subset(d1):
    count1 = 0
    count2 = 0
    for d in d1:
        if d[0] < 0:
            count1 += 1
        elif d[0] > 0:
            count2 += 1
    to_remove1 = round(count1 * 0.2)
    to_remove2 = round(count2 * 0.8)
    deleted1 = 0
    deleted2 = 0
    i = 0
    while deleted1 < to_remove1 or deleted2 < to_remove2:
        if d1[i][0] < 0 and deleted1 < to_remove1:
            np.delete(d1, i)
            deleted1 += 1
            continue

        elif d1[i][0] > 0 and deleted2 < to_remove2:
            np.delete(d1, i)
            deleted2 += 1
            continue
        i += 1

def visualize_data(data1, data2):
    plt.scatter(data1[:, 0], data1[:, 1], c='red')
    plt.scatter(data2[:, 0], data2[:, 1], c='blue')
    plt.show()

def concatenate_and_shuffle(d1, d2, class1, class2):
    concatenated = np.concatenate((d1, d2), axis=0)
    concatenated = concatenated.tolist()
    labels = [class1] * d1.shape[0] + [class2] * d2.shape[0]
    data = list(zip(concatenated, labels))
    random.shuffle(data)
    data, labels = list(zip(*data))
    return np.array(data), np.array(labels)

def add_bias(data_set):
    N = data_set.shape[0]
    biases = np.ones((N, 1))
    data_set = np.concatenate((data_set, biases), axis=1)
    return data_set