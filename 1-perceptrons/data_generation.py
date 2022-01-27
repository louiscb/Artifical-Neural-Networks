import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data_points(param1, param2, n):
    distr1 = multivariate_normal(param1[0], param1[1])
    distr2 = multivariate_normal(param2[0], param2[1])
    data1 = np.zeros((n, 2))
    data2 = np.zeros((n, 2))
    for i in range(n):
            data1[i] = distr1.rvs()
            data2[i] = distr2.rvs()
    return data1, data2

def visualize_data(data1, data2):
    plt.scatter(data1[:, 0], data1[:, 1], c='red')
    plt.scatter(data2[:, 0], data2[:, 1], c='blue')
    plt.show()
