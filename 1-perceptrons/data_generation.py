import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data_points(param1, param2, n):
    distr1 = multivariate_normal(param1[0], param1[1])
    distr2 = multivariate_normal(param2[0], param2[1])
    data = np.zeros((n, 2))
    for i in range(n):
        if i % 2 == 0:
            data[i] = distr1.rvs()
        else:
            data[i] = distr2.rvs()
    return data

def visualize_data(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
