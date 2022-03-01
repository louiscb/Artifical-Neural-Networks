import numpy as np
from rbf_net import RBFNetwork
import matplotlib.pyplot as plt

def sin2(x):
    return np.sin(2 * x)

def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1


#f = open("part1results/results3_1.txt", 'w')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100,150,200])
mses=np.zeros(no_of_units.shape)


print("sin(2x) function:\n")
for i in range(len(no_of_units)):

    network = RBFNetwork(n_inputs=1, n_rbf= no_of_units[i], n_outputs=1)
    x_train = np.arange(0, 2*np.pi, 0.1)
    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_test = np.arange(0.05, 2*np.pi, 0.1)
    y_test_target = list(map(sin2, x_test))



    network.train_leastsquaresbatch(x_train, y_train_target)

    mse= network.test_leastsquaresbatch(x_test, y_test_target)
    mses[i] = mse
    
    print("no. units: " + str(no_of_units[i]) + "  mse: " + str(mses[i]) + '\n')







no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50])
mses=np.zeros(no_of_units.shape)

print("\nsquare(2x) function:\n")
for i in range(len(no_of_units)):

    network2 = RBFNetwork(n_inputs=1, n_rbf= no_of_units[i], n_outputs=1)
    x_train = np.arange(0, 2*np.pi, 0.1)
    y = np.zeros(x_train.shape)
    y_train_target = list(map(square, x_train))

    x_test = np.arange(0.05, 2*np.pi, 0.1)
    y_test_target = list(map(square, x_test))



    network2.train_leastsquaresbatch(x_train, y_train_target)

    mse= network2.test_leastsquaresbatch(x_test, y_test_target)
    mses[i] = mse
    
    print("no. units: " + str(no_of_units[i]) + "  mse: " + str(mses[i]) + '\n')










