import numpy as np
from rbf_net import RBFNetwork
from scipy.stats import norm

def sin2(x):
    return np.sin(2 * x)

def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1

def add_noise(data, variance):
    noise_rv = norm(loc=0, scale=variance**2)
    data += noise_rv.rvs(size=len(data))
    return data


f = open("part1results/results3_2.txt", 'w')
f.write('\nwidth=0.1\n')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100])
mses=np.zeros(no_of_units.shape)


for i in range(len(no_of_units)):
    network = RBFNetwork(n_inputs=1, n_rbf=no_of_units[i], n_outputs=1, rbf_var=0.2)
    x_train = np.arange(0, 2*np.pi, 0.1)

    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_train = add_noise(x_train, 0.1)

    x_test = np.arange(0.05, 2*np.pi, 0.1)


    y_test_target = list(map(sin2, x_test))
    x_test = add_noise(x_test, 0.1)


    network.train_sequential_delta(x_train, y_train_target)

    mse = network.test_sequential_delta(x_test, y_test_target)
    mses[i] = mse
    f.write('no. of units: '+ str(no_of_units[i]) + '  mse: ' + str(mses[i]) + '\n')



f.write('\nwidth=0.2\n')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100])
mses=np.zeros(no_of_units.shape)


for i in range(len(no_of_units)):
    network2 = RBFNetwork(n_inputs=1, n_rbf=no_of_units[i], n_outputs=1, rbf_var=0.2)
    x_train = np.arange(0, 2*np.pi, 0.1)

    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_train = add_noise(x_train, 0.1)

    x_test = np.arange(0.05, 2*np.pi, 0.1)


    y_test_target = list(map(sin2, x_test))
    x_test = add_noise(x_test, 0.1)


    network2.train_sequential_delta(x_train, y_train_target)

    mse = network2.test_sequential_delta(x_test, y_test_target)
    mses[i] = mse
    f.write('no. of units: '+ str(no_of_units[i]) + '  mse: ' + str(mses[i]) + '\n')


f.write('\nwidth=0.3\n')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100])
mses=np.zeros(no_of_units.shape)


for i in range(len(no_of_units)):
    network3 = RBFNetwork(n_inputs=1, n_rbf=no_of_units[i], n_outputs=1, rbf_var=0.3)
    x_train = np.arange(0, 2*np.pi, 0.1)

    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_train = add_noise(x_train, 0.1)

    x_test = np.arange(0.05, 2*np.pi, 0.1)


    y_test_target = list(map(sin2, x_test))
    x_test = add_noise(x_test, 0.1)


    network3.train_sequential_delta(x_train, y_train_target)

    mse = network3.test_sequential_delta(x_test, y_test_target)
    mses[i] = mse
    f.write('no. of units: '+ str(no_of_units[i]) + '  mse: ' + str(mses[i]) + '\n')



f.write('\nwidth=0.4\n')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100])
mses=np.zeros(no_of_units.shape)


for i in range(len(no_of_units)):
    network4 = RBFNetwork(n_inputs=1, n_rbf=no_of_units[i], n_outputs=1, rbf_var=0.4)
    x_train = np.arange(0, 2*np.pi, 0.1)

    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_train = add_noise(x_train, 0.1)

    x_test = np.arange(0.05, 2*np.pi, 0.1)


    y_test_target = list(map(sin2, x_test))
    x_test = add_noise(x_test, 0.1)


    network4.train_sequential_delta(x_train, y_train_target)

    mse = network4.test_sequential_delta(x_test, y_test_target)
    mses[i] = mse
    f.write('no. of units: '+ str(no_of_units[i]) + '  mse: ' + str(mses[i]) + '\n')


f.write('\nwidth=0.6\n')
no_of_units = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100])
mses=np.zeros(no_of_units.shape)


for i in range(len(no_of_units)):
    network5 = RBFNetwork(n_inputs=1, n_rbf=no_of_units[i], n_outputs=1, rbf_var=0.6)
    x_train = np.arange(0, 2*np.pi, 0.1)

    y = np.zeros(x_train.shape)
    y_train_target = list(map(sin2, x_train))

    x_train = add_noise(x_train, 0.1)

    x_test = np.arange(0.05, 2*np.pi, 0.1)


    y_test_target = list(map(sin2, x_test))
    x_test = add_noise(x_test, 0.1)


    network5.train_sequential_delta(x_train, y_train_target)

    mse = network5.test_sequential_delta(x_test, y_test_target)
    mses[i] = mse
    f.write('no. of units: '+ str(no_of_units[i]) + '  mse: ' + str(mses[i]) + '\n')








