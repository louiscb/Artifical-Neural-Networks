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


f = open("part1results/results3_2_positioning_RBFs_effect.txt", 'w')
f.write('\nwe positioned them evenly over interval 0.05 to 2pi, after 20 epochs the mse on testset is\n')


network = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1, rbf_var=0.1 )
x_train = np.arange(0, 2*np.pi, 0.1)

y = np.zeros(x_train.shape)
y_train_target = list(map(sin2, x_train))

x_train = add_noise(x_train, 0.1)

x_test = np.arange(0.05, 2*np.pi, 0.1)


y_test_target = list(map(sin2, x_test))
x_test = add_noise(x_test, 0.1)


mses = network.train_sequential_delta(x_train, y_train_target)

mse = network.test_sequential_delta(x_test, y_test_target)

f.write('mse:' + str(mse) + '\n')



f.write('\nwhen positioned randomly over interval 0.05 to 2pi after 2p epochs the mse on testset is \n')

network2 = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1, rbf_var=0.1, rbf_positioning='random' )
x_train = np.arange(0, 2*np.pi, 0.1)

y = np.zeros(x_train.shape)
y_train_target = list(map(sin2, x_train))

x_train = add_noise(x_train, 0.1)

x_test = np.arange(0.05, 2*np.pi, 0.1)


y_test_target = list(map(sin2, x_test))
x_test = add_noise(x_test, 0.1)


mses = network2.train_sequential_delta(x_train, y_train_target)

mse = network2.test_sequential_delta(x_test, y_test_target)

f.write('mse:' + str(mse) + '\n')








