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



f = open("part1results/results3_3_CL_vs_no_CL.txt", 'w')
f.write('\nwith noise after 20 epochs, no. rbfs=40, width=0.1, learningrate=0.4\n')

network = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1)
network2 = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1)
x_train = np.arange(0, 2*np.pi, 0.1)

y_train_target = list(map(sin2, x_train))

x_train = add_noise(x_train, 0.1)

x_test = np.arange(0.05, 2*np.pi, 0.1)
y_test_target = list(map(sin2, x_test))

x_test = add_noise(x_test, 0.1)


MSEs= network.train_sequential_delta(x_train, y_train_target, CL_iterations=20)
MSEs2= network2.train_sequential_delta(x_train, y_train_target, CL_iterations=0)



mse = network.test_sequential_delta(x_test, y_test_target)
mse2 = network2.test_sequential_delta(x_test, y_test_target)

f.write('mses showing convergence NO CL:' + str(MSEs2) + '\n')
f.write('mses showing convergence with CL:' + str(MSEs) + '\n')

f.write('final mse on testset NO CL:' + str(mse2) + '\n')
f.write('final mse on testset with CL:' + str(mse) + '\n')


f.write('\nwithout noise after 20 epochs, no. rbfs=40, width=0.1, learningrate=0.4\n')

network3 = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1)
network4 = RBFNetwork(n_inputs=1, n_rbf=40, n_outputs=1)
x_train = np.arange(0, 2*np.pi, 0.1)

y_train_target = list(map(sin2, x_train))



x_test = np.arange(0.05, 2*np.pi, 0.1)
y_test_target = list(map(sin2, x_test))




MSEs= network3.train_sequential_delta(x_train, y_train_target, CL_iterations=20)
MSEs2= network4.train_sequential_delta(x_train, y_train_target, CL_iterations=0)



mse = network3.test_sequential_delta(x_test, y_test_target)
mse2 = network4.test_sequential_delta(x_test, y_test_target)

f.write('mses showing convergence NO CL:' + str(MSEs2) + '\n')
f.write('mses showing convergence with CL:' + str(MSEs) + '\n')

f.write('final mse on testset NO CL:' + str(mse2) + '\n')
f.write('final mse on testset with CL:' + str(mse) + '\n')





