import matplotlib.pyplot as plt
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_size, output_dimensions, training_set, learning_rate, alpha, val_set):
        self.h_out = None
        self.h_in = None
        self.o_in = None
        self.o_out = None
        self.delta_v = None
        self.delta_w = None
        self.input_dimensions = input_dimensions
        self.hidden_layer_size = hidden_layer_size
        self.output_dimensions = output_dimensions
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.val_set= val_set
        self.w = np.random.uniform(-1, 1, (input_dimensions, hidden_layer_size))
        self.v = np.random.uniform(-1, 1, (hidden_layer_size + 1, output_dimensions))

    def forward_pass(self, validation_set=None):
        if validation_set is None:
            self.h_in = np.matmul(self.training_set, self.w)
            self.h_in = self.add_bias_row(self.h_in)
            self.h_out = self.phi(self.h_in)
            self.o_in = np.matmul(self.h_out, self.v)
            self.o_out = self.phi(self.o_in)
            return self.o_out
        else:
            h_in = np.matmul(validation_set, self.w)
            h_in = self.add_bias_row(h_in)
            h_out = self.phi(h_in)
            o_in = np.matmul(h_out, self.v)
            o_out = self.phi(o_in)
            return o_out

    def forward_pass_val(self):
        hin = np.matmul(self.val_set, self.w)
        hin = self.add_bias_row(hin)
        hout = self.phi(hin)
        oin = np.matmul(hout, self.v)
        oout = self.phi(oin)
        return oout

    def add_bias_row(self, data):
        N = data.shape[0]
        bias_row = np.ones((N, 1))
        return np.concatenate((data, bias_row), axis=1)

    def phi(self, data):
        return (2 / (np.exp(-1 * data) + 1)) - 1

    def backward_pass(self, labels):
        labels = np.reshape(labels, (len(labels), 1))
        delta_o = self.o_out - labels
        delta_o = delta_o * self.phi_prime(self.o_in)
        delta_h = np.matmul(delta_o, self.v.T) * self.phi_prime(self.h_in)
        # remove bias row
        delta_h = np.delete(delta_h, delta_h.shape[1] - 1, axis=1)

        # weight update
        if self.delta_w is None:
            self.delta_w = -self.learning_rate * np.matmul(self.training_set.T, delta_h)
            self.delta_v = -self.learning_rate * np.matmul(self.h_out.T, delta_o)
        else:
            self.delta_w = self.learning_rate * (self.delta_w * self.alpha) - (1 - self.alpha) * np.matmul(
                self.training_set.T, delta_h)
            self.delta_v = self.learning_rate * (self.delta_v * self.alpha) - (1 - self.alpha) * np.matmul(self.h_out.T,
                                                                                                           delta_o)
        self.w = self.w + self.delta_w
        self.v = self.v + self.delta_v

    def phi_prime(self, input):
        return 0.5 * (1 + self.phi(input)) * (1 - self.phi(input))

    def mse(self, target):
        return np.mean((self.o_out - target) ** 2)

    def visualize_function_approx(self):
        N = 21
        x = np.linspace(-5.0, 5.0, num=N).reshape((N, 1))
        y = np.linspace(-5.0, 5.0, num=N).reshape((N, 1))
        zz = np.reshape(self.o_out, (N, N))
        xx, yy = np.meshgrid(x, y)
        plt.contour(xx, yy, zz)
        plt.show()
