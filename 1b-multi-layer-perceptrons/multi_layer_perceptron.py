import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_size, output_dimensions, training_set, learning_rate, iterations):
        self.h_out = None
        self.o_out = None
        self.input_dimensions = input_dimensions
        self.hidden_layer_size = hidden_layer_size
        self.output_dimensions = output_dimensions
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = np.uniform(-1, 1, (input_dimensions, hidden_layer_size))
        self.v = np.uniform(-1, 1, (hidden_layer_size + 1, output_dimensions))

    def forward_pass(self):
        h_in = np.matmul(self.training_set, self.w)
        h_in = self.add_bias_row(h_in)
        self.h_out = self.sigmoid(h_in)
        o_in = np.matmul(self.h_out, self.v)
        self.o_out = self.sigmoid(o_in)
        return self.o_out

    def add_bias_row(self, data):
        N = data.shape[1]
        bias_row = np.ones((N, 1))
        return np.concatenate((data, bias_row), axis=1)

    def sigmoid(self, data):
        return (2 / (np.exp(-1 * data) + 1)) - 1

    def backward_pass(self, labels):
        delta_o = self.o_out - labels
        delta_o = delta_o * self.sigmoid_prime(self.o_out)
        delta_h = np.matmul(self.v, delta_o.T) * self.sigmoid_prime(self.h_out)
        #remove bias row
        #weight update
        delta_w = -self.learning_rate * np.matmul(delta_h, self.training_set.T)
        delta_v = -self.learning_rate * np.matmul(delta_o, self.h_out.T)
        #add momentum
        self.w = self.w + delta_w
        self.v = self.v + delta_v

    def sigmoid_prime(self, input):
        return 0.5 * (1 + self.sigmoid(input)) * (1 - self.sigmoid(input))