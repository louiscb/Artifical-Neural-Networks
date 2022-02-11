import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_size, output_dimensions, training_set, learning_rate, alpha, iterations):
        self.h_out = None
        self.o_out = None
        self.delta_v = None
        self.delta_w = None
        self.input_dimensions = input_dimensions
        self.hidden_layer_size = hidden_layer_size
        self.output_dimensions = output_dimensions
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.iterations = iterations
        self.w = np.random.uniform(-1, 1, (input_dimensions, hidden_layer_size))
        self.v = np.random.uniform(-1, 1, (hidden_layer_size + 1, output_dimensions))

    def forward_pass(self):
        h_in = np.matmul(self.training_set, self.w)
        h_in = self.add_bias_row(h_in)
        self.h_out = self.sigmoid(h_in)
        o_in = np.matmul(self.h_out, self.v)
        self.o_out = self.sigmoid(o_in)
        return self.o_out

    def add_bias_row(self, data):
        N = data.shape[0]
        bias_row = np.ones((N, 1))
        return np.concatenate((data, bias_row), axis=1)

    def sigmoid(self, data):
        return (2 / (np.exp(-1 * data) + 1)) - 1

    def backward_pass(self, labels):
        labels = np.reshape(labels, (len(labels), 1))
        delta_o = self.o_out - labels
        delta_o = delta_o * self.sigmoid_prime(self.o_out)
        delta_h = np.matmul(delta_o, self.v.T) * self.sigmoid_prime(self.h_out)
        delta_h = np.delete(delta_h, delta_h.shape[1] - 1, axis=1)
        #remove bias row
        #weight update
        if self.delta_w == None:
            self.delta_w = -self.learning_rate * np.matmul(self.training_set.T, delta_h)
            self.delta_v = -self.learning_rate * np.matmul(self.h_out.T, delta_o)
        #add momentum
        else:
            self.delta_w = self.learning_rate * (self.delta_w * self.alpha) - (1 - self.alpha) * np.matmul(self.training_set.T, delta_h)
            self.delta_v = self.learning_rate * (self.delta_v * self.alpha) - (1 - self.alpha) * np.matmul(self.h_out.T, delta_o)

        self.w = self.w + self.delta_w
        self.v = self.v + self.delta_v

    def sigmoid_prime(self, input):
        return 0.5 * (1 + self.sigmoid(input)) * (1 - self.sigmoid(input))