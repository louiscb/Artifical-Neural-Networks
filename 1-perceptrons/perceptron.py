import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(forward_pass, weights, training_set):
    class1 = []
    class2 = []
    for data_point in training_set:
        if forward_pass(data_point) <= 0:
            class1.append(data_point)
        else:
            class2.append(data_point)
    class1 = np.array(class1)
    class2 = np.array(class2)
    plt.scatter(class1[:, 0], class1[:, 1], c='red')
    plt.scatter(class2[:, 0], class2[:, 1], c='blue')
    x = np.linspace(-10, 10, 1000)
    y = -1 * weights[0] / weights[1] * x + -1 * weights[2] / weights[1]
    plt.plot(x, y, '-g')
    plt.grid()
    plt.xlim((-15, 15))
    plt.ylim((-15, 15))
    plt.show()


class PerceptronLearningRuleOnline:

    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.weights = np.random.uniform(-1, 1, (dimensions, 1))
        self.learning_rate = learning_rate

    def iteratively_update_weights(self, training_set, labels):
        assert len(training_set) == len(labels)
        # visualize_predictions(self.forward_pass, self.weights, training_set)
        for data_point, label in zip(training_set, labels):
            prediction = self.forward_pass(data_point)
            if prediction != label:
                delta_w = self.learning_rate * data_point
                delta_w = np.reshape(delta_w, (3, 1))
                if prediction == 0:
                    self.weights = np.add(self.weights, delta_w)
                else:
                    self.weights = np.subtract(self.weights, delta_w)
        #visualize_predictions(self.forward_pass, self.weights, training_set)

    def forward_pass(self, data_point):
        prediction = np.matmul(self.weights.T, data_point)
        prediction = 0 if prediction < 0 else 1
        return prediction


class DeltaRule:

    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.weights = np.random.uniform(-1, 1, (dimensions, 1))
        self.learning_rate = learning_rate

    def forward_pass(self, data_point):
        prediction = np.matmul(self.weights.T, data_point)
        return prediction[0]

    def iteratively_update_weights(self, training_set, labels):
        #visualize_predictions(self.forward_pass, self.weights, training_set)

        for data_point, label in zip(training_set, labels):
            prediction = self.forward_pass(data_point)
            error = prediction - label
            delta_w = -self.learning_rate * error * data_point
            delta_w = np.reshape(delta_w, (3, 1))
            self.weights = np.add(self.weights, delta_w)

        #visualize_predictions(self.forward_pass, self.weights, training_set)

    def batch_update_weights(self, training_set, labels):
        #visualize_predictions(self.forward_pass, self.weights, training_set)

        prediction = np.matmul(training_set, self.weights)
        labels = np.reshape(labels, (200, 1))
        error = prediction - labels
        delta_w = np.matmul(training_set.T, error) * - self.learning_rate
        self.weights = self.weights + delta_w

        #visualize_predictions(self.forward_pass, self.weights, training_set)
