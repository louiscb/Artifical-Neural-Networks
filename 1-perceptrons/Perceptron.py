import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearningRuleOnline:

    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.weights = np.random.uniform(-1, 1, (dimensions, 1))
        self.learning_rate = learning_rate

    def iteratively_update_weights(self, training_set, labels):
        assert len(training_set) == len(labels)
        self.visualize_predictions(training_set)
        for data_point, label in zip(training_set, labels):
            prediction = self.forward_pass(data_point)
            if prediction != label:
                delta_w = self.learning_rate * data_point
                delta_w = np.reshape(delta_w, (3, 1))
                if prediction == 0:
                    self.weights = np.add(self.weights, delta_w)
                else:
                    self.weights = np.subtract(self.weights, delta_w)
        self.visualize_predictions(training_set)

    def visualize_predictions(self, training_set):
        class1 = []
        class2 = []
        for data_point in training_set:
            if self.forward_pass(data_point) == 0:
                class1.append(data_point)
            else:
                class2.append(data_point)
        class1 = np.array(class1)
        class2 = np.array(class2)
        plt.scatter(class1[:, 0], class1[:, 1], c='red')
        plt.scatter(class2[:, 0], class2[:, 1], c='blue')
        x = np.linspace(-10, 10, 1000)
        y = -1 * self.weights[0] / self.weights[1] * x + -1 * self.weights[2] / self.weights[1]
        plt.plot(x, y, '-g')
        plt.grid()
        plt.xlim((-15, 15))
        plt.ylim((-15, 15))
        plt.show()



    def forward_pass(self, data_point):
        prediction = np.matmul(self.weights.T, data_point)
        prediction = 0 if prediction < 0 else 1
        return prediction