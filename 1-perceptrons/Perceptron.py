import numpy as np

class PerceptronLearningRuleOnline:

    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.weights = np.random.rand((dimensions, 1))
        self.learning_rate = learning_rate

    def iteratively_update_weights(self, training_set, labels):
        assert len(training_set) == len(labels)
        for data_point, label in zip(training_set, labels):
            prediction = self.forward_pass(data_point)
            if prediction != label:
                delta_w = self.learning_rate * data_point
                if prediction == 0:
                    self.weights += delta_w
                else:
                    self.weights -= delta_w
        self.visualize_predictions(training_set)

    def visualize_predictions(self, training_set):
       pass

    def forward_pass(self, data_point):
        prediction = np.matmul(self.weights, data_point)
        prediction = 0 if prediction < 0 else 1
        return prediction