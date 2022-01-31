import numpy as np

class Perceptron_Learning_Rule_Online():

    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.weights = np.random.rand((dimensions, 1))
        self.learning_rate = learning_rate