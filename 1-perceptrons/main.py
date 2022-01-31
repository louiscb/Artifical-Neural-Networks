import data_generation
from Perceptron import PerceptronLearningRuleOnline
import matplotlib.pyplot as plt

def main():
    d1, d2 = data_generation.generate_data_points(([-5, -5], [1, 1]), ([5, 5], [1, 1]), 100)
    data_generation.visualize_data(d1, d2)
    data, label = data_generation.concatenate_and_shuffle(d1, d2, 0, 1)
    data = data_generation.add_bias(data)
    perceptron = PerceptronLearningRuleOnline(data.shape[1], 0.1)
    perceptron.iteratively_update_weights(data, label)



main()