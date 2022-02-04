import data_generation_lib
import perceptron
from perceptron import PerceptronLearningRuleOnline, DeltaRule
import numpy as np
import time
import random


def main():
    N = 100
    learning_rate = 0.1
    epochs = 20
    iterations = 100
    get_results(N, learning_rate, iterations, epochs)

    # delta rule sequential
    #data, label = generate_data_with_bias(100)
    #delta = DeltaRule(data.shape[1], learning_rate)
    #delta.iteratively_update_weights(data, label)


# delta rule batch
# delta = DeltaRule(data.shape[1], learning_rate)
# delta.batch_update_weights(data, label)

def get_results(N, learning_rate, iterations, epochs):
    ratios = np.zeros((iterations, epochs))
    for i in range(iterations):
        random.seed(time.time())
        data, label = generate_data_with_bias(N)
        # perceptron learning rule
        perceptron = PerceptronLearningRuleOnline(data.shape[1], learning_rate)
        ratios_per_epoch = number_of_misclassified_samples(perceptron, data, label, epochs)
        ratios[i] = ratios_per_epoch
    ratios = np.mean(ratios, axis=0)
    write_results_to_file('sequential_perceptron_learning_not_linearly_separable_2_25_both_classes' + str(time.time()) + '.txt', learning_rate, ratios)


def write_results_to_file(filename, learning_rate, ratios):
    f = open(filename, 'w')
    f.write(str(learning_rate) + '\n')
    for ratio in ratios:
        f.write(str(ratio))
        f.write('\n')
    f.flush()


def number_of_misclassified_samples(model, training_set, labels, epochs):
    prediction_ratios = np.zeros(epochs)
    for epoch in range(epochs):
        count = 0
        for data_point, label in zip(training_set, labels):
            prediction = model.forward_pass(data_point)
            if prediction <= 0 and (label == 0 or label == -1):
                count += 1
            elif label == 1 and prediction > 0:
                count += 1
        prediction_ratios[epoch] = count / len(training_set)
        model.iteratively_update_weights(training_set, labels)
    #perceptron.visualize_predictions(model.forward_pass, model.weights, training_set)
    return prediction_ratios


def generate_data_with_bias(N):
    d1, d2 = data_generation_lib.generate_data_points(([1, 0.3], [0.2, 0.2]), ([0, -0.1], [0.3, 0.3]), N, class2_removal=0.25, class1_removal=0.25)
    #data_generation_lib.visualize_data(d1, d2)
    data, label = data_generation_lib.concatenate_and_shuffle(d1, d2, -1, 1)
    data = data_generation_lib.add_bias(data)
    return data, label


main()
