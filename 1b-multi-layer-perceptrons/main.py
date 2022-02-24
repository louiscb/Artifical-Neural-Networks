import time

from data_generation_lib import *
from multi_layer_perceptron import MultiLayerPerceptron
import numpy as np


def main():
    generate_data_vary_number_of_hidden_nodes()


def generate_data_vary_number_of_hidden_nodes():
    # for i in [1, 2, 4, 8, 16]:
    get_results(100, 0.001, 100, 20, 16)


def generate_data_with_bias(N):
    d1, d2 = generate_data_points([[0, 0], [2, 2]], [[3, 3], [1, 1]], N)
    subset_N = int(N / 4)
    subset_d1 = d1[:subset_N]
    subset_d2 = d2[:subset_N]
    d1 = d1[subset_N:]
    d2 = d2[subset_N:]
    # visualize_data(d1, d2)
    data, labels = concatenate_and_shuffle(d1, d2, -1, 1)
    data_subset, labels_subset = concatenate_and_shuffle(subset_d1, subset_d2, -1, 1)
    data = add_bias(data)
    data_subset = add_bias(data_subset)
    return data, labels, data_subset, labels_subset


def training(data_subset, labels_subset, num_of_hidden_nodes, validation_set, validation_labels, epochs):
    model = MultiLayerPerceptron(data_subset.shape[1], hidden_layer_size=num_of_hidden_nodes, output_dimensions=1,
                                 training_set=data_subset,
                                 learning_rate=0.001, alpha=0.8)
    predictions = np.zeros(epochs)

    for i in range(epochs):
        model.forward_pass()
        model.backward_pass(labels_subset)
        prediction = number_of_misclassified_samples_per_class(model, validation_set, validation_labels)
        predictions[i] = prediction

    return predictions


def get_results(N, learning_rate, iterations, epochs, num_of_hidden_nodes):
    ratios = np.zeros((iterations, epochs, 2))

    for i in range(iterations):
        random.seed(time.time())
        data, label, data_subset, labels_subset = generate_data_with_bias(N)
        prediction_ratios = training(data_subset, labels_subset, num_of_hidden_nodes, data, label, epochs)
        # prediction_ratios = number_of_misclassified_samples_per_class(model, data, label)
        ratios[i] = prediction_ratios
    ratios = np.mean(ratios, axis=0)
    write_results_to_file(str(num_of_hidden_nodes) + '_num_of_hidden_nodes' + str(time.time()) + '.txt', learning_rate,
                          num_of_hidden_nodes, ratios)


def write_results_to_file(filename, learning_rate, num_of_hidden_nodes, ratios):
    f = open(filename, 'w')
    f.write(str(learning_rate) + " " + str(num_of_hidden_nodes) + '\n')
    for ratio in ratios:
        f.write(str(ratio))
        f.write('\n')
    f.flush()


def number_of_misclassified_samples_per_class(model, validation_set, validation_labels):
    count1 = 0
    correct1 = 0
    count2 = 0
    correct2 = 0
    predictions = model.forward_pass(validation_set)
    for prediction, label in zip(predictions, validation_labels):
        if label == 0 or label == -1:
            count1 += 1
            if prediction <= 0:
                correct1 += 1
        else:
            count2 += 1
            if prediction > 0:
                correct2 += 1
    prediction_ratio_1 = correct1 / count1
    prediction_ratio_2 = correct2 / count2
    return prediction_ratio_1, prediction_ratio_2

main()
