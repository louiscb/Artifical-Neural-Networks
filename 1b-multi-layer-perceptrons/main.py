from data_generation_lib import *
from multi_layer_perceptron import MultiLayerPerceptron
import numpy as np


def main():
    d1, d2 = generate_data_points([[-3.0, -3], [1, 1]], [[3, 3], [1, 1]], 100)
    data, labels = concatenate_and_shuffle(d1, d2, -1, 1)
    data = add_bias(data)

    model = MultiLayerPerceptron(data.shape[1], hidden_layer_size=3, output_dimensions=1, training_set=data,
                                learning_rate=0.001, alpha=0.8, val_set=None)
    ratios_per_epoch = number_of_misclassified_samples_per_class(model, data, labels, epochs=20)
    print(ratios_per_epoch)

    #sampling data
    d1_train, d2_train, d1_val, d2_val = remove_samples(d1, d2, 100, 0.25, 0.25)
    data_train, labels_train = concatenate_and_shuffle(d1_train, d2_train, -1, 1)
    data_val, labels_val = concatenate_and_shuffle(d1_val, d2_val, -1, 1)
    data_train = add_bias(data_train)
    data_val = add_bias(data_val)
    sampled_model = MultiLayerPerceptron(data_train.shape[1], hidden_layer_size=3, output_dimensions=1, training_set=data_train,
                                learning_rate=0.001, alpha=0.8, val_set= data_val )

    number_of_misclassified_samples_per_class(sampled_model, data_train, labels_train, epochs=20)
    val_prediction = sampled_model.forward_pass_val()



    count1 = 0
    correct1 = 0
    count2 = 0
    correct2 = 0
    index = 0
    prediction_ratios = np.zeros(2)
    for data_point, label in zip(data_val, labels):
        if label == 0 or label == -1:
            count1 += 1
            if val_prediction[index] <= 0:
                correct1 += 1
        else:
            count2 += 1
            if val_prediction[index] > 0:
                correct2 += 1
        index += 1

    prediction_ratios[0] = correct1 / count1
    prediction_ratios[1] = correct2 / count2
    total_prediction_ratio = (correct1 + correct2) / (count1 + count2)

    print("classification ratio on val set is: ", prediction_ratios, total_prediction_ratio)


    








def number_of_misclassified_samples_per_class(model, training_set, labels, epochs):
    prediction_ratios = np.zeros((epochs, 2))
    total_prediction_ratio = np.zeros(epochs)
    for epoch in range(epochs):
        count1 = 0
        correct1 = 0
        count2 = 0
        correct2 = 0
        prediction = model.forward_pass()
        index = 0
        for data_point, label in zip(training_set, labels):
            if label == 0 or label == -1:
                count1 += 1
                if prediction[index] <= 0:
                    correct1 += 1
            else:
                count2 += 1
                if prediction[index] > 0:
                    correct2 += 1
            index += 1
        prediction_ratios[epoch][0] = correct1 / count1
        prediction_ratios[epoch][1] = correct2 / count2
        total_prediction_ratio[epoch] = (correct1 + correct2) / (count1 + count2)
        model.backward_pass(labels)
    return total_prediction_ratio, prediction_ratios


def remove_samples(d1, d2, n, class1_removal, class2_removal):
    class1_adjusted_len = n - round(n * class1_removal)
    d1_train = d1[:class1_adjusted_len]
    d1_val = d1[class1_adjusted_len:]

    class2_adjusted_len = n - round(n * class2_removal)
    d2_train = d2[:class2_adjusted_len]
    d2_val = d2[class2_adjusted_len:]

    return d1_train, d2_train, d1_val, d2_val





main()
