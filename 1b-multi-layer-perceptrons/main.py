from data_generation_lib import *
from multi_layer_perceptron import MultiLayerPerceptron


def main():
    d1, d2 = generate_data_points([[1.0, 0.3], [0.2, 0.2]], [[0, -0.1], [0.3, 0.3]], 100)
    data, labels = concatenate_and_shuffle(d1, d2, 0, 1)
    data = add_bias(data)
    model = MultiLayerPerceptron(data.shape[1], hidden_layer_size=8, output_dimensions=1, training_set=data,
                                 learning_rate=0.01, iterations=20)
    model.forward_pass()
    model.backward_pass(labels)


main()
