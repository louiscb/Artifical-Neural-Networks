import numpy as np

from hopfield_lib import *
from hopfield_net import HopfieldNet


def main():
    images = load_data()
    model = HopfieldNet(max_iter=200)
    distortion_percentage = 0.3
    for i in range(3, 11):
        model.fit(images[:i])
        print('\n')
        print("model trained with", i + 1, "patterns")
        for j in range(len(images[:3])):
            prediction = model.predict(add_noise_to_image(images[j], distortion_percentage))
            accuracy = calc_element_accuracy(images[j].reshape((1, 1024)), prediction)
            print(j + 1, distortion_percentage, accuracy)

    print("\nRANDOM PATTERNS")
    patterns = np.random.uniform(-1, 1, (11, 1024))
    patterns = np.sign(patterns)
    rand_model = HopfieldNet(max_iter=200)
    distortion_percentage = 0.3
    for i in range(3, 11):
        rand_model.fit(patterns[:i])
        print('\n')
        print("model trained with", i + 1, "patterns")
        for j in range(len(patterns[:3])):
            prediction = rand_model.predict(add_noise_to_image(patterns[j], distortion_percentage))
            accuracy = calc_element_accuracy(patterns[j].reshape((1, 1024)), prediction)
            print(j + 1, distortion_percentage, accuracy)

main()
