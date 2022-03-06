import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    with open('pict.dat', 'r') as f:
        text = str(f.read())
        value_list = np.array([int(val) for val in text.split(',')])
        images = []
        for n in range(11):
            start_index = 1024 * n
            end_index = 1024 * (n + 1)
            images.append(value_list[start_index:end_index])
        return np.array(images)


def add_noise_to_image(image, percentage):
    noisy_image = image.copy()
    noisy_image.reshape((-1, 1))
    K = round(percentage * (len(noisy_image) - 1))
    indices_to_flip = random.sample(range(0, len(noisy_image) - 1), K)
    for i in indices_to_flip:
        noisy_image[i] *= -1
    noisy_image.reshape(image.shape)
    return noisy_image


def showimage(image):
    image = np.reshape(image, (32, 32)).T
    plt.imshow(image)
    plt.show()


def calc_element_accuracy(patterns, preds):
    n_total = patterns.shape[0] * patterns.shape[1]
    n_correct = np.sum(patterns == preds)
    return n_correct / n_total