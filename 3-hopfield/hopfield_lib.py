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


# sparsify a binary matrix
def sparsify(pattern, activity):
    sparse_pattern = pattern.copy()
    sparse_pattern = sparse_pattern.reshape((-1, 1))
    nonzeros = np.nonzero(sparse_pattern)[0]
    target_nonzeros = round(activity * sparse_pattern.shape[0])
    num_indices_to_flip = len(nonzeros) - target_nonzeros
    if num_indices_to_flip < 0:
        return None
    indices_to_zeroify = random.sample(nonzeros.tolist(), num_indices_to_flip)
    for i in indices_to_zeroify:
        sparse_pattern[i] = 0
    return sparse_pattern.reshape(pattern.shape)






def calc_element_accuracy(patterns, preds):
    n_total = patterns.shape[0] * patterns.shape[1]
    n_correct = np.sum(patterns == preds)
    return n_correct / n_total
