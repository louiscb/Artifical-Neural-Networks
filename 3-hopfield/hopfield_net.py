import numpy as np
import random

class HopfieldNet:
    def __init__(self, max_iter=200):
        self.w = None
        self.n_elements = None
        self.max_iter = max_iter
        self.hundreth_images=None

    def energy(self, pattern):
        pattern = pattern.reshape((1, -1))
        return -1 * pattern @ self.w @ pattern.T


    def fit(self, patterns):
        patterns = np.array(patterns)
        if len(patterns.shape) == 1:
            patterns = np.reshape(patterns, (1, -1))
        self.n_elements = patterns.shape[1]
        self.w = np.zeros((self.n_elements, self.n_elements))
        for pattern in patterns:
            pattern = np.reshape(pattern, (-1, 1))
            self.w += (pattern @ pattern.T)/self.n_elements


    def predict(self, pattern, method='batch', show_energy=False, collect_hundredth_image=False):
        input_pattern = pattern.reshape((-1, 1024))
        current_pattern = input_pattern.copy()
        self.hundreth_images=[]
        i = 0.0
        while i <= self.max_iter:
            if method == 'batch':
                current_pattern = self._batch_update(current_pattern)
                i += 1
            if method == 'sequential':
                current_pattern = self._sequential_update(current_pattern)
                i += 0.1
            if show_energy and round(i, ndigits=3) % 10 == 0:
                print(self.energy(current_pattern))
            if collect_hundredth_image and round(i, ndigits=3) % 10 == 0:
                self.hundreth_images.append(current_pattern.copy())
        return current_pattern

    def _sequential_update(self, pattern):
        index = random.randint(0, pattern.shape[1] - 1)
        pattern[0][index] = np.sign(np.dot(pattern[0], self.w[index]))
        return pattern


    def _batch_update(self, pattern):
        return np.sign(pattern @ self.w)
