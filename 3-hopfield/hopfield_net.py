import numpy as np

class HopfieldNet:
    def __init__(self, max_iter=50):
        self.w = None
        self.n_elements = None
        self.max_iter = max_iter
        self.hundreth_images=None

    def fit(self, patterns):
        patterns = np.array(patterns)
        if len(patterns.shape) == 1:
            patterns = np.reshape(patterns, (1, -1))
        self.n_elements = patterns.shape[1]
        self.w = np.zeros((self.n_elements, self.n_elements))
        for pattern in patterns:
            pattern = np.reshape(pattern, (-1, 1))
            self.w += (pattern @ pattern.T)/self.n_elements


    def predict(self, pattern, method='batch'):
        input_pattern = np.array(pattern)
        current_pattern = input_pattern.copy()
        iter = 0
        self.hundreth_images=[]
 


        while ( (iter < self.max_iter) ):
            if method == 'batch':
                current_pattern = self._batch_update(current_pattern)
            elif method == 'sequential':
                current_pattern = self.sequential_update(current_pattern)

            iter += 1

        return current_pattern

    
    def _batch_update(self, pattern):
        return np.sign(pattern @ self.w)

    def sequential_update(self, pattern):
        current_pattern = pattern.copy()
        unit_i = np.array(range(0, self.n_elements))
        np.random.shuffle(unit_i)

        for i in unit_i:
            current_pattern[i] = np.sign(self.w[i,:].dot(current_pattern))
            if (i%100) == 0:
                self.hundreth_images.append(current_pattern.copy())

        return current_pattern