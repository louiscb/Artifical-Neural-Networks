import numpy as np


def euclidean_distance(p1, p2):
    difference_vec = p1 - p2
    return np.linalg.norm(difference_vec)


class SelfOrganizingMapCircular:
    def __init__(self, input_dims, granularity):
        self.input_dims = input_dims
        self.granularity = granularity
        self.w = np.random.uniform(0, 1, (granularity, input_dims))

    def evaluate(self, data):
        predictions = np.zeros(data.shape[0])
        for i in range(len(data)):
            predictions[i] = self.get_nearest_node(data[i])
        return predictions

    def fit(self, data):
        epochs = 29
        while epochs >= 0:
            neighbourhood_size = int(epochs/10)
            for dp in data:
                index = self.get_nearest_node(dp)
                left_neighbours, right_neighbours = self.get_neighbours(index, neighbourhood_size)
                self.adjust_weights(dp, [index], discount=0.2)
                self.adjust_weights(dp, left_neighbours, discount=0.2)
                self.adjust_weights(dp, right_neighbours, discount=0.2)
            epochs -= 1

    def adjust_weights(self, target, vectors, discount):
        for index in vectors:
            self.w[index] = self.w[index] + discount * (target - self.w[index])

    def get_neighbours(self, index, max_neighbourhood_size):
        lower_bound = index - max_neighbourhood_size / 2
        upper_bound = (index + max_neighbourhood_size / 2 % self.granularity - 1) + 1
        return list(range(round(lower_bound), index)), list(
            range(min(index + 1, int(upper_bound)), max(index + 1, int(upper_bound))))

    def get_nearest_node(self, data_point):
        index = None
        min_distance = float('inf')
        for i in range(self.granularity):
            d = euclidean_distance(data_point, self.w[i])
            if d < min_distance:
                index = i
                min_distance = d
        return index
