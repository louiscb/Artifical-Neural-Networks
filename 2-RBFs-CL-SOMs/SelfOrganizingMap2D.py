import numpy as np


def euclidean_distance(p1, p2):
    difference_vec = p1 - p2
    return np.linalg.norm(difference_vec)


class SelfOrganizingMap2D:
    def __init__(self, input_dims, granularity):
        self.input_dims = input_dims
        self.granularity = granularity
        self.w = np.random.uniform(0, 1, (granularity, granularity, input_dims))

    def evaluate(self, data):
        predictions = np.zeros((data.shape[0], 2))
        for i in range(len(data)):
            predictions[i] = self.get_nearest_node(data[i])
        return predictions

    def fit(self, data):
        epochs = 25
        while epochs >= 0:
            for dp in data:
                index = self.get_nearest_node(dp)
                neighbours = self.get_neighbours(index, epochs * 2)
                self.adjust_weights(dp, [index], discount=0.2)
                self.adjust_weights(dp, neighbours, discount=0.1)
            epochs -= 1

    def adjust_weights(self, target, vectors, discount):
        for index in vectors:
            self.w[index[0]][index[1]] = self.w[index[0]][index[1]] + discount * (target - self.w[index[0]][index[1]])

    def get_neighbours(self, index, max_neighbourhood_size):
        neighbours = []
        to_visit = []
        curr_node = index
        while len(neighbours) < max_neighbourhood_size:
            adjacent_nodes = self.get_adjacent(curr_node)
            for node in adjacent_nodes:
                if node not in to_visit:
                    to_visit.append(node)
            neighbours.append(curr_node)
            curr_node = to_visit.pop(0)
        return neighbours

    def get_adjacent(self, node):
        i = node[0]
        j = node[1]
        adjacent = [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]
        legal_neighbours = []
        for neighbour in adjacent:
            if 0 <= neighbour[0] < self.granularity and 0 <= neighbour[1] < self.granularity:
                legal_neighbours.append(neighbour)
        return legal_neighbours


    def get_nearest_node(self, data_point):
        index = None
        min_distance = float('inf')
        for i in range(self.granularity):
            for j in range(self.granularity):
                d = euclidean_distance(data_point, self.w[i][j])
                if d < min_distance:
                    index = [i, j]
                    min_distance = d
        return index
