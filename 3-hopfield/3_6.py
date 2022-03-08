from hopfield_net import HopfieldNetBinary
from hopfield_lib import *


def main():
    print('sparsity = 0.1')
    thetas = [0, 0.2, 0.4, 0.6]
    for theta in thetas:
        print('theta', theta)
        model = HopfieldNetBinary(max_iter=100, theta=theta)
        patterns = 0.5 + 0.5 * np.sign(np.random.uniform(-1, 1, (30, 100)))
        sparse_patterns = sparsify(patterns, 0.1)
        for i in range(1, 30):
            model.fit(sparse_patterns[:i])
            model.remove_diagonals()
            count = 0
            for j in range(i):
                prediction = model.predict(sparse_patterns[j])
                if np.array_equiv(sparse_patterns[j], prediction):
                    count += 1
            print(i, count)

    print('sparsity = 0.05')
    thetas = [0, 0.2, 0.4, 0.6]
    for theta in thetas:
        print('theta', theta)
        model = HopfieldNetBinary(max_iter=100, theta=theta)
        patterns = 0.5 + 0.5 * np.sign(np.random.uniform(-1, 1, (30, 100)))
        sparse_patterns = sparsify(patterns, 0.1)
        for i in range(1, 30):
            model.fit(sparse_patterns[:i])
            model.remove_diagonals()
            count = 0
            for j in range(i):
                prediction = model.predict(sparse_patterns[j])
                if np.array_equiv(sparse_patterns[j], prediction):
                    count += 1
            print(i, count)

main()
