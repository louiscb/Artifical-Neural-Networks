import numpy as np
import itertools
from hopfield_net import HopfieldNet


def main():
    test_convergence()
    print('\n///////////\n')
    find_attractors()
    print('\n///////////\n')
    test_noisier()



def get_clean_data():
    patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
                        [-1, -1, -1, -1, -1,  1, -1, -1],
                        [-1,  1,  1, -1, -1,  1, -1,  1]])
    
    return patterns


def get_noisy_data():
    patterns = np.array([[ 1, -1,  1, -1,  1, -1, -1,  1],
                        [ 1,  1, -1, -1, -1,  1, -1, -1],
                        [ 1,  1,  1, -1,  1,  1, -1,  1]])
    return patterns

def calc_element_accuracy(patterns, preds):
    n_total = patterns.shape[0] * patterns.shape[1]
    n_correct = np.sum(patterns == preds)
    return n_correct / n_total

def calc_pattern_accuracy(patterns, preds):
    n_total = patterns.shape[0]
    n_correct = 0
    for pattern, pred in zip(patterns, preds):
        if (pattern == pred).all():
            n_correct += 1
    return n_correct / n_total




def test_convergence():
    clean_patterns = get_clean_data()
    noisy_patterns = get_noisy_data()


    net = HopfieldNet()
    net.fit(clean_patterns)
    noisy_preds= net.predict(noisy_patterns)


    elements_converged= calc_element_accuracy(clean_patterns, noisy_preds)
    patterns_converged = calc_pattern_accuracy(clean_patterns, noisy_preds)

    print("percentage of converged patterns is ", patterns_converged)
    print("sample accuracy is ", elements_converged)


def find_attractors():

    clean_patterns = get_clean_data()
    net = HopfieldNet()
    net.fit(clean_patterns)

    combs = []
    combinations = itertools.product([-1,1], repeat=8)
    for comb in combinations:

        prediction = net.predict(comb)

        combs.append(net.predict(comb))
        unique_preds = list(set(tuple(x) for x in combs))
    
    no_attractors= len(list(unique_preds))
    print('number of attractors:', no_attractors)

def test_noisier():
    clean_patterns= get_clean_data()
    noisy_patterns= np.array([[-1, -1,  1, -1,  -1, 1, 1,  -1],
                              [-1, -1, -1, -1, 1,  -1, 1, 1],
                              [-1,  1,  1, -1, 1,  -1, 1,  -1]])

    

    
    net = HopfieldNet()
    net.fit(clean_patterns)
    noisy_preds= net.predict(noisy_patterns)

    elements_converged= calc_element_accuracy(clean_patterns, noisy_preds)
    patterns_converged = calc_pattern_accuracy(clean_patterns, noisy_preds)

    print("percentage of converged patterns is ", patterns_converged)
    print("sample accuracy is ", elements_converged)








main()


