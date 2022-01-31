import data_generation

def main():
    d1, d2 = data_generation.generate_data_points(([-10, -10], [1, 1]), ([10, 10], [1, 1]), 100)
    data_generation.visualize_data(d1, d2)
    data, label = data_generation.concatenate_and_shuffle(d1, d2)
    data = data_generation.add_bias(data)



main()