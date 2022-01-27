import data_generation

def main():
    d1, d2 = data_generation.generate_data_points(([-2, -2], [1, 1]), ([2, 2], [1, 1]), 100)
    data_generation.visualize_data(d1, d2)

main()