import data_generation

def main():
    data = data_generation.generate_data_points(([-10, -10], [1, 1]), ([10, 10], [1, 1]), 100)
    data_generation.visualize_data(data)

main()