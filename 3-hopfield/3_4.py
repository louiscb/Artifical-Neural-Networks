from hopfield_lib import *
from hopfield_net import HopfieldNet


def main():
    images = load_data()
    model = HopfieldNet(max_iter=200)
    model.fit(patterns=images[:3])
    distortion_percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("distortion_percentage" "prediction_accuracy")
    for i, image in enumerate(images[:3]):
        print("Image", i+1)
        for distortion_percentage in distortion_percentages:
            prediction = model.predict(add_noise_to_image(image, distortion_percentage))
            accuracy = calc_element_accuracy(image.reshape((1, 1024)), prediction)
            print(distortion_percentage, accuracy)
main()