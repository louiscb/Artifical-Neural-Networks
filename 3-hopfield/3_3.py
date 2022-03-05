from hopfield_lib import *
from hopfield_net import HopfieldNet
def main():
    images=load_data()
    images = images[:3]
    for image in images:
        showimage(image)


main()