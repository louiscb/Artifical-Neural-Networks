from hopfield_lib import *
from hopfield_net import HopfieldNet


def main():
    images = load_data()
    model = HopfieldNet(max_iter=400)
    print(images.shape)
    model.fit(patterns=images[:3])
    print("Energy at p1", model.energy(images[0]))
    print("Energy at p2", model.energy(images[1]))
    print("Energy at p3", model.energy(images[2]))
    print("Energy of distorted p1", model.energy(images[9]))
    print("Energy of mixture of p2 and p3", model.energy(images[10]))
    print("Energy progression towards p1")
    model.predict(images[9], method='sequential', show_energy=True)
    print("Energy progression of p2 and p3 mixture")
    model.predict(images[10], method='sequential', show_energy=True)
    model.random_weights()
    print("Random weight values energy progression")
    model.predict(images[0], show_energy=True)
    model.make_weights_symmetric()
    print("Random symmetric weight matrix")
    model.predict(images[0], show_energy=True)



main()
