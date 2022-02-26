import numpy as np
from SelfOrganizingMap import *

def main():
    topological_ordering_of_animal_species()

def topological_ordering_of_animal_species():
    animal_names = None
    animal_props = None
    with open('data_set/animalnames.txt') as f:
        animal_names = [line.rstrip() for line in f]
    with open('data_set/animals.dat') as f:
        animal_props = f.readline()
        animal_props = animal_props.split(',')
    animal_props = np.array(animal_props, dtype=float)
    animal_props = animal_props.reshape((32, 84))
    model = SelfOrganizingMap(84, 100)
    model.fit(animal_props)
    predictions = model.evaluate(animal_props)
    animal_topology = {}
    for animal, mapping in zip(animal_names, predictions):
        animal_topology[animal] = mapping
    sorted_animals = sorted(animal_topology, key=animal_topology.get)
    print(sorted_animals)

main()