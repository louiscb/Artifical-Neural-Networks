from SelfOrganizingMap import *
from SelfOrganizingMapCircular import *
import csv


def main():
    cyclic_tour()


def cyclic_tour():
    with open('data_set/cities.dat') as f:
        cities = []
        reader = csv.reader(f)
        for line in reader:
            cities.append(line)
        cities = np.array(cities).astype(float)
        model = SelfOrganizingMapCircular(2, 10)
        model.fit(cities)


def topological_ordering_of_animal_species():
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
