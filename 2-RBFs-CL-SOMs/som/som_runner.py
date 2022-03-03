from SelfOrganizingMap import *
from SelfOrganizingMapCircular import *
from SelfOrganizingMap2D import *
import csv
import matplotlib.pyplot as plt


def main():
    data_clustering_mps()


def data_clustering_mps():
    with open('data_set/votes.dat') as props_file, open('data_set/mpsex.dat') as gender_file, open('data_set/mpparty.dat') as party_file, open('data_set/mpdistrict.dat') as district_file:
        vote_props = props_file.readline().split(',')
        genders = [int(x.rstrip()) for x in gender_file]
        parties = [int(x.rstrip()) for x in party_file]
        district = [int(x.rstrip()) for x in district_file]
        vote_props = np.array(vote_props, dtype=float).reshape((349, 31))
        model = SelfOrganizingMap2D(31, 10)
        model.fit(vote_props)
        predictions = model.evaluate(vote_props)
        cm = []
        for i in range(len(genders)):
            if genders[i] == 0:
                cm.append('red')
            else:
                cm.append('blue')
        plt.scatter(predictions[:, 0], predictions[:, 1], c=cm)
        plt.show()
        plt.scatter(predictions[:, 0], predictions[:, 1], c=parties)
        plt.show()
        plt.scatter(predictions[:, 0], predictions[:, 1], c=district)
        plt.show()



def cyclic_tour():
    with open('data_set/cities.dat') as f:
        cities = []
        reader = csv.reader(f)
        for line in reader:
            cities.append(line)
        cities = np.array(cities).astype(float)
        model = SelfOrganizingMapCircular(2, 10)
        model.fit(cities)
        plt.plot(model.w[:, 0], model.w[:, 1])
        plt.scatter(cities[:, 0], cities[:, 1])
        plt.show()


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
