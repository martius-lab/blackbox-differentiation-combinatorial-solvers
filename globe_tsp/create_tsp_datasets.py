import numpy as np
import os
import random
from scipy import spatial
from comb_modules import tsp

# path to the working directory
data_dir = "./data/globe_tsp/40_countries_from_100_small"


def createTrain():
    # countries.npy file with country details must exist (see globe_tsp dataset here:
    # https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S
    country_names, gps, flags = zip(*np.load(os.path.join(data_dir, "countries.npy"), allow_pickle=True))
    gps = np.array(gps)

    num_examples = 1000

    indices = np.zeros([num_examples, 40]).astype(np.uint8)
    distances = np.zeros([num_examples, 40, 40]).astype(np.float32)
    tours = np.zeros([num_examples, 40, 40]).astype(np.uint8)
    for i in range(num_examples):
        indices[i] = random.sample(range(100), 40)
        distances[i] = spatial.distance.cdist(gps[indices[i]], gps[indices[i]])
        tours[i] = tsp.gurobi_tsp(distances[i], {"time_limit": None, "mip_gap_abs": None, "mip_gap": None})

    np.save(os.path.join(data_dir, "train_indices.npy"), indices)
    np.save(os.path.join(data_dir, "train_tsp_tours.npy"), tours)
    np.save(os.path.join(data_dir, "train_distance_matrices.npy"), distances)


def createTest():
    # countries.npy file with country details must exist (see globe_tsp dataset here:
    # https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S
    country_names, gps, flags = zip(*np.load(os.path.join(data_dir, "countries.npy"), allow_pickle=True))
    gps = np.array(gps)

    num_examples = 500

    indices = np.zeros([num_examples, 40]).astype(np.uint8)
    distances = np.zeros([num_examples, 40, 40]).astype(np.float32)
    tours = np.zeros([num_examples, 40, 40]).astype(np.uint8)
    for i in range(num_examples):
        indices[i] = random.sample(range(100), 40)
        distances[i] = spatial.distance.cdist(gps[indices[i]], gps[indices[i]])
        tours[i] = tsp.gurobi_tsp(distances[i], {"time_limit": None, "mip_gap_abs": None, "mip_gap": None})

    np.save(os.path.join(data_dir, "test_indices.npy"), indices)
    np.save(os.path.join(data_dir, "test_tsp_tours.npy"), tours)
    np.save(os.path.join(data_dir, "test_distance_matrices.npy"), distances)


if __name__ == "__main__":
    createTrain()
    createTest()
