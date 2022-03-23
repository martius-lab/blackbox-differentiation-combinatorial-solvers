import numpy as np
import os
from decorators import input_to_numpy
from utils import TrainingIterator


# globe_tsp dataset from here: https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S
def load_dataset(data_dir, use_test_set, evaluate_with_extra, normalize, use_local_path):
    if not os.path.exists(data_dir):
        raise Exception(f"Cannot find {data_dir}")

    train_prefix = "train"
    val_prefix = ("test" if use_test_set else "val") + ("_extra" if evaluate_with_extra else "")

    country_names, gps, flags = zip(*np.load(os.path.join(data_dir, "countries.npy"), allow_pickle=True))
    flags = np.array(flags)

    train_indices = np.load(os.path.join(data_dir, train_prefix + "_indices.npy")).astype(np.int)
    train_tours = np.load(os.path.join(data_dir, train_prefix + "_tsp_tours.npy")).astype(np.int)
    train_distances = np.load(os.path.join(data_dir, train_prefix + "_distance_matrices.npy")).astype(np.float32)
    train_flags = flags[train_indices].astype(np.float64).transpose(0, 1, 4, 2, 3)  # examples,k,channels,height,width

    val_indices = np.load(os.path.join(data_dir, val_prefix + "_indices.npy")).astype(np.int)
    val_tours = np.load(os.path.join(data_dir, val_prefix + "_tsp_tours.npy")).astype(np.int)
    val_distances = np.load(os.path.join(data_dir, val_prefix + "_distance_matrices.npy")).astype(np.float32)
    val_flags = flags[val_indices].astype(np.float64).transpose(0, 1, 4, 2, 3)  # examples,k,channels,height,width

    mean, std = (
        np.mean(train_flags, keepdims=True),
        np.std(train_flags, keepdims=True),
    )
    if normalize:
        train_flags -= mean
        train_flags /= std
        val_flags -= mean
        val_flags /= std

    @input_to_numpy
    def denormalize(x):
        return (x * std) + mean

    train_iterator = TrainingIterator(dict(flags=train_flags, labels=train_tours, true_distances=train_distances))
    eval_iterator = TrainingIterator(dict(flags=val_flags, labels=val_tours, true_distances=val_distances))

    metadata = {
        "num_examples": train_flags.shape[0],
        "num_flags": train_flags.shape[1],
        "num_channels": train_flags.shape[2],
        "flag_height": train_flags.shape[3],
        "flag_width": train_flags.shape[4],
        "denormalize": denormalize
    }

    return train_iterator, eval_iterator, metadata
