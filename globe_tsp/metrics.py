import numpy as np

from comb_modules.tsp import gurobi_tsp
from decorators import input_to_numpy, none_if_missing_arg
from utils import all_accuracies


@none_if_missing_arg
def perfect_match_accuracy(true_tours, suggested_tours):
    matching_correct = np.sum(np.abs(true_tours - suggested_tours), axis=-1)
    avg_matching_correct = (matching_correct < 0.5).mean()
    return avg_matching_correct


@none_if_missing_arg
def cost_ratio(true_distances, true_tours, suggested_tours):
    suggested_paths_costs = suggested_tours * true_distances
    true_paths_costs = true_tours * true_distances
    return (np.sum(suggested_paths_costs, axis=1) / np.sum(true_paths_costs, axis=1)).mean()


@input_to_numpy
def compute_metrics(true_tours, suggested_tours, true_distances):
    batch_size = true_distances.shape[0]
    metrics = {
        "perfect_match_accuracy": perfect_match_accuracy(true_tours.reshape(batch_size, -1),
                                                         suggested_tours.reshape(batch_size, -1)),
        "cost_ratio_suggested_true": cost_ratio(true_distances, true_tours, suggested_tours),
        **all_accuracies(true_tours, suggested_tours, true_distances, is_valid_label_fn, 6)
    }
    return metrics


def is_valid_label_fn(suggested_tour):
    return np.count_nonzero(suggested_tour) > 0
