import numpy as np
from collections import Counter, defaultdict
import random
import argparse


def stratified_group_k_fold(X, y, groups, k, seed=None):
    """Create stratified, grouped data indices for K-fold cv.
    Credit goes to: https://www.kaggle.com/code/jakubwasikowski/stratified-group-k-fold-cross-validation/notebook

    Args:
        X (list): List of data points
        y (list): List of categorical data labels
        groups (list): List of group IDs
        k (int): Number of folds
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        NoneType: No return values

    Yields:
        Tuple: Tuple of training and testing indices for the current fold.
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def parser_builder():
    parser = argparse.ArgumentParser('Options')

    # Paths
    parser.add_argument(
        '--data_path', type=str,
        help="Path to the HDF5 storage object",
        required=True
    )
    parser.add_argument(
        '--network_path', type=str,
        help="Optional, path to where the EventNet network weights should be stored."
    )
    parser.add_argument(
        '--log_path', type=str,
        help="Optional, path to where to store loss logs."
    )

    # Training details
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help="Base learning rate. Will decay throughout training"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--duration_factor', type=int, default=10,
        help="How large should the training window be? Recommended to not change"
    )

    # Losses
    parser.add_argument(
        '--lambda_r', type=float, default=5.,
        help="Relative regression loss weight."
    )

    # Network details
    parser.add_argument(
        '--duration_threshold', type=float, default=10.,
        help='Maximum duration for EventNet'
    )

    return parser
