from functools import lru_cache

import pandas as pd

from math import floor

import numpy as np

from config import *


@lru_cache
def get_dataset(dataset):
    if dataset == 'kdd':
        return read_and_prep_kdd()
    elif dataset == 'gaussian':
        return generate_k_gaussians()
    else:
        raise RuntimeError(f"bad dataset {dataset}")


def read_and_prep_kdd():
    full_data = pd.read_csv(KDD_DATASET_FILE, nrows=KDD_SUBSET_SIZE)
    N = full_data.select_dtypes([np.number])
    return N


def generate_k_gaussians():
    N = np.random.normal(scale=GAUSSIANS_STD_DEV, size=(GAUSSIANS_NUM_POINTS, GAUSSIANS_DIMENSIONS,))
    centers = np.random.uniform(size=(GAUSSIANS_K, GAUSSIANS_DIMENSIONS,))
    the_sum = sum(i ** GAUSSIANS_ALPHA for i in range(1, GAUSSIANS_K + 1))
    cluster_sizes = [floor(GAUSSIANS_NUM_POINTS * ((i ** GAUSSIANS_ALPHA) / the_sum)) for i in range(1, GAUSSIANS_K + 1)]
    N = N[:sum(cluster_sizes)]

    cluster_slices = []
    next_element = 0
    for cluster_size in cluster_sizes:
        cluster_slices.append((slice(next_element, next_element + cluster_size)))
        next_element += cluster_size

    for k, cluster_slice in enumerate(cluster_slices):
        N[cluster_slice] += centers[k]

    return pd.DataFrame(N)


if __name__ == '__main__':
    print(generate_k_gaussians())
