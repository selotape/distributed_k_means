from functools import lru_cache
from math import floor

from sklearn.preprocessing import MinMaxScaler  # , StandardScaler,

import numpy as np
import pandas as pd

from soccer.config import *


@lru_cache
def get_dataset(logger):
    logger.info(f"Loading Dataset {DATASET}...")
    if DATASET == 'kdd':
        N = read_and_prep_kdd()
    elif DATASET.startswith('gaussian'):
        N = generate_k_gaussians()
    elif DATASET == 'bigcross':
        N = read_and_prep_bigcross()
    elif DATASET == 'census1990':
        N = read_and_prep_census1990()
    elif DATASET == 'higgs':
        N = read_and_prep_higgs()
    else:
        raise RuntimeError(f"bad dataset {DATASET}")
    logger.info(f'len(N)={len(N)}')
    if SCALE_DATASET:
        N = scale_dataset(N)

    return N


def scale_dataset(N):
    scaler = MinMaxScaler()
    scaler.fit(N)
    N = scaler.transform(N)
    return N


def read_and_prep_kdd():
    full_data = pd.read_csv(KDD_DATASET_FILE, nrows=DATASET_SIZE)
    N = full_data.select_dtypes([np.number])
    return N


def read_and_prep_bigcross():
    full_data = pd.read_csv(BIGCROSS_DATASET_FILE, nrows=DATASET_SIZE)
    N = full_data.select_dtypes([np.number])
    return N


def read_and_prep_census1990():
    census = pd.read_csv(CENSUS1990_DATASET_FILE, nrows=DATASET_SIZE, skiprows=1)
    census = pd.get_dummies(census)
    census = census.dropna()
    census = census.iloc[:, 1:]  # throw away first index column
    return census


def read_and_prep_higgs():
    full_data: pd.DataFrame = pd.read_csv(HIGGS_DATASET_FILE, nrows=DATASET_SIZE)
    N: pd.DataFrame = full_data.select_dtypes([np.number])
    N = N.dropna()
    return N


def generate_k_gaussians():
    rng = np.random.RandomState(GAUSSIANS_RANDOM_SEED or None)
    N = rng.normal(scale=GAUSSIANS_STD_DEV, size=(DATASET_SIZE, GAUSSIANS_DIMENSIONS,))
    gaussians_k = determine_gaussians_k()
    centers = rng.uniform(size=(gaussians_k, GAUSSIANS_DIMENSIONS,), )
    cluster_sizes = get_cluster_sizes(gaussians_k)
    N = N[:sum(cluster_sizes)]

    cluster_slices = []
    next_element = 0
    for cluster_size in cluster_sizes:
        cluster_slices.append((slice(next_element, next_element + cluster_size)))
        next_element += cluster_size

    for k, cluster_slice in enumerate(cluster_slices):
        N[cluster_slice] += centers[k]

    return pd.DataFrame(N)


def determine_gaussians_k():
    tokens = DATASET.split(sep='_')
    if len(tokens) == 1:  # "gaussian"
        return GAUSSIANS_K
    elif len(tokens) == 2:
        return int(tokens[1])  # "gaussian_25"
    else:
        raise NotImplementedError(f"Bad gaussian dataset name \"{DATASET}\". Should be e.g. \"gaussian_25\"")


def get_cluster_sizes(gaussians_k):
    if GAUSSIANS_TYPE == 'gamma':
        the_sum = sum(i ** GAUSSIANS_GAMMA for i in range(1, gaussians_k + 1))
        cluster_sizes = [floor(DATASET_SIZE * ((i ** GAUSSIANS_GAMMA) / the_sum)) for i in range(1, gaussians_k + 1)]
    elif GAUSSIANS_TYPE == 'exp':
        the_sum = sum(2 ** i for i in range(1, gaussians_k + 1))
        cluster_sizes = [floor(DATASET_SIZE * ((2 ** i) / the_sum)) for i in range(1, gaussians_k + 1)]
    else:
        raise NotImplementedError(f"No such gaussian type {GAUSSIANS_TYPE}")
    return cluster_sizes


if __name__ == '__main__':
    print(generate_k_gaussians())
