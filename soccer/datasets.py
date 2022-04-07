from functools import lru_cache
from math import floor
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from soccer.config import *


@lru_cache
def get_dataset(dataset, logger):
    logger.info(f"Loading Dataset {dataset}...")
    if path.isfile(dataset):
        N = read_and_prep_unknown_file(dataset)
    elif dataset == 'kdd':
        N = read_and_prep_kdd()
    elif dataset.startswith('gaussian'):
        gaussian_k = determine_gaussians_k(dataset)
        N = generate_k_gaussians(gaussian_k)
    elif dataset == 'mnist':
        N = read_and_prep_mnist()
    elif dataset == 'bigcross':
        N = read_and_prep_bigcross()
    elif dataset == 'census1990':
        N = read_and_prep_census1990()
    elif dataset == 'higgs':
        N = read_and_prep_higgs()
    else:
        raise RuntimeError(f"bad dataset {dataset}")
    logger.info(f'len(N)={len(N)}')
    if SCALE_DATASET:
        N = scale_dataset(N)

    return N


def scale_dataset(N):
    scaler = MinMaxScaler()
    scaler.fit(N)
    N = scaler.transform(N)
    return N


def read_and_prep_unknown_file(dataset_csv):
    skip_rows = 1 if NEW_DATASET_SKIP_HEADER else 0
    N = pd.read_csv(dataset_csv, nrows=DATASET_SIZE, skiprows=skip_rows)
    if NEW_DATASET_CONVERT_CATEGORICAL_TO_DUMMIES:
        N = pd.get_dummies(N)
    if NEW_DATASET_RETAIN_ONLY_NUMERIC_COLUMNS:
        N = N.select_dtypes([np.number])
    if NEW_DATASET_DROP_NA:
        N = N.dropna()
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


def read_and_prep_mnist():
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    X = train_X.reshape(len(train_X),-1)
    X = X.astype(float) / 255.
    return pd.DataFrame(X)


def read_and_prep_higgs():
    full_data: pd.DataFrame = pd.read_csv(HIGGS_DATASET_FILE, nrows=DATASET_SIZE)
    N: pd.DataFrame = full_data.select_dtypes([np.number])
    N = N.dropna()
    return N


def generate_k_gaussians(gaussian_k):
    rng = np.random.RandomState(GAUSSIANS_RANDOM_SEED or None)
    N = rng.normal(scale=GAUSSIANS_STD_DEV, size=(DATASET_SIZE, GAUSSIANS_DIMENSIONS,))
    centers = rng.uniform(size=(gaussian_k, GAUSSIANS_DIMENSIONS,), )
    the_sum = sum(i ** GAUSSIANS_GAMMA for i in range(1, gaussian_k + 1))
    cluster_sizes = [floor(DATASET_SIZE * ((i ** GAUSSIANS_GAMMA) / the_sum)) for i in range(1, gaussian_k + 1)]
    N = N[:sum(cluster_sizes)]

    cluster_slices = []
    next_element = 0
    for cluster_size in cluster_sizes:
        cluster_slices.append((slice(next_element, next_element + cluster_size)))
        next_element += cluster_size

    for k, cluster_slice in enumerate(cluster_slices):
        N[cluster_slice] += centers[k]

    return pd.DataFrame(N)


def determine_gaussians_k(dataset):
    tokens = dataset.split(sep='_')
    if len(tokens) == 2:
        return int(tokens[1])  # "gaussian_25"
    else:
        raise NotImplementedError(f"Bad gaussian dataset name \"{dataset}\". Should be e.g. \"gaussian_25\"")
