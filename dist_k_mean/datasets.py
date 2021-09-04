from functools import lru_cache
from math import floor

from sklearn.preprocessing import MinMaxScaler  # , StandardScaler,

import numpy as np
import pandas as pd

from dist_k_mean.config import *


@lru_cache
def get_dataset(dataset, logger):
    logger.info(f"Loading Dataset {DATASET}...")
    if dataset == 'kdd':
        N = read_and_prep_kdd()
    elif dataset == 'gaussian':
        N = generate_k_gaussians()
    elif dataset == 'power':
        N = read_and_prep_power_consumption()
    elif dataset == 'covtype':
        N = read_and_prep_covtype()
    elif dataset == 'bigcross':
        N = read_and_prep_bigcross()
    elif dataset == 'census1990':
        N = read_and_prep_census1990()
    elif dataset == 'skin':
        N = read_and_prep_skin()
    elif dataset == 'higgs':
        N = read_and_prep_higgs()
    elif dataset == 'activity':
        N = read_and_prep_activity_recognition()
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


def read_and_prep_kdd():
    full_data = pd.read_csv(KDD_DATASET_FILE, nrows=DATASET_SIZE)
    N = full_data.select_dtypes([np.number])
    return N


def read_and_prep_covtype():
    full_data = pd.read_csv(COVTYPE_DATASET_FILE, nrows=DATASET_SIZE)
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
    census = census.iloc[: , 1:]  # throw away first index column
    return census


def read_and_prep_power_consumption():
    dtypes = {"Date": 'str',
              'Time': 'str',
              'Global_active_power': 'float64',
              'Global_reactive_power': 'float64',
              'Voltage': 'float64',
              'Global_intensity': 'float64',
              'Sub_metering_1': 'float64',
              'Sub_metering_2': 'float64',
              'Sub_metering_3': 'float64',
              }
    full_data: pd.DataFrame = pd.read_csv(POWER_DATASET_FILE, nrows=DATASET_SIZE, skiprows=1, sep=';', dtype=dtypes)
    N: pd.DataFrame = full_data.select_dtypes([np.number])
    N = N.dropna()
    return N


def read_and_prep_skin():
    full_data: pd.DataFrame = pd.read_csv(SKIN_DATASET_FILE, nrows=DATASET_SIZE, sep='\t')
    N = full_data.iloc[:, :-1]  # drop labels
    return N


def read_and_prep_higgs():
    full_data: pd.DataFrame = pd.read_csv(HIGGS_DATASET_FILE, nrows=DATASET_SIZE)
    N: pd.DataFrame = full_data.select_dtypes([np.number])
    N = N.dropna()
    return N

def read_and_prep_activity_recognition():
    full_data: pd.DataFrame = pd.read_csv(ACTIVITY_DATASET_FILE, nrows=DATASET_SIZE, sep='\t')
    N: pd.DataFrame = full_data.select_dtypes([np.number])
    N = N.dropna()
    return N


def generate_k_gaussians():
    N = np.random.normal(scale=GAUSSIANS_STD_DEV, size=(DATASET_SIZE, GAUSSIANS_DIMENSIONS,))
    centers = np.random.uniform(size=(GAUSSIANS_K, GAUSSIANS_DIMENSIONS,))
    cluster_sizes = get_cluster_sizes()
    N = N[:sum(cluster_sizes)]

    cluster_slices = []
    next_element = 0
    for cluster_size in cluster_sizes:
        cluster_slices.append((slice(next_element, next_element + cluster_size)))
        next_element += cluster_size

    for k, cluster_slice in enumerate(cluster_slices):
        N[cluster_slice] += centers[k]

    return pd.DataFrame(N)


def get_cluster_sizes():
    if GAUSSIANS_TYPE == 'alpha':
        the_sum = sum(i ** GAUSSIANS_GAMMA for i in range(1, GAUSSIANS_K + 1))
        cluster_sizes = [floor(DATASET_SIZE * ((i ** GAUSSIANS_GAMMA) / the_sum)) for i in range(1, GAUSSIANS_K + 1)]
    elif GAUSSIANS_TYPE == 'exp':
        the_sum = sum(2 ** i for i in range(1, GAUSSIANS_K + 1))
        cluster_sizes = [floor(DATASET_SIZE * ((2 ** i) / the_sum)) for i in range(1, GAUSSIANS_K + 1)]
    else:
        raise NotImplementedError(f"No such gaussian type {GAUSSIANS_TYPE}")
    return cluster_sizes


if __name__ == '__main__':
    print(generate_k_gaussians())
