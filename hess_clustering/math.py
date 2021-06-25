from math import log, pow
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids


def kplus_formula(k: int, dt: float):
    """
    The allowed size of the "k+" clusters group
    """
    return int(k + 38 * log(32 * k / dt))  # TODO - ask Tom if round up or down


def max_subset_size_formula(n: int, k: int, ep: float, dt: float):
    """
    The size above which data doesn't fit inside a single machine,
    so clustering must be distributed.
    """
    return 10 * k * pow(n, ep) * log(8 * k / dt)


def alpha_formula(n, k, ep, dt, N_size):
    """
    The probability to draw a datum into P1/P2 samples

    10k n^ep log(8k/dt) / |N|
    """
    return max_subset_size_formula(n, k, ep, dt) / N_size


def distance(x: np.array, y: np.array):
    """
    the distance func for any two points in the space.
    For now, it's a simple euclidean distance.
    """
    return np.linalg.norm(x - y)


def A(N: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    The blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    return pd.DataFrame(KMedoids(n_clusters=k).fit(N).cluster_centers_)


def EstProc(P1: pd.DataFrame, P2: pd.DataFrame, alpha: float, dt: float, k: int, kp: int) -> Tuple[float, pd.DataFrame]:
    """
    calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
    Emits the cluster and the ~risk.
    """
    raise NotImplementedError


def risk(N, C):
    """
    Sum of distances of samples to their closest cluster center.
    """
    raise NotImplementedError
