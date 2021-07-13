from math import log, pow
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn_extra.cluster import KMedoids


def kplus_formula(k: int, dt: float):
    """
    The allowed size of the "k+" clusters group
    """
    return int(k + 10 * log(8 * k / dt))


def max_subset_size_formula(n: int, k: int, ep: float, dt: float):
    """
    The size above which data doesn't fit inside a single machine,
    so clustering must be distributed.
    """
    return 10 * k * pow(n, ep) * log(8 * k / dt)


def alpha_formula(n, k, ep, dt, N_current_size):
    """
    The probability to draw a datum into P1/P2 samples

    10k n^ep log(8k/dt) / |N|
    """
    return max_subset_size_formula(n, k, ep, dt) / N_current_size


distance = 'euclidean'
Blackbox = [KMeans, KMedoids][0]


def A(N: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    The blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    return pd.DataFrame(Blackbox(n_clusters=k)
                        .fit(N)
                        .cluster_centers_)


def risk_kmeans(N: pd.DataFrame, C: pd.DataFrame):
    """
    Sum of distances of samples to their closest cluster center.
    """
    _, distances = pairwise_distances_argmin_min(N, C, metric=distance)

    return _risk_of_distances(distances)


def risk_truncated(P2, C, r):
    _, distances = pairwise_distances_argmin_min(P2, C, metric=distance)
    distances.sort()

    if r >= len(P2):
        return 0  # The "trivial risk"

    return _risk_of_distances(distances[:len(distances) - r])


def _risk_of_distances(distances):
    return np.sum(np.square(distances))


risk = risk_kmeans


def phi_alpha_formula(alpha: float, k: int, dt: float):
    """
    The size of the already-handled clusters
    """
    return (10 / alpha) * log(8 * k / dt)


def r_formula(alpha: float, k: int, phi_alpha: float) -> int:
    return int(2 * alpha * (k + 1) * phi_alpha)


def v_formula(psi: float, k: int, phi_alpha: float):
    return psi / (k * phi_alpha)


def EstProc(P1: pd.DataFrame, P2: pd.DataFrame, alpha: float, dt: float, k: int, kp: int) -> Tuple[float, pd.DataFrame]:
    """
    calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
    Emits the cluster and the ~risk.
    """
    Ta = A(P1, kp)

    phi_alpha = phi_alpha_formula(alpha, k, dt)
    r = r_formula(alpha, k, phi_alpha)
    Rr = risk_truncated(P2, Ta, r)

    psi = (1 / (3 * alpha)) * Rr
    return v_formula(psi, k, phi_alpha), Ta
