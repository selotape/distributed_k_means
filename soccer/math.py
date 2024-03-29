import logging
from math import log, pow

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, pairwise_distances_argmin

from soccer.config import PHI_ALPHA_C, MAX_SS_SIZE_C, KPLUS_C, KPLUS_SCALER


def kplus_formula(k: int, dt: float, ep: float):
    """
    The allowed size of the "k+" clusters group
    """
    return KPLUS_SCALER * int(k + KPLUS_C * log(1.1 * k / (dt * ep)))


def max_subset_size_formula(n: int, k: int, ep: float, dt: float):
    """
    The size above which data doesn't fit inside a single machine,
    so clustering must be distributed.
    """
    return int(MAX_SS_SIZE_C * k * pow(n, ep) * log(1.1 * k / dt))


def alpha_formula(n, k, ep, dt, N_current_size):
    """
    The probability to draw a datum into P1/P2 samples

    10k n^ep log(8k/dt) / |N|
    """
    return max_subset_size_formula(n, k, ep, dt) / N_current_size


def risk_kmeans(N: pd.DataFrame, C: pd.DataFrame):
    """
    Sum of distances of samples to their closest cluster center.
    """
    distances = pairwise_distances_argmin_min_squared(N, C)

    return np.sum(distances)


def risk_truncated(P2, C, r):

    if r >= len(P2):
        return 0  # The "trivial risk"

    distances = pairwise_distances_argmin_min_squared(P2, C)
    distances.sort()

    return np.sum(distances[:len(distances) - r])


risk = risk_kmeans


def phi_alpha_formula(alpha: float, k: int, dt: float, ep: float):
    """
    In the paper, is equal to (d_k / alpha)
    """
    return (PHI_ALPHA_C / alpha) * log(1.1 * k / (dt * ep))


def r_formula(alpha: float, k: int, phi_alpha: float) -> int:
    return int(1.5 * alpha * (k + 1) * phi_alpha)


def v_formula(psi: float, k: int, phi_alpha: float):
    return psi / (k * phi_alpha)


def pairwise_distances_argmin_min_squared(X, Y) -> ndarray:
    linear_dists = pairwise_distances_argmin_min(X, Y)[1]
    square_dists = np.square(linear_dists)
    return square_dists


def alpha_s_formula(k, n, ep, len_R):
    return 9 * k * (n ** ep) * log(n) / len_R


def alpha_h_formula(n, ep, len_R):
    return 4 * (n ** ep) * log(n) / len_R


def Select(S, H, n):
    dists: np.ndarray = pairwise_distances_argmin_min_squared(H, S)
    dists.sort()

    if len(dists) < 8 * log(n):
        logging.warning("len(dists) < 8*log(n) ☹️💔")
        return dists[0]

    return dists[int(-8 * log(n))]


def measure_weights(N, C):
    chosen_centers = pairwise_distances_argmin(N, C)
    center_weights = np.zeros((len(C),), dtype=np.intc)
    for cc in chosen_centers:
        center_weights[cc] += 1
    return center_weights
