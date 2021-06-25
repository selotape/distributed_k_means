import pandas as pd
import numpy as np
from math import log, pow
from typing import Tuple


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


def EstProc(P1: pd.DataFrame, P2: pd.DataFrame, alpha: float, dt: float, k: int, kp: int) -> Tuple[float, pd.DataFrame]:
    """
    calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
    Emits the cluster and the ~risk.
    """
    pass


def A(N, k):
    """
    The blackbox clustering algorithm
    """
    pass


def risk(N, C):
    """
    Sum of distances of samples to their closest cluster center.
    """
    pass
