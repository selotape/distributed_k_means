import pandas as pd
import numpy as np
from typing import Tuple


def sample_P1_P2(alpha: float, N: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples two independent subsets P1 & P2 where each member of N has
    probability alpha to appear in P1/P2.
    """
    return N, pd.DataFrame()


def split(N, m):
    """
    Takes all data N and returns N/m equasized subsets of all data
    """
    return [N] + [pd.DataFrame()] * (m - 1)


def kplus_formula(k: int, dt: float):
    """
    The allowed size of the "k+" clusters group

    k+38log(32k/delta)
    """
    return k + 10


def max_subset_size_formula(k: int, ep: float, dt: float):
    """
    The size above which data doesn't fit inside a single machine,
    so clustering must be distributed.

    10k n^ep log(8k/dt)
    """
    return k + ep + dt


def alpha_formula(k, ep, dt, N_size):
    """
    The probability to draw a datum into P1/P2 samples

    10k n^ep log(8k/dt) / |N|
    """
    return max_subset_size_formula(k, ep, dt) / N_size


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