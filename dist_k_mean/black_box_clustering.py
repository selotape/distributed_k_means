import inspect
import time
from dataclasses import dataclass
from functools import partial
from random import choice
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from dist_k_mean.config import MINI_BATCH_SIZE, INNER_BLACKBOX, FINALIZATION_BLACKBOX
from dist_k_mean.math import risk, pairwise_distances_argmin_min_squared, measure_weights


class ScalableKMeans:
    def __init__(self, n_clusters, iterations, m, l):
        self.m = m
        self.k = n_clusters
        self.l = l
        self.iterations = iterations
        self.cluster_centers_: Union[pd.DataFrame, None] = None
        self._cluster_centers_pre_finalization: Union[pd.DataFrame, None] = None
        self._timing: Union[SkmTiming, None] = None

    def fit(self, N, sample_weight=None):
        if sample_weight:
            raise NotImplementedError("ScalableKMeans doesn't support sample_weights")
        C, C_final, timing = _scalable_k_means(N, self.iterations, self.l, self.k, self.m)
        self.cluster_centers_ = C_final
        self._cluster_centers_pre_finalization = C
        self._timing = timing
        return self


BlackBoxes = {
    'KMeans': KMeans,
    'MiniBatchKMeans': partial(MiniBatchKMeans, batch_size=MINI_BATCH_SIZE),
    'ScalableKMeans': ScalableKMeans,
}


def getAppliedBlackBox(blackbox_name, kwargs):
    BlackBox = BlackBoxes[blackbox_name]
    black_box_args = inspect.getfullargspec(BlackBox)[0]
    relevant_kwargs = {kw: val for kw, val in kwargs.items() if kw in black_box_args}
    return partial(BlackBox, **relevant_kwargs)


def A_inner(N: pd.DataFrame, k: int, sample_weight=None, **kwargs) -> pd.DataFrame:
    BlackBox = getAppliedBlackBox(INNER_BLACKBOX, kwargs)
    return _A(N, k, BlackBox, sample_weight)


def A_final(N: pd.DataFrame, k: int, sample_weight=None) -> pd.DataFrame:
    return _A(N, k, BlackBoxes[FINALIZATION_BLACKBOX], sample_weight)


def _A(N: pd.DataFrame, k: int, Blackbox, sample_weight=None) -> pd.DataFrame:
    """
    Runs the blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    return pd.DataFrame(Blackbox(n_clusters=k)
                        .fit(N, sample_weight=sample_weight)
                        .cluster_centers_)


@dataclass
class SkmTiming:
    reducers_time: float = 0
    finalization_time: float = 0

    def total_time(self):
        return self.reducers_time + self.finalization_time


def _scalable_k_means(N: pd.DataFrame, iterations: int, l: int, k: int, m) -> Tuple[pd.DataFrame, pd.DataFrame, SkmTiming]:
    start = time.time()
    timing = SkmTiming()
    C = pd.DataFrame().append(N.iloc[[choice(range(len(N)))]])
    Ctmp = C
    prev_distances_to_C = None
    for i in range(iterations):
        psii = risk(N, C)
        distances_to_Ctmp = pairwise_distances_argmin_min_squared(N, Ctmp)
        prev_distances_to_C = np.minimum(distances_to_Ctmp, prev_distances_to_C) if prev_distances_to_C is not None else distances_to_Ctmp
        probabilities = prev_distances_to_C / psii

        draws = np.random.choice(len(N), l, p=probabilities, replace=False)
        Ctmp = N.iloc[draws]
        C = C.append(Ctmp)

    C_weights = measure_weights(N, C)

    timing.reducers_time = (time.time() - start) / m

    start = time.time()
    C_final = A_final(C, k, C_weights)
    timing.finalization_time = time.time() - start

    return C, C_final, timing
