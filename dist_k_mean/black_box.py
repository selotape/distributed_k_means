import inspect
from functools import partial
from typing import Union

import faiss
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from dist_k_mean.competitors.competitors import scalable_k_means
from dist_k_mean.config import MINI_BATCH_SIZE, INNER_BLACKBOX, FINALIZATION_BLACKBOX
from dist_k_mean.utils import Measurement


def getAppliedBlackBox(blackbox_name, kwargs, k):
    kwargs.update({'k': k})
    BlackBox = BlackBoxes[blackbox_name]
    black_box_args = inspect.getfullargspec(BlackBox)[0]
    relevant_kwargs = {kw: val for kw, val in kwargs.items() if kw in black_box_args}
    return partial(BlackBox, **relevant_kwargs)


def A_inner(N: pd.DataFrame, k: int, sample_weight=None, **kwargs) -> pd.DataFrame:
    kwargs.update({'n_dims': len(N.columns)})

    BlackBox = getAppliedBlackBox(INNER_BLACKBOX, kwargs, k)
    return _A(N, k, BlackBox, sample_weight)


def A_final(N: pd.DataFrame, k: int, sample_weight=None) -> pd.DataFrame:
    BlackBox = getAppliedBlackBox(FINALIZATION_BLACKBOX, {}, k)
    return _A(N, k, BlackBox, sample_weight)


def _A(N: pd.DataFrame, k: int, Blackbox, sample_weight=None) -> pd.DataFrame:
    """
    Runs the blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    return pd.DataFrame(Blackbox(n_clusters=k)
                        .fit(N, sample_weight=sample_weight)
                        .cluster_centers_)


class ScalableKMeans:
    def __init__(self, n_clusters, iterations, m, l):
        self.m = m
        self.k = n_clusters
        self.l = l
        self.iterations = iterations
        self.cluster_centers_: Union[pd.DataFrame, None] = None
        self._cluster_centers_pre_finalization: Union[pd.DataFrame, None] = None
        self._timing: Union[Measurement, None] = None

    def fit(self, N, sample_weight=None):
        if sample_weight:
            raise NotImplementedError("ScalableKMeans doesn't support sample_weights")
        C, C_final, timing = scalable_k_means(N, self.iterations, self.l, self.k, self.m, A_final)
        self.cluster_centers_ = C_final
        self._cluster_centers_pre_finalization = C
        self._timing = timing
        return self


class FaissKMeans:
    def __init__(self, n_clusters, n_dims):
        self.d = n_dims
        self.k = n_clusters
        self.cluster_centers_: Union[pd.DataFrame, None] = None
        self._cluster_centers_pre_finalization: Union[pd.DataFrame, None] = None

    def fit(self, N, sample_weight=None):
        if sample_weight:
            raise NotImplementedError("Faiss doesn't support sample_weights")
        kmeans = faiss.Kmeans(self.d, self.k)
        kmeans.train(np.float32(np.ascontiguousarray(N)), sample_weight)
        self.cluster_centers_ = kmeans.centroids
        return self


BlackBoxes = {
    'KMeans': KMeans,
    'MiniBatchKMeans': partial(MiniBatchKMeans, batch_size=MINI_BATCH_SIZE),
    'ScalableKMeans': ScalableKMeans,
    'FaissKMeans': FaissKMeans,
}
