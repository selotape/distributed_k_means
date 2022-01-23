import inspect
from collections import namedtuple
from functools import partial
from typing import Union

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from soccer.competitors.competitors import scalable_k_means
from soccer.config import MINI_BATCH_SIZE
from soccer.utils import Measurement


CentersAndMeasurement = namedtuple("CentersAndMeasurement", ['centers', 'measurement'])


def getAppliedBlackBox(blackbox_name, kwargs, k):
    kwargs.update({'k': k})
    BlackBox = BlackBoxes[blackbox_name]
    black_box_args = inspect.getfullargspec(BlackBox)[0]
    relevant_kwargs = {kw: val for kw, val in kwargs.items() if kw in black_box_args}
    return partial(BlackBox, **relevant_kwargs)

DEFAULT_BLACKBOX = 'KMeans'

def A_inner(N: pd.DataFrame, k: int, blackbox=DEFAULT_BLACKBOX, sample_weight=None, **kwargs) -> CentersAndMeasurement:
    kwargs.update({'n_dims': len(N.columns)})

    BlackBox = getAppliedBlackBox(blackbox, kwargs, k)
    return _A(N, k, BlackBox, sample_weight)

def A_final(N: pd.DataFrame, k: int, blackbox=DEFAULT_BLACKBOX, sample_weight=None) -> pd.DataFrame:
    BlackBox = getAppliedBlackBox(blackbox, {}, k)
    return _A(N, k, BlackBox, sample_weight).centers


def _A(N: pd.DataFrame, k: int, Blackbox, sample_weight=None) -> CentersAndMeasurement:
    """
    Runs the blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    blackbox = Blackbox(n_clusters=k)
    blackbox.fit(N, sample_weight=sample_weight)
    return CentersAndMeasurement(pd.DataFrame(blackbox.cluster_centers_), getattr(blackbox, "_measurement", None))



class ScalableKMeans:
    def __init__(self, n_clusters, iterations, m, l):
        self.m = m
        self.k = n_clusters
        self.l = l
        self.iterations = iterations
        self.cluster_centers_: Union[pd.DataFrame, None] = None
        self._cluster_centers_pre_finalization: Union[pd.DataFrame, None] = None
        self._measurement: Union[Measurement, None] = None

    def fit(self, N, sample_weight=None):
        if sample_weight:
            raise NotImplementedError("ScalableKMeans doesn't support sample_weights")
        C, C_final, measurement = scalable_k_means(N, self.iterations, self.l, self.k, self.m, A_final)
        self.cluster_centers_ = C_final
        self._cluster_centers_pre_finalization = C
        self._measurement = measurement
        return self



BlackBoxes = {
    'KMeans': KMeans,
    'MiniBatchKMeans': partial(MiniBatchKMeans, batch_size=MINI_BATCH_SIZE),
    'ScalableKMeans': ScalableKMeans,
}
