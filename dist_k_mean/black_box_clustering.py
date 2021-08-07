from functools import partial

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from dist_k_mean.config import MINI_BATCH_SIZE, INNER_BLACKBOX, FINALIZATION_BLACKBOX

BlackBoxes = {
    'KMeans': KMeans,
    'MiniBatchKMeans': partial(MiniBatchKMeans, batch_size=MINI_BATCH_SIZE),
    'ScalableKMeans': None,
}

InnerBlackbox = BlackBoxes[INNER_BLACKBOX]
FinalizationBlackbox = BlackBoxes[FINALIZATION_BLACKBOX]


def A_inner(N: pd.DataFrame, k: int, sample_weight=None) -> pd.DataFrame:
    return _A(N, k, InnerBlackbox, sample_weight)


def A_final(N: pd.DataFrame, k: int, sample_weight=None) -> pd.DataFrame:
    return _A(N, k, FinalizationBlackbox, sample_weight)


def _A(N: pd.DataFrame, k: int, Blackbox, sample_weight=None) -> pd.DataFrame:
    """
    Runs the blackbox offline clustering algorithm. Returns the k chosen clusters
    """
    return pd.DataFrame(Blackbox(n_clusters=k)
                        .fit(N, sample_weight=sample_weight)
                        .cluster_centers_)
