import time
from dataclasses import dataclass
from random import choice
from typing import Tuple

from dist_k_mean.black_box_clustering import A_final
from dist_k_mean.math import risk, pairwise_distances_argmin_min_squared, measure_weights
import numpy as np
import pandas as pd


@dataclass
class SkmTiming:
    reducers_time: float = 0
    finalization_time: float = 0

    def total_time(self):
        return self.reducers_time + self.finalization_time


def scalable_k_means(N: pd.DataFrame, iterations: int, l: int, k: int, m) -> Tuple[pd.DataFrame, pd.DataFrame, SkmTiming]:
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

