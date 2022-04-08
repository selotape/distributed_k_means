import time
from random import choice
from typing import Tuple

import numpy as np
import pandas as pd

from soccer.math import pairwise_distances_argmin_min_squared, measure_weights, \
    risk
from soccer.utils import SimpleMeasurement, Measurement


def scalable_k_means(N: pd.DataFrame, iterations: int, l: int, k: int, m, finalize) -> Tuple[pd.DataFrame, pd.DataFrame, Measurement]:
    start = time.time()
    measurement = SimpleMeasurement()
    measurement.iterations_ = iterations
    C = pd.DataFrame().append(N.iloc[[choice(range(len(N)))]])
    Ctmp = C
    prev_distances_to_C = None
    for i in range(iterations):
        psii = risk(N, C)
        if psii == 0.0:
            break
        measurement.dist_comps_ += int((len(N) * len(Ctmp)) / m)
        distances_to_Ctmp = pairwise_distances_argmin_min_squared(N, Ctmp)
        prev_distances_to_C = np.minimum(distances_to_Ctmp, prev_distances_to_C) if prev_distances_to_C is not None else distances_to_Ctmp
        probabilities: np.ndarray = prev_distances_to_C / psii

        draws = draw_from_N(N, l, probabilities)
        Ctmp = N.iloc[draws]
        C = C.append(Ctmp)

    measurement.num_centers_unfinalized_ = len(C)
    measurement.reducers_time_ = (time.time() - start) / m

    C_weights = measure_weights(N, C)
    start = time.time()
    C_final = finalize(C, k, C_weights)
    measurement.finalization_time_ = time.time() - start

    return C, C_final, measurement


def draw_from_N(N, l, probabilities):
    try:
        draws = np.random.choice(len(N), l, p=probabilities, replace=False)
    except ValueError as e:
        print(e)
        if 'Fewer non-zero entries in p than size' in str(e):
            draws = N[probabilities != 0.0]
        else:
            print('Major bummer. This shouldn\'t happen')
    return draws
