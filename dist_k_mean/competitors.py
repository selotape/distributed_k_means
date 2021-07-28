import logging
from math import log
from typing import List, Tuple

import pandas as pd
import numpy as np
from random import choice

from dist_k_mean.math import Blackbox, risk, Select, pairwise_distances_argmin_min_squared, alpha_s_formula, alpha_h_formula, measure_weights, A
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession


def spark_kmeans(N, k):
    spark = SparkSession.builder.getOrCreate()
    dataset = spark.createDataFrame(N)
    kmeans = KMeans().setK(k)
    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    return silhouette


def blackbox(N, k):
    kmeans = Blackbox(n_clusters=k, n_jobs=-1).fit(N)
    return risk(N, kmeans.cluster_centers_)


def scalable_k_means_clustering(N: pd.DataFrame, iterations: int, l: int, k: int):
    C = pd.DataFrame().append(N.iloc[[choice(range(len(N)))]])
    psii = risk(N, C)
    for i in range(iterations):
        calcs = pd.DataFrame()

        calcs['dists'] = pairwise_distances_argmin_min_squared(N, C)
        calcs['probs'] = (calcs['dists']) / psii
        # TODO - figure out why this didnt work -  calcs['coin_toss'] = np.random.uniform(size=len(N), )
        draws = np.random.choice(len(N), l, p=calcs['probs'], replace=False)
        C = C.append(N.iloc[draws])
        psii = risk(N, C)


    C_weights = measure_weights(N, C)

    C_final = A(C, k, C_weights)

    return C, C_final


class _FastClusteringReducer:

    def __init__(self, Ri):
        self.Ri: pd.DataFrame = Ri

    def sample_Ss_and_Hs(self, alpha_s, alpha_h) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.Ri.sample(frac=alpha_s), self.Ri.sample(frac=alpha_h)  # TODO - figure out why this always returns exactly alpha

    def remove_handled_points_and_return_remaining(self, S: pd.DataFrame, v: float) -> int:
        if len(self.Ri) == 0:
            return 0

        distances = pairwise_distances_argmin_min_squared(self.Ri, S)
        remaining_points = distances > v
        self.Ri = self.Ri[remaining_points]
        return len(self.Ri)


class _FastClusteringCoordinator:

    def __init__(self, n):
        self.n = n
        self.S = pd.DataFrame()

    def iterate(self, Ss: List[pd.DataFrame], Hs: List[pd.DataFrame]) -> float:
        Stmp = pd.concat(Ss)
        self.S = pd.concat([self.S, Stmp], ignore_index=True)
        H = pd.concat(Hs)  # TODO - do these copy the data? if so, avoid it

        logging.info(f"============ Select start ============")
        v = Select(self.S, H, self.n)
        logging.info(f"============ Select end ============")
        if v == 0.0:
            logging.error("Bad! v == 0.0")

        return v


def fast_clustering(R: pd.DataFrame, k: int, ep: float, m: int):
    n = len(R)
    Rs = np.array_split(R, m)
    reducers = [_FastClusteringReducer(Ri) for Ri in Rs]
    coordinator = _FastClusteringCoordinator(n)

    remaining_elements_count = len(R)
    loop_count = 0
    max_subset_size = (4 / ep) * k * (n ** ep) * log(n)

    logging.info(f"max_subset_size:{max_subset_size}")

    alpha_s, alpha_h = alpha_s_formula(k, n, ep, len(R)), alpha_h_formula(n, ep, len(R))

    while remaining_elements_count > max_subset_size:
        logging.info(f"============ Starting LOOP {loop_count} ============")
        logging.info(f"============ Sampling Ss & Hs ============")
        Ss_and_Hs = [r.sample_Ss_and_Hs(alpha_s, alpha_h) for r in reducers]
        logging.info(f"============ Sampling done ============")

        Ss = [h_and_s[0] for h_and_s in Ss_and_Hs]
        Hs = [h_and_s[1] for h_and_s in Ss_and_Hs]

        logging.info(f"============ iterate start ============")
        v = coordinator.iterate(Ss, Hs)

        logging.info(f"============ len(S): {len(coordinator.S)} ============")
        logging.info(f"============ iterate end ============")

        logging.info(f"============ remove_handled_points_and_return_remaining start ============")
        remaining_elements_count = sum(r.remove_handled_points_and_return_remaining(coordinator.S, v) for r in reducers)
        logging.info(f"============ remove_handled_points_and_return_remaining end ============")
        alpha_s, alpha_h = alpha_s_formula(k, n, ep, remaining_elements_count), alpha_h_formula(n, ep, remaining_elements_count)
        logging.info(f"============ END OF LOOP {loop_count}. "
                     f"remaining_elements_count:{remaining_elements_count}."
                     f" alpha_s:{alpha_s}. alpha_h:{alpha_h}. v:{v}. len(S):{len(coordinator.S)}. "
                     f" len(Ss):{sum(len(s) for s in Ss)}. len(Hs):{sum(len(h) for h in Hs)}"
                     f" max_subset_size:{max_subset_size}. remaining_elements_count:{remaining_elements_count}."
                     f"  ============")
        loop_count += 1

    coordinator.S = pd.concat([r.Ri for r in reducers] + [coordinator.S])

    logging.info(f'loop_count: {loop_count}. len(S):{len(coordinator.S)}')
    return coordinator.S
