from itertools import product
from time import strftime

from k_median_clustering import competitors
from k_median_clustering.algo import distributed_k_median_clustering
from k_median_clustering.math import risk
from k_median_clustering.utils import setup_logger
import numpy as np
import pandas as pd

log_time = strftime('%Y%m%d%H%M')
logger = setup_logger('full_log', f'k_median_clustering_log_{log_time}.log', with_console=True)
results = setup_logger('results_log', f'k_median_clustering_results_{log_time}.log')

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 6000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logger.info(f"Data size: {len(full_data):,}")

ks = [500, ]
epsilons = [0.2]
deltas = [0.1]
ms = [50]


def summarize(i, name, k, dt, m, ep, len_dkm_C, iters, l, len_skm_C, dkm_risk, skm_risk, dkm_risk_final, skm_risk_final):
    return f'{i}, {name}, k={k}, dt={dt}, m={m}, ep={ep}, len(dkm_C)={len_dkm_C}, iters={iters}, l={l}, len(skm_C)={len_skm_C},' \
           f' (dkm_risk/skm_risk)={dkm_risk / skm_risk:,}, (dkm_risk_final/skm_risk_final)={dkm_risk_final / skm_risk_final:,}\n'


def main():
    for i, (k, dt, m, ep) in enumerate(product(ks, deltas, ms, epsilons)):
        logger.info(f'=============================================================================================')
        logger.info(f"======== Starting distributed k median with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
        logger.info(f'=============================================================================================')
        results.info('Starting...\n')
        dkm_C, dkm_C_final, iters = distributed_k_median_clustering(N, k, ep, dt, m)
        dkm_risk = risk(N, dkm_C)
        dkm_risk_f = risk(N, dkm_C_final)
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')
        results.info(f'len(N):{len(N)}. dkm_risk:{dkm_risk:,}. dkm_risk_final:{dkm_risk_f:,}. len(dkm_C):{len(dkm_C)}. len(dkm_C_final):{len(dkm_C_final)}')

        l = int(len(dkm_C) / iters)

        logger.info(f"1. Starting scalable_k_mean with {iters} iterations and l=={l}")
        skm_C, skm_C_final = competitors.scalable_k_means_clustering(N, iters, l, k)
        skm_risk = risk(N, skm_C)
        skm_risk_f = risk(N, skm_C_final)
        logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
        logger.info(f'=============================================================================================')
        test_summary = summarize(i, 'skm 1', k, dt, m, ep, len(dkm_C), iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f)
        logger.info(test_summary)
        results.info(test_summary)
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')

        iters *= 2
        logger.info(f"2. Starting scalable_k_mean with {iters} iterations and l=={l}")
        skm_C, skm_C_final = competitors.scalable_k_means_clustering(N, iters, l, k)
        skm_risk = risk(N, skm_C)
        skm_risk_f = risk(N, skm_C_final)
        logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
        logger.info(f'=============================================================================================')
        test_summary = summarize(i, 'skm 1', k, dt, m, ep, len(dkm_C), iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f)
        logger.info(test_summary)
        results.info(test_summary)
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')
        logger.info(f'=============================================================================================')

        iters *= 2
        logger.info(f"3. Starting scalable_k_mean with {iters} iterations and l=={l}")
        skm_C, skm_C_final = competitors.scalable_k_means_clustering(N, iters, l, k)
        skm_risk = risk(N, skm_C)
        skm_risk_f = risk(N, skm_C_final)
        logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
        logger.info(f'=============================================================================================')
        test_summary = summarize(i, 'skm 1', k, dt, m, ep, len(dkm_C), iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f)
        logger.info(test_summary)
        results.info(test_summary)
        logger.info(f'=============================================================================================')
        #
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        #
        # logger.info(f"Starting fast clustering")
        # fast_clustering_S = competitors.fast_clustering(N, k, ep, m)
        # fast_clustering_risk = risk(N, fast_clustering_S)
        # logger.info(f'The fast clustering risk is {fast_clustering_risk:,} and size of S is {len(fast_clustering_S)}')
        #
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')

        # logger.info(f"Starting {Blackbox.__name__}")
        # blackbox_risk = competitors.blackbox(N, k, risk)
        # logger.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')

        # try:
        #     logger.info(f"Starting Spark KMeans")
        #     blackbox_risk = competitors.spark_kmeans(N, k)
        #     logger.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')
        # except Exception as e:
        #     logger.error(e)


if __name__ == "__main__":
    # execute only if run as a script
    main()
