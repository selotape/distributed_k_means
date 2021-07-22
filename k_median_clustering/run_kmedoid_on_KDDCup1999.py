from itertools import product
import logging
from time import strftime

from k_median_clustering import competitors
from k_median_clustering.algo import distributed_k_median_clustering
from k_median_clustering.math import risk
import numpy as np
import pandas as pd

log_time = strftime('%Y%m%d%H%M')
logging.basicConfig(filename=f'k_median_clustering_{log_time}.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 6000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logging.info(f"Data size: {len(full_data):,}")

ks = [5]
epsilons = [0.05]
deltas = [0.1]
ms = [50]

for k, dt, m, ep in product(ks, deltas, ms, epsilons):
    logging.info(f'===============================================')
    logging.info(f"======== Starting distributed k median with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
    logging.info(f'===============================================')
    C, iterations = distributed_k_median_clustering(N, k, ep, dt, m)
    dist_k_median_risk = risk(N, C)
    logging.info(f'The k_median_clustering risk is {dist_k_median_risk:,}. the size of C is {len(C)} (where k:{k} after {iterations} iterations')
    logging.info(f'=============================================================================================')

    l = int(len(C) / iterations)

    logging.info(f"1. Starting scalable_k_mean with {iterations} iterations and l=={l}")
    scalable_k_means_C = competitors.scalable_k_means_clustering(N, iterations, l)
    scalable_k_means_risk = risk(N, scalable_k_means_C)
    logging.info(f'The scalable_k_means risk is {scalable_k_means_risk:,} and size of C is {len(scalable_k_means_C)}')
    logging.info(f'=============================================================================================')
    logging.info(f'The ratio (dist_k_median_risk / scalable_k_means) is {dist_k_median_risk / scalable_k_means_risk:,}')
    logging.info(f'=============================================================================================')

    iterations *= 2
    logging.info(f"2. Starting scalable_k_mean with {iterations} iterations and l=={l}")
    scalable_k_means_C = competitors.scalable_k_means_clustering(N, iterations, l)
    scalable_k_means_risk = risk(N, scalable_k_means_C)
    logging.info(f'The scalable_k_means risk is {scalable_k_means_risk:,} and size of C is {len(scalable_k_means_C)}')
    logging.info(f'=============================================================================================')
    logging.info(f'The ratio (dist_k_median_risk / scalable_k_means) is {dist_k_median_risk / scalable_k_means_risk:,}')
    logging.info(f'=============================================================================================')

    iterations *= 2
    logging.info(f"3. Starting scalable_k_mean with {iterations} iterations and l=={l}")
    scalable_k_means_C = competitors.scalable_k_means_clustering(N, iterations, l)
    scalable_k_means_risk = risk(N, scalable_k_means_C)
    logging.info(f'The scalable_k_means risk is {scalable_k_means_risk:,} and size of C is {len(scalable_k_means_C)}')
    logging.info(f'=============================================================================================')
    logging.info(f'The ratio (dist_k_median_risk / scalable_k_means) is {dist_k_median_risk / scalable_k_means_risk:,}')
    logging.info(f'=============================================================================================')
    #
    # logging.info(f'=============================================================================================')
    # logging.info(f'=============================================================================================')
    # logging.info(f'=============================================================================================')
    #
    # logging.info(f"Starting fast clustering")
    # fast_clustering_S = competitors.fast_clustering(N, k, ep, m)
    # fast_clustering_risk = risk(N, fast_clustering_S)
    # logging.info(f'The fast clustering risk is {fast_clustering_risk:,} and size of S is {len(fast_clustering_S)}')
    #
    # logging.info(f'=============================================================================================')
    # logging.info(f'=============================================================================================')
    # logging.info(f'=============================================================================================')

    # logging.info(f"Starting {Blackbox.__name__}")
    # blackbox_risk = competitors.blackbox(N, k, risk)
    # logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')

    # try:
    #     logging.info(f"Starting Spark KMeans")
    #     blackbox_risk = competitors.spark_kmeans(N, k)
    #     logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')
    # except Exception as e:
    #     logging.error(e)
