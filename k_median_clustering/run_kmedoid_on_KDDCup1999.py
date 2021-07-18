import logging

from k_median_clustering import competitors
from k_median_clustering.algo import k_median_clustering
from k_median_clustering.math import risk, Blackbox
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 6000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logging.info(f"Data size: {len(full_data):,}")

k = 50
dt = 0.1
m = 50
ep = 0.2



logging.info(f"Starting distributed k median")
C = k_median_clustering(N, k, ep, dt, m)
k_median_risk = risk(N, C)
logging.info(f'The k_median_clustering risk is {k_median_risk:,} and the size of C is {len(C)} (where k:{k}')
logging.info(f'=============================================================================================')
logging.info(f'=============================================================================================')
logging.info(f'=============================================================================================')


iterations = 3
l = int(2.1 * k)

logging.info(f"Starting scalable_k_mean")
scalable_k_means_C = competitors.scalable_k_means_clustering(N, iterations, l)
scalable_k_means_risk = risk(N, scalable_k_means_C)
logging.info(f'The scalable_k_means risk is {scalable_k_means_risk:,} and size of C is {len(scalable_k_means_C)}')


# logging.info(f"Starting fast clustering")
# fast_clustering_S = competitors.fast_clustering(N, k, ep, m)
# fast_clustering_risk = risk(N, fast_clustering_S)
# logging.info(f'The fast clustering risk is {fast_clustering_risk:,} and size of S is {len(fast_clustering_S)}')

# logging.info(f"Starting {Blackbox.__name__}")
# blackbox_risk = competitors.blackbox(N, k, risk)
# logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')


# try:
#     logging.info(f"Starting Spark KMeans")
#     blackbox_risk = competitors.spark_kmeans(N, k)
#     logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')
# except Exception as e:
#     logging.error(e)
