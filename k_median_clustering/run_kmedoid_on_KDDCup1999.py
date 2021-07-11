import logging
from random import choice

from k_median_clustering.algo import k_median_clustering
from k_median_clustering.math import risk, Blackbox
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data_10_percent_corrected.csv"
SUBSET_SIZE = 600000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logging.info(f"Data size: {len(full_data):,}")

Ks = [50]  # , 50, 100]
Deltas = [0.1]  # , 0.01]
Ms = [50]  # , 200]
epsilons = [0.2]  # , 0.15, 0.2]

k = choice(Ks)
dt = choice(Deltas)
m = choice(Ms)
ep = choice(epsilons)

logging.info(f"Starting distributed k median")
C = k_median_clustering(N, k, ep, dt, m)
logging.info(f"Final size of C:{len(C)} x(and k:{k})")
k_median_risk = risk(N, C)
logging.info(f'The k_median_clustering risk is {k_median_risk:,}')

logging.info(f"Starting Blackbox")
kmeans = Blackbox(n_clusters=k, n_jobs=-1).fit(N)
kmeans_risk = risk(N, kmeans.cluster_centers_)
logging.info(f'The {Blackbox.__name__} risk is {kmeans_risk:,}')
