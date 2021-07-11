import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from random import choice

from k_median_clustering.algo import k_median_clustering
from k_median_clustering.math import risk

import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data_10_percent_corrected.csv"
SUBSET_SIZE = 600000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])

Ks = [5]  # , 50, 100]
Deltas = [0.1]  # , 0.01]
Ms = [50]  # , 200]
epsilons = [0.2]  # , 0.15, 0.2]

k = choice(Ks)
dt = choice(Deltas)
m = choice(Ms)
ep = choice(epsilons)

C = k_median_clustering(N, k, ep, dt, m)
logging.info(f"Final size of C:{len(C)} x(and k:{k})")
k_median_risk = risk(N, C)
logging.info(f'The k_median_clustering risk is {k_median_risk}')

# kmedoids = KMedoids(n_clusters=k).fit(N)
# logging.info(f'The kmedoids.inertia is {kmedoids.inertia_}')

kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(N)
kmeans_risk = risk(N, kmeans.cluster_centers_)
logging.info(f'The kmeans_risk is {kmeans_risk}')
