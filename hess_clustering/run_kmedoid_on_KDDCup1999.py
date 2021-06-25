import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

from random import choice

from hess_clustering.algo import hess_clustering, risk

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data_10_percent_corrected.csv"
SUBSET_SIZE = 100

Ks = [20, 50, 100]
Deltas = [0.1, 0.01]
Ms = [100, 200]
epsilons = [0.1, 0.15, 0.2]

k = choice(Ks)
dt = choice(Deltas)
m = choice(Ms)
ep = choice(epsilons)

full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])

kmedoids = KMedoids(n_clusters=k, random_state=0).fit(N)
print(f'the kmedoids.inertia is {kmedoids.inertia_}')

C = hess_clustering(N, k, ep, dt, m)
hess_risk = risk(N, C)
print(f'the hess_clustering risk is {hess_risk}')
