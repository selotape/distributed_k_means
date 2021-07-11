import logging

from k_median_clustering.algo import k_median_clustering
from k_median_clustering import competitors
from k_median_clustering.math import risk, Blackbox
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

# DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data_10_percent_corrected.csv"
DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 60000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logging.info(f"Data size: {len(full_data):,}")

k = 50
dt = 0.1
m = 50
ep = 0.2

logging.info(f"Starting distributed k median")
C = k_median_clustering(N, k, ep, dt, m)
logging.info(f"Final size of C:{len(C)} (where k:{k})")
k_median_risk = risk(N, C)
logging.info(f'The k_median_clustering risk is {k_median_risk:,}')

logging.info(f"Starting {Blackbox.__name__}")
blackbox_risk = competitors.blackbox(N, k, risk)
logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')

logging.info(f"Starting Spark KMeans")
blackbox_risk = competitors.spark_kmeans(N, k)
logging.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')
