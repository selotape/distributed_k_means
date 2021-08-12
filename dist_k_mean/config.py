import os
import sys

RUN_NAME = 'SCHMOD' if len(sys.argv) < 2 else sys.argv[1]

###### DATA SETS ######
DATASET = os.getenv('DATASET', default='gaussian')  # 'gaussian', 'kdd'

KDD_DATASET_FILE = os.getenv('KDD_DATASET_FILE', default="data_samples/kddcup99/kddcup.data")  # const
KDD_SUBSET_SIZE = int(os.getenv('KDD_SUBSET_SIZE', default=6_000_000))

GAUSSIANS_DIMENSIONS = int(os.getenv('GAUSSIANS_DIMENSIONS', default=15))
GAUSSIANS_NUM_POINTS = int(os.getenv('GAUSSIANS_NUM_POINTS', default=10_000_000))
GAUSSIANS_K = int(os.getenv('GAUSSIANS_K', default=100))
GAUSSIANS_ALPHA = float(os.getenv('GAUSSIANS_ALPHA', default=0.0))
GAUSSIANS_STD_DEV = float(os.getenv('GAUSSIANS_STD_DEV', default=0.1))

###### BLACK_BOXES ######
INNER_BLACKBOX = os.getenv('INNER_BLACKBOX', default='KMeans')  # 'KMeans' 'MiniBatchKMeans' 'ScalableKMeans'
INNER_BLACKBOX_ITERATIONS = int(os.getenv('INNER_BLACKBOX_ITERATIONS', default=4))
INNER_BLACKBOX_L_TO_K_RATIO = int(os.getenv('INNER_BLACKBOX_L_TO_K_RATIO', default=1))

FINALIZATION_BLACKBOX = os.getenv('FINALIZATION_BLACKBOX', default='KMeans')

MINI_BATCH_SIZE = int(os.getenv('MINI_BATCH_SIZE', default=1000))

###### DISTRIBUTED PARAMS ######
ROUNDS = int(os.getenv('ROUNDS', default=5))
KS = [25, 50, 100]
EPSILONS = [0.1, 0.2]
DELTAS = [0.1]
MS = [50]

CONST_MODE = 'fast'  # 'fast' 'strict'
PHI_ALPHA = {
    'strict': 6.5,
    'fast': 5.0,
}
MAX_SS_SIZE = {
    'strict': 38,
    'fast': 16,
}
KPLUS = {
    'strict': 9,
    'fast': 8,
}

###### SKM PARAMS ######
L_TO_K_RATIOS = [1, 2, 5]
SKM_ITERATIONS = [2, 3, 4, 5]
