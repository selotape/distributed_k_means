import os
import sys
from time import strftime

"""
The default config values. Override these via ENVIRONMENT VARIABLES.
"""

RUN_NAME = 'RUN_NAME' if len(sys.argv) < 2 else sys.argv[1]

###### DATA SETS ######
KDD_DATASET_FILE = "datasets/kddcup99/kddcup.data"
CENSUS1990_DATASET_FILE = "datasets/census1990/USCensus1990.data.txt"
BIGCROSS_DATASET_FILE = "datasets/bigcross/BigCross.data"
HIGGS_DATASET_FILE = "datasets/higgs/HIGGS.csv"
NEW_DATASET_SKIP_HEADER = bool(os.getenv('NEW_DATASET_SKIP_HEADER', default=False))
NEW_DATASET_CONVERT_CATEGORICAL_TO_DUMMIES = bool(os.getenv('NEW_DATASET_CONVERT_CATEGORICAL_TO_DUMMIES', default=False))
NEW_DATASET_RETAIN_ONLY_NUMERIC_COLUMNS = bool(os.getenv('NEW_DATASET_RETAIN_ONLY_NUMERIC_COLUMNS', default=True))
NEW_DATASET_DROP_NA = bool(os.getenv('NEW_DATASET_DROP_NA', default=True))
DATASET_SIZE = int(os.getenv('DATASET_SIZE', default=10_000_000))
SCALE_DATASET = bool(os.getenv('SCALE_DATASET', default=False))

###### Gaussian dataset ######
GAUSSIANS_DIMENSIONS = int(os.getenv('GAUSSIANS_DIMENSIONS', default=15))
GAUSSIANS_GAMMA = float(os.getenv('GAUSSIANS_GAMMA', default=1.5))
GAUSSIANS_STD_DEV = float(os.getenv('GAUSSIANS_STD_DEV', default=0.001))
GAUSSIANS_LOW = float(os.getenv('GAUSSIANS_LOW', default=0.0))
GAUSSIANS_HIGH = float(os.getenv('GAUSSIANS_HIGH', default=1.0))
GAUSSIANS_RANDOM_SEED = int(os.getenv('GAUSSIANS_RANDOM_SEED', default=1234))  # set 0 for random

###### BLACK_BOXES ######
INNER_BLACKBOX_ITERATIONS = int(os.getenv('INNER_BLACKBOX_ITERATIONS', default=5))
INNER_BLACKBOX_L_TO_K_RATIO = float(os.getenv('INNER_BLACKBOX_L_TO_K_RATIO', default=2))
FINALIZATION_BLACKBOX = os.getenv('FINALIZATION_BLACKBOX', default='KMeans')
MINI_BATCH_SIZE = int(os.getenv('MINI_BATCH_SIZE', default=1000))

###### EXPERIMENT PARAMS ######
ROUNDS = int(os.getenv('ROUNDS', default=10))
M = int(os.getenv('M', default=50))
TIMESTAMP = os.getenv('TIMESTAMP', default=strftime('%m_%d_%H_%M'))

##### SOCCER CONSTS ######
EPSILON = float(os.getenv('EPSILON', default=0.1))  # 0.2
DELTA = float(os.getenv('DELTA', default=0.1))
PHI_ALPHA_C = 6.5
MAX_SS_SIZE_C = 36
KPLUS_C = 9
KPLUS_SCALER = int(os.getenv('KPLUS_SCALER', default=1))

###### SKM PARAMS ######
L_TO_K_RATIO = float(os.getenv('L_TO_K_RATIO', default=2.0))
SKM_ITERATIONS = int(os.getenv('SKM_ITERATIONS', default=5))
