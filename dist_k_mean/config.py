import os
import sys

RUN_NAME = 'SCHMOD' if len(sys.argv) < 2 else sys.argv[1]

###### DATA SETS ######
DATASET = os.getenv('DATASET', default='bigcross')  # 'gaussian', 'kdd', 'covtype', 'power', 'skin', 'bigcross'
ALGO = os.getenv('ALGO', default='DKM')  # 'SKM', 'DKM', 'ENE'

KDD_DATASET_FILE = os.getenv('KDD_DATASET_FILE', default="data_samples/kddcup99/kddcup.data")
BIGCROSS_DATASET_FILE = os.getenv('BIGCROSS_DATASET_FILE', default="data_samples/bigcross/BigCross.data")
COVTYPE_DATASET_FILE = os.getenv('COVTYPE_DATASET_FILE', default="data_samples/covtype/covtype.data")
POWER_DATASET_FILE = os.getenv('POWER_DATASET_FILE', default="data_samples/power/household_power_consumption.txt")
SKIN_DATASET_FILE = os.getenv('SKIN_DATASET_FILE', default="data_samples/skin/Skin_NonSkin.txt")
DATASET_SIZE = int(os.getenv('DATASET_SIZE', default=100_000_000))

GAUSSIANS_DIMENSIONS = int(os.getenv('GAUSSIANS_DIMENSIONS', default=15))
GAUSSIANS_K = int(os.getenv('GAUSSIANS_K', default=100))
GAUSSIANS_ALPHA = float(os.getenv('GAUSSIANS_ALPHA', default=0.0))
GAUSSIANS_STD_DEV = float(os.getenv('GAUSSIANS_STD_DEV', default=0.1))

###### BLACK_BOXES ######
INNER_BLACKBOX = os.getenv('INNER_BLACKBOX', default='KMeans')  # 'KMeans' 'MiniBatchKMeans' 'ScalableKMeans'
INNER_BLACKBOX_ITERATIONS = int(os.getenv('INNER_BLACKBOX_ITERATIONS', default=4))
INNER_BLACKBOX_L_TO_K_RATIO = float(os.getenv('INNER_BLACKBOX_L_TO_K_RATIO', default=1))

FINALIZATION_BLACKBOX = os.getenv('FINALIZATION_BLACKBOX', default='KMeans')

MINI_BATCH_SIZE = int(os.getenv('MINI_BATCH_SIZE', default=1000))

###### DISTRIBUTED PARAMS ######
ROUNDS = int(os.getenv('ROUNDS', default=10))
K = int(os.getenv('K', default=50))
EPSILON = float(os.getenv('EPSILON', default=0.1))  # 0.2
DELTA = float(os.getenv('DELTA', default=0.1))
M = int(os.getenv('M', default=50))

PHI_ALPHA_C = 6.5
MAX_SS_SIZE_C = 38
KPLUS_C = 9

###### SKM PARAMS ######
L_TO_K_RATIO = float(os.getenv('L_TO_K_RATIO', default=2.0))
SKM_ITERATIONS = int(os.getenv('SKM_ITERATIONS', default=5))
