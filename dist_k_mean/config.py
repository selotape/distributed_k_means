import os
import sys
from time import strftime

RUN_NAME = 'SCHMOD' if len(sys.argv) < 2 else sys.argv[1]

###### DATA SETS ######
DATASET = os.getenv('DATASET', default='gaussian_100')  # 'kdd', 'covtype', 'power', 'skin', 'bigcross', 'census1990', 'activity', 'higgs', 'gaussian_25', 'gaussian_50', 'gaussian_100', 'gaussian_200'
ALGO = os.getenv('ALGO', default='DKM')  # 'SKM', 'DKM', 'ENE'
TIMESTAMP = os.getenv('TIMESTAMP', default=strftime('%m_%d_%H_%M'))

KDD_DATASET_FILE = os.getenv('KDD_DATASET_FILE', default="data_samples/kddcup99/kddcup.data")
CENSUS1990_DATASET_FILE = os.getenv('CENSUS1990_DATASET_FILE', default="data_samples/census1990/USCensus1990.data.txt")
BIGCROSS_DATASET_FILE = os.getenv('BIGCROSS_DATASET_FILE', default="data_samples/bigcross/BigCross.data")
COVTYPE_DATASET_FILE = os.getenv('COVTYPE_DATASET_FILE', default="data_samples/covtype/covtype.data")
POWER_DATASET_FILE = os.getenv('POWER_DATASET_FILE', default="data_samples/power/household_power_consumption.txt")
SKIN_DATASET_FILE = os.getenv('SKIN_DATASET_FILE', default="data_samples/skin/Skin_NonSkin.txt")
HIGGS_DATASET_FILE = os.getenv('HIGGS_DATASET_FILE', default="data_samples/higgs/HIGGS.csv")
ACTIVITY_DATASET_FILE = os.getenv('ACTIVITY_DATASET_FILE', default="data_samples/activity/Activity_recognition_exp.csv")
DATASET_SIZE = int(os.getenv('DATASET_SIZE', default=10_000_000))
SCALE_DATASET = bool(os.getenv('SCALE_DATASET', default=False))

GAUSSIANS_DIMENSIONS = int(os.getenv('GAUSSIANS_DIMENSIONS', default=15))
GAUSSIANS_K = int(os.getenv('GAUSSIANS_K', default=100))
GAUSSIANS_TYPE = os.getenv('GAUSSIANS_TYPE', default='gamma')  # 'exp' 'gamma'
GAUSSIANS_GAMMA = float(os.getenv('GAUSSIANS_GAMMA', default=1.5))
GAUSSIANS_STD_DEV = float(os.getenv('GAUSSIANS_STD_DEV', default=0.001))
GAUSSIANS_LOW = float(os.getenv('GAUSSIANS_LOW', default=0.0))
GAUSSIANS_HIGH = float(os.getenv('GAUSSIANS_HIGH', default=1.0))
GAUSSIANS_RANDOM_SEED = int(os.getenv('GAUSSIANS_RANDOM_SEED', default=1234))

###### BLACK_BOXES ######
INNER_BLACKBOX = os.getenv('INNER_BLACKBOX', default='KMeans')  # 'KMeans' 'MiniBatchKMeans' 'ScalableKMeans' 'FaissKMeans'
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
MAX_SS_SIZE_C = 36
KPLUS_C = 9

###### SKM PARAMS ######
L_TO_K_RATIO = float(os.getenv('L_TO_K_RATIO', default=2.0))
SKM_ITERATIONS = int(os.getenv('SKM_ITERATIONS', default=5))
