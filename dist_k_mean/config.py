KDD_DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
KDD_SUBSET_SIZE = 6000000

DATASETS = ['gaussian', 'kdd', ]
KS = [100]
EPSILONS = [0.1]
DELTAS = [0.1]
MS = [50]
AUTO_COMPUTE_L = -1
L_TO_K_RATIOS = [AUTO_COMPUTE_L, 1, ]
SKM_ITERATIONS = [2, 3, 4, 5]
ROUNDS = 2
MINI_BATCH_SIZE = 1000

GAUSSIANS_DIMENSIONS = 10
GAUSSIANS_NUM_POINTS = 10_000_000
GAUSSIANS_K = 100
GAUSSIANS_ALPHA = 0.0
GAUSSIANS_STD_DEV = 0.1