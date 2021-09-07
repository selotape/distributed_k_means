# SOCCER

SOCCER (Sampling Optimal Clustering Cost Estimation) is a fast and accurate distributed clustering algorithm.

## Install Soccer

1. Install, create & activate a python 3.8+ virtualenv.
2. Run:

```bash
pip install -r requirements.txt
```

## Experiments

To show and compare SOCCER's performance, we scripted an experiment which reads a chosen dataset, clusters it using a chosen 
algorithm and prints the results. This code allows to easily reproduce all experiments reported in the paper.

### 1. Configure the experiment

The experiment script is highly configurable - e.g. which algorithm, how exactly to run it, which dataset to cluster, etc. The experiment is configured via environment variables.

**See all configuration options in `soccer/config.py`.**

### 2. Run the experiment

**To run one clustering experiment**, use `run_experiment.py`, e.g.:

```bash
export ALGO=SOCCER
export DATASET=kdd
export K=100
python3.9 run_experiment.py "exp13___soccer_kdd_100k"
```

**To run "ScalableKMeans vs SOCCER"** with different setups, use `run_meta_experiment.sh`, e.g.:

```bash
DATASET=gaussians_50 K=50 ./scripts/experiments/run_meta_experiment.sh
```

### 3. Read experiment output

Experiment outputs are written in 3 files:

1. {run_name}.log - the full log output
2. {run_name}_results.csv - the results of each single clustering iteration
3. {DATASET}\_{K}K\_{TIMESTAMP}_summary.csv - summary results of all iterations

### About Datasets

In the experiment script you can choose one of many well-known datasets. The list of supported datasets is in `soccer/config.py`. 

Before using a dataset for the first time, you should download and
extract it. **See how to download and extract all datasets in `scripts/download_and_extract_all_datasets.sh`**

#### Gaussian Dataset Seed

For reproducibility, we used a default `GAUSSIANS_RANDOM_SEED=1234` when generating gaussian datasets. To reuse our gaussian datasets, don't change the gaussians configurations (just choose
e.g. `DATASET=gaussians_50`). To use a random seed (and dataset), set `GAUSSIANS_RANDOM_SEED=0`.
