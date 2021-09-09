# SOCCER

SOCCER (Sampling Optimal Clustering Cost Estimation) is a fast and accurate distributed clustering algorithm.

## Install Soccer

1. Install, create & activate a python 3.8+ virtualenv.
2. Run:

```bash
pip install -r ./requirements.txt
```

## Experiments

To show and compare SOCCER's performance, we scripted an experiment which reads a chosen dataset, clusters it using a chosen 
algorithm and prints the results. This code allows to easily reproduce all experiments reported in the paper.

### 1. Configure the experiment

The experiment script is highly configurable - e.g. which algorithm, how exactly to run it, which dataset to cluster, etc. The experiment is configured via environment variables.

The most common configurations are:

* `K` - The number of clusters to calculate
* `DATASET` - Which known dataset to load and cluster. Alternatively, a new dataset can be specified as a csv file from commandline.
* `EPSILON` - A value in range (0, 1) which links the coordinator size with the data set size 
* `DELTA` - The confidence parameter.
* `ALGO` - Which clustering algorithm to run. Supported values are "soccer" & "skm" (ScalableKMeans)

**See all configuration options in `soccer/config.py`.**

### 2. Usage

#### Run SOCCER

**To run one clustering experiment**, enable your virtualenv and use `run_a_soccer_experiment.py` - 

e.g. - 
```bash
[ CONF1=val1 CONF2=val2 ]  ./run_a_soccer_experiment.py "experiment_name" [path/to/your_data.csv]

K=100 EPSILON=0.1 DELTA=0.1 DATASET=higgs ./run_a_soccer_experiment.py "soccer_with_higgs_and_100k"
K=100 EPSILON=0.1 DELTA=0.1 ALGO=skm      ./run_a_soccer_experiment.py "skm_with_100k_on_my_data" my/custom/data.csv
```

#### Run all experiments reported in the paper

**To reproduce all the experiments reported in the SOCCER paper**, enable your virtualenv and the run:

```bash
./run_all_soccer_paper_experiments.py
```

Note - to run all experiments you must first download the datasets (see "About Datasets" section).

### 3. Read experiment output

Experiment outputs are written in 3 files:

1. {run_name}_{timestamp}.log - the full log output
2. {run_name}_{timestamp}_results.csv - the results of each single clustering iteration (out of 10*)
3. {run_name}_{timestamp}_summary.csv - summary results of all (10) iterations

\* configuable via ITERATIONS environment variable

### About Datasets

In the experiment script you can choose one of many well-known datasets. The list of supported datasets is in `soccer/config.py`. 

Before using a dataset for the first time, you should download and
extract it. **See how to download and extract all datasets in `scripts/download_and_extract_all_datasets.sh`**

#### Gaussian Dataset Seed

For reproducibility, we used a default `GAUSSIANS_RANDOM_SEED=1234` when generating gaussian datasets. To reuse our gaussian datasets, don't change the gaussians configurations (just choose
e.g. `DATASET=gaussians_50`). To use a random seed (and dataset), set `GAUSSIANS_RANDOM_SEED=0`.
