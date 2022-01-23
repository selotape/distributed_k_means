# SOCCER

SOCCER (Sampling Optimal Clustering Cost Estimation) is a fast and accurate distributed clustering algorithm.

## Install Soccer

1. Install, create & activate a python 3.8+ Anaconda virtualenv.
2. Run:

```bash
pip install -r ./requirements.txt
```

## Experiments

To show and compare SOCCER's performance, we scripted an experiment which reads a dataset, clusters it using a chosen 
algorithm and prints the results. This code allows to easily reproduce all experiments reported in the paper.

### Usage   
```text
usage: run_all_soccer_paper_experiments.py [-h]
                                           [--blackbox {KMeans,MiniBatchKMeans,ScalableKMeans}]
                                           [--datasets {kdd,bigcross,census1990,higgs} [{kdd,bigcross,census1990,higgs} ...]]
                                           [--custom-dataset-csvs CUSTOM_DATASET_CSVS [CUSTOM_DATASET_CSVS ...]]
Runs all SOCCER experiments. If used without any parameters, it runs all experiments exactly as reported in the paper.

The one time prerequisites are:
1. Install Anaconda
2. create a conda env - `conda create --name soccer python=3.8`
3. Install requirements - `conda activate soccer && pip3 install -r requirements.txt`
4. run `scripts/download_and_extract_all_datasets.sh`.
Finally, remember before every run to execute - `conda activate soccer`


Examples:
1. *run all experiments*: `./run_all_soccer_paper_experiments.py`
2. run one dataset with scalable-kmeans blackbox: `./run_all_soccer_paper_experiments.py --datasets kdd --blackbox ScalableKMeans`
3. run a new custom csv dataset: `./run_all_soccer_paper_experiments.py --custom-dataset-csvs ./my/custom/data.csv`
optional arguments:
  -h, --help            show this help message and exit
  --blackbox {KMeans,MiniBatchKMeans,ScalableKMeans}
                        [optional] which internal algorithm to run as the black-box clustering algorithm used by SOCCER. The default is KMeans
  --datasets {kdd,bigcross,census1990,higgs} [{kdd,bigcross,census1990,higgs} ...]
                        [optional] run the experiment only on these datasets. If unspecified, runs all datasets.
  --custom-dataset-csvs CUSTOM_DATASET_CSVS [CUSTOM_DATASET_CSVS ...]
                        [optional] run on your custom local CSVs
```

### Configuration

The experiment script is highly configurable - which algorithm, how to run it, which dataset to cluster, etc.

The most common configurations are:

* `K` - The number of clusters to calculate
* `DATASET` - Which known dataset to load and cluster. Alternatively, a new dataset can be specified as a csv file from commandline.
* `BLACKBOX` - Which internal algorithm to run as the black-box clustering algorithm used by SOCCER
* `EPSILON` - A value in range (0, 1) which links the coordinator size with the data set size
* `DELTA` - The confidence parameter.
* `ALGO` - Which clustering algorithm to run - SOCCER or Scalable KMeans.

You can configure many other parameters via environment variables. For the full list, see `soccer/config.py`.


### Reading experiment output

Experiment outputs are written in 3 files:

1. {run_name}_{timestamp}.log - the full log output
2. {run_name}_{timestamp}_results.csv - the results of each single clustering iteration (out of 10)
3. {run_name}_{timestamp}_summary.csv - summary results of all (10) iterations

#### Gaussian Dataset Seed

For reproducibility, we used a default `GAUSSIANS_RANDOM_SEED=1234` when generating gaussian datasets. To reuse our gaussian datasets, don't change the gaussians configurations (just choose
e.g. `DATASET=gaussians_50`). To use a random seed (and dataset), set `GAUSSIANS_RANDOM_SEED=0`.
