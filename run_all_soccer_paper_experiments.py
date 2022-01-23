#!/usr/bin/env python
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from itertools import product

from soccer import run_a_soccer_experiment
from soccer.black_box import DEFAULT_BLACKBOX


def main():
    parser = ArgumentParser(description='Runs all SOCCER experiments. If used without any parameters, it '
                                        'runs all experiments exactly as reported in the paper.\n\n'
                                        'The one time prerequisites are:\n'
                                        '1. Install Anaconda\n'
                                        '2. create a conda env - `conda create --name soccer python=3.8`\n'
                                        '3. Install requirements - `conda activate soccer && pip3 install -r requirements.txt`\n'
                                        '4. run `scripts/download_and_extract_all_datasets.sh`.\n'
                                        'Finally, remember before every run to execute - `conda activate soccer`\n\n\n'
                                        'Examples:\n'
                                        '1. *run all experiments*: `./run_all_soccer_paper_experiments.py`\n'
                                        '2. run one dataset with scalable-kmeans blackbox: `./run_all_soccer_paper_experiments.py --datasets kdd --blackbox ScalableKMeans`\n'
                                        '3. run a new custom csv dataset: `./run_all_soccer_paper_experiments.py --custom-dataset-csvs ./my/custom/data.csv`',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('--blackbox', default=DEFAULT_BLACKBOX,
                        choices=['KMeans', 'MiniBatchKMeans', 'ScalableKMeans'],
                        help='[optional] which internal algorithm to run as the black-box'
                             ' clustering algorithm used by SOCCER. The default is '
                             + DEFAULT_BLACKBOX)
    parser.add_argument('--datasets', nargs='+', default=[],
                        choices=['kdd', 'bigcross', 'census1990', 'higgs'],
                        help='[optional] run the experiment only on these datasets. '
                             'If unspecified, runs all datasets.')

    parser.add_argument('--custom-dataset-csvs', nargs='+', default=[],
                        help='[optional] run on your custom local CSVs')

    args = parser.parse_args()

    if args.datasets or args.custom_dataset_csvs:
        run_but_only_datasets(args.datasets + args.custom_dataset_csvs, args.blackbox)
    else:
        run_all_experiments(args.blackbox)


_GAUSSIAN_EXPERIMENTS = (
    ("gaussian_25", 25), ("gaussian_50", 50), ("gaussian_100", 100),
    ("gaussian_200", 200))


def run_but_only_datasets(datasets, blackbox):
    for dataset in datasets:
        if dataset == 'gaussian':
            for gaussian, k in _GAUSSIAN_EXPERIMENTS:
                run_meta_experiment(gaussian, k, blackbox)
        else:
            other_experiments = product((dataset,), (25, 50, 100, 200))
            for other, k in other_experiments:
                run_meta_experiment(other, k, blackbox)


def run_all_experiments(blackbox):
    for dataset, k in _GAUSSIAN_EXPERIMENTS:
        run_meta_experiment(dataset, k, blackbox)
    other_experiments = product(("higgs", "kdd", "census1990", "bigcross"),
                                (25, 50, 100, 200))
    for dataset, k in other_experiments:
        run_meta_experiment(dataset, k, blackbox)


def get_blackbox():
    if '--skm_blackbox' in sys.argv:
        return 'ScalableKMeans'
    elif '--minibatch_blackbox' in sys.argv:
        return 'MiniBatchKMeans'
    elif '--kmeans_blackbox' in sys.argv:
        return 'KMeans'
    else:
        print('Unkown/specified blackbox. Using default KMeans.')
        return 'KMeans'


def run_meta_experiment(dataset, k, blackbox):
    for epsilon in (0.01, 0.05, 0.1, 0.2,):
        if (k, epsilon) == (200, 0.2):
            continue
        run_soccer(k, dataset, epsilon, blackbox)

    for skm_iters in (1, 2, 3, 4, 5):
        run_skm(k, dataset, skm_iters)


def run_soccer(k, dataset, epsilon, blackbox):
    run_name = f"{dataset}_{k}K_{epsilon}ep_soccer"
    run_a_soccer_experiment.main(run_name, 'SOCCER', k, dataset, epsilon, blackbox)


def run_skm(k, dataset, epsilon, blackbox, skm_iters):
    run_name = f"{dataset}_{k}K_skm_{skm_iters}iters"
    run_a_soccer_experiment.main(run_name, 'SKM', k, dataset, epsilon, blackbox, skm_iters)


if __name__ == '__main__':
    main()
