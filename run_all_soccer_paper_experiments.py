#!/usr/bin/env python
import os
import subprocess
import sys
from itertools import product



def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--but-only':
        datasets = sys.argv[2]
        run_but_only_datasets(datasets)
    else:
        run_all_experiments()


_GAUSSIAN_EXPERIMENTS = (
    ("gaussian_25", 25), ("gaussian_50", 50), ("gaussian_100", 100),
    ("gaussian_200", 200))


def run_but_only_datasets(datasets):
    for dataset in datasets.split(","):
        if dataset == 'gaussian':
            for gaussian, k in _GAUSSIAN_EXPERIMENTS:
                run_meta_experiment(gaussian, k)
        else:
            other_experiments = product((dataset,), (25, 50, 100, 200))
            for other, k in other_experiments:
                run_meta_experiment(other, k)


def run_all_experiments():
    for dataset, k in _GAUSSIAN_EXPERIMENTS:
        run_meta_experiment(dataset, k)
    other_experiments = product(("higgs", "kdd", "census1990", "bigcross"),
                                (25, 50, 100, 200))
    for dataset, k in other_experiments:
        run_meta_experiment(dataset, k)


def get_blackbox():
    if '--skm_blackbox' in sys.argv:
        return 'ScalableKMeans'
    elif '--faiss_blackbox' in sys.argv:
        return 'FaissKMeans'
    elif '--minibatch_blackbox' in sys.argv:
        return 'MiniBatchKMeans'
    elif '--kmeans_blackbox' in sys.argv:
        return 'KMeans'
    else:
        print('Unkown/specified blackbox. Using default KMeans.')
        return 'KMeans'

def run_meta_experiment(dataset, k):
    blackbox = get_blackbox()
    if '--no-soccer' not in sys.argv:
        for epsilon in (0.01, 0.05, 0.1, 0.2,):
            if (k, epsilon) == (200, 0.2):
                continue
            run_soccer(k, dataset, epsilon, blackbox)

    if '--no-skm' not in sys.argv:
        for skm_iters in (1, 2, 3, 4, 5):
            run_skm(k, dataset, skm_iters)


def run_soccer(k, dataset, epsilon, blackbox):
    subprocess.call(f"K={k} DATASET={dataset} INNER_BLACKBOX={blackbox} EPSILON={epsilon} ./run_a_soccer_experiment.py {dataset}_{k}K_{os.getenv('KP_SCALER')}kpscale_{epsilon}ep_soccer", shell=True)


def run_skm(k, dataset, skm_iters):
    subprocess.call(f"K={k} DATASET={dataset} ALGO=SKM SKM_ITERATIONS={skm_iters} ./run_a_soccer_experiment.py {dataset}_{k}K_skm_{skm_iters}iters", shell=True)


if __name__ == '__main__':
    main()
