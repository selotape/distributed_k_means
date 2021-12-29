#!/usr/bin/env python

import subprocess
from itertools import product


def main():
    gaussian_experiments = (("gaussian_200", 200))
    for dataset, k in gaussian_experiments:
        run_meta_experiment(dataset, k)

    # other_experiments = product(("higgs", "kdd", "census", "bigcross"), (25, 50, 100, 200))
    # for dataset, k in other_experiments:
    #     run_meta_experiment(dataset, k)


def run_meta_experiment(dataset, k):
    for epsilon in (0.1, ):
        if (k, epsilon) == (200, 0.2):
            continue
        run_soccer(k, dataset, epsilon)
    for skm_iters in ( 3,):
        run_skm(k, dataset, skm_iters)


def run_soccer(k, dataset, epsilon):
    subprocess.call(f"K={k} DATASET={dataset} INNER_BLACKBOX=FaissKMeans EPSILON={epsilon} ./run_a_soccer_experiment.py {dataset}_{k}K_{epsilon}ep_soccer", shell=True)


def run_skm(k, dataset, skm_iters):
    subprocess.call(f"K={k} DATASET={dataset} SKM_ITERATIONS={skm_iters} ./run_a_soccer_experiment.py {dataset}_{k}K_skm_{skm_iters}iters", shell=True)


if __name__ == '__main__':
    main()
