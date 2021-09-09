#!/usr/bin/env python

import subprocess
from itertools import product


def main():
    gaussian_experiments = (("gaussian_25", 25), ("gaussian_50", 50), ("gaussian_100", 100), ("gaussian_200", 200))
    for dataset, k in gaussian_experiments:
        run_meta_experiment(dataset, k)

    other_experiments = product(("higgs", "kdd", "census", "bigcross"), (25, 50, 100, 200))
    for dataset, k in other_experiments:
        run_meta_experiment(dataset, k)


def run_meta_experiment(dataset, k):
    for epsilon in (0.01, 0.05, 0.1, 0.2,):
        if (k, epsilon) == (200, 0.2):
            continue
        run_soccer(k, dataset, epsilon)
    for skm_iters in (1, 2, 3, 4, 5):
        run_skm(k, dataset, skm_iters)


def run_soccer(k, dataset, epsilon):
    subprocess.run([f"K={k}", f'DATASET={dataset}', f'EPSILON={epsilon}', './run_a_soccer_experiment.py', f'{dataset}_{k}K_{epsilon}ep_soccer'])


def run_skm(k, dataset, skm_iters):
    subprocess.run([f"K={k}", f'DATASET={dataset}', f'SKM_ITERATIONS={skm_iters}', './run_a_soccer_experiment.py', f'{dataset}_{k}K_skm_{skm_iters}iters"'])


if __name__ == '__main__':
    main()
