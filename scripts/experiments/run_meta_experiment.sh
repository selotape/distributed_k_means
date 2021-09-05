#!/bin/bash
set -eu

function run_exp() {
  nohup python3.9 run_experiment.py "$1" &
}

TIMESTAMP="$(date +'%m-%d-%H-%M')"
export TIMESTAMP

pushd "$(git rev-parse --show-cdup)"

ALGO=SKM SKM_ITERATIONS=1 run_exp "${DATASET}_${K}K_skm_1iters"
ALGO=SKM SKM_ITERATIONS=2 run_exp "${DATASET}_${K}K_skm_2iters"
ALGO=SKM SKM_ITERATIONS=3 run_exp "${DATASET}_${K}K_skm_3iters"
ALGO=SKM SKM_ITERATIONS=4 run_exp "${DATASET}_${K}K_skm_4iters"
ALGO=SKM SKM_ITERATIONS=5 run_exp "${DATASET}_${K}K_skm_5iters"

ALGO=SOCCER EPSILON=0.2 run_exp "${DATASET}_${K}K_02ep_dkm"
ALGO=SOCCER EPSILON=0.1 run_exp "${DATASET}_${K}K_01ep_dkm"
ALGO=SOCCER EPSILON=0.05 run_exp "${DATASET}_${K}K_005ep_dkm"
ALGO=SOCCER EPSILON=0.01 run_exp "${DATASET}_${K}K_001ep_dkm"

popd