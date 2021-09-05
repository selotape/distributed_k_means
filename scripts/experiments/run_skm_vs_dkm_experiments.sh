#!/bin/bash
set -eu

function run_exp() {
  nohup python3.9 run_experiment.py "$1" &
}


TIMESTAMP="$(date +'%m-%d-%H-%M')"
export TIMESTAMP
export ALGO=SKM
export DATASET=gaussian

pushd "$(git rev-parse --show-cdup)"

K=25 SKM_ITERATIONS=2 L_TO_K_RATIO=1.5 run_exp "${DATASET}_25K_${ALGO}"
K=50 SKM_ITERATIONS=2 L_TO_K_RATIO=1.5 run_exp "${DATASET}_50K_${ALGO}"
K=100 SKM_ITERATIONS=2 L_TO_K_RATIO=1.5 run_exp "${DATASET}_100K_${ALGO}"
K=500 SKM_ITERATIONS=2 L_TO_K_RATIO=1.5 run_exp "${DATASET}_500K_${ALGO}"

popd