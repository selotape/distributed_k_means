#!/bin/bash
set -eu

function run_exp() {
  nohup python3.9 run_experiments.py "$1" &
}

ALGO=SKM SKM_ITERATIONS=2 run_exp "{$DATASET}_{$K}K_skm_2iters"
ALGO=SKM SKM_ITERATIONS=3 run_exp "{$DATASET}_{$K}K_skm_3iters"
ALGO=SKM SKM_ITERATIONS=4 run_exp "{$DATASET}_{$K}K_skm_4iters"
ALGO=SKM SKM_ITERATIONS=5 run_exp "{$DATASET}_{$K}K_skm_5iters"

ALGO=DKM INNER_BLACKBOX=KMeans CONST_MODE=strict run_exp "{$DATASET}_{$K}K_dkm_strict_KMBlackBox"
ALGO=DKM INNER_BLACKBOX=ScalableKMeans CONST_MODE=strict run_exp "{$DATASET}_{$K}K_dkm_strict_SKMBlackBox"
ALGO=DKM INNER_BLACKBOX=KMeans CONST_MODE=fast run_exp "{$DATASET}_{$K}K_dkm_fast_KMBlackBox"
ALGO=DKM INNER_BLACKBOX=ScalableKMeans CONST_MODE=fast run_exp "{$DATASET}_{$K}K_dkm_fast_SKMBlackBox"
