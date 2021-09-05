#!/bin/bash
set -eu

function run_exp() {
  nohup python run_experiments.py "$1"
}

TIMESTAMP="$(date +'%m-%d-%H-%M')"
export TIMESTAMP

#__conda_setup="$('/home/anaconda3.8/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/anaconda3.8/etc/profile.d/conda.sh" ]; then
#        . "/home/anaconda3.8/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/anaconda3.8/bin:$PATH"
#    fi
#fi
#unset __conda_setup
#conda activate soccer

ALGO=SKM SKM_ITERATIONS=1 run_exp "${DATASET}_${K}K_skm_1iters"
#ALGO=SKM SKM_ITERATIONS=2 run_exp "${DATASET}_${K}K_skm_2iters"
#ALGO=SKM SKM_ITERATIONS=3 run_exp "${DATASET}_${K}K_skm_3iters"
#ALGO=SKM SKM_ITERATIONS=4 run_exp "${DATASET}_${K}K_skm_4iters"
#ALGO=SKM SKM_ITERATIONS=5 run_exp "${DATASET}_${K}K_skm_5iters"
#
#ALGO=DKM EPSILON=0.2 run_exp "${DATASET}_${K}K_02ep_dkm"
#ALGO=DKM EPSILON=0.1 run_exp "${DATASET}_${K}K_01ep_dkm"
#ALGO=DKM EPSILON=0.05 run_exp "${DATASET}_${K}K_005ep_dkm"
#ALGO=DKM EPSILON=0.01 run_exp "${DATASET}_${K}K_001ep_dkm"
