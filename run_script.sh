#!/bin/bash

function run_exp() {
  nohup python3.9 run_experiments.py "$1" &
}

# kdds
DATASET=kdd run_exp kdd_default
DATASET=kdd INNER_BLACKBOX=ScalableKMeans run_exp kdd_scalable_blackbox_l1
DATASET=kdd INNER_BLACKBOX=ScalableKMeans INNER_BLACKBOX_L_TO_K_RATIO=2 run_exp kdd_scalable_blackbox_l2

# gaussian
run_exp gaussian_default
INNER_BLACKBOX=ScalableKMeans run_exp gaussian_scalable_blackbox_l1
INNER_BLACKBOX=ScalableKMeans INNER_BLACKBOX_L_TO_K_RATIO=2 run_exp gaussian_scalable_blackbox_l2

GAUSSIANS_ALPHA=0.5 GAUSSIANS_STD_DEV=0.2 run_exp gaus_alpha05_std02
INNER_BLACKBOX=ScalableKMeans GAUSSIANS_ALPHA=0.5 GAUSSIANS_STD_DEV=0.2 run_exp gaus_alpha05_std02_scalable_blackbox_l1
INNER_BLACKBOX=ScalableKMeans INNER_BLACKBOX_L_TO_K_RATIO=2 GAUSSIANS_ALPHA=0.5 GAUSSIANS_STD_DEV=0.2 run_exp gaus_alpha05_std02_scalable_blackbox_l2
