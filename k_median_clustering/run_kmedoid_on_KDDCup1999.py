from collections import defaultdict
from itertools import product
from time import strftime
from statistics import mean
from k_median_clustering import competitors
from k_median_clustering.algo import distributed_k_median_clustering
from k_median_clustering.math import risk
from k_median_clustering.utils import setup_logger
import numpy as np
import pandas as pd
import subprocess


log_time = strftime('%Y%m%d%H%M')
logger = setup_logger('full_log', f'k_median_clustering_log_{log_time}.log', with_console=True)
results = setup_logger('results_log', f'k_median_clustering_results_{log_time}.log', another_file=f'k_median_clustering_log_{log_time}.log' ,with_console=True)

label = subprocess.check_output(["git", "describe"]).strip()
results.info(f"Running experiment with git label {label}")

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 6000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logger.info(f"Data size: {len(full_data):,}")

ks = [10, 50, 100, 500]
epsilons = [0.15, 0.2]
deltas = [0.1]
ms = [50]
more = [1, 2, 3, 4]


# 1. avg of ratio of risks & risk_f
# ... future: variance of ratios
# 2. sum/avg_iters(max(sample_phase) + bb_phase + max(trim_phase) + )
# 3. make skm ""distributed"". measure sum+avg like above
# 4. new data set

def main():
    skm_runs = 4
    risks = defaultdict(list)
    for k, dt, m, ep, _ in product(ks, deltas, ms, epsilons, more):
        try:
            logger.info(f'=============================================================================================')
            logger.info(f"======== Starting distributed k median with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
            logger.info(f'=============================================================================================')
            results.info('Starting...\n')
            dkm_C, dkm_C_final, dkm_iters = distributed_k_median_clustering(N, k, ep, dt, m, logger, results)
            dkm_risk = risk(N, dkm_C)
            dkm_risk_f = risk(N, dkm_C_final)

            risks['dkm'].append(dkm_risk), risks['dkm_f'].append(dkm_risk_f)

            logger.info(f'=============================================================================================')
            logger.info(f'=============================================================================================')
            logger.info(f'=============================================================================================')
            results.info(f'len(N):{len(N)}. dkm_risk:{dkm_risk:,}. dkm_risk_final:{dkm_risk_f:,}. len(dkm_C):{len(dkm_C)}. len(dkm_C_final):{len(dkm_C_final)}')

            l = int(len(dkm_C) / dkm_iters)


            for i in range(skm_runs):
                skm_iters = dkm_iters + i
                logger.info(f"{i}. Starting scalable_k_mean with {skm_iters} iterations and l=={l}")
                skm_C, skm_C_final = competitors.scalable_k_means_clustering(N, skm_iters, l, k)
                skm_risk = risk(N, skm_C)
                skm_risk_f = risk(N, skm_C_final)
                risks[f'skm_{i}'].append(skm_risk), risks[f'skm_f_{i}'].append(skm_risk_f)
                logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
                logger.info(f'=============================================================================================')
                test_summary = summarize(i, 'skm 1', k, dt, m, ep, len(dkm_C), dkm_iters, skm_iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f)
                results.info(test_summary)
                logger.info(f'=============================================================================================')
                logger.info(f'=============================================================================================')
                logger.info(f'=============================================================================================')
        except Exception as e:
            results.error("BAD!" + str(e))


        risk_avg_str = ', '.join(f'dkm_r/smk_{i}_r' + str(mean(risks['dkm']) / mean(risks[f'skm_{i}']) for i in range(skm_runs)))
        risk_f_avg_str = ', '.join(f'dkm_f_r/smk_{i}_f_r' + str(mean(risks['dkm_f']) / mean(risks[f'skm_f_{i}']) for i in range(skm_runs)))


        #
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        #
        # logger.info(f"Starting fast clustering")
        # fast_clustering_S = competitors.fast_clustering(N, k, ep, m)
        # fast_clustering_risk = risk(N, fast_clustering_S)
        # logger.info(f'The fast clustering risk is {fast_clustering_risk:,} and size of S is {len(fast_clustering_S)}')
        #
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')
        # logger.info(f'=============================================================================================')

        # logger.info(f"Starting {Blackbox.__name__}")
        # blackbox_risk = competitors.blackbox(N, k, risk)
        # logger.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')

        # try:
        #     logger.info(f"Starting Spark KMeans")
        #     blackbox_risk = competitors.spark_kmeans(N, k)
        #     logger.info(f'The {Blackbox.__name__} risk is {blackbox_risk:,}')
        # except Exception as e:
        #     logger.error(e)


def summarize(i, name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, dkm_risk, skm_risk, dkm_risk_final, skm_risk_final):
    return f'{i}, {name}, k={k}, dt={dt}, m={m}, ep={ep}, len(dkm_C)={len_dkm_C}, dkm_iters={dkm_iters}, skm_iters={skm_iters}, l={l}, len(skm_C)={len_skm_C},' \
           f' (dkm_risk/skm_risk)={dkm_risk / skm_risk:,}, (dkm_risk_final/skm_risk_final)={dkm_risk_final / skm_risk_final:,}\n'


if __name__ == "__main__":
    # execute only if run as a script
    main()
