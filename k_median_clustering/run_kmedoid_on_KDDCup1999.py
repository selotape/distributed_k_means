from collections import defaultdict
from itertools import product
from statistics import mean
import subprocess
from time import strftime

from k_median_clustering import competitors
from k_median_clustering.algo import distributed_k_median_clustering
from k_median_clustering.math import risk
from k_median_clustering.utils import setup_logger
import numpy as np
import pandas as pd

label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
log_time = strftime('%Y%m%d%H%M')
run_name = f'k_median_clustering_{log_time}_{label}'
logger = setup_logger('full_log', f'{run_name}.log', with_console=True)
csv = open(f"{run_name}_results.csv", "a")

DATASET_FILE = "/home/ronvis/private/distributed_k_median/data_samples/kddcup99/kddcup.data.corrected"
SUBSET_SIZE = 6000000
full_data = pd.read_csv(DATASET_FILE, nrows=SUBSET_SIZE)
N = full_data.select_dtypes([np.number])
logger.info(f"Data size: {len(full_data):,}")

ks = [10, 50, 100, 500]
epsilons = [0.15, 0.2]
deltas = [0.1]
ms = [50]
repetitions = [0, 1, 2, 3, 4]
skm_runs = 3


HEADER = 'test_name,k,dt,m,ep,len(dkm_C),dkm_iters,skm_iters,l,len(skm_C),(dkm_r/skm_r),(dkm_r_f/skm_r_f),avg(dkm_r/skm_r),avg(dkm_r_f/skm_r_f)\n'
csv.write(HEADER)

def format_as_csv(test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, dkm_risk, skm_risk, dkm_risk_final, skm_risk_final, risks, skm_run):
    return ','.join(str(s) for s in
                    [test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, (dkm_risk / skm_risk), (dkm_risk_final / skm_risk_final), avg_r(risks, skm_run), avg_r_f(risks, skm_run)])


# 1. avg of ratio of risks & risk_f DONE
# 2. results as CSV DONE
# ... future: variance of ratios
# 2. sum/avg_iters(max(sample_phase) + bb_phase + max(trim_phase))
# 3. make skm ""distributed"". measure sum+avg like above
# 4. new data set

def main():
    risks = defaultdict(list)
    for k, dt, m, ep, rep in product(ks, deltas, ms, epsilons, repetitions):
        try:
            logger.info(f'===========================================================================================')
            logger.info(f"======== Starting distributed k median with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
            logger.info(f'===========================================================================================')
            logger.info('Starting...\n')
            dkm_C, dkm_C_final, dkm_iters = distributed_k_median_clustering(N, k, ep, dt, m, logger)
            dkm_risk = risk(N, dkm_C)
            dkm_risk_f = risk(N, dkm_C_final)

            risks['dkm'].append(dkm_risk), risks['dkm_f'].append(dkm_risk_f)

            logger.info(f'===========================================================================================')
            logger.info(f'===========================================================================================')
            logger.info(f'===========================================================================================')
            logger.info(f'len(N):{len(N)}. dkm_risk:{dkm_risk:,}. dkm_risk_final:{dkm_risk_f:,}. len(dkm_C):{len(dkm_C)}. len(dkm_C_final):{len(dkm_C_final)}')

            l = int(len(dkm_C) / dkm_iters)

            for skm_run in range(skm_runs):
                skm_iters = dkm_iters + skm_run
                logger.info(f"{skm_run}. Starting scalable_k_mean with {skm_iters} iterations and l=={l}")
                skm_C, skm_C_final = competitors.scalable_k_means_clustering(N, skm_iters, l, k)
                skm_risk = risk(N, skm_C)
                skm_risk_f = risk(N, skm_C_final)
                risks[f'skm_{skm_run}'].append(skm_risk), risks[f'skm_f_{skm_run}'].append(skm_risk_f)
                logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
                logger.info(f'===========================================================================================')
                test_summary = format_as_csv(f'{skm_run}th_skm_run_in_rep_{rep}', k, dt, m, ep, len(dkm_C), dkm_iters, skm_iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f, risks,
                                             skm_run)
                csv.write(test_summary + '\n')
                logger.info(test_summary)
                logger.info(f'===========================================================================================')
                logger.info(f'===========================================================================================')
                logger.info(f'===========================================================================================')

        except Exception:
            logger.exception("BAD!")


def avg_r(risks, skm_run):
    return mean(risks['dkm']) / mean(risks[f'skm_{skm_run}'])


def avg_r_f(risks, skm_run):
    return mean(risks['dkm_f']) / mean(risks[f'skm_f_{skm_run}'])


if __name__ == "__main__":
    # execute only if run as a script
    main()
    csv.close()
