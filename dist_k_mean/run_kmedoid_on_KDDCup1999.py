from itertools import product
from statistics import mean
import subprocess
from time import strftime

from dist_k_mean import competitors
from dist_k_mean.algo import distributed_k_means
from dist_k_mean.datasets import generate_k_gaussians, read_and_prep_kdd
from dist_k_mean.math import risk
from dist_k_mean.utils import setup_logger
from config import *

# 1. avg of ratio of risks & risk_f DONE
# 2. results as CSV DONE
# 3. new data set DONE
# 4. sum/avg_iters(  max(sample_phase) + bb_phase + max(trim_phase)) DONE
# 5. Timing for skm (total_time/num_machines) DONE
# 6. super-comfortable config for running tests: gaussian params. l as param.
# 7. run on Server
# 8. calculate distances to Ctmp...
# 9. find another BB
# 10. fast_clustering


label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
log_time = strftime('%Y%m%d%H%M')
run_name = f'dist_k_mean_{log_time}_git{label}'
logger = setup_logger('full_log', f'{run_name}.log', with_console=True)

HEADER = 'test_name,k,dt,m,ep,len(dkm_C),dkm_iters,skm_iters,l,len(skm_C),(dkm_r/skm_r),(dkm_r_f/skm_r_f)'


def format_as_csv(test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, dkm_risk, skm_risk, dkm_risk_final, skm_risk_final):
    return ','.join(str(s) for s in
                    [test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, (dkm_risk / skm_risk), (dkm_risk_final / skm_risk_final)])


def main(kdd=True):
    logger.info("Loading Data...")
    N = read_and_prep_kdd() if kdd else generate_k_gaussians()
    logger.info(f"len(N)={len(N)}")

    csv = open(f"{run_name}_results.csv", "a")
    csv.write(HEADER)
    for k, dt, m, ep, l_ratio, rep in product(KS, DELTAS, MS, EPSILONS, L_TO_K_RATIOS, range(ROUNDS)):
        try:
            logger.info(f"======== Starting distributed k means with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
            dkm_C, dkm_C_final, dkm_iters, dkm_timing = distributed_k_means(N, k, ep, dt, m, logger)
            logger.info(f'dkm_timing:{dkm_timing}')

            dkm_risk = risk(N, dkm_C)
            dkm_risk_f = risk(N, dkm_C_final)

            logger.info(f'len(N):{len(N)}. dkm_risk:{dkm_risk:,}. dkm_risk_final:{dkm_risk_f:,}. len(dkm_C):{len(dkm_C)}. len(dkm_C_final):{len(dkm_C_final)}')

            logger.info(f'===========================================================================================')
            l = int(len(dkm_C) / dkm_iters) if l_ratio == AUTO_COMPUTE_L else l_ratio * k
            logger.info(f'===========Starting scalable_k_mean with {SKM_ITERATIONS} iterations and l=={l}==============')
            skm_C, skm_C_final, skm_timing = competitors.scalable_k_means(N, SKM_ITERATIONS, l, k, m)
            skm_run_name = f'{SKM_ITERATIONS}-iter_skm_in_rep_{rep}'
            logger.info(f'{skm_run_name}_timing:{skm_timing}')
            skm_risk = risk(N, skm_C)
            skm_risk_f = risk(N, skm_C_final)
            logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
            test_summary = format_as_csv(skm_run_name, k, dt, m, ep, len(dkm_C), dkm_iters, SKM_ITERATIONS, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f)
            csv.write(test_summary + '\n')
            logger.info('\n' + HEADER + '\n' + test_summary)
            logger.info(f'===========================================================================================')

        except Exception:
            logger.exception("BAD!")
    csv.close()


def avg_r(risks, skm_run):
    return mean(risks['dkm']) / mean(risks[f'skm_{skm_run}'])


def avg_r_f(risks, skm_run):
    return mean(risks['dkm_f']) / mean(risks[f'skm_f_{skm_run}'])


if __name__ == "__main__":
    main(kdd=True)
