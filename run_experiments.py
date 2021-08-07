import sys
from itertools import product
from statistics import mean
from time import strftime

from dist_k_mean.algo import distributed_k_means
from dist_k_mean.black_box_clustering import _scalable_k_means

from dist_k_mean.config import *
from dist_k_mean.datasets import get_dataset
from dist_k_mean.math import risk
from dist_k_mean.utils import setup_logger, log_config_file

log_time = strftime('%m_%d_%H_%M')
run_name = f'dist_k_mean_{log_time}_{RUN_NAME}'
logger = setup_logger('full_log', f'{run_name}.log', with_console=True)

log_config_file(logger)

HEADER = "test_name,k,dt,m,ep,l,len(C),iterations,risk,risk_final,reducers_time,total_time"


def format_as_csv(test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time, total_time):
    return ','.join(str(s) for s in [test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time, total_time])


def main():
    logger.info(sys.argv)
    csv = open(f"{run_name}_results.csv", "a")
    csv.write(HEADER + '\n')

    for k, dt, m, ep in product(KS, DELTAS, MS, EPSILONS):
        try:
            N = get_dataset(DATASET)
            for the_round in range(ROUNDS):
                logger.info(f"Loading Dataset {DATASET}...")

                logger.info(f"len(N)={len(N)}")

                logger.info(f"======== Starting round {the_round} of distributed k means with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
                dkm_C, dkm_C_final, dkm_iters, dkm_timing = distributed_k_means(N, k, ep, dt, m, logger)
                logger.info(f'dkm_timing:{dkm_timing}')

                dkm_risk = risk(N, dkm_C)
                dkm_risk_f = risk(N, dkm_C_final)

                write_csv_line(csv, logger, f'us_round_{the_round}', k, dt, m, ep, -1, len(dkm_C), dkm_iters, dkm_risk, dkm_risk_f, dkm_timing.reducers_time(), dkm_timing.total_time())
                logger.info(f'===========================================================================================')

            for l_ratio in L_TO_K_RATIOS:
                for skm_iters in SKM_ITERATIONS:
                    for the_round in range(ROUNDS):
                        l = l_ratio * k
                        logger.info(f'===========Starting round {the_round} of scalable_k_mean with {skm_iters} iterations and l=={l}==============')
                        skm_C, skm_C_final, skm_timing = _scalable_k_means(N, skm_iters, l, k, m)
                        skm_run_name = f'{skm_iters}-iter_skm_{dataset}_round_{the_round}'
                        logger.info(f'{skm_run_name}_timing:{skm_timing}')
                        skm_risk = risk(N, skm_C)
                        skm_risk_f = risk(N, skm_C_final)
                        logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
                        write_csv_line(csv, logger, f'skm_round_{the_round}', k, dt, m, ep, -1, len(skm_C), skm_iters, skm_risk, skm_risk_f, skm_timing.reducers_time, skm_timing.total_time())
                        logger.info(f'===========================================================================================')

        except Exception:
            logger.exception("BAD!")
    csv.close()


def write_csv_line(csv, the_logger, test_name, k, dt, m, ep, l, len_C, iterations, risk, risk_final, reducers_time, total_time):
    test_summary = format_as_csv(test_name, k, dt, m, ep, l, len_C, iterations, risk, risk_final, reducers_time, total_time)
    the_logger.info('\n' + HEADER + '\n' + test_summary)
    csv.write(test_summary + '\n')
    csv.flush()


def avg_r(risks, skm_run):
    return mean(risks['dkm']) / mean(risks[f'skm_{skm_run}'])


def avg_r_f(risks, skm_run):
    return mean(risks['dkm_f']) / mean(risks[f'skm_f_{skm_run}'])


if __name__ == "__main__":
    main()
