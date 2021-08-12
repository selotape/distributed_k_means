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

SINGLE_HEADER = "test_name,k,dt,m,ep,l,len(C),iterations,risk,risk_final,reducers_time,total_time"
SUMMARY_HEADER = "test_name,reducers_time_avg,total_time_avg,risk_avg,risk_final_avg"


def format_as_csv(test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time, total_time):
    return ','.join(str(s) for s in [test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time, total_time])


def print_summary(csv, risks, risks_final, timings):
    csv.write('\n' + SUMMARY_HEADER + '\n')
    csv.write(','.join(str(x) for x in [RUN_NAME, mean(t.reducers_time() for t in timings), mean(t.total_time() for t in timings), mean(risks), mean(risks_final)]) + '\n')


def main():
    logger.info(sys.argv)

    csv = open(f"{run_name}_results.csv", "a")
    csv.write(SINGLE_HEADER + '\n')

    logger.info(f"Loading Dataset {DATASET}...")
    N = get_dataset(DATASET)
    logger.info(f"len(N)={len(N)}")

    run_exp = create_exp_runner(N, csv)

    risks, risks_final, timings = run_all_rounds(run_exp)

    print_summary(csv, risks, risks_final, timings)

    csv.close()


def run_all_rounds(run_exp):
    timings = []
    risks = []
    risks_final = []
    for the_round in range(ROUNDS):
        risk, risk_final, timing = run_exp(the_round)
        timings.append(timing), risks.append(risk), risks_final.append(risk_final)
        logger.info(f'===========================================================================================')
    return risks, risks_final, timings


def create_exp_runner(N, csv):
    if ALGO == 'DKM':
        def run_exp(the_round):
            return run_dkm_exp(N, csv, DELTA, EPSILON, K, M, the_round)
    elif ALGO == 'SKM':
        def run_exp(the_round):
            return run_skm_exp(N, csv, DELTA, EPSILON, K, L_TO_K_RATIO, M, SKM_ITERATIONS, the_round)
    else:
        raise NotImplementedError(f"Algo {ALGO} is not implemented")
    return run_exp


def run_skm_exp(N, csv, dt, ep, k, l_ratio, m, skm_iters, the_round):
    l = l_ratio * k
    logger.info(f'===========Starting round {the_round} of scalable_k_mean with {skm_iters} iterations and l=={l}==============')
    skm_C, skm_C_final, timing = _scalable_k_means(N, skm_iters, l, k, m)
    skm_run_name = f'{skm_iters}-iter_skm_{DATASET}_round_{the_round}'
    logger.info(f'{skm_run_name}_timing:{timing}')
    the_risk = risk(N, skm_C)
    risk_final = risk(N, skm_C_final)
    logger.info(f'The scalable_k_means risk is {the_risk:,} and size of C is {len(skm_C)}')
    write_csv_line(csv, logger, f'skm_round_{the_round}', k, dt, m, ep, -1, len(skm_C), skm_iters, the_risk, risk_final, timing.reducers_time_, timing.total_time())
    return the_risk, risk_final, timing


def run_dkm_exp(N, csv, dt, ep, k, m, the_round):
    logger.info(f"======== Starting round {the_round} of distributed k means with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
    dkm_C, dkm_C_final, dkm_iters, timing = distributed_k_means(N, k, ep, dt, m, logger)
    logger.info(f'dkm_timing:{timing}')
    the_risk = risk(N, dkm_C)
    risk_final = risk(N, dkm_C_final)
    write_csv_line(csv, logger, f'us_round_{the_round}', k, dt, m, ep, -1, len(dkm_C), dkm_iters, risk, risk_final, timing.reducers_time(), timing.total_time())
    return the_risk, risk_final, timing


def write_csv_line(csv, the_logger, test_name, k, dt, m, ep, l, len_C, iterations, risk, risk_final, reducers_time, total_time):
    test_summary = format_as_csv(test_name, k, dt, m, ep, l, len_C, iterations, risk, risk_final, reducers_time, total_time)
    the_logger.info('\n' + SINGLE_HEADER + '\n' + test_summary)
    csv.write(test_summary + '\n')
    csv.flush()


def avg_r(risks, skm_run):
    return mean(risks['dkm']) / mean(risks[f'skm_{skm_run}'])


def avg_r_f(risks, skm_run):
    return mean(risks['dkm_f']) / mean(risks[f'skm_f_{skm_run}'])


if __name__ == "__main__":
    main()
