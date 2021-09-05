from logging import Logger
from statistics import mean, stdev
from typing import Tuple, List

from soccer.algo import distributed_k_means
from soccer.black_box import A_final
from soccer.competitors.competitors import ene_clustering, scalable_k_means
from soccer.config import *
from soccer.datasets import get_dataset
from soccer.math import risk
from soccer.utils import setup_logger, log_config_file, Measurement

run_name = f'{RUN_NAME}_{TIMESTAMP}'
logger = setup_logger('full_log', f'{run_name}.log', with_console=True)

log_config_file(logger)

SINGLE_HEADER = "test_name,k,dt,m,ep,l,len(C),iterations,risk,risk_final,reducers_time,total_time"


def main():
    logger.info(sys.argv)

    csv = open(f"{run_name}_results.csv", "a")
    summary_f = open(f"{DATASET}_{K}K_{TIMESTAMP}_summary.csv", "a")
    csv.write(SINGLE_HEADER + '\n')

    N = get_dataset(logger)

    run_experiment = create_experiment_runner(N, csv)

    risks, risks_final, timings = run_all_rounds(run_experiment)

    print_summary(summary_f, risks_final, timings)

    csv.close()
    summary_f.close()


def create_experiment_runner(N, csv):
    if ALGO == 'SOCCER':
        def run_exp(the_round):
            return run_dkm_exp(N, csv, DELTA, EPSILON, K, M, the_round)
    elif ALGO == 'SKM':
        def run_exp(the_round):
            return run_skm_exp(N, csv, DELTA, EPSILON, K, L_TO_K_RATIO, M, SKM_ITERATIONS, the_round)
    elif ALGO == 'ENE':
        def run_exp(the_round):
            return run_fast_exp(N, csv, K, EPSILON, M, the_round)
    else:
        raise NotImplementedError(f"Algo {ALGO} is not implemented")
    return run_exp


def run_fast_exp(N, csv, k, ep, m, the_round) -> Tuple[float, float, Measurement]:
    logger.info(f"======== Starting round {the_round} of fast_clustering with len(N)={len(N)} k={k} ep={ep} & m={m} ========")
    C, C_final, iterations, timing = ene_clustering(N, k, ep, m, A_final)
    logger.info(f'fast_timing:{timing}')
    the_risk = risk(N, C)
    risk_final = risk(N, C_final)
    write_csv_line(csv, logger, f'fast_round_{the_round}', k, -1, m, ep, -1, len(C), iterations, the_risk, risk_final, timing.reducers_time(), timing.total_time())
    return the_risk, risk_final, timing


def run_skm_exp(N, csv, dt, ep, k, l_ratio, m, skm_iters, the_round) -> Tuple[float, float, Measurement]:
    l = int(l_ratio * k)
    logger.info(f'===========Starting round {the_round} of scalable_k_mean with {skm_iters} iterations and l=={l}==============')
    skm_C, skm_C_final, timing = scalable_k_means(N, skm_iters, l, k, m, A_final)
    skm_run_name = f'{skm_iters}-iter_skm_{DATASET}_round_{the_round}'
    logger.info(f'{skm_run_name}_timing:{timing}')
    the_risk = risk(N, skm_C)
    risk_final = risk(N, skm_C_final)
    logger.info(f'The scalable_k_means risk is {the_risk:,} and size of C is {len(skm_C)}')
    write_csv_line(csv, logger, f'skm_round_{the_round}', k, dt, m, ep, -1, len(skm_C), skm_iters, the_risk, risk_final, timing.reducers_time(), timing.total_time())
    return the_risk, risk_final, timing


def run_dkm_exp(N, csv, dt, ep, k, m, the_round) -> Tuple[float, float, Measurement]:
    logger.info(f"======== Starting round {the_round} of distributed k means with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
    dkm_C, dkm_C_final, dkm_iters, timing = distributed_k_means(N, k, ep, dt, m, logger)
    logger.info(f'dkm_timing:{timing}')
    the_risk = risk(N, dkm_C)
    risk_final = risk(N, dkm_C_final)
    write_csv_line(csv, logger, f'us_round_{the_round}', k, dt, m, ep, -1, len(dkm_C), dkm_iters, the_risk, risk_final, timing.reducers_time(), timing.total_time())
    return the_risk, risk_final, timing


def run_all_rounds(run_exp):
    timings = []
    risks = []
    risks_final = []
    for the_round in range(ROUNDS):
        risk, risk_final, timing = run_exp(the_round)
        timings.append(timing), risks.append(risk), risks_final.append(risk_final)
        logger.info(f'===========================================================================================')
    return risks, risks_final, timings


def write_csv_line(csv, the_logger: Logger, test_name: str, k: int, dt, m: int, ep, l, len_C: int, iterations: int, the_risk: float, risk_final: float, reducers_time, total_time):
    test_summary = ','.join(str(s) for s in [test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time, total_time])
    the_logger.info('\n' + SINGLE_HEADER + '\n' + test_summary)
    csv.write(test_summary + '\n')
    csv.flush()


SUMMARY_HEADER = 'algorithm,k,epsilon,coord_mem,num_centers_avg,num_centers_stdv,rounds_avg,rounds_stdv,final_risk_avg,final_risk_stdv,comps_pm_avg,comps_pm_stdv,comps_tot_avg,comps_tot_stdv'


def print_summary(summary_f, risks_final, timings: List[Measurement]):
    coordinator_memory = timings[0].coord_memory()
    rounds_avg = mean(t.iterations() for t in timings)
    rounds_stdv = stdev(t.iterations() for t in timings)
    num_centers_avg = mean(t.num_centers_unfinalized() for t in timings)
    num_centers_stdv = stdev(t.num_centers_unfinalized() for t in timings)
    final_risk_avg = mean(risks_final)
    final_risk_stdv = stdev(risks_final)
    comps_pm_avg = mean(t.total_comps_per_machine() for t in timings)
    comps_pm_stdv = stdev(t.total_comps_per_machine() for t in timings)
    comps_tot_avg = mean(t.total_comps() for t in timings)
    comps_tot_stdv = stdev(t.total_comps() for t in timings)

    test_summary = f'{ALGO},{K},{EPSILON},{coordinator_memory},{num_centers_avg},{num_centers_stdv},{rounds_avg},{rounds_stdv},{final_risk_avg},{final_risk_stdv},{comps_pm_avg},{comps_pm_stdv},{comps_tot_avg},{comps_tot_stdv}'
    logger.info('\n' + SUMMARY_HEADER + '\n' + test_summary)
    summary_f.write(test_summary + '\n')


if __name__ == "__main__":
    main()
