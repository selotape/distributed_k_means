from logging import Logger
from statistics import mean, stdev
from typing import Tuple, List

from soccer.algo import run_soccer
from soccer.black_box import A_final
from soccer.competitors.competitors import scalable_k_means, ene_clustering
from soccer.config import *
from soccer.datasets import get_dataset
from soccer.math import risk
from soccer.utils import setup_logger, Measurement

SINGLE_HEADER = "test_name,k,dt,m,ep,l,len(C),iterations,risk,risk_final,reducers_time,total_time"
SUMMARY_HEADER = 'algorithm,k,epsilon,coord_mem,num_centers_avg,num_centers_stdv,rounds_avg,rounds_stdv,final_risk_avg,final_risk_stdv,comps_pm_avg,comps_pm_stdv,comps_tot_avg,comps_tot_stdv,reducers_time_avg,reducers_time_stdv,total_time_avg,total_time_stdv'


def main(run_name, algo, k, dataset, epsilon, blackbox, skm_iters=None):
    logger = setup_logger('full_log', f'{run_name}.log', with_console=True)

    csv = open(f"{run_name}_results.csv", "a")
    summary_f = open(f"{run_name}_summary.csv", "a")
    csv.write(SINGLE_HEADER + '\n')

    N = get_dataset(dataset, logger)

    run_experiment = create_experiment_runner(algo, k, N, csv, epsilon, skm_iters, blackbox, logger)

    risks, risks_final, measurements = run_all_rounds(run_experiment, logger)

    print_summary(algo, k, epsilon, summary_f, risks_final, measurements, logger)

    csv.close()
    summary_f.close()


SKM = 'SKM'
SOCCER = 'SOCCER'
ENE = 'ENE'


def create_experiment_runner(algo, k, N, csv, epsilon, skm_iters, blackbox, logger):
    if algo == SOCCER:
        def run_exp(the_round):
            return run_soccer_exp(N, csv, DELTA, epsilon, k, M, the_round, blackbox, logger)
    elif algo == SKM:
        def run_exp(the_round):
            return run_skm_exp(N, csv, DELTA, epsilon, k, L_TO_K_RATIO, M, skm_iters, the_round, logger)
    elif algo == ENE:
        def run_exp(the_round):
            return run_ene_exp(N, csv, k, epsilon, M, the_round, logger)
    else:
        raise NotImplementedError(f"Algo {algo} is not implemented")
    return run_exp


def run_soccer_exp(N, csv, dt, ep, k, m, the_round, blackbox, logger) -> Tuple[float, float, Measurement]:
    logger.info(
        f"======== Starting round {the_round} of SOCCER with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
    soccer_C, soccer_C_final, soccer_iters, measurement = run_soccer(N, k, ep, dt, m, blackbox, logger)
    logger.info(f'soccer_measurement:{measurement}')
    the_risk = risk(N, soccer_C)
    risk_final = risk(N, soccer_C_final)
    write_csv_line(csv, logger, f'us_round_{the_round}', k, dt, m, ep, -1, len(soccer_C), soccer_iters, the_risk,
                   risk_final, measurement.reducers_time(), measurement.total_time())
    return the_risk, risk_final, measurement


def run_skm_exp(N, csv, dt, ep, k, l_ratio, m, skm_iters, the_round, logger) -> Tuple[float, float, Measurement]:
    l = int(l_ratio * k)
    logger.info(
        f'===========Starting round {the_round} of KMeans|| with {skm_iters} iterations and l=={l}==============')
    skm_C, skm_C_final, measurement = scalable_k_means(N, skm_iters, l, k, m, A_final)
    skm_run_name = f'{skm_iters}-iter_skm_round_{the_round}'
    logger.info(f'{skm_run_name}_measurement:{measurement}')
    the_risk = risk(N, skm_C)
    risk_final = risk(N, skm_C_final)
    logger.info(f'The scalable_k_means risk is {the_risk:,} and size of C is {len(skm_C)}')
    write_csv_line(csv, logger, f'skm_round_{the_round}', k, dt, m, ep, -1, len(skm_C), skm_iters, the_risk, risk_final,
                   measurement.reducers_time(), measurement.total_time())
    return the_risk, risk_final, measurement


def run_ene_exp(N, csv, ep, k, m, the_round, logger) -> Tuple[float, float, Measurement]:
    logger.info(
        f"======== Starting round {the_round} of fast_clustering with len(N)={len(N)} k={k} ep={ep} & m={m} ========")
    C, C_final, iterations, measurement = ene_clustering(N, k, ep, m, A_final)
    logger.info(f'fast_measurement:{measurement}')
    the_risk = risk(N, C)
    risk_final = risk(N, C_final)
    write_csv_line(csv, logger, f'fast_round_{the_round}', k, -1, m, ep, -1, len(C), iterations, the_risk, risk_final,
                   measurement.reducers_time(), measurement.total_time())
    return the_risk, risk_final, measurement


def run_all_rounds(run_exp, logger):
    measurements = []
    risks = []
    risks_final = []
    for the_round in range(ROUNDS):
        the_risk, risk_final, measurement = run_exp(the_round)
        measurements.append(measurement), risks.append(the_risk), risks_final.append(risk_final)
        logger.info(f'===========================================================================================')
    return risks, risks_final, measurements


def write_csv_line(csv, the_logger: Logger, test_name: str, k: int, dt, m: int, ep, l, len_C: int, iterations: int,
                   the_risk: float, risk_final: float, reducers_time, total_time):
    test_summary = ','.join(str(s) for s in
                            [test_name, k, dt, m, ep, l, len_C, iterations, the_risk, risk_final, reducers_time,
                             total_time])
    the_logger.info('\n' + SINGLE_HEADER + '\n' + test_summary)
    csv.write(test_summary + '\n')
    csv.flush()


def print_summary(algo, k, epsilon, summary_f, risks_final, measurements: List[Measurement], logger):
    coordinator_memory = measurements[0].coord_memory()
    rounds_avg = mean(t.iterations() for t in measurements)
    rounds_stdv = stdev(t.iterations() for t in measurements)
    num_centers_avg = mean(t.num_centers_unfinalized() for t in measurements)
    num_centers_stdv = stdev(t.num_centers_unfinalized() for t in measurements)
    final_risk_avg = mean(risks_final)
    final_risk_stdv = stdev(risks_final)
    comps_pm_avg = mean(t.total_comps_per_machine() for t in measurements)
    comps_pm_stdv = stdev(t.total_comps_per_machine() for t in measurements)
    comps_tot_avg = mean(t.total_comps() for t in measurements)
    comps_tot_stdv = stdev(t.total_comps() for t in measurements)

    reducers_time_avg = mean(t.reducers_time() for t in measurements)
    reducers_time_stdv = stdev(t.reducers_time() for t in measurements)
    total_time_avg = mean(t.total_time() for t in measurements)
    total_time_stdv = stdev(t.total_time() for t in measurements)

    test_summary = f'{algo},{k},{epsilon},{coordinator_memory},{num_centers_avg},{num_centers_stdv},{rounds_avg},{rounds_stdv},{final_risk_avg},{final_risk_stdv},{comps_pm_avg},{comps_pm_stdv},{comps_tot_avg},{comps_tot_stdv},{reducers_time_avg},{reducers_time_stdv},{total_time_avg},{total_time_stdv}'
    logger.info('\n' + SUMMARY_HEADER + '\n' + test_summary)
    summary_f.write(test_summary + '\n')
