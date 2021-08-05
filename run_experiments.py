from itertools import product
from statistics import mean
import subprocess
from time import strftime

from dist_k_mean.config import *
from dist_k_mean import competitors
from dist_k_mean.algo import distributed_k_means, DkmTiming
from dist_k_mean.competitors import SkmTiming
from dist_k_mean.datasets import get_dataset
from dist_k_mean.math import risk
from dist_k_mean.utils import setup_logger, log_config_file

# 1. avg of ratio of risks & risk_f DONE
# 2. results as CSV DONE
# 3. new data set DONE
# 4. sum/avg_iters(  max(sample_phase) + bb_phase + max(trim_phase)) DONE
# 5. Timing for skm (total_time/num_machines) DONE
# 6. super-comfortable config for running tests: gaussian params. l as param. DONE
# 7. calculate distances to Ctmp... DONE
# 8. find another BB DONE
# 9. run on Server
# 10. split internal BB & finalization BB
# 11. fast_clustering


label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
log_time = strftime('%Y%m%d%H%M')
run_name = f'dist_k_mean_{log_time}_git{label}'
logger = setup_logger('full_log', f'{run_name}.log', with_console=True)

log_config_file(logger)

HEADER = 'test_name,k,dt,m,ep,len(dkm_C),dkm_iters,skm_iters,l,len(skm_C),(dkm_r/skm_r),(dkm_r_f/skm_r_f),dkmr_avg_time,dkmc_avg_time,dkmt_time,skmi_total_time,skmf_time,skm_total_time'


def format_as_csv(test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, dkm_risk, skm_risk, dkm_risk_final, skm_risk_final, dkm_timing: DkmTiming, skm_timing: SkmTiming):
    return ','.join(str(s) for s in
                    [test_name, k, dt, m, ep, len_dkm_C, dkm_iters, skm_iters, l, len_skm_C, (dkm_risk / skm_risk), (dkm_risk_final / skm_risk_final),
                     dkm_timing.reducer_avg_time(), dkm_timing.coordinator_avg_time(), dkm_timing.total_time(), skm_timing.iterate_total_time, skm_timing.finalization_time, skm_timing.total_time(), ])


def main():
    csv = open(f"{run_name}_results.csv", "a")
    csv.write(HEADER + '\n')
    for the_round in range(ROUNDS):
        for k, dt, m, ep, l_ratio, dataset in product(KS, DELTAS, MS, EPSILONS, L_TO_K_RATIOS, DATASETS):
            try:
                logger.info(f"Loading Dataset {dataset}...")
                N = get_dataset(dataset)
                logger.info(f"len(N)={len(N)}")

                logger.info(f"======== Starting distributed k means with len(N)={len(N)} k={k} dt={dt} ep={ep} & m={m} ========")
                dkm_C, dkm_C_final, dkm_iters, dkm_timing = distributed_k_means(N, k, ep, dt, m, logger)
                logger.info(f'dkm_timing:{dkm_timing}')

                dkm_risk = risk(N, dkm_C)
                dkm_risk_f = risk(N, dkm_C_final)

                logger.info(f'len(N):{len(N)}. dkm_risk:{dkm_risk:,}. dkm_risk_final:{dkm_risk_f:,}. len(dkm_C):{len(dkm_C)}. len(dkm_C_final):{len(dkm_C_final)}')

                logger.info(f'===========================================================================================')
                l = int(len(dkm_C) / dkm_iters) if l_ratio == AUTO_COMPUTE_L else l_ratio * k
                for skm_iters in SKM_ITERATIONS:
                    logger.info(f'===========Starting scalable_k_mean with {skm_iters} iterations and l=={l}==============')
                    skm_C, skm_C_final, skm_timing = competitors.scalable_k_means(N, skm_iters, l, k, m)
                    skm_run_name = f'{skm_iters}-iter_skm_{dataset}_round_{the_round}'
                    logger.info(f'{skm_run_name}_timing:{skm_timing}')
                    skm_risk = risk(N, skm_C)
                    skm_risk_f = risk(N, skm_C_final)
                    logger.info(f'The scalable_k_means risk is {skm_risk:,} and size of C is {len(skm_C)}')
                    test_summary = format_as_csv(skm_run_name, k, dt, m, ep, len(dkm_C), dkm_iters, skm_iters, l, len(skm_C), dkm_risk, skm_risk, dkm_risk_f, skm_risk_f, dkm_timing, skm_timing)
                    csv.write(test_summary + '\n')
                    csv.flush()
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
    main()
