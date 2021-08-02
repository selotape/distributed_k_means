from dataclasses import dataclass, field
import time
from statistics import mean
from typing import List, Iterable, Tuple

from dist_k_mean.disk_k_mean_math import *
from dist_k_mean.utils import keep_time, get_kept_time


@dataclass
class DkmTiming:
    sample_times: List[float] = field(default_factory=list)
    iterate_times: List[float] = field(default_factory=list)
    remove_handled_times: List[float] = field(default_factory=list)
    final_iter_time: float = 0.0
    weighing_time: float = 0.0
    finalization_time: float = 0.0

    def reducer_avg_time(self):
        return mean(self.sample_times) + mean(self.remove_handled_times)

    def coordinator_avg_time(self):
        return mean(self.iterate_times)

    def total_time(self):
        return sum(self.sample_times + self.iterate_times + self.remove_handled_times + [self.final_iter_time, self.finalization_time, self.weighing_time])


class Reducer:

    def __init__(self, Ni):
        self.Ni_orig: pd.DataFrame = Ni
        self.Ni: pd.DataFrame = Ni
        self._prev_distances_to_C = None

    @keep_time
    def sample_P1_P2(self, alpha) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.Ni.sample(frac=alpha), self.Ni.sample(frac=alpha)  # TODO - figure out why this always returns exactly alpha

    @keep_time
    def remove_handled_points_and_return_remaining(self, Ctmp: pd.DataFrame, v: float) -> int:
        """
        removes from _Ni all points further than v from C.
        returns the number of remaining elements.

        TODO - measure distance to Ctmp and do min with the prev distances (to C before adding Ctmp)
        """
        if len(self.Ni) == 0:
            return 0
        distances_to_Ctmp = pairwise_distances_argmin_min_squared(self.Ni, Ctmp)
        distances_to_C = np.minimum(distances_to_Ctmp, self._prev_distances_to_C) if self._prev_distances_to_C is not None else distances_to_Ctmp
        remaining_points: pd.DataFrame[bool] = distances_to_Ctmp > v
        self.Ni = self.Ni[remaining_points]
        self._prev_distances_to_C = distances_to_C[remaining_points]
        return len(self.Ni)

    @keep_time
    def measure_weights(self, C):
        return measure_weights(self.Ni_orig, C)


class Coordinator:

    def __init__(self, k, kp, dt, logger):
        self.C = pd.DataFrame()
        self._k = k
        self._kp = kp
        self._dt = dt
        self._logger = logger
        self._psi = 0.0

    @keep_time
    def iterate(self, P1s: List[pd.DataFrame], P2s: List[pd.DataFrame], alpha) -> Tuple[float, pd.DataFrame]:
        P1 = pd.concat(P1s)
        P2 = pd.concat(P2s)  # TODO - do these copy the data? if so, avoid it

        v, Ctmp = self.EstProc(P1, P2, alpha, self._dt, self._k, self._kp)
        if v == 0.0:
            logging.error("Bad! v == 0.0")
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)
        return v, Ctmp

    @keep_time
    def last_iteration(self, Nis: Iterable[pd.DataFrame]):
        self._logger.info('starting last iteration...')
        N_remaining = pd.concat(Nis)
        Ctmp = A(N_remaining, self._k) if len(N_remaining) > self._k else N_remaining
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)

    def EstProc(self, P1: pd.DataFrame, P2: pd.DataFrame, alpha: float, dt: float, k: int, kp: int) -> Tuple[float, pd.DataFrame]:
        """
        calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
        Emits the cluster and the ~risk.
        """
        Ta = A(P1, kp)

        phi_alpha = phi_alpha_formula(alpha, k, dt)
        r = r_formula(alpha, k, phi_alpha)
        Rr = risk_truncated(P2, Ta, r)

        self._psi = max((1 / (3 * alpha)) * Rr, self._psi)
        return v_formula(self._psi, k, phi_alpha), Ta


def distributed_k_means(N: pd.DataFrame, k: int, ep: float, dt: float, m: int, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, int, DkmTiming]:
    n = len(N)
    logger.info("starting to split")
    Ns = np.array_split(N, m)
    logger.info("finished splitting")
    reducers = [Reducer(Ni) for Ni in Ns]
    kp = kplus_formula(k, dt)
    coordinator = Coordinator(k, kp, dt, logger)
    alpha = alpha_formula(n, k, ep, dt, len(N))
    timing = DkmTiming()
    remaining_elements_count = len(N)
    iteration = 0
    max_subset_size = max_subset_size_formula(n, k, ep, dt)
    logger.info(f"max_subset_size:{max_subset_size}")

    if n ** ep < 4:
        err_msg = f"n ** ep < 4 !! n:{n}. ep:{ep}"
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    while remaining_elements_count > max_subset_size:
        iteration += 1
        logger.info(f"============ Starting LOOP {iteration} ============")
        P1s_and_P2s = [r.sample_P1_P2(alpha) for r in reducers]
        timing.sample_times.append(max(get_kept_time(r, 'sample_P1_P2') for r in reducers))

        P1s = [p1p2[0] for p1p2 in P1s_and_P2s]
        P2s = [p1p2[1] for p1p2 in P1s_and_P2s]

        v, Ctmp = coordinator.iterate(P1s, P2s, alpha)
        timing.iterate_times.append(get_kept_time(coordinator, 'iterate'))

        remaining_elements_count = sum(r.remove_handled_points_and_return_remaining(Ctmp, v) for r in reducers)
        timing.remove_handled_times.append(max(get_kept_time(r, 'remove_handled_points_and_return_remaining') for r in reducers))

        if remaining_elements_count == 0:
            logger.info("remaining_elements_count == 0!!")
            break

        alpha = alpha_formula(n, k, ep, dt, remaining_elements_count)
        end_of_loop = f"============ END OF LOOP {iteration}. " + \
                      f"remaining_elements_count:{remaining_elements_count}." + \
                      f" alpha:{alpha}. v:{v}. len(Ctmp):{len(Ctmp)}. " + \
                      f" len(P2s):{sum(len(P2) for P2 in P2s)}. max_subset_size:{max_subset_size}" + \
                      f" r:{r_formula(alpha, k, phi_alpha_formula(alpha, k, dt))}" + \
                      f"  ============"
        logger.info(end_of_loop)

    logger.info(f"Finished while-loop after {iteration} iterations")
    if remaining_elements_count > 0:
        coordinator.last_iteration([r.Ni for r in reducers])
        iteration += 1
        timing.final_iter_time = get_kept_time(coordinator, 'last_iteration')

    start = time.time()
    logger.info("Calculating center-weights...")
    C_weights = calculate_center_weights(coordinator, reducers)
    timing.weighing_time = (time.time() - start) / m

    logger.info("Calculating C_final")
    start = time.time()
    C_final = A(coordinator.C, k, C_weights)
    timing.finalization_time = time.time() - start

    logger.info(f'iteration: {iteration}. len(C):{len(coordinator.C)}. len(C_final)={len(C_final)}')
    return coordinator.C, C_final, iteration, timing


def calculate_center_weights(coordinator, reducers: Iterable[Reducer]):
    return np.sum([r.measure_weights(coordinator.C) for r in reducers], axis=0)
