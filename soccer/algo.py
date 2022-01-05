import time
from dataclasses import dataclass, field
from typing import List, Iterable, Tuple

from soccer.black_box import A_inner, A_final
from soccer.config import INNER_BLACKBOX_ITERATIONS, INNER_BLACKBOX_L_TO_K_RATIO
from soccer.math import *
from soccer.utils import keep_time, get_kept_time, Measurement


@dataclass
class SoccerMeasurement(Measurement):
    sample_times: List[float] = field(default_factory=list)
    iterate_times: List[float] = field(default_factory=list)
    remove_handled_times: List[float] = field(default_factory=list)
    total_comps_per_machine_: float = 0.0
    total_comps_: float = 0.0
    final_iter_time: float = 0.0
    finalization_time: float = 0.0
    iterations_: int = 0
    num_centers_unfinalized_: int = 0
    coord_memory_: float = -1.0

    def reducers_time(self):
        return sum(self.sample_times) + sum(self.remove_handled_times)

    def total_time(self):
        return sum(self.sample_times + self.iterate_times + self.remove_handled_times + [self.final_iter_time, self.finalization_time])

    def total_comps_per_machine(self):
        return self.total_comps_per_machine_

    def total_comps(self):
        return self.total_comps_

    def coord_memory(self):
        return self.coord_memory_

    def num_centers_unfinalized(self):
        return self.num_centers_unfinalized_

    def iterations(self):
        return self.iterations_


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

    def __init__(self, k, kp, dt, ep, m, inner_iterations, logger):
        self.C = pd.DataFrame()
        self._k = k
        self._kp = kp
        self._dt = dt
        self._ep = ep
        self._logger = logger
        self._psi = 0.0
        self._m = m
        self._inner_iterations = inner_iterations

    @keep_time
    def iterate(self, P1s: List[pd.DataFrame], P2s: List[pd.DataFrame], alpha) -> Tuple[float, pd.DataFrame]:
        P1 = pd.concat(P1s)
        P2 = pd.concat(P2s)

        v, Ctmp = self.EstProc(P1, P2, alpha, self._dt, self._k, self._kp)
        if v == 0.0:
            logging.error("Bad! v == 0.0")
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)
        return v, Ctmp

    @keep_time
    def last_iteration(self, Nis: Iterable[pd.DataFrame]):
        self._logger.info('starting last iteration...')
        N_remaining = pd.concat(Nis)
        if len(N_remaining) > self._k:
            Ctmp = A_inner(N_remaining, self._k, m=self._m, iterations=self._inner_iterations, l=int(self._k * INNER_BLACKBOX_L_TO_K_RATIO)) if len(N_remaining) > self._k else N_remaining
        else:
            Ctmp = N_remaining
        Ctmp.columns = self.C.columns
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)

    def EstProc(self, P1: pd.DataFrame, P2: pd.DataFrame, alpha: float, dt: float, k: int, kp: int) -> Tuple[float, pd.DataFrame]:
        """
        calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
        Emits the cluster and the ~risk.
        """
        Ta = A_inner(P1, kp, m=self._m, iterations=self._inner_iterations, l=int(self._k * INNER_BLACKBOX_L_TO_K_RATIO))

        phi_alpha = phi_alpha_formula(alpha, k, dt, self._ep)
        r = r_formula(alpha, k, phi_alpha)
        Rr = risk_truncated(P2, Ta, r)

        self._psi = max((2 / (3 * alpha)) * Rr, self._psi)
        return v_formula(self._psi, k, phi_alpha), Ta


def run_soccer(N: pd.DataFrame, k: int, ep: float, dt: float, m: int, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, int, SoccerMeasurement]:
    n = len(N)
    logger.info("starting to split")
    Ns = np.array_split(N, m)
    logger.info("finished splitting")
    reducers = [Reducer(Ni) for Ni in Ns]
    kp = kplus_formula(k, dt, ep)
    coordinator = Coordinator(k, kp, dt, ep, m, INNER_BLACKBOX_ITERATIONS, logger)
    alpha = alpha_formula(n, k, ep, dt, len(N))
    measurement = SoccerMeasurement()
    remaining_elements_count = len(N)
    max_subset_size = max_subset_size_formula(n, k, ep, dt)
    measurement.coord_memory_ = max_subset_size
    logger.info(f"max_subset_size:{max_subset_size}")

    while remaining_elements_count > max_subset_size:
        measurement.iterations_ += 1
        logger.info(f"============ Starting LOOP {measurement.iterations_} ============")
        P1s_and_P2s = [r.sample_P1_P2(alpha) for r in reducers]
        measurement.sample_times.append(max(get_kept_time(r, 'sample_P1_P2') for r in reducers))

        P1s = [p1p2[0] for p1p2 in P1s_and_P2s]
        P2s = [p1p2[1] for p1p2 in P1s_and_P2s]

        v, Ctmp = coordinator.iterate(P1s, P2s, alpha)
        measurement.iterate_times.append(get_kept_time(coordinator, 'iterate'))

        measurement.total_comps_per_machine_ += max(len(r.Ni) * len(Ctmp) for r in reducers)
        measurement.total_comps_ += remaining_elements_count * len(Ctmp)

        remaining_elements_count = sum(r.remove_handled_points_and_return_remaining(Ctmp, v) for r in reducers)
        measurement.remove_handled_times.append(max(get_kept_time(r, 'remove_handled_points_and_return_remaining') for r in reducers))

        if remaining_elements_count == 0:
            logger.info("remaining_elements_count == 0!!")
            break

        alpha = alpha_formula(n, k, ep, dt, remaining_elements_count)
        end_of_loop = f"============ END OF LOOP {measurement.iterations_}. " + \
                      f"remaining_elements_count:{remaining_elements_count}." + \
                      f" alpha:{alpha}. v:{v}. len(Ctmp):{len(Ctmp)}. " + \
                      f" len(P2s):{sum(len(P2) for P2 in P2s)}. max_subset_size:{max_subset_size}" + \
                      f" r:{r_formula(alpha, k, phi_alpha_formula(alpha, k, dt, ep))}" + \
                      f"  ============"
        logger.info(end_of_loop)

    logger.info(f"Finished while-loop after {measurement.iterations_} iterations")

    coordinator.last_iteration([r.Ni for r in reducers])
    measurement.final_iter_time = get_kept_time(coordinator, 'last_iteration')

    logger.info("Calculating C_final")
    C_weights = calculate_center_weights(coordinator, reducers)
    start = time.time()
    C_final = A_final(coordinator.C, k, C_weights)
    measurement.num_centers_unfinalized_ = len(coordinator.C)
    measurement.finalization_time = time.time() - start

    logger.info(f'iteration: {measurement.iterations_}. len(C):{len(coordinator.C)}. len(C_final)={len(C_final)}')
    return coordinator.C, C_final, measurement.iterations_, measurement


def calculate_center_weights(coordinator, reducers: Iterable[Reducer]):
    return np.sum([r.measure_weights(coordinator.C) for r in reducers], axis=0)
