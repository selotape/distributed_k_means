from typing import List, Iterable

from k_median_clustering.math import *
from k_median_clustering.utils import keep_time


class Reducer:

    def __init__(self, Ni):
        self.Ni_orig: pd.DataFrame = Ni
        self.Ni: pd.DataFrame = Ni

    @keep_time
    def sample_P1_P2(self, alpha) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.Ni.sample(frac=alpha), self.Ni.sample(frac=alpha)  # TODO - figure out why this always returns exactly alpha

    @keep_time
    def remove_handled_points_and_return_remaining(self, C: pd.DataFrame, v: float) -> int:
        """
        removes from _Ni all points further than v from C.
        returns the number of remaining elements.

        Using Ctmp instead of C might improve runtime but harm number of removed elements (and thus num interations)
        """
        if len(self.Ni) == 0:
            return 0
        distances = pairwise_distances_argmin_min_squared(self.Ni, C)
        remaining_points = distances > v
        self.Ni = self.Ni[remaining_points]
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

    @keep_time
    def iterate(self, P1s: List[pd.DataFrame], P2s: List[pd.DataFrame], alpha) -> Tuple[float, pd.DataFrame]:
        P1 = pd.concat(P1s)
        P2 = pd.concat(P2s)  # TODO - do these copy the data? if so, avoid it

        v, Ctmp = EstProc(P1, P2, alpha, self._dt, self._k, self._kp)
        if v == 0.0:
            logging.error("Bad! v == 0.0")
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)
        return v, Ctmp

    @keep_time
    def last_iteration(self, Nis: Iterable[pd.DataFrame]):
        self._logger.info('starting last iteration...')
        N_remaining = pd.concat(Nis)
        Ctmp = A(N_remaining, self._k)
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)


def distributed_k_median_clustering(N: pd.DataFrame, k: int, ep: float, dt: float, m: int, logger: logging.Logger, results: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    n = len(N)
    logger.info("starting to split")
    Ns = np.array_split(N, m)
    logger.info("finished splitting")
    reducers = [Reducer(Ni) for Ni in Ns]
    kp = kplus_formula(k, dt)
    coordinator = Coordinator(k, kp, dt, logger)
    alpha = alpha_formula(n, k, ep, dt, len(N))

    remaining_elements_count = len(N)
    iteration = 0
    max_subset_size = max_subset_size_formula(n, k, ep, dt)
    logger.info(f"max_subset_size:{max_subset_size}")

    if n ** ep < 8:
        err_msg = f"n ** ep < 8 !! n:{n}. ep:{ep}"
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    # while remaining_elements_count > 4 * max_subset_size and max_subset_size > 5 * r_formula(alpha, k, phi_alpha_formula(alpha, k, dt)):
    while remaining_elements_count > max_subset_size:
        logger.info(f"============ Starting LOOP {iteration} ============")
        P1s_and_P2s = [r.sample_P1_P2(alpha) for r in reducers]

        P1s = [p1p2[0] for p1p2 in P1s_and_P2s]
        P2s = [p1p2[1] for p1p2 in P1s_and_P2s]

        v, Ctmp = coordinator.iterate(P1s, P2s, alpha)

        remaining_elements_count = sum(r.remove_handled_points_and_return_remaining(coordinator.C, v) for r in reducers)
        alpha = alpha_formula(n, k, ep, dt, remaining_elements_count)
        end_of_loop = f"============ END OF LOOP {iteration}. " + \
                      f"remaining_elements_count:{remaining_elements_count}." + \
                      f" alpha:{alpha}. v:{v}. len(Ctmp):{len(Ctmp)}. " + \
                      f" len(P2s):{sum(len(P2) for P2 in P2s)}. max_subset_size:{max_subset_size}" + \
                      f" r:{r_formula(alpha, k, phi_alpha_formula(alpha, k, dt))}" + \
                      f"  ============"
        results.info(end_of_loop)
        iteration += 1

    logger.info(f"Finished while-loop after {iteration-1} iterations")
    coordinator.last_iteration([r.Ni for r in reducers])

    logger.info(f"Calculating center-weights")
    C_weights = calculate_center_weights(coordinator, reducers)
    logger.info(f"Calculating C_final")
    C_final = A(coordinator.C, k, C_weights)

    iteration += 1

    logger.info(f'iteration: {iteration}. len(C):{len(coordinator.C)}. len(C_final)={len(C_final)}')
    return coordinator.C, C_final, iteration


def calculate_center_weights(coordinator, reducers: Iterable[Reducer]):
    return np.sum([r.measure_weights(coordinator.C) for r in reducers], axis=0)
