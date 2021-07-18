import logging
from typing import List, Iterable

from k_median_clustering.math import *


class Reducer:

    def __init__(self, Ni):
        self.Ni: pd.DataFrame = Ni

    def sample_P1_P2(self, alpha) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.Ni.sample(frac=alpha), self.Ni.sample(frac=alpha)  # TODO - figure out why this always returns exactly alpha

    def remove_handled_points_and_return_remaining(self, Ctmp: pd.DataFrame, v: float) -> int:
        """
        removes from _Ni all points further than v from Ctmp.
        returns the number of remaining elements.
        """
        if len(self.Ni) == 0:
            return 0
        distances = pairwise_distances_argmin_min_squared(self.Ni, Ctmp)
        remaining_points = distances > v
        self.Ni = self.Ni[remaining_points]
        return len(self.Ni)


class Coordinator:

    def __init__(self, k, kp, dt):
        self.C = pd.DataFrame()
        self._k = k
        self._kp = kp
        self._dt = dt

    def iterate(self, P1s: List[pd.DataFrame], P2s: List[pd.DataFrame], alpha) -> Tuple[float, pd.DataFrame]:
        P1 = pd.concat(P1s)
        P2 = pd.concat(P2s)  # TODO - do these copy the data? if so, avoid it

        v, Ctmp = EstProc(P1, P2, alpha, self._dt, self._k, self._kp)
        if v == 0.0:
            logging.error("Bad! v == 0.0")
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)
        return v, Ctmp

    def last_iteration(self, Nis: Iterable[pd.DataFrame]):
        logging.info('starting last iteration...')
        N_remaining = pd.concat(Nis)
        Ctmp = A(N_remaining, self._k)
        self.C = pd.concat([self.C, Ctmp], ignore_index=True)


def k_median_clustering(N: pd.DataFrame, k: int, ep: float, dt: float, m: int):
    n = len(N)
    logging.info("starting to split")
    Ns = np.array_split(N, m)
    logging.info("finished splitting")
    reducers = [Reducer(Ni) for Ni in Ns]
    kp = kplus_formula(k, dt)
    coordinator = Coordinator(k, kp, dt)
    alpha = alpha_formula(n, k, ep, dt, len(N))

    remaining_elements_count = len(N)
    loop_count = 0
    max_subset_size = max_subset_size_formula(n, k, ep, dt)
    logging.info(f"max_subset_size:{max_subset_size}")

    while remaining_elements_count > 4 * max_subset_size and max_subset_size > 10 * r_formula(alpha, k, phi_alpha_formula(alpha, k, dt)):
        logging.info(f"============ Starting LOOP {loop_count} ============")
        P1s_and_P2s = [r.sample_P1_P2(alpha) for r in reducers]

        P1s = [p1p2[0] for p1p2 in P1s_and_P2s]
        P2s = [p1p2[1] for p1p2 in P1s_and_P2s]

        v, Ctmp = coordinator.iterate(P1s, P2s, alpha)

        remaining_elements_count = sum(r.remove_handled_points_and_return_remaining(Ctmp, v) for r in reducers)
        alpha = alpha_formula(n, k, ep, dt, remaining_elements_count)
        logging.info(f"============ END OF LOOP {loop_count}. "
                     f"remaining_elements_count:{remaining_elements_count}."
                     f" alpha:{alpha}. v:{v}. len(Ctmp):{len(Ctmp)}. "
                     f" len(P2s):{sum(len(P2) for P2 in P2s)}. max_subset_size:{max_subset_size}"
                     f" r:{r_formula(alpha, k, phi_alpha_formula(alpha, k, dt))}"
                     f"  ============")
        loop_count += 1

    coordinator.last_iteration([r.Ni for r in reducers])

    logging.info(f'loop_count: {loop_count}. len(C):{len(coordinator.C)}')
    return coordinator.C
