from hess_clustering.math import *
from typing import List, Iterable


class Reducer:

    def __init__(self, Ni):
        self.Ni = Ni

    def sample_P1_P2(self, alpha) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return sample_P1_P2(alpha, self.Ni)

    def remove_handled_points(self, Ctmp: pd.DataFrame, v: float) -> int:
        """
        removes from _Ni all points further than v from Ctmp.
        returns the number of remaining elements.
        """
        pass

    def size_of_Ni(self) -> int:
        return len(self.Ni)


class Coordinator:

    def __init__(self, k):
        self.C = pd.DataFrame()
        self._k = k

    def iterate(self, P1s: List[pd.DataFrame], P2s: List[pd.DataFrame], alpha) -> Tuple[float, pd.DataFrame]:
        P1 = pd.concat(P1s)
        P2 = pd.concat(P2s)
        v, Ctmp = EstProc(P1, P2, alpha, 0, 0, 0)
        self.C.append(Ctmp)
        return v, Ctmp

    def last_iteration(self, Nis: Iterable[pd.DataFrame]):
        N_remaining = pd.concat(Nis)
        Ctmp = A(N_remaining, self._k)
        self.C.append(Ctmp)


def hess_clustering(N: pd.DataFrame, k: int, ep: float, dt: float, m: int):
    Ns = split(N, m)
    reducers = [Reducer(Ni) for Ni in Ns]
    coordinator = Coordinator(k)

    remaining_elements_count = len(N)

    while remaining_elements_count > max_subset_size_formula(k, ep, dt):
        alpha = alpha_formula(k, ep, dt, len(N))
        P1s_and_P2s = [r.sample_P1_P2(alpha) for r in reducers]

        P1s = [p1p2[0] for p1p2 in P1s_and_P2s]
        P2s = [p1p2[1] for p1p2 in P1s_and_P2s]

        v, Ctmp = coordinator.iterate(P1s, P2s, alpha)

        remaining_elements_count = sum(r.remove_handled_points(Ctmp, v) for r in reducers)

    coordinator.last_iteration(r.Ni for r in reducers)

    return coordinator.C
