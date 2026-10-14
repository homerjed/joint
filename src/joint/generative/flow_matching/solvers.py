from typing import NamedTuple

import diffrax as dfx


class SolutionKwargs(NamedTuple):
    """Diffrax ODE solver settings used during flow-matching sampling."""

    solver: dfx.AbstractSolver
    t0: float = 0.0
    t1: float = 1.0
    dt0: float = 0.05
