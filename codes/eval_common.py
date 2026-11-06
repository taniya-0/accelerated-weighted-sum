"""
eval_common.py
==============
This code has the OOD, heuristics. Clustered distribution are in clustered.py code. This code is used by eval_target.py and eval_budget.py
In this code RaceTracker matches every solve result against the ground truth which is the exact algo's result. It also computed HV.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import clustered as cl
from problems import Instance, Problem, Solver
from exa import ExactSearch, final_points, point_key, run_exact

HERE = Path(__file__).resolve().parent

MATCH_TOL = 0.5                 # integer objectives: same point if every coordinate differs < 0.5
EXACT_TIME_LIMIT = 5400.0
TARGET_SEED = 20260828          # seed series of the fixed-target race and OOD runs
BUDGET_SEED = 20260919          # seed series of the fixed-budget ablation


# =============================================================================
# POINT MATCHING
# =============================================================================

def points_match(a, b, tol: float = MATCH_TOL) -> bool:
    """Same point if every coordinate differs by less than tol."""
    return len(a) == len(b) and all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))


def find_index(point, truth_points, tol: float = MATCH_TOL) -> Optional[int]:
    """Index of the first truth point that matches `point` (None if none)."""
    if point is None:
        return None
    for i, t in enumerate(truth_points):
        if points_match(point, t, tol):
            return i
    return None


class RaceTracker:
    """Cumulative matching of every solve result against the truth.

    observe(z) -> (new true ESN point?, new spurious point?). The time spent
    here is summed in check_time, so it can be subtracted from a method's time.
    """

    def __init__(self, truth_points, tol: float = MATCH_TOL):
        self.oracle = [tuple(float(v) for v in z) for z in truth_points]
        self.tol = float(tol)
        self.used: set = set()
        self.matched: list = []
        self.spurious: list = []
        self.check_time = 0.0

    def observe(self, point):
        start = time.perf_counter()
        new_true = new_spurious = False
        if point is not None:
            pt = tuple(float(v) for v in point)
            idx = find_index(pt, self.oracle, self.tol)
            if idx is not None:
                if idx not in self.used:
                    self.used.add(idx)
                    self.matched.append(pt)
                    new_true = True
            elif not any(points_match(pt, old, self.tol) for old in self.spurious):
                self.spurious.append(pt)
                new_spurious = True
        self.check_time += time.perf_counter() - start
        return new_true, new_spurious


# =============================================================================
# HYPERVOLUME (all objectives maximised)
# =============================================================================

def hv_reference(points, p: int) -> tuple:
    """Reference point dominated by every point: min of each objective - 1."""
    if not points:
        return tuple(0.0 for _ in range(p))
    arr = np.asarray(points, dtype=np.float64)
    return tuple(float(arr[:, q].min() - 1.0) for q in range(p))


def hypervolume_2d(points, ref) -> float:
    pts = sorted(((z1, z2) for z1, z2 in points if z1 > ref[0] and z2 > ref[1]), key=lambda z: -z[0])
    hv, prev_z2 = 0.0, ref[1]
    for z1, z2 in pts:                       # z1 descending: add the new horizontal strip
        if z2 > prev_z2:
            hv += (z1 - ref[0]) * (z2 - prev_z2)
            prev_z2 = z2
    return hv


def _nondominated_2d(pts):
    nd, best_z2 = [], -float("inf")
    for z1, z2 in sorted(pts, key=lambda z: (-z[0], -z[1])):
        if z2 > best_z2:
            nd.append((z1, z2))
            best_z2 = z2
    return nd


def hypervolume_3d(points, ref) -> float:
    """Sweep over z3 (descending), adding 2-D slabs area * thickness."""
    pts = sorted(((float(z[0]), float(z[1]), float(z[2])) for z in points
                  if z[0] > ref[0] and z[1] > ref[1] and z[2] > ref[2]), key=lambda z: -z[2])
    hv, front, i = 0.0, [], 0
    while i < len(pts):
        z3 = pts[i][2]
        while i < len(pts) and abs(pts[i][2] - z3) < 1e-9:
            front.append((pts[i][0], pts[i][1]))
            i += 1
        front = _nondominated_2d(front)
        next_z3 = pts[i][2] if i < len(pts) else ref[2]
        hv += hypervolume_2d(front, (ref[0], ref[1])) * (z3 - next_z3)
    return hv


def hypervolume(points, ref, p: int) -> float:
    if not points:
        return 0.0
    if p == 2:
        return hypervolume_2d(points, ref)
    if p == 3:
        return hypervolume_3d(points, ref)
    raise NotImplementedError(f"hypervolume for p={p}")

# =============================================================================
# INSTANCES OF DIFFERENT DISTRIBUTIONS
# =============================================================================
# "id" is the training distribution. 

KS_SCENARIOS = {                         # profits AND item weights
    "id":                          {"kind": "uniform", "low": 60, "high": 100},
    "uniform_40_80":               {"kind": "uniform", "low": 40, "high": 80},
    "uniform_80_120":              {"kind": "uniform", "low": 80, "high": 120},
    "uniform_40_120":              {"kind": "uniform", "low": 40, "high": 120},
    "gaussian_80_8_range_60_100":  {"kind": "truncated_normal", "mean": 80.0, "sd": 8.0, "low": 60, "high": 100},
    "gaussian_80_15_range_40_120": {"kind": "truncated_normal", "mean": 80.0, "sd": 15.0, "low": 40, "high": 120},
}
TSP_SCENARIOS = {                        # city coordinates
    "id":                      {"kind": "uniform", "low": 0, "high": 999},
    "coords_0_499":            {"kind": "uniform", "low": 0, "high": 499},
    "coords_0_1499":           {"kind": "uniform", "low": 0, "high": 1499},
    "coords_250_749":          {"kind": "uniform", "low": 250, "high": 749},
    "gaussian_500_150_0_999":  {"kind": "truncated_normal", "mean": 500.0, "sd": 150.0, "low": 0, "high": 999},
    "gaussian_500_300_0_1499": {"kind": "truncated_normal", "mean": 500.0, "sd": 300.0, "low": 0, "high": 1499},
}


def _scenarios(problem: Problem) -> dict:
    return KS_SCENARIOS if problem.kind == "knapsack" else TSP_SCENARIOS


def _draw_values(rng: np.random.Generator, scen: dict, size) -> np.ndarray:
    """Integers from the scenario distribution."""
    if scen["kind"] == "uniform":
        return np.asarray(rng.integers(int(scen["low"]), int(scen["high"]) + 1, size=size), dtype=np.int64)
    total = int(np.prod(size))
    accepted = []
    while len(accepted) < total:
        draws = rng.normal(float(scen["mean"]), float(scen["sd"]), size=max(2 * (total - len(accepted)), 64))
        accepted.extend(draws[(draws >= scen["low"]) & (draws <= scen["high"])].tolist())
    return np.asarray(np.rint(np.asarray(accepted[:total])).reshape(size), dtype=np.int64)


def _distances(coords: np.ndarray) -> np.ndarray:
    """Rounded Euclidean distances, negated for TSP."""
    p, n, _ = coords.shape
    dist = np.zeros((p, n, n))
    for q in range(p):
        diff = coords[q, :, None, :] - coords[q, None, :, :]
        dist[q] = np.floor(np.sqrt(np.sum(diff ** 2, axis=-1)) + 0.5)
    return -dist


def scenario_instance(problem: Problem, n: int, trial: int, scenario: str, base_seed: int) -> Instance:
    """Instance of size n from a distribution."""
    scenarios = _scenarios(problem)
    scen = scenarios[scenario]
    s_idx = list(scenarios).index(scenario)
    p = problem.p
    seq = np.random.SeedSequence([int(base_seed), int(p), int(n), int(trial), int(s_idx)])
    instance_id = 90_000_000 + p * 1_000_000 + s_idx * 10_000 + int(trial)
    if problem.kind == "knapsack":
        profit_seed, weight_seed = seq.spawn(2)
        C = _draw_values(np.random.default_rng(profit_seed), scen, (p, n))
        weights = _draw_values(np.random.default_rng(weight_seed), scen, n)
        return Instance("knapsack", n, p, instance_id, scenario,
                        C=C, weights=weights, capacity=int(np.floor(0.5 * weights.sum())))
    coords = _draw_values(np.random.default_rng(seq), scen, (p, n, 2))
    return Instance("tsp", n, p, instance_id, scenario, coords=coords, dist=_distances(coords))


def scenario_names(problem: Problem) -> list:
    """id and the five shifted value distributions."""
    return list(_scenarios(problem))


def all_scenarios(problem: Problem) -> list:
    """id, the five shifted value distributions and the two clustered placements (refer clustered.py)."""
    return [*scenario_names(problem), *cl.SCENARIOS]


def make_instance(problem: Problem, n: int, trial: int, scenario: str, base_seed: int) -> Instance:
    """Instance of size n of any scenario of all_scenarios().
    The same seeds give the same instances in every script and in the DRL
    evaluation (ood_eval_v6.py, drl_eval_clustered.py).
    """
    if scenario in cl.SCENARIOS:
        data = cl.generate(problem.kind, problem.p, n, trial, scenario, base_seed)
        return Instance(problem.kind, n, problem.p, cl.instance_id(problem.p, scenario, trial), scenario, **data)
    return scenario_instance(problem, n, trial, scenario, base_seed)


# =============================================================================
# GROUND TRUTH
# =============================================================================

@dataclass
class Truth:
    points: list                 # true ESN points (tuples)
    n_solves: int                # solves of the exact algorithm
    time: float                  # wall time of the exact algorithm (incl. final filtering)
    hv_ref: tuple                # hypervolume reference point (min of each objective - 1)
    hv: float                    # hypervolume of the full ESN set
    complete: bool


def exact_truth(problem: Problem, inst: Instance) -> Truth:
    res = run_exact(problem, inst, time_limit=EXACT_TIME_LIMIT)
    hv_ref = hv_reference(res.points, problem.p)
    hv = hypervolume(res.points, hv_ref, problem.p) if res.points else 0.0
    return Truth(res.points, res.n_solves, res.total_time, hv_ref, hv, res.stopped_by == "complete")

# =============================================================================
# THE TARGET / BUDGET Q (shared by eval_fixed_target.py and eval_fixed_budget.py)
# =============================================================================

TARGET_FRACTION = 0.75
TARGET_INSTANCES = 10           # instances per size and scenario in the fixed-target race


def target_q(mean_m: float) -> int:
    """Q = max(2, floor(0.75 * mean |Y_E|))."""
    return max(2, math.floor(TARGET_FRACTION * float(mean_m) + 1e-9))


def fixed_target_q(problem: Problem, n: int, scenario: str, n_instances: int = TARGET_INSTANCES) -> int:
    """The Q of eval_fixed_target.py for (n, scenario), from the race's own instances."""
    truths = [exact_truth(problem, make_instance(problem, n, t, scenario, TARGET_SEED)) for t in range(n_instances)]
    return target_q(np.mean([len(t.points) for t in truths]))


# This was just for testing, not needed anymore.
def reference_v8(problem: Problem, inst: Instance) -> dict:
    res = run_exact(problem, inst, time_limit=EXACT_TIME_LIMIT, exa_rule2=True)
    return {"n_esn": len(res.points), "n_solves": res.n_solves, "time": res.total_time, "points": res.points}
    
# =============================================================================
# MATCHING AGAINST THE TRUTH
# =============================================================================

def matched_indices(points, truth_points) -> set:
    """Indices of the truth points that appear in `points`."""
    keys = {point_key(z): i for i, z in enumerate(truth_points)}
    hit = set()
    for z in points:
        i = keys.get(point_key(z))
        if i is None:  # fall back to the tolerance rule (identical for integer data)
            i = find_index(tuple(z), truth_points)
        if i is not None:
            hit.add(i)
    return hit


def hv_fraction(points, truth: Truth) -> float:
    """Hypervolume of points as a fraction of the hypervolume of the full ESN set."""
    if not points or truth.hv <= 0:
        return 0.0
    pts = [tuple(float(v) for v in z) for z in points]
    return hypervolume(pts, truth.hv_ref, len(pts[0])) / truth.hv
    
# =============================================================================
# SOLVING A FIXED LIST OF WEIGHTS (uniform and bisection heuristics)
# =============================================================================

@dataclass
class WeightRun:
    returned: list               # point returned by every solve
    solves: int
    time: float                  # wall time of the solves
    solver_time: float
    check_time: float            # time spent inside on_solve, to subtract


def solve_weights(problem: Problem, inst: Instance, weights,
                  on_solve: Optional[Callable] = None) -> WeightRun:
    """Solve every weight once, in order. on_solve(z) -> True stops early."""
    solver = Solver(problem, inst, EXACT_TIME_LIMIT)
    returned, check = [], 0.0
    t0 = time.perf_counter()
    try:
        for lam in weights:
            z = solver.solve(lam)
            returned.append(z)
            if on_solve is not None:
                c0 = time.perf_counter()
                stop = on_solve(z)
                check += time.perf_counter() - c0
                if stop:
                    break
        wall = time.perf_counter() - t0
        return WeightRun(returned, solver.n_solves, wall, solver.solver_time, check)
    finally:
        solver.close()


def output_set(returned: list) -> list:
    """nondominated extreme points among its results."""
    return final_points([z for z in returned if z is not None])


# =============================================================================
# HEURISTIC WEIGHTS: uniform design and bisection order
# =============================================================================

WEIGHT_EPS = 1e-6
MAX_BISECTION_WEIGHTS = 2500    # bisection stops at the finest level within this many weights


def simplex_lattice(p: int, R: int) -> np.ndarray:
    """All points (i_1, ..., i_p)/R with i >= 0 and sum = R."""
    pts = []

    def rec(remaining, depth, current):
        if depth == p - 1:
            pts.append(current + [remaining])
            return
        for i in range(remaining + 1):
            rec(remaining - i, depth + 1, current + [i])

    rec(int(R), 0, [])
    return np.asarray(pts, dtype=np.float64) / float(R)


def positive_weights(W) -> np.ndarray:
    W = np.clip(np.atleast_2d(np.asarray(W, dtype=np.float64)), WEIGHT_EPS, None)
    return W / W.sum(axis=1, keepdims=True)


def farthest_point_order(C: np.ndarray, selected: Optional[np.ndarray] = None,
                         limit: Optional[int] = None) -> list:
    """Greedy max-min order of the rows of C: next = the row farthest from everything chosen (ties: earliest)."""
    if len(C) == 0:
        return []
    limit = len(C) if limit is None else max(0, min(int(limit), len(C)))
    if selected is not None and len(selected):
        dmin = ((C[:, None, :] - selected[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    else:
        dmin = np.full(len(C), np.inf)
    alive = np.ones(len(C), dtype=bool)
    order = []
    for _ in range(limit):
        k = int(np.argmax(np.where(alive, dmin, -1.0)))
        order.append(k)
        alive[k] = False
        dmin = np.minimum(dmin, ((C - C[k]) ** 2).sum(axis=1))
    return order


def _lattice_size(p: int, R: int) -> int:
    return math.comb(R + p - 1, p - 1)


def uniform_weights(p: int, S: int) -> np.ndarray:
    """S weights spread uniformly over the simplex.
    The p extreme weights first, then the smallest regular simplex lattice with
    at least S points, thinned by farthest-point sampling.
    """
    S = int(S)
    if S <= 0:
        return np.zeros((0, p))
    vertices = np.eye(p)
    if S <= p:
        return positive_weights(vertices[:S])
    R = 1
    while _lattice_size(p, R) < S:
        R += 1
    lattice = simplex_lattice(p, R)
    rest = lattice[lattice.max(axis=1) < 1.0]
    order = farthest_point_order(rest, vertices, limit=S - p)
    return positive_weights(np.vstack([vertices, rest[order]]))


def bisection_weights(p: int, max_weights: int = MAX_BISECTION_WEIGHTS) -> np.ndarray:
    """Weights in hierarchical bisection order (extremes, 1/2, 1/4 and 3/4, ...).
    Level R = 1, 2, 4, 8, ... adds the lattice points with denominator R that
    are new, in farthest-point order; it stops at the last level whose lattice
    has at most max_weights points.
    """
    order = np.zeros((0, p))
    R = 1
    while _lattice_size(p, R) <= max_weights:
        lattice = simplex_lattice(p, R)
        seen = {tuple(np.round(w * R).astype(int)) for w in order}
        new = np.asarray([w for w in lattice if tuple(np.round(w * R).astype(int)) not in seen]).reshape(-1, p)
        order = np.vstack([order, new[farthest_point_order(new, order)]])
        R *= 2
    return positive_weights(order)


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def pooled(df, num: str, den: str, scale: float = 1.0) -> float:
    """sum(num) / sum(den) over the instances in df (how every table pools); NaN if not measured."""
    num_sum = float(df[num].sum(min_count=1))
    den_sum = float(df[den].sum(min_count=1))
    if np.isnan(num_sum) or np.isnan(den_sum) or den_sum <= 0:
        return float("nan")
    return scale * num_sum / den_sum


# =============================================================================
# SEARCH-BASED METHODS
# =============================================================================
# E    CAN / ExA in its own (depth-first) order              (exa.ExactSearch)
# SSA  SSAv3 facet search, scorer order (tau = 0 in the target / budget races)
#                                                             (ssa.FacetSearch)
# A    largest-area heuristic: the same facet search, but the "score" is the
#      size of the facet (triangle area for p=3, segment length for p=2), i.e.
#      always the largest open facet next; it never stops on its score
#      (tau = -inf). SSA and A differ only in which open facet is solved next.

SIZE_FEATURE = {2: "log_length", 3: "log_area"}     # monotone in the segment length / triangle area


class AreaScorer:
    """Score = log10 of the facet's size (a feature of exa.stage_features): largest facet first."""

    def __init__(self, p: int):
        from ssa import FEATURE_NAMES
        self.j = FEATURE_NAMES[p].index(SIZE_FEATURE[p])

    def __call__(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.j]


def make_search(method: str, problem: Problem, inst: Instance, scorer, **kwargs):
    """The search object of E, SSA (tau = 0) or A; kwargs: max_solves, on_solve, record, ..."""
    from ssa import FacetSearch
    if method == "E":
        return ExactSearch(problem, inst, order="dfs", **kwargs)
    if method == "SSA":
        return FacetSearch(problem, inst, order="score", scorer=scorer, tau=0.0, **kwargs)
    if method == "A":
        return FacetSearch(problem, inst, order="score", scorer=AreaScorer(problem.p), tau=float("-inf"),
                           **kwargs)
    raise ValueError(method)


def stop_at_target(tracker: "RaceTracker", Q: int):
    """on_solve callback: record the returned point, stop once Q reference points are found."""
    def on_solve(z):
        tracker.observe(None if z is None else tuple(z))
        return len(tracker.matched) >= Q
    return on_solve


def observe_only(tracker: "RaceTracker"):
    """on_solve callback: record the returned point, never stop."""
    def on_solve(z):
        tracker.observe(None if z is None else tuple(z))
        return False
    return on_solve


# =============================================================================
# LATEX HELPERS (knapsack and TSP side by side)
# =============================================================================

SCENARIO_LABEL = {
    # table order: training distribution, knapsack from the narrowest to the widest values, then RC and C
    "knapsack": {"id": r"$\mathcal{U}[60,100]$ (ID)",
                 "gaussian_80_8_range_60_100": r"$\mathcal{N}(80,8^2)_{[60,100]}$",
                 "uniform_80_120": r"$\mathcal{U}[80,120]$",
                 "gaussian_80_15_range_40_120": r"$\mathcal{N}(80,15^2)_{[40,120]}$",
                 "uniform_40_80": r"$\mathcal{U}[40,80]$", "uniform_40_120": r"$\mathcal{U}[40,120]$",
                 "random_clustered": "Random-clustered (RC)", "clustered": "Clustered (C)"},
    "tsp": {"id": r"$\mathcal{U}[0,999]$ (ID)", "coords_0_499": r"$\mathcal{U}[0,499]$",
            "coords_0_1499": r"$\mathcal{U}[0,1499]$", "coords_250_749": r"$\mathcal{U}[250,749]$",
            "gaussian_500_150_0_999": r"$\mathcal{N}(500,150^2)_{[0,999]}$",
            "gaussian_500_300_0_1499": r"$\mathcal{N}(500,300^2)_{[0,1499]}$",
            "random_clustered": "Random-clustered (RC)", "clustered": "Clustered (C)"},
}
SCENARIO_SHIFT = {   # what the scenario changes with respect to the training distribution
    "knapsack": {"id": "Training", "gaussian_80_8_range_60_100": "Narrower", "uniform_80_120": "Shifted up",
                 "gaussian_80_15_range_40_120": "Wider", "uniform_40_80": "Shifted down",
                 "uniform_40_120": r"Wider ($\times2$)", "clustered": "Clustered", "random_clustered": "Half clustered"},
    "tsp": {"id": "Training", "coords_0_499": r"Scaled $\times\frac12$", "coords_0_1499": r"Scaled $\times1.5$",
            "coords_250_749": r"Centre, $\times\frac12$", "gaussian_500_150_0_999": "Gaussian",
            "gaussian_500_300_0_1499": r"Gaussian, $\times1.5$", "clustered": "Clustered",
            "random_clustered": "Half clustered"},
}


def fmt(x, d: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"

HV_GAP_DECIMALS = 2
HV_GAP_DEF = (r"HV gap $=100\,(1-\mathrm{HV}/\mathrm{HV}^*)$, HV$^*$ the hypervolume of the full ESN set " r"(\%, lower is better)")


def hv_gap(fraction) -> float:
    """HV gap in %: 100 (1 - HV / HV of the full ESN set); fraction = HV / HV of the full ESN set."""
    return float("nan") if fraction is None or (isinstance(fraction, float) and math.isnan(fraction)) \
        else 100.0 * (1.0 - float(fraction))
        
def bold_best(cells: list, best: str = "max") -> list:
    """Bold every cell (formatted number strings) equal to the best value; '--' cells are ignored."""
    vals = [float(c) for c in cells if c != "--"]
    if not vals:
        return cells
    b = max(vals) if best == "max" else min(vals)
    return [rf"\textbf{{{c}}}" if c != "--" and float(c) == b else c for c in cells]


def side_by_side(left: list, right: list, ncols: int) -> list:
    """Pair the rows of the knapsack (left) and TSP (right) halves of a table."""
    out = []
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else [""] * ncols
        b = right[i] if i < len(right) else [""] * ncols
        out.append(" & ".join(a + b) + r" \\")
    return out
