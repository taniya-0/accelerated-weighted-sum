"""
exa.py -- the exact benchmark: CAN (p=2) and ExA (p=3)
=====================================================
The CAN / ExA engine of SSAv2 (SSA/ssa.py)
"""
from __future__ import annotations

import heapq
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from problems import Instance, Problem, Solver, exa_dummies, gurobi_env, z_max

HERE = Path(__file__).resolve().parent #The folder containing this exa.py.

try:   # the LAPACK solver behind np.linalg.solve, called directly (no wrapper overhead)
    from numpy.linalg import _umath_linalg
    _LAPACK_SOLVE = _umath_linalg.solve
except (ImportError, AttributeError):
    _LAPACK_SOLVE = None
    
# a singular stage matrix gives NaNs (-> "no unique plane"); do not warn about it
"""
k is the number of triangles solved in one call.
ExA: a new point found inside a triangle splits it into 3 new triangles, so k = 3
SSA: a new point added to the hull creates one new facet for each hull edge around it. That can be 3, 5, 8 or more facets, so k varies
The function is handed a batch of triangles at once, not just one, because new facets always arrive in groups
A simple version builds one 3×3 matrix per triangle and calls np.linalg.solve k times in a Python loop. The LAPACK version stacks all k matrices into one array and makes one call.
Examplewith k=2:
A.shape = (2, 3, 3)          <- 2 matrices, each 3x3
A[0] = [[ 135. -128.  -24.]      a-b
        [  -4.  128.  -63.]      b-c
        [   1.    1.    1.]]     sum = 1
A[1] = [[   0.    0.    0.]      a-a  -> a row of zeros: singular
        [ 135. -128.  -24.]      a-b
        [   1.    1.    1.]]
_RHS[2].shape = (2, 3, 1)    <- one right-hand side (0, 0, 1) per matrix # Solves lam_vec.(A-B) = lam_vec.(B-C) = lam_vec.(C-A) = 0 given that lam1 + lam2 + lam3 = 1
"""
warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__) #
_RHS = {k: np.zeros((k, 3, 1)) for k in (1, 2, 3)}      # right-hand sides (0, 0, 1) of the lambda systems
for _b in _RHS.values():
    _b[:, 2, 0] = 1.0


# =============================================================================
# SETTINGS
# =============================================================================

class Config:
    # --- tolerances copied from the reference CAN / ExA code ---
    LAM_TOL = 1e-9          # a weight vector is valid only if every component > LAM_TOL
    Z_TOL = 1e-6            # ExA: two weighted-sum values are equal if closer than this
    POINT_TOL = 0.5         # two objective vectors are the same point (integer data)
    CAN_EPS = 1e-6          # CAN: the two extreme weights are (1-eps, eps) and (eps, 1-eps)
    TIME_LIMIT = 5400.0     # seconds, same as the benchmarks

    EXA_RULE2 = False       #HERE1 (Always keep this false)

    # --- data / calibration ---
    COLLECT_INSTANCES = 20          # training instances per size
    CALIB_INSTANCES = 8             # validation instances per size
    TARGET_RECALL = 0.8             # tau keeps the 10th-percentile recall >= this
    TAU_GRID = [round(t, 2) for t in np.arange(0.0, 0.99, 0.02)] # Tries tau values 0.0, 0.02, 0.04 so on

    # --- scorer network and training ---
    HIDDEN = 64
    EPOCHS = 60
    LR = 2e-3
    WEIGHT_DECAY = 1e-5
    BATCH = 512
    SEED = 20261004

    # --- files (inside SSA/runs/<problem>/) ---
    DATA_FILE = "data.npz"
    SCORER_FILE = "scorer.npz"
    CALIB_FILE = "calibration.json"


def run_dir(problem: Problem) -> Path:
    d = HERE / "runs" / problem.key
    d.mkdir(parents=True, exist_ok=True)
    return d
    
# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def point_key(z) -> tuple:
    return tuple(np.round(np.asarray(z, dtype=np.float64), 6).tolist()) # round to 6 decimals, return a tuple of Python floats


def same_point(a, b) -> bool:
    return all(abs(x - y) < Config.POINT_TOL for x, y in zip(a, b)) # True if every coordinate differs by < Config.POINT_TOL, i.e they are different

# Solves lam_vec.(A-B) = lam_vec.(B-C) = lam_vec.(C-A) = 0 given that lam1 + lam2 + lam3 = 1
def exa_lambda_rows(corner_sets: list) -> np.ndarray:
    """ExA Step 3 (p=3) for several stages at once: solve lambda.(r_1 - r_2) = 0, lambda.(r_2 - r_3) = 0,
    sum(lambda) = 1, the linear system of _compute_lambda in v8.py / tsp_exa_v2.py.
    corner_sets: list of 3-tuples of corners, each corner a tuple of 3 Python floats. Returns a (k, 3) array with one raw solution per stage. The systems
    are solved together by LAPACK.
    """
    k = len(corner_sets)
    rows = []
    for (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) in corner_sets:
        rows.append(((a0 - b0, a1 - b1, a2 - b2), (b0 - c0, b1 - c1, b2 - c2), (1.0, 1.0, 1.0)))
    A = np.array(rows)
    if _LAPACK_SOLVE is not None:
        if k not in _RHS:                                # SSA can split into more than 3 facets at once
            _RHS[k] = np.zeros((k, 3, 1))
            _RHS[k][:, 2, 0] = 1.0
        #_LAPACK_SOLVE(A, _RHS[2], signature="dd->d") returns shape (2, 3, 1). [:, :, 0] drops the last axis of length 1 to give (2, 3) which is the 3 lambda components of 2 triangles.
        return _LAPACK_SOLVE(A, _RHS[k], signature="dd->d")[:, :, 0]  #signature="dd->d" for “double-precision inputs, double-precision output”.
        
    # This below is a fallback. SImilar to the simplified exa_lambda_rows commented out below.
    X = np.full((k, 3), np.nan)                         
    for i in range(k):
        try:
            X[i] = np.linalg.solve(A[i], _RHS[1][0, :, 0]) 
        except np.linalg.LinAlgError:
            pass
    return X
    
# def exa_lambda_rows(corner_sets: list) -> np.ndarray:
    # X = np.full((len(corner_sets), 3), np.nan)
    # for i, (a, b, c) in enumerate(corner_sets):
        # A = np.array([np.subtract(a, b), np.subtract(b, c), [1.0, 1.0, 1.0]])
        # try:
            # X[i] = np.linalg.solve(A, np.array([0.0, 0.0, 1.0]))
        # except np.linalg.LinAlgError:
            # pass
    # return X
    
def can_lambda(yL: np.ndarray, yU: np.ndarray) -> Optional[np.ndarray]:
    """CAN: the weight orthogonal to the segment yL-yU (None if degenerate)"""
    if yL[0] < yU[0]:
        l1, l2 = yL[1] - yU[1], yU[0] - yL[0]
    else:
        l1, l2 = yU[1] - yL[1], yL[0] - yU[0]
    if l1 <= 0 or l2 <= 0:
        return None
    return np.array([l1 / (l1 + l2), l2 / (l1 + l2)])


def can_between(z, yL, yU) -> bool:
    """CAN: is z strictly between the two segment end points (0.5 tolerance)?"""
    if yL[0] < yU[0]:
        return (z[0] > yL[0] + 0.5 and z[0] < yU[0] - 0.5 and
                z[1] < yL[1] - 0.5 and z[1] > yU[1] + 0.5)
    return (z[0] < yL[0] - 0.5 and z[0] > yU[0] + 0.5 and
            z[1] > yL[1] + 0.5 and z[1] < yU[1] - 0.5)
            
            
# order matters: deduplicate, then remove dominated points, then remove non-extreme ones.

def filter_dominated(points: list) -> list:
    """Point a is dropped if some other point b has b ≥ a in every objective and b > a in at least one (we maximise)."""
    if not points:
        return []
    Z = np.vstack(points)
    keep = []
    for i, a in enumerate(points):
        weakly_better = (Z >= Z[i]).all(axis=1)             # b >= a everywhere
        strictly_somewhere = (Z > Z[i]).any(axis=1)         # b > a somewhere
        same = (np.abs(Z[i] - Z) < Config.POINT_TOL).all(axis=1)
        weakly_better[i] = False                            # a does not dominate itself
        if not np.any(weakly_better & strictly_somewhere & ~same):
            keep.append(a)
    return keep

"""
Let points be {P0 = (10,0,0), P1 = (0,10,0), P2 = (0,0,10), P3 = (4,3,3)}, so m = 4 points and p = 3
Z = np.vstack(points). m_pts = 4 (number of points) and dim = 3 (number of objectives).
Z = [[10,  0,  0],    # P0
     [ 0, 10,  0],    # P1
     [ 0,  0, 10],    # P2
     [ 4,  3,  3]]    # P3
a = m.addVars(m_pts, lb=0.0, ub=1.0) -> Four variables a[0] … a[3], each between 0 and 1. a[k] is the weight on point k
rows = [m.addConstr(gp.quicksum(a[k] * float(Z[k, d]) for k in range(m_pts)) == 0.0) for d in range(dim)] does
d=0:  10·a0 +  0·a1 +  0·a2 + 4·a3 = 0      (RHS 0 = placeholder)
d=1:   0·a0 + 10·a1 +  0·a2 + 3·a3 = 0
d=2:   0·a0 +  0·a1 + 10·a2 + 3·a3 = 0

"""

def filter_nonextreme(points: list, env: Optional[gp.Env] = None) -> list:
    """Keep only extreme points (vertices of the convex hull of the set).
    p=2: upper convex hull of the front, exact arithmetic on integer points. By checking cross product.
    p=3: drop z if it is a convex combination of the other points. Say for k known points, For each of teh q objective if we do a_k*z_k and add them up for all k points,
    will it be equal to z_i's qth objective? a_k remainins same across objectives and sum of a_k over all k = 1. This is what the LP mdoel solves.
    """
    if len(points) <= 2:
        return list(points)
    if len(points[0]) == 2:
        pts = sorted(points, key=lambda z: (z[0], -z[1]))   # z1 ascending
        hull = []
        for z in pts:
            # pop the last hull point while it is not strictly above the line hull[-2] -> z
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                cross = (b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0])
                if cross >= 0:
                    hull.pop()
                else:
                    break
            hull.append(z)
        keys = {point_key(z) for z in hull}
        return [z for z in points if point_key(z) in keys]
    own_env = env is None
    env = gurobi_env() if own_env else env
    try:
        Z = np.vstack(points)
        m_pts, dim = Z.shape
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0
        a = m.addVars(m_pts, lb=0.0, ub=1.0)
        m.addConstr(gp.quicksum(a[k] for k in range(m_pts)) == 1)
        rows = [m.addConstr(gp.quicksum(a[k] * float(Z[k, d]) for k in range(m_pts)) == 0.0) for d in range(dim)]
        m.setObjective(0, GRB.MINIMIZE)                  # We ask whether a feasible a exists.
        keep = []                                        
        for i, z in enumerate(points):                   # Test each point in turn: i = 0, 1, 2, 3.
            a[i].UB = 0.0                                # only the OTHER points may be combined so Force a_i = 0
            for d in range(dim):
                rows[d].RHS = float(Z[i, d])             # Set the right-hand sides to z_i.
                """
                10·a0 +  0·a1 +  0·a2 + 4·a3 = 4
                0·a0 + 10·a1 +  0·a2 + 3·a3 = 3
                0·a0 +  0·a1 + 10·a2 + 3·a3 = 3
                a0 + a1 + a2 + a3 = 1,   a3 = 0 (UB), all >= 0
                """
            m.reset(1)
            m.optimize()
            if m.Status != GRB.OPTIMAL:                  # infeasible -> not a convex combination -> extreme
                keep.append(z)
            a[i].UB = 1.0
        m.dispose()
        return keep
    finally:
        if own_env:
            env.dispose()


def final_points(points: list, env: Optional[gp.Env] = None) -> list:
    """The output set of a search: distinct, nondominated and extreme points, as tuples."""
    unique = {}
    for z in points:                                     
        unique.setdefault(point_key(z), np.asarray(z, dtype=np.float64))
    pts = filter_nonextreme(filter_dominated(list(unique.values())), env)
    return [tuple(float(v) for v in z) for z in pts]
    
# =============================================================================
# STAGE FEATURES
# =============================================================================
# All lengths are divided by S = z_max(instance) so that features are comparable between sizes. 

FEATURE_NAMES = {
    2: [
        "log_min_lam",   # log10 of the smallest weight: ~0 = balanced, very negative = near an axis
        "log_length",    # log10 of the Euclidean segment length / S
        "spread",        # largest coordinate difference of the two end points / S
        "log_room",      # log10 of the area of the triangle in which a new point could still lie
        "depth",         # how many splits produced this segment (search depth) / 50
        "parent_gain",   # how far beyond its stage the point that created this segment was
        "n_size",        # n / 200 (knapsack) or n / 30 (TSP)
    ],
    3: [
        "n_dummy",       # share of the three corners that are ExA dummy points
        "log_min_lam",   # log10 of the smallest weight
        "log_area",      # log10 of the triangle area / S^2
        "real_spread",   # largest coordinate difference between REAL corners / S
        "log_room",      # log10 of (lambda.u - lambda.r1)/S, u = component-wise max of the corners
        "depth",         # generation of the stage / 50
        "parent_gain",   # how far beyond its stage the point that created this stage was
        "n_size",        # n / 200 (knapsack) or n / 30 (TSP)
    ],
}


def signed_log(x: float) -> float:
    """sign(x) * log10(1 + |x|): keeps the sign, squashes huge values (dummy stages)."""
    return math.copysign(math.log10(1.0 + abs(x)), x)


def stage_features(stages: list, S: float, n_size: float) -> np.ndarray:
    rows = []
    for st in stages:
        lam = st.lam.tolist()
        corners = st.coords if st.coords is not None else [z.tolist() for z in st.points]
        P = [[v / S for v in z] for z in corners]          # corners in units of S
        log_min_lam = math.log10(max(min(lam), 1e-12)) / 12.0
        depth = min(st.depth, 50) / 50.0
        parent_gain = signed_log(st.gain()) / 6.0
        if len(lam) == 2:
            d0, d1 = abs(P[0][0] - P[1][0]), abs(P[0][1] - P[1][1])
            rows.append((
                log_min_lam,
                math.log10(math.sqrt(d0 * d0 + d1 * d1) + 1e-12) / 12.0,   # log_length
                max(d0, d1),                                               # spread
                # a new supported point between the two end points must lie in the triangle (end point, end point, their component-wise maximum)
                math.log10(0.5 * (d0 * d1) + 1e-30) / 10.0,                # log_room
                depth, parent_gain, n_size))
            continue
        (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = P
        e0, e1, e2 = b0 - a0, b1 - a1, b2 - a2                  # edge corner 1 -> 2
        f0, f1, f2 = c0 - a0, c1 - a1, c2 - a2                  # edge corner 1 -> 3
        x, y, z = e1 * f2 - e2 * f1, e2 * f0 - e0 * f2, e0 * f1 - e1 * f0   # cross product
        area = 0.5 * math.sqrt(x * x + y * y + z * z)
        real = [p for p, dummy in zip(P, st.is_dummy) if not dummy]
        real_spread = 0.0
        if len(real) >= 2:                                      # largest coordinate difference of real corners
            real_spread = max(max(abs(u - v) for u, v in zip(real[i], real[j])) for i in range(len(real)) for j in range(i + 1, len(real)))
        l0, l1, l2 = lam
        u0, u1, u2 = max(a0, b0, c0), max(a1, b1, c1), max(a2, b2, c2)   # component-wise best corner
        room = (l0 * u0 + l1 * u1 + l2 * u2) - (l0 * a0 + l1 * a1 + l2 * a2)  # height of u above the plane
        rows.append((
            sum(st.is_dummy) / 3.0,                             # n_dummy
            log_min_lam,
            math.log10(area + 1e-30) / 10.0,                    # log_area
            real_spread,
            math.log10(max(room, 0.0) + 1e-12) / 12.0,          # log_room
            depth, parent_gain, n_size))
    return np.array(rows, dtype=np.float32)
    
# =============================================================================
# THE SCORER (a 2-hidden-layer MLP)
# =============================================================================
class Scorer:
    """score(X) = sigmoid(MLP((X - mu) / sd)), X = (n_stages, n_features).
    Layer 0	8 features to 64 hidden	W0 (64, 8)	b0 (64,)
    Layer 1	64 to 64 hidden	W1 (64, 64)	b1 (64,)
    Layer 2	64 to 1 logit	W2 (1, 64)	b2 (1,)
    h (input)	(k, 8)	k facets, 8 features each
    self.W[0]	(64, 8)	one row of 8 weights per neuron (torch stores it this way)
    self.W[0].T	(8, 64)	transposed so the matrix product fits
    h @ W0.T	(k, 64)	64 weighted sums for each facet
    + self.b[0]	(64,)	added to every row (numpy broadcasting
    """
    def __init__(self, params: dict):
        self.mu, self.sd = params["mu"], params["sd"]
        self.W = [params[f"W{i}"] for i in range(3)]
        self.b = [params[f"b{i}"] for i in range(3)]
        self.feature_names = [str(s) for s in params["feature_names"]]

    @classmethod
    def load(cls, problem: Problem) -> "Scorer":
        path = run_dir(problem) / Config.SCORER_FILE
        params = dict(np.load(path, allow_pickle=False))
        scorer = cls(params)
        if scorer.feature_names != FEATURE_NAMES[problem.p]:
            raise ValueError(f"{path} was trained with other features; retrain it")
        return scorer

    def __call__(self, X: np.ndarray) -> np.ndarray:
        h = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        h = np.maximum(h @ self.W[0].T + self.b[0], 0.0)      # Linear + ReLU
        h = np.maximum(h @ self.W[1].T + self.b[1], 0.0)      # Linear + ReLU
        logit = (h @ self.W[2].T + self.b[2])[:, 0]           # Linear -> 1 logit
        return 1.0 / (1.0 + np.exp(-logit))                   # probability


def load_tau(problem: Problem) -> float:
    with open(run_dir(problem) / Config.CALIB_FILE) as f:
        return float(json.load(f)["tau"])
        
# =============================================================================
# THE SEARCH (CAN for p=2, ExA for p=3)
# =============================================================================
"""
1. self.keys is a tuple with the point_key of each corner, in corner order. It is computed once, when the stage is created. For the facet (a, b, r*):
keys = ((2359.0, 2276.0, 2217.0),     # point_key(a)
        (2224.0, 2404.0, 2241.0),     # point_key(b)
        (2305.0, 2330.0, 2284.0))     # point_key(r*)
self.key is the identity of the whole stage: the same three keys as a frozenset. key = frozenset({(2359.0, 2276.0, 2217.0), (2224.0, 2404.0, 2241.0), (2305.0, 2330.0, 2284.0)})
2. Precalculation asks: “among all known real points, does any of them lie beyond teh stage in question. each stage has its own pre_m and pre_best
pre_m    = how many known points there were when the answer was computed (-1 = never)
pre_best = the answer: index of the best known point
if R.pre_m != known.m:                               # points were added since -> recompute
    R.pre_m, R.pre_best = known.m, known.best(R.lam)
i_best = R.pre_best                                  # otherwise reuse
Example:
At creation there are 10 known points. best returns index 4, so pre_m = 10 and pre_best = 4.
At pop time, if there are still 10 points, the cached 4 is reused and no matrix product is needed.
If a new point was found in between (11 points), it is recomputed.
"""

class Stage:
    """One open stage"""
    __slots__ = ("points", "coords", "keys", "is_dummy", "key", "depth", "parent_gain", "lam", "feats","score", "pre_m", "pre_best") #A fixed list of attributes

    def __init__(self, points, keys, is_dummy, depth, parent_gain, lam, coords=None, key=None):
        self.points = points            # the p corner points (np arrays)
        self.coords = coords            # the same corners as tuples of Python floats (ExA; faster)
        self.keys = keys                # point_key of every corner (computed once) (a,b,r*) != (r*,a,b), look at individual corners
        self.is_dummy = is_dummy        # which corners are ExA dummies (always False for CAN)
        self.key = frozenset(keys) if key is None else key   # the stage identity = set of corner keys (a,b,r*) = (r*,a,b), answers if teh facet ahs been seen before
        self.depth = depth              # 0 for the first stage, parent depth + 1 otherwise
        self.parent_gain = parent_gain
        self.lam = lam                  # normalised weight vector, None if not valid
        self.feats = None               # features, computed once when first needed
        self.score = math.inf           # inf = "free" stage (needs no solve), popped first
        self.pre_m = -1                 # precalculation cache: number of known points it was computed with
        self.pre_best = None            # ... and its result (index of the best known point)

    def gain(self) -> float:
        """How far beyond its parent stage the point that created this stage was (feature parent_gain)."""
        g = self.parent_gain
        if g.__class__ is tuple:
            lam, corner, r_star, S = g
            g = self.parent_gain = (float(lam @ r_star) - float(lam @ corner)) / S
        return g

@dataclass
class SolveRecord:
    """One Gurobi solve, as seen by the search."""
    step: int                      # 1, 2, 3, ... (solve counter)
    lam: np.ndarray                # weight vector that was solved
    z: Optional[np.ndarray]        # returned objective vector
    feats: Optional[np.ndarray]    # stage features (None for forced solves / dfs runs)
    score: float                   # scorer output (nan if no scorer)
    new_point: bool                # z was not known to the search before this solve
    forced: bool                   # CAN's two extreme solves: always made, never scored
    time: float                    # seconds since the search started


@dataclass
class SearchResult:
    points: list                   # output set: nondominated + extreme (tuples)
    found: list                    # every distinct point returned by a solve (np arrays)
    n_solves: int
    solver_time: float             # pure Gurobi time
    search_time: float             # wall time of the search loop
    total_time: float              # search_time + final filtering of the output set
    stopped_by: str                # complete | tau | budget | target | time_limit | solver_failed
    trace: list = field(default_factory=list)
    
class Worklist:
    """The list L of open stages and the rule that picks the next one.
    dfs:     last in, first out: exactly the reference CAN/ExA
    random:  uniformly random stage (training data collection)
    score:   highest scorer output first (SSA); ties -> most recent (like dfs)

    Score order is a max-heap on (score, push counter). Free stages have score = inf, so they always come first, the most recent one first. They are kept
    on a separate stack (LIFO) instead of the heap: same pop order, but most
    ExA stages are free and a stack is much cheaper than a heap.
    """

    def __init__(self, order: str, seed: int):
        if order not in ("dfs", "random", "score"):
            raise ValueError(order)
        self.order = order
        self.rng = np.random.default_rng(seed)
        self.items: list = []           # self.items is a plain Python list holding the open stages:
        self.free: list = []            # score order: stack of the free stages (score = inf)
        self.counter = 0                # records the order in which stages entered the worklist

    def push(self, stage: Stage):
        self.counter += 1
        if self.order == "score":
            if stage.score == math.inf: 
                self.free.append(stage)
            else:
                heapq.heappush(self.items, (-stage.score, -self.counter, stage)) #score	is a triplet of (−score, −counter, stage), arranged as a heap
        else:
            self.items.append(stage)

    def pop(self) -> Stage:
        if self.order == "score":
            if self.free:
                return self.free.pop()
            return heapq.heappop(self.items)[2]
        if self.order == "random":
            i = int(self.rng.integers(len(self.items)))
            self.items[i], self.items[-1] = self.items[-1], self.items[i] #swaps position i(random) and the last position. Then items.pop() removes the last entry, which is now the randomly chosen stage.
        return self.items.pop()

    def __len__(self):
        return len(self.items) + len(self.free)

    def __bool__(self):
        return bool(self.items) or bool(self.free)


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _KnownPoints:
    """ExA's real points found so far: a growing matrix (preallocated, doubled when full) and their keys."""
    # Z is a preallocated matrix with spare rows, and m counts how many rows are in use. Z contains the points. 
    def __init__(self, p: int):
        self.Z = np.empty((64, p))
        self.m = 0
        self.filled = self.Z[:0]        # view of the filled rows that is the real points 
        self.keys: list = []

    def add(self, z, k):
        if self.m == len(self.Z):
            self.Z = np.vstack([self.Z, np.empty_like(self.Z)])
        self.Z[self.m] = z
        self.m += 1
        self.filled = self.Z[:self.m]
        self.keys.append(k)

    def best(self, lam):
        """Index of the known point with the largest lambda-value (first one on ties), None if none."""
        if self.m == 0:
            return None
        return int((self.filled @ lam).argmax())
        
class ExactSearch:
    """CAN (p=2) or ExA (p=3) with a chosen stage order and optional early stop.

    order      "dfs" (reference), "random" (collect) or "score" (SSA)
    scorer     needed for order="score"
    tau        stop when the best stage that needs a solve scores below tau
    max_solves stop after this many solves (fixed-budget experiments)
    on_solve   Returns True once 4 distinct points are seen (fixed-target experiment)
    record     keep a SolveRecord per solve (with features) in result.trace
    exa_rule2  always False.
    """

    def __init__(self, problem: Problem, inst: Instance, *, order: str = "score",
                 scorer=None, tau: float = 0.0, seed: int = 0,
                 max_solves: Optional[int] = None,
                 on_solve: Optional[Callable[[Optional[np.ndarray]], bool]] = None,
                 record: bool = False, time_limit: float = Config.TIME_LIMIT,
                 exa_rule2: bool = Config.EXA_RULE2):
        if order == "score" and scorer is None:
            raise ValueError("order='score' needs a scorer")
        self.problem, self.inst = problem, inst
        self.p = problem.p
        self.exa_rule2 = bool(exa_rule2)
        self.order, self.scorer, self.tau = order, scorer, float(tau)
        self.max_solves, self.on_solve, self.record = max_solves, on_solve, record
        self.time_limit = float(time_limit)
        self.L = Worklist(order, seed)
        self.S = z_max(inst)                              # feature scale
        self.n_size = inst.n / problem.n_ref
        self.solver = Solver(problem, inst, time_limit)   
        self.trace: list = []
        self.found: list = []                             # distinct solve results
        self.found_keys: set = set()                      # # points the algorithm keeps as its ESN candidates
        self.accepted: list = []

    # ------------------------------------------------------------------ #
    # stage creation and scoring
    # ------------------------------------------------------------------ #
    def _make_can_stage(self, points, keys, depth, parent_gain) -> Stage:
        return Stage(points, keys, (False, False), depth, parent_gain, can_lambda(points[0], points[1]))

    def _features(self, stage: Stage) -> np.ndarray:
        if stage.feats is None:
            stage.feats = stage_features([stage], self.S, self.n_size)[0]
        return stage.feats

    def _score(self, stages: list):
        """Score the stages (all need a solve) with one feature pass and one scorer call."""
        X = stage_features(stages, self.S, self.n_size)
        for st, x, v in zip(stages, X, self.scorer(X).tolist()):
            st.feats, st.score = x, v

    def _push(self, stages: list, is_free: Optional[Callable[[Stage], bool]] = None):
        """Add new stages to L.

        Stages that are certainly free keep score = inf and are therefore processed first: stages with an invalid lambda, and
        (ExA) stages already beaten by a known point, which the precalculation resolves without a solve (this stays true later,
        because the set of known points only grows).
        """
        if self.order == "score":
            to_score = [s for s in stages if s.lam is not None and not (is_free is not None and is_free(s))]
            if to_score:
                self._score(to_score)
        for s in stages:
            self.L.push(s)

    # ------------------------------------------------------------------ #
    # one solve: stop checks, Gurobi, bookkeeping, callback
    # ------------------------------------------------------------------ #
    def _solve(self, stage: Optional[Stage], lam: np.ndarray, forced: bool = False):
        if not forced:
            if self.order == "score" and stage.score < self.tau:
                raise _Stop("tau")
            if self.max_solves is not None and self.solver.n_solves >= self.max_solves:
                raise _Stop("budget")
        if time.perf_counter() - self.t0 > self.time_limit:
            raise _Stop("time_limit")
        # the features of the stage are known BEFORE the solve (training rows)
        feats = self._features(stage) if (self.record and not forced) else None
        z = self.solver.solve(lam)
        k = None if z is None else point_key(z)
        new = k is not None and k not in self.found_keys
        if new:
            self.found.append(z)
            self.found_keys.add(k)
        if self.record:
            self.trace.append(SolveRecord(
                step=self.solver.n_solves, lam=np.asarray(lam, dtype=np.float64).copy(), z=z,
                feats=feats, score=float("nan") if (forced or self.order != "score") else stage.score,
                new_point=new, forced=forced, time=time.perf_counter() - self.t0))
        if z is None:
            raise _Stop("solver_failed")
        return z, k

    def _after_solve(self, z):
        """Ask the caller whether to stop (fixed-target and budget race) and check the budget."""
        if self.on_solve is not None and self.on_solve(z):
            raise _Stop("target")
        if self.max_solves is not None and self.solver.n_solves >= self.max_solves and len(self.L):
            raise _Stop("budget")

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def run(self, postprocess: bool = True) -> SearchResult:
        self.t0 = time.perf_counter()
        stopped_by = "complete"
        try:
            if self.p == 2:
                self._run_can()
            else:
                self._run_exa()
        except _Stop as stop:
            stopped_by = stop.reason
        search_time = time.perf_counter() - self.t0
        points = final_points(self.accepted, self.solver.env) if postprocess else []
        total_time = time.perf_counter() - self.t0
        result = SearchResult(points=points, found=list(self.found), n_solves=self.solver.n_solves,
                              solver_time=self.solver.solver_time, search_time=search_time,
                              total_time=total_time, stopped_by=stopped_by, trace=self.trace)
        self.solver.close()
        return result

    # ------------------------------------------------------------------ #
    # CAN (p=2), as dichotomic_search in the reference code
    # ------------------------------------------------------------------ #
    def _run_can(self):
        eps = Config.CAN_EPS
        # the two extreme points are always needed: forced solves, never scored or stopped
        y1, k1 = self._solve(None, np.array([1.0 - eps, eps]), forced=True)
        self.accepted.append(y1)
        self._after_solve(y1)
        y2, k2 = self._solve(None, np.array([eps, 1.0 - eps]), forced=True)
        if same_point(y1, y2):
            self._after_solve(y2)
            return                                        # a single nondominated point
        self.accepted.append(y2)
        self._after_solve(y2)
        esn = self.accepted                               # points CAN has accepted so far
        self._push([self._make_can_stage((y1, y2), (k1, k2), depth=1, parent_gain=0.0)])
        while len(self.L):
            seg = self.L.pop()
            if seg.lam is None:                           # degenerate segment: CAN skips it
                continue
            yL, yU = seg.points
            z, kz = self._solve(seg, seg.lam)
            if can_between(z, yL, yU) and not any(same_point(z, e) for e in esn):
                esn.append(z)
                gain = float(seg.lam @ z - seg.lam @ yL) / self.S
                # CAN pushes (yL, z) then (z, yU): dfs pops (z, yU) first
                self._push([self._make_can_stage((yL, z), (seg.keys[0], kz), seg.depth + 1, gain),
                            self._make_can_stage((z, yU), (kz, seg.keys[1]), seg.depth + 1, gain)])
            self._after_solve(z)

    # ------------------------------------------------------------------ #
    # ExA (p=3), as exa_algorithm in v8.py / tsp_exa_v2.py
    # ------------------------------------------------------------------ #
    def _run_exa(self):
        assert self.p == 3, "ExA is used for p=3 (CAN for p=2)"
        LAM_TOL, Z_TOL, INF = Config.LAM_TOL, Config.Z_TOL, math.inf
        dummies = exa_dummies(self.inst)
        dummy_keys = [point_key(d) for d in dummies]
        dummy_key_set = set(dummy_keys)
        dummy_coords = [tuple(d.tolist()) for d in dummies]
        y_em = list(zip(dummies, dummy_keys, dummy_coords))  # Y'_EM: dummies, then real points (array, key, coords)
        known = _KnownPoints(3)                               # real points found so far
        known_coords = []                                     # and their coordinates as Python floats
        esn_keys = set()
        visited = set()                 # V: stage keys already processed
        cf, nf = {}, {}                 # CF'/NF': pairs that do / do not share a facet
        score_order = self.order == "score"
        push, pop = self.L.push, self.L.pop

        def record_point(z, k):
            if k in esn_keys or k in dummy_key_set:
                return
            esn_keys.add(k)
            known.add(z, k)
            known_coords.append(tuple(z.tolist()))
            y_em.append((z.copy(), k, known_coords[-1]))
            self.accepted.append(z.copy())

        def precalc_point(R: Stage):
            """Precalculation: the known point with the best lambda-value, if it is not a corner of R."""
            if R.pre_m != known.m:
                R.pre_m, R.pre_best = known.m, known.best(R.lam)
            i_best = R.pre_best
            if i_best is None or known.keys[i_best] in R.key:
                return None
            return i_best

        def add_stages(cands: list, depth: int, gain: float):
            """Make the new stages (one batched lambda solve) and put them into L.
            cands: (points, coords, keys, is_dummy, key) per new stage.
            """
            X = exa_lambda_rows([c[1] for c in cands])
            normalised = X / X.sum(axis=1, keepdims=True)     # = row / row.sum() for every row
            stages = []
            for (pts, coords, keys, flags, key), lam, vals in zip(cands, normalised, X.tolist()):
                if not (LAM_TOL < vals[0] < INF and LAM_TOL < vals[1] < INF and LAM_TOL < vals[2] < INF):
                    lam = None                                # ExA Step 5 stage (free)
                stages.append(Stage(pts, keys, flags, depth, gain, lam, coords, key))
            if score_order:
                to_score = [st for st in stages if st.lam is not None and precalc_point(st) is None]
                if to_score:
                    self._score(to_score)
            for st in stages:
                push(st)

        def spawn(R: Stage, r_star, r_key, r_coords, gain):
            """Replace each corner of R by r_star.
            A new stage is skipped if r_star is already one of its corners or if
            the same triangle was already processed (in V).
            """
            k0, k1, k2 = R.keys
            cands = []
            for q, keys in ((0, (r_key, k1, k2)), (1, (k0, r_key, k2)), (2, (k0, k1, r_key))):
                key = frozenset(keys)
                if len(key) == 3 and key not in visited:
                    pts, coords, flags = list(R.points), list(R.coords), list(R.is_dummy)
                    pts[q], coords[q], flags[q] = r_star, r_coords, r_key in dummy_key_set
                    cands.append((tuple(pts), tuple(coords), keys, tuple(flags), key))
            if cands:
                add_stages(cands, R.depth + 1, gain)

        def step5_inject(R: Stage):
            """When lambda not positive, Step 5 (no solve): take the most recently found point that is not a corner of R, and spawn new stages with it"""
            for cand, k, c in reversed(y_em):
                if k not in R.key:
                    spawn(R, cand, k, c, 0.0)
                    return
        # CF′ records pairs of points known to share a facet, and NF′ records pairs that cannot. Rule 1 skips any stage containing an NF′ pair.
        def mark_cf(ka, kb):
            if ka != kb:
                cf.setdefault(ka, set()).add(kb)
                cf.setdefault(kb, set()).add(ka)

        def mark_nf(ka, kb):
            if ka != kb:
                nf.setdefault(ka, set()).add(kb)
                nf.setdefault(kb, set()).add(ka)

        def mark_all_cf(keys):
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    mark_cf(keys[i], keys[j])

        def has_nf_pair(R: Stage) -> bool:
            k0, k1, k2 = R.keys
            n0 = nf.get(k0, ())
            return k1 in n0 or k2 in n0 or k2 in nf.get(k1, ())

        def try_mark_nf(R: Stage):
            ks = R.keys
            non_cf = [(ks[i], ks[j]) for i in range(3) for j in range(i + 1, 3)
                      if ks[j] not in cf.get(ks[i], ())]
            if len(non_cf) == 1:
                mark_nf(*non_cf[0])

        # the first stage is the triangle of the dummies
        add_stages([(tuple(dummies), tuple(dummy_coords), tuple(dummy_keys), (True, True, True),
                     frozenset(dummy_keys))], depth=0, gain=0.0)

        L = self.L
        while L:
            R = pop()                                         # Step 2: select a stage
            if R.key in visited:
                continue
            visited.add(R.key)
            if nf and has_nf_pair(R):                         # Rule 1: cannot be a facet
                continue
            if R.lam is None:                                 # Step 5: lambda not positive
                if self.exa_rule2:                            # Rule 2 (off by default, see Config)
                    try_mark_nf(R)
                step5_inject(R)
                continue
            lam = R.lam

            # Precalculation: a known point beyond the plane makes the solve unnecessary
            i_best = precalc_point(R)
            used_precalc = i_best is not None
            if used_precalc:
                r_star, r_key = known.Z[i_best], known.keys[i_best]   # rows of known.Z never change
                r_coords = known_coords[i_best]
            else:                                             # Step 4.1.3: solve
                r_star, r_key = self._solve(R, lam)
                r_coords = tuple(r_star.tolist())

            if r_key in R.key:                                # Step 4.2: R is a facet
                mark_all_cf(R.keys)
            else:                                             # Step 4.3: new stages
                if not used_precalc:
                    lam0 = float(lam @ R.points[0])           # value of the stage plane
                    r_val = float(lam @ r_star)
                    if abs(r_val - lam0) < Z_TOL:             # r* on the plane of R
                        mark_all_cf(list(R.keys) + [r_key])
                    elif r_key in esn_keys:                   # known point beyond R
                        try_mark_nf(R)
                record_point(r_star, r_key)
                # gain of the new stages: (lambda.r* - lambda.corner) / S, computed when needed
                spawn(R, r_star, r_key, r_coords, (lam, R.points[0], r_star, self.S))
            if not used_precalc:
                self._after_solve(r_star)


def run_exact(problem: Problem, inst: Instance, **kwargs) -> SearchResult:
    """The reference CAN/ExA (dfs order, no stop) with our persistent solver."""
    return ExactSearch(problem, inst, order="dfs", **kwargs).run()
    
"""
ExA in brief:
Start. Make the triangle of the three dummy points (m1, m2, m3), with λ = (⅓, ⅓, ⅓), and put it in the worklist L.
Pick. Pop the newest stage R from L (dfs).
Already seen? If R’s triangle was processed before (same three corners in any order), skip it.
No valid weight (Step 5)? If λ has a component ≤ 0, don’t solve. Take the most recently found point that is not a corner of R and make new stages with it.
Precalculation. If some already-known point lies beyond R , use it as r* without solving. 
Otherwise solve with λ. That gives r*.
Is r* a corner of R? Then nothing lies beyond R: R is a final facet and is closed.
Otherwise record r* as an ESN point (if new) and spawn up to 3 new stages, each replacing one corner of R by r*. Skip triangles already seen.
Repeat steps 2–8 until L is empty. Every final facet has then been closed, and the recorded points are all the ESN points.

"""