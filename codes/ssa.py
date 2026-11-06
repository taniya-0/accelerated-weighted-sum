"""
ssa.py -- Stage Search Algorithm (SSA)
====================================================================================
SSA is its own algorithm, not CAN/ExA with another order. It keeps the convex hull of the points found so far (Qhull) and
repeatedly solves the weighted-sum problem for the hull facet the scorer rates
highest:
    1. start   p=3: three dummy points give ONE facet (the dummy triangle,lambda = (1/3, 1/3, 1/3)).
               p=2: CAN's two extreme solves (lambda = (1-eps, eps) and (eps, 1-eps), forced, never scored) give ONE facet, the segment between the two extreme points.
    2. score   every open facet gets the geometric features and a score from the neural network.
    3. solve   the open facet with the highest score is solved
    4. update  if the returned point z* lies beyond the facet it is a new hull vertex: Qhull adds it, the facets it can see disappear (also open ones, which then
               need no solve), and the new facets (z* joined) become open facets. Otherwise nothing lies beyond the facet: it
               is a facet of the final hull and is closed.
    5. stop    when no open facet is left (complete; with tau = 0 the result is exact) or when the best open facet scores below tau.

HOW TO RUN? 
Pipeline (one model per problem; 'pipeline --problem all' = all three steps, 4 problems):
    python ssa.py collect   --problem ks_p3            # random-order facet search, labels from ExA/CAN -> data.npz
    python ssa.py train     --problem ks_p3            # MLP scorer               -> scorer.npz
    python ssa.py calibrate --problem ks_p3            # threshold tau            -> calibration.json
    python ssa.py run       --problem ks_p3            # SSA vs ExA on one test instance
Outputs go to SSAv3/runs/<problem>/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Callable, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from exa import (FEATURE_NAMES, Config, Scorer, SearchResult, SolveRecord, Stage, Worklist,  # noqa: F401
                 _Stop, can_lambda, exa_lambda_rows, final_points, load_tau, point_key, run_dir,
                 run_exact, same_point, stage_features)
from problems import PROBLEMS, Instance, Problem, Solver, test_instance, train_instance, val_instance, z_max

class HullConfig:
    DUMMY_M = 1e5           # p=3 dummies d_q = c - M e_q, M = DUMMY_M * z_max (ExA's M), SSA’s dummies c − M·e_q go infinitely far down in one objective, so the real points form the top of the hull.
    DUMMY_C = 2.0           # ... with c = DUMMY_C * z_max * (1, 1, 1), above every objective value
    FLOOR = -2.0            # p=2 floor point FLOOR * z_max * (1, 1), below every objective value, check below
    PROJ_R = 1.0            # p=3 projective map for Qhull: phi(x) = u / (1 + sum(u) / R), u = c - x, R = PROJ_R * z_max
    QHULL_OPTIONS = {2: "", 3: "QJ"}
    CODE_BASE = 1 << 20     # facet identity: its sorted vertex numbers in base 2^20 (one int64)

"""
Reason for FLOOR = -2.0 for p=2
Qhull computes a closed polygon, and it can’t build one from two points. At the start SSA has only y1 and y2, which give a line, not a polygon. 
Adding a point far below and to the left of everything closes the shape into a triangle F–y1–y2. As more front points are added, the polygon always 
has the shape “F plus the front”

Code base:
Qhull describes each facet by its three vertex row numbers, in no particular order: (5, 2, 9) and (9, 5, 2) are the same facet. SSA needs one fixed 
ID per facet, to compare “facets before” with “facets after” an update. So it sorts the row numbers and packs them into one integer, like the digits
of a number in base B = 1,048,576: code = 5·B² + 2·B + 9
"""

# =============================================================================
# THE FACET SEARCH
# =============================================================================

class FacetSearch:
    """Facet search on the convex hull.Same interface and result as exa.ExactSearch.
    order      "score" (SSA, or the area heuristic with eval_common.AreaScorer) or "random" (training data collection)
    scorer     needed for order="score"
    tau        stop when the best open facet scores below tau
    max_solves stop after this many solves (fixed-budget experiments)
    on_solve   callback(z) after every solve; return True to stop ("targetexperiment")
    record     keep a SolveRecord per solve (with features) in result.trace
    """

    def __init__(self, problem: Problem, inst: Instance, *, order: str = "score",
                 scorer=None, tau: float = 0.0, seed: int = 0,
                 max_solves: Optional[int] = None,
                 on_solve: Optional[Callable[[Optional[np.ndarray]], bool]] = None,
                 record: bool = False, time_limit: float = Config.TIME_LIMIT):
        if order not in ("score", "random"):
            raise ValueError(f"FacetSearch order must be 'score' or 'random', not {order!r}")
        if order == "score" and scorer is None:
            raise ValueError("order='score' needs a scorer")
        self.problem, self.inst = problem, inst
        self.p = problem.p
        self.order, self.scorer, self.tau = order, scorer, float(tau)
        self.max_solves, self.on_solve, self.record = max_solves, on_solve, record
        self.time_limit = float(time_limit)
        self.L = Worklist(order, seed)                    # open facets, in the chosen order. Unlike self.open this can have stale entries (facets Qhull already deleted, waiting to be skipped)
        self.open: dict = {}                              # facet code -> Stage, open facets only
        self.S = z_max(inst)                              # feature scale
        self.n_size = inst.n / problem.n_ref
        self.solver = Solver(problem, inst, time_limit)   # built before the clock
        self.trace: list = []
        self.found: list = []                             
        self.found_keys: set = set()
        self.accepted: list = []                          # real hull vertices ever added (output candidates)
        self.rows: list = []                              # coordinates (np arrays; p=3 dummies at c - M e_q)
        self.row_coords: list = []                        # coords as tuples of Python floats (features, lambda)
        self.qrows: list = []                             # the coordinates Qhull sees (p=3: projective image)
        self.row_dummy: list = []                         # is the row a dummy?
        self.row_of_key: dict = {}                        # point_key -> row, real points only
        self.hull: Optional[ConvexHull] = None
        self.n_rebuilds = 0                               # Qhull rebuilds after an incremental error
        self.codes: set = set()                           # codes of all facets of the current hull
        
    # ------------------------------------------------------------------ #
    # facets: identity, creation, scoring, the open list
    # ------------------------------------------------------------------ #
    @staticmethod
    def _codes(simplices: np.ndarray) -> np.ndarray:
        """Coding the facet using CODE_BASE discussed earlier."""
        s = np.sort(simplices, axis=1).astype(np.int64)
        code = s[:, 0]
        for j in range(1, s.shape[1]):
            code = code * HullConfig.CODE_BASE + s[:, j]
        return code

    def _qhull_coords(self, z, dummy_axis: Optional[int] = None) -> np.ndarray:
        """Coordinates for Qhull. p=2: the point itself. p=3: the projective image phi (see the docstring)."""
        if self.p == 2:
            return np.asarray(z, dtype=np.float64)
        R = HullConfig.PROJ_R * self.S                    # R = z_max = 6285
        if dummy_axis is not None:                        # dummy at infinity in direction -e_q -> R e_q
            return R * np.eye(3)[dummy_axis]              # d1 -> (6285, 0, 0), d2 -> (0, 6285, 0), d3 -> (0, 0, 6285)
        u = self.c - np.asarray(z, dtype=np.float64)       # u > 0 for every real point, c = (12570, 12570, 12570) lies above every real value, so u is “how far below the ceiling” the point is in each objective,
        return u / (1.0 + u.sum() / R)                   # Divide by a number that grows with the size of u: the farther away, the more it is shrunk
        # Near points barely change in order of magnitude, while far points all end up just below 6285. After the map, every number Qhull sees is between 0 and 6285.

    def _add_row(self, z, dummy: bool, key=None, dummy_axis: Optional[int] = None) -> int:
        self.qrows.append(self._qhull_coords(z, dummy_axis))
        self.rows.append(np.asarray(z, dtype=np.float64))
        self.row_coords.append(tuple(float(v) for v in z))
        self.row_dummy.append(dummy)
        if key is not None:
            self.row_of_key[key] = len(self.rows) - 1
        return len(self.rows) - 1
    
    
    """
    Example: p=2 
    Say the rows (points) are 0 = floor (−8244, −8244), 1 = y1 = (2287, 2194) and 2 = y2 = (2153, 2381). 
    For teh segment between points 2 and 1, We need to compute lambda between 2 and 1. Qhull reports the open edge as rows (i, j) = (2, 1) with code 1,048,578. 
    This is the first facet, so depth = 1 and gain = 0. sililarly for other (i,j) point combinations of this hull.
    
    Example: p=3
    points	(array([−6.2849e+08, 12570, 12570]), array([−2777, −1697, −2963]), array([−3118, −1794, −2651]))
    keys	(0, 3, 4)
    is_dummy	(True, False, False): row 0 is a dummy
    depth, parent_gain	2, 0.01 (example values)
    lam	array([2.3178e-05, 0.76280, 0.23718]) , normal to the facet made by these 3 points
    coords	((−628487430.0, 12570.0, 12570.0), (−2777.0, −1697.0, −2963.0), (−3118.0, −1794.0, −2651.0))
    key	3145732 (= 0*B^2 + 3*B + 4) (indicating the facet)
    """
    def _make_facets(self, vertex_rows: list, codes: list, depth: int, gain: float) -> list:
        """Stages for new positive-normal facets; a facet without a valid positive lambda is dropped."""
        stages = []
        if self.p == 2:
            for (i, j), code in zip(vertex_rows, codes):
                lam = can_lambda(self.rows[i], self.rows[j])       # CAN's weight of the segment
                if lam is not None:
                    stages.append(Stage((self.rows[i], self.rows[j]), (i, j), (False, False), depth, gain,
                                        lam, (self.row_coords[i], self.row_coords[j]), code))
            return stages
        X = exa_lambda_rows([tuple(self.row_coords[v] for v in vr) for vr in vertex_rows])  # ExA's lambda
        for vr, code, vals in zip(vertex_rows, codes, X.tolist()): 
            if not all(Config.LAM_TOL < v < math.inf for v in vals): #NaN and non-positive weights are rejected
                continue                                           
            total = vals[0] + vals[1] + vals[2]
            lam = np.array([vals[0] / total, vals[1] / total, vals[2] / total])
            stages.append(Stage(tuple(self.rows[v] for v in vr), tuple(vr),
                                tuple(self.row_dummy[v] for v in vr), depth, gain, lam,
                                tuple(self.row_coords[v] for v in vr), code))
        return stages
    
    def _features(self, stage: Stage) -> np.ndarray:
        """Features of one facet"""
        if stage.feats is None:
            stage.feats = stage_features([stage], self.S, self.n_size)[0]
        return stage.feats

    def _push(self, stages: list):
        """Open new facets. In score order they are scored together (one feature pass, one scorer call)."""
        if not stages:
            return
        if self.order == "score":
            X = stage_features(stages, self.S, self.n_size)
            for st, x, v in zip(stages, X, self.scorer(X).tolist()):
                st.feats, st.score = x, v
        for st in stages:
            self.open[st.key] = st
            self.L.push(st)

    def _pop(self) -> Stage:
        """The next open facet. Facets that disappeared from the hull are skipped"""
        while True:
            st = self.L.pop()
            if self.open.get(st.key) is st:
                del self.open[st.key]
                return st
                
    """
    Note: Several facets are scored at once. stage_features(stages, S, n_size) puts one row per facet into a single matrix. Say there are 5 facets then for the 5 facets , 
    that is a (5, 8) matrix is passed to scorer and every step is a matrix operation that treats each row separately
    """
    
    # ------------------------------------------------------------------ #
    # the hull (Qhull)
    # ------------------------------------------------------------------ #
    def _update_hull(self, depth: int, gain: float) -> list:
        """After a change of the hull: close the facets that disappeared, return the new open facets."""
        simplices = self.hull.simplices                   # The current facets
        codes = self._codes(simplices).tolist()
        current = set(codes)
        for code in self.codes - current:                 # facets the new point could see: gone
            self.open.pop(code, None)
        idx = [i for i, code in enumerate(codes) if code not in self.codes]   # Positions of the new facets in Qhull’s list.
        self.codes = current
        if self.p == 2:                                   # open: outward normal strictly positive
            positive = (self.hull.equations[idx, :2] > 0).all(axis=1).tolist() #equations has one row (lam_1, lam_2, offset) per facet, and Qhull’s normals point outward. Both components > 0 means the edge faces the front.
            idx = [i for i, ok in zip(idx, positive) if ok]
        else:
            # every facet of the free-disposal hull, exclude the face at infinity. far_code is set in _start
            # triangle); a facet whose lambda is not strictly positive is dropped in _make_facets
            idx = [i for i in idx if codes[i] != self.far_code]
        return self._make_facets(simplices[idx].tolist(), [codes[i] for i in idx], depth, gain)

    def _add_point(self, z, key, depth: int, gain: float) -> list:
        """Add a new hull vertex (Qhull, incremental) and return the new open facets."""
        row = self._add_row(z, dummy=False, key=key) #The new point becomes the next row,
        if self.hull is not None:
            try:
                self.hull.add_points(self.qrows[row][None, :])
                return self._update_hull(depth, gain)
            except QhullError:                          # rare: rebuild the hull from all points instead
                self.n_rebuilds += 1
                self.hull.close()
        self.hull = ConvexHull(np.vstack(self.qrows), incremental=True, qhull_options=HullConfig.QHULL_OPTIONS[self.p])
        return self._update_hull(depth, gain)
        
    # ------------------------------------------------------------------ #
    # one solve: stop checks, Gurobi, bookkeeping, callback (as exa.ExactSearch)
    # ------------------------------------------------------------------ #
    def _solve(self, stage: Optional[Stage], lam: np.ndarray, forced: bool = False):
        if not forced:
            if self.order == "score" and stage.score < self.tau:
                raise _Stop("tau")
            if self.max_solves is not None and self.solver.n_solves >= self.max_solves:
                raise _Stop("budget")
        if time.perf_counter() - self.t0 > self.time_limit:
            raise _Stop("time_limit")
        # the features of the facet are known BEFORE the solve (training rows)
        feats = self._features(stage) if (self.record and not forced) else None
        z = self.solver.solve(lam)
        k = None if z is None else point_key(z)
        new = k is not None and k not in self.found_keys
        if new:
            self.found.append(z)
            self.found_keys.add(k)
        if self.record:
            self.trace.append(SolveRecord(   # This record is read during data collection and caliberation
                step=self.solver.n_solves, lam=np.asarray(lam, dtype=np.float64).copy(), z=z,
                feats=feats, score=float("nan") if (forced or self.order != "score") else stage.score,
                new_point=new, forced=forced, time=time.perf_counter() - self.t0))
        if z is None:
            raise _Stop("solver_failed")
        return z, k

    def _after_solve(self, z):
        """Ask the caller whether to stop (fixed-target race) and check the budget."""
        if self.on_solve is not None and self.on_solve(z):
            raise _Stop("target")
        if self.max_solves is not None and self.solver.n_solves >= self.max_solves and self.open:
            raise _Stop("budget")
    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def run(self, postprocess: bool = True) -> SearchResult:
        self.t0 = time.perf_counter()
        stopped_by = "complete"
        try:
            if self._start():
                self._search()
        except _Stop as stop:
            stopped_by = stop.reason
        search_time = time.perf_counter() - self.t0
        points = final_points(self.accepted, self.solver.env) if postprocess else []
        total_time = time.perf_counter() - self.t0
        """At the end of run, everything a caller needs is packed into one SearchResult object and returned. That object is the only thing evaluation scripts 
        get from a run, and ExactSearch returns the same type, so every script can treat E and SSA the same way.
        Found: every distinct point any solve returned, whether or not it became a hull vertex. used for diagnstics to identify what the solver returned
        POints: FInal ESN points"""
        result = SearchResult(points=points, found=list(self.found), n_solves=self.solver.n_solves,
                              solver_time=self.solver.solver_time, search_time=search_time,
                              total_time=total_time, stopped_by=stopped_by, trace=self.trace)
        if self.hull is not None:
            self.hull.close()
        self.solver.close()
        return result

    def _start(self) -> bool:
        """Step 1: the first facet. Returns False if the search is already complete."""
        S = self.S
        if self.p == 3:
            # three dummies d_q = c - M e_q (no solve); their triangle is the first facet
            self.c = np.full(3, HullConfig.DUMMY_C * S)
            M = HullConfig.DUMMY_M * S
            for q in range(3):
                self._add_row(self.c - M * np.eye(3)[q], dummy=True, dummy_axis=q)
            self.far_code = int(self._codes(np.array([[0, 1, 2]]))[0]) #This computes the code of the face at infinity
            first = self._make_facets([[0, 1, 2]], [self.far_code], 0, 0.0)
            self._push(first)
            return True
        # p=2: CAN's two extreme points are always needed: forced solves, never scored or stopped
        eps = Config.CAN_EPS
        self._add_row(np.full(2, HullConfig.FLOOR * S), dummy=True)      
        y1, k1 = self._solve(None, np.array([1.0 - eps, eps]), forced=True)
        self.accepted.append(y1)
        self._add_row(y1, dummy=False, key=k1)
        self._after_solve(y1)
        y2, k2 = self._solve(None, np.array([eps, 1.0 - eps]), forced=True)
        if same_point(y1, y2):
            self._after_solve(y2)
            return False                                  # a single nondominated point
        self.accepted.append(y2)
        self._after_solve(y2)
        # the first facet: the segment y1-y2 (the hull of floor, y1, y2 is a triangle)
        self._push(self._add_point(y2, k2, depth=1, gain=0.0))
        return True

    def _search(self):
        """Steps 2-5: solve the best open facet, add a point beyond it to the hull, until none is left."""
        Z_TOL = Config.Z_TOL
        while self.open:
            F = self._pop()
            z, k = self._solve(F, F.lam)
            lam = F.lam
            gap = float(lam @ z) - float(lam @ F.points[0])    # how far z lies beyond the facet's plane
            if gap > Z_TOL and k not in self.row_of_key:       # beyond: a new hull vertex
                self.accepted.append(z)
                self._push(self._add_point(z, k, depth=F.depth + 1, gain=gap / self.S))
            # else: nothing lies beyond F, so F is a facet of the final hull (closed)
            self._after_solve(z)


def run_ssa(problem: Problem, inst: Instance, scorer: Scorer, tau: float, **kwargs) -> SearchResult:
    """SSA = score order + threshold tau."""
    return FacetSearch(problem, inst, order="score", scorer=scorer, tau=tau, **kwargs).run()


###############____NOW THE ACTUAL DATA COLLECTION TRAINING AND CALLIBERATION HAPPENS____#######################################

# =============================================================================
# STEP 1: COLLECT TRAINING DATA (random-order facet searches)
# =============================================================================

def labelled_rows(result: SearchResult, true_points):
    """marks (features, label) of every scored solve of a run.
    label = 1  if the solve returned a true ESN point that no earlier solve had returned,
    label = 0  otherwise.
    """
    true_keys = {point_key(z) for z in true_points}  # True actual ESN points identified by the exact algo.
    seen, X, Y = set(), [], [] # seen holds the points returned so far in this run.
    for rec in result.trace:
        k = None if rec.z is None else point_key(rec.z)
        label = int(k is not None and k in true_keys and k not in seen)
        if k is not None:
            seen.add(k) #Added after labelling, so this record’s own point doesn’t count against it. Forced solves also mark their points as seen.
        if not rec.forced:
            X.append(rec.feats)
            Y.append(label)
    return X, Y
    

def collect(problem: Problem, sizes, n_instances: int):
    X, Y, G, N = [], [], [], []
    group = 0 # group is liek an identifier for which instance teh record came from
    n_differ = 0                      # instances where the complete facet search and ExA / CAN disagree, just for a check.
    for n in sizes:
        for k in range(n_instances):
            inst = train_instance(problem, n, k) # generates instance
            res = FacetSearch(problem, inst, order="random", seed=k, record=True).run() # does all the qhull update , pushign popping all of that
            if res.stopped_by != "complete":
                print(f"[collect] n={n} k={k}: run stopped by {res.stopped_by}, skipped")
                continue
            # ground truth for the labels: the exact benchmark (ExA for p=3, CAN for p=2)
            truth = run_exact(problem, inst)
            if truth.stopped_by != "complete":
                print(f"[collect] n={n} k={k}: {problem.exact_name} stopped by {truth.stopped_by}, skipped")
                continue
            if {point_key(z) for z in res.points} != {point_key(z) for z in truth.points}:
                n_differ += 1         # should not happen: the complete facet search is exact
                print(f"[collect] WARNING n={n} k={k}: the facet search found {len(res.points)} points, "
                      f"{problem.exact_name} {len(truth.points)}; labels use {problem.exact_name}'s set")
            x, y = labelled_rows(res, truth.points)
            X += x
            Y += y
            G += [group] * len(y)
            N += [n] * len(y)
            group += 1
            print(f"[collect] {problem.key} n={n} k={k} |Y_E|={len(truth.points)} solves={res.n_solves} "
                  f"rows={len(y)} positive={np.mean(y):.2f} t={res.total_time:.1f}s", flush=True)
    path = run_dir(problem) / Config.DATA_FILE
    np.savez_compressed(path, X=np.vstack(X).astype(np.float32), Y=np.array(Y, dtype=np.int8),
                        G=np.array(G, dtype=np.int32), N=np.array(N, dtype=np.int32),
                        feature_names=np.array(FEATURE_NAMES[problem.p])) #np.vstack turns the list of feature rows into one matrix
    print(f"[collect] {len(Y)} rows ({np.mean(Y):.1%} positive) -> {path}")
    print(f"[collect] facet search vs {problem.exact_name}: different ESN set on {n_differ} of {group} instances")

# =============================================================================
# STEP 2: TRAIN THE SCORER
# the loss is binary cross-entropy (BCE). For a probability p and label y: loss = −[y·log p + (1 − y)·log(1 − p)] BCEWithLogitsLoss takes the logit directly and applies the sigmoid inside
# =============================================================================

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability that a random positive row scores higher than a random negative row.
    y = (1, 0, 1, 0), s = (0.9, 0.3, 0.4, 0.5). The positive scores are {0.9, 0.4} and the negative scores {0.3, 0.5}
    Pair (positive vs negative). is positive higher? 0.9 vs 0.3 yes, 0.9 vs 0.5	yes,0.4 vs 0.3yes,	0.4 vs 0.5 no. so AUC = 3/4
    AUC = 0.83 means say if there are only 2 facets and scorer pick one facet that finds a new ESN point and one that doesn’t, at random. The scorer gives the first one the higher score 83% of the time, and gets the order wrong 17% of the time.
    """
    from scipy.stats import rankdata
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(problem: Problem, epochs: int = Config.EPOCHS):
    import torch
    import torch.nn as nn

    torch.manual_seed(Config.SEED)
    rng = np.random.default_rng(Config.SEED)
    d = np.load(run_dir(problem) / Config.DATA_FILE)
    if [str(s) for s in d["feature_names"]] != FEATURE_NAMES[problem.p]:
        raise SystemExit(f"{run_dir(problem) / Config.DATA_FILE} has other features; run 'collect' again")
    X, Y, G = d["X"].astype(np.float32), d["Y"].astype(np.float32), d["G"]

    # hold out 20% of the INSTANCES (not rows) to pick the best epoch
    groups = np.unique(G)
    val_groups = rng.choice(groups, size=max(1, len(groups) // 5), replace=False)
    is_val = np.isin(G, val_groups)
    Xtr, Ytr, Xva, Yva = X[~is_val], Y[~is_val], X[is_val], Y[is_val]

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6               # constant features become 0 after scaling
    net = nn.Sequential(nn.Linear(X.shape[1], Config.HIDDEN), nn.ReLU(),
                        nn.Linear(Config.HIDDEN, Config.HIDDEN), nn.ReLU(),
                        nn.Linear(Config.HIDDEN, 1))
    opt = torch.optim.Adam(net.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(Ytr)
    xv = torch.tensor((Xva - mu) / sd)
    yv = torch.tensor(Yva)

    best_loss, best_state = math.inf, None
    for ep in range(epochs):
        net.train()
        # A new random order each epoch. Per batch: clear old gradients, forward pass, loss, backward pass, Adam step.
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), Config.BATCH):
            idx = perm[i:i + Config.BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
        net.eval() # The held-out loss.
        with torch.no_grad():
            logits = net(xv).squeeze(-1)
            vl = float(loss_fn(logits, yv))
        if vl < best_loss:                                 # keep the best epoch on held-out instances
            best_loss = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            pv = torch.sigmoid(logits).numpy()
            print(f"[train] epoch {ep:3d} val_bce={vl:.4f} val_auc={auc(Yva, pv):.3f} "
                  f"val_acc={((pv > 0.5) == Yva).mean():.3f}")
    net.load_state_dict(best_state)

    # save as plain numpy arrays: the search never needs torch
    layers = [m for m in net if isinstance(m, nn.Linear)]
    params = {"mu": mu.astype(np.float64), "sd": sd.astype(np.float64),
              "feature_names": np.array(FEATURE_NAMES[problem.p])}
    for i, layer in enumerate(layers):
        params[f"W{i}"] = layer.weight.detach().numpy().astype(np.float64)
        params[f"b{i}"] = layer.bias.detach().numpy().astype(np.float64)
    path = run_dir(problem) / Config.SCORER_FILE
    np.savez(path, **params)
    pv = Scorer(params)(Xva)
    print(f"[train] best val_bce={best_loss:.4f} val_auc={auc(Yva, pv):.3f} "
          f"(train rows={len(Ytr)}, val rows={len(Yva)}, positive={Y.mean():.1%}) -> {path}")

# =============================================================================
# STEP 3: CALIBRATE THE STOP THRESHOLD TAU
# =============================================================================
# SSA with threshold tau is exactly the tau=0 run cut at the first solve whose
# stage scores below tau (the order of the stages does not depend on tau). So
# ONE complete run per validation (def tau_curve) instance gives recall and solves for every
# tau on the grid, without needing to re-run anything.

def tau_curve(result: SearchResult, true_points) -> dict:
    """{tau: (recall, solves, share of complete-run solves)} for one complete SSA run.
    true_points: the true ESN set of the instance (exact benchmark, see collect).
    solves(τ) = the position of the FIRST record with score < τ (or all of them);
    recall(τ) = the share of true ESN points first found before that position.
    the selection rule is this: Choose the largest τ (means the fewest solves) whose 10th-percentile recall (means about 90% of instances reach at least 80% recall) over the validation instances is ≥ 0.8.
    """
    true_keys = {point_key(z) for z in true_points}
    first_hit, seen = [], set()                   # solve index of every newly found true point
    for i, rec in enumerate(result.trace):
        k = None if rec.z is None else point_key(rec.z)
        if k is not None and k in true_keys and k not in seen:
            first_hit.append(i)
        if k is not None:
            seen.add(k)
    first_hit = np.array(first_hit)
    scores = np.array([np.inf if rec.forced else rec.score for rec in result.trace])
    out = {}
    for tau in Config.TAU_GRID:
        below = np.nonzero(scores < tau)[0]
        stop = int(below[0]) if len(below) else len(scores)  # number of solves SSA makes
        recall = float((first_hit < stop).sum()) / max(len(true_keys), 1)
        out[tau] = (recall, stop, stop / max(len(scores), 1))
    return out


def calibrate(problem: Problem, sizes, n_instances: int, target_recall: float):
    scorer = Scorer.load(problem)
    curves, sizes_of = [], []
    for n in sizes:
        for k in range(n_instances):
            inst = val_instance(problem, n, k)
            res = FacetSearch(problem, inst, order="score", scorer=scorer, tau=0.0, record=True).run()
            if res.stopped_by != "complete":
                print(f"[calibrate] n={n} k={k}: stopped by {res.stopped_by}, skipped")
                continue
            truth = run_exact(problem, inst)          # ground truth: ExA (p=3) / CAN (p=2)
            if truth.stopped_by != "complete":
                print(f"[calibrate] n={n} k={k}: {problem.exact_name} stopped by {truth.stopped_by}, skipped")
                continue
            c = tau_curve(res, truth.points) # calculate tau_curve for each instance
            curves.append(c)
            sizes_of.append(n)
            print(f"[calibrate] {problem.key} n={n} k={k} |Y_E|={len(truth.points)} complete solves={res.n_solves} "
                  f"| tau=0.5: recall={c[0.5][0]:.1%} solves={c[0.5][1]}", flush=True)
    rows = []
    for tau in Config.TAU_GRID:
        r = np.array([c[tau][0] for c in curves]) #recall for all instances at that tau
        share = np.array([c[tau][2] for c in curves]) #solve share for all instances at that tau
        row = {"tau": tau, "recall_mean": float(r.mean()), "recall_p10": float(np.percentile(r, 10)),
               "recall_min": float(r.min()), "solve_share_mean": float(share.mean())} #averaged
        for n in sizes:   # per-size 10th percentile, to see whether one tau suits every n
            rn = np.array([c[tau][0] for c, m in zip(curves, sizes_of) if m == n]) #sort recall of 10 instances for this particular tau, and identify the 10th percentile value
            if len(rn):
                row[f"recall_p10_n{n}"] = float(np.percentile(rn, 10))
        rows.append(row)
    ok = [row for row in rows if row["recall_p10"] >= target_recall] # if tenth percentile value is > 80 add this tau to possible taus
    chosen = max(ok, key=lambda row: row["tau"]) if ok else rows[0] # select the max tau from the possible taus
    print(f"\n{'tau':>5} {'recall':>7} {'p10':>6} {'min':>6} {'solves/complete':>16}")
    for row in rows:
        mark = "  <== chosen" if row is chosen else ""
        print(f"{row['tau']:>5.2f} {row['recall_mean']:>6.1%} {row['recall_p10']:>6.1%} "
              f"{row['recall_min']:>6.1%} {row['solve_share_mean']:>16.2f}{mark}")
    path = run_dir(problem) / Config.CALIB_FILE
    with open(path, "w") as f:
        json.dump({"tau": chosen["tau"], "target_recall": target_recall, "sizes": list(sizes),
                   "n_instances": n_instances, "table": rows}, f, indent=2)
    print(f"\n[calibrate] tau={chosen['tau']} (p10 recall {chosen['recall_p10']:.1%}, "
          f"mean recall {chosen['recall_mean']:.1%}) -> {path}")


# =============================================================================
# FINAL WHERE EVERYTHING IS CALLED AND RAN
# =============================================================================

def pipeline(problem: Problem, sizes, collect_instances: int, calib_instances: int, epochs: int,
             target_recall: float):
    """collect -> train -> calibrate for one problem."""
    collect(problem, sizes, collect_instances)
    train(problem, epochs=epochs)
    calibrate(problem, sizes, calib_instances, target_recall)


def main():
    ap = argparse.ArgumentParser(description="SSAv3: learned facet search on the convex hull (Qhull).")
    ap.add_argument("command", choices=["collect", "train", "calibrate", "pipeline", "run"],
                    help="pipeline = collect, then train, then calibrate")
    ap.add_argument("--problem", choices=["all", *PROBLEMS], required=True,
                    help="'all' = ks_p2, ks_p3, tsp_p2, tsp_p3 one after the other (not for 'run')")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="default: the problem's training sizes")
    ap.add_argument("--instances", type=int, default=None,
                    help=f"instances per size (default: collect {Config.COLLECT_INSTANCES}, "
                         f"calibrate {Config.CALIB_INSTANCES})")
    ap.add_argument("--epochs", type=int, default=Config.EPOCHS)
    ap.add_argument("--target-recall", type=float, default=Config.TARGET_RECALL)
    ap.add_argument("--n", type=int, default=None, help="run: instance size")
    ap.add_argument("--k", type=int, default=0, help="run: test instance index")
    ap.add_argument("--tau", type=float, default=None, help="run: override the calibrated tau")
    a = ap.parse_args()
    if a.problem == "all" and a.command == "run":
        raise SystemExit("'run' needs one problem, e.g. --problem ks_p3")
    if a.problem == "all" and a.sizes:
        raise SystemExit("--sizes differs between knapsack and TSP: give it with a single problem only")
    keys = list(PROBLEMS) if a.problem == "all" else [a.problem]

    for key in keys:                                  # one problem after the other
        problem = PROBLEMS[key]
        sizes = a.sizes or list(problem.sizes)
        t0 = time.perf_counter()
        if a.command != "run":
            print(f"\n{'=' * 70}\n{a.command} {key}  (sizes {sizes})\n{'=' * 70}", flush=True)
        if a.command == "collect":
            collect(problem, sizes, a.instances or Config.COLLECT_INSTANCES)
        elif a.command == "train":
            train(problem, epochs=a.epochs)
        elif a.command == "calibrate":
            calibrate(problem, sizes, a.instances or Config.CALIB_INSTANCES, a.target_recall)
        elif a.command == "pipeline":
            pipeline(problem, sizes, a.instances or Config.COLLECT_INSTANCES,
                     a.instances or Config.CALIB_INSTANCES, a.epochs, a.target_recall)
        else:
            n = a.n or problem.sizes[0]
            inst = test_instance(problem, n, a.k)
            tau = a.tau if a.tau is not None else load_tau(problem)
            res = run_ssa(problem, inst, Scorer.load(problem), tau)
            ref = run_exact(problem, inst)
            true_keys = {point_key(z) for z in ref.points}
            hit = sum(point_key(z) in true_keys for z in res.points)
            print(f"{problem.key} n={n} test k={a.k} tau={tau}")
            print(f"  {problem.exact_name}: |Y_E|={len(ref.points)} solves={ref.n_solves} t={ref.total_time:.2f}s")
            print(f"  SSA: found={len(res.points)} true={hit} recall={hit / max(len(true_keys), 1):.1%} "
                  f"solves={res.n_solves} t={res.total_time:.2f}s stopped_by={res.stopped_by}")
            continue
        print(f"[{a.command}] {key} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
