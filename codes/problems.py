"""
problems.py
===========
Everything that is specific to one of the four problems, kept in one place so
that the search code (ssa.py) and the experiment scripts never have to care
whether they work on a knapsack or a TSP:

    key       problem     p   exact benchmark   sizes
    ks_p2     knapsack    2   CAN               50 75 100 150 200
    ks_p3     knapsack    3   ExA               50 75 100 150 200
    tsp_p2    TSP         2   CAN               5 10 15 25 30
    tsp_p3    TSP         3   ExA               5 10 15 25 30

In this code you will find:
1. Instance generators
2. Solver - one persistent Gurobi model per instance. Only the objective coefficients change between solves
3. Scale + dummies z_max (the scale used by ExA) and ExA's dummy points.

All objectives are MAXIMISED. TSP distances are stored negated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

@dataclass(frozen=True)
class Problem:
    key: str                  # "ks_p2", "ks_p3", "tsp_p2", "tsp_p3"
    kind: str                 # "knapsack" or "tsp"
    p: int                    # number of objectives
    exact_name: str           # "CAN" (p=2) or "ExA" (p=3)
    sizes: tuple              # sizes used in training and in the main table
    extra_sizes: tuple        # unseen sizes (marked with a dagger in the paper)
    mip_gap: float            # MIPGap of the reference solver (see Solver)
    n_ref: int                # n is divided by this to make the n_size feature. It is max n value.
    
PROBLEMS = {
    "ks_p2": Problem("ks_p2", "knapsack", 2, "CAN", (50, 75, 100, 150, 200), (65, 90, 125, 175), 1e-6, 200),
    "ks_p3": Problem("ks_p3", "knapsack", 3, "ExA", (50, 75, 100, 150, 200), (65, 90, 125, 175), 0.0, 200),
    "tsp_p2": Problem("tsp_p2", "tsp", 2, "CAN", (5, 10, 15, 25, 30), (8, 13, 20, 28), 0.0, 30),
    "tsp_p3": Problem("tsp_p3", "tsp", 3, "ExA", (5, 10, 15, 25, 30), (8, 13, 20, 28), 0.0, 30),
}

# =============================================================================
# INSTANCE GENERATION
# =============================================================================

# These IDs are used to distinguish between instances used for training, validation(caliberation), and evaluation.
TRAIN_ID_START = 0
VAL_ID_START = 80_000_000
TEST_ID_START = 90_000_000

@dataclass
class Instance:
    kind: str                              # "knapsack" or "tsp"
    n: int                                 # items (knapsack) or cities (TSP)
    p: int                                 # number of objectives
    instance_id: int
    scenario: str = "id"                   # distribution name (OOD experiments) "id" means training instance distribution (in-distribution)
    # knapsack data
    C: Optional[np.ndarray] = None         # (p, n) profits
    weights: Optional[np.ndarray] = None   # (n) item weights
    capacity: Optional[int] = None
    # TSP data
    coords: Optional[np.ndarray] = None    # (p, n, 2) city coordinates
    dist: Optional[np.ndarray] = None      # (p, n, n) NEGATED distances because w eare doing max -f 
    
def _knapsack_seed(n: int, instance_id: int) -> int:
    ss = np.random.SeedSequence([int(n), int(instance_id)]) #hashes the (n, id) pair into one 64-bit seed. Every (size, id) pair gets an independent seed making it easy to replicate.
    return int(ss.generate_state(1, dtype=np.uint64)[0])
    
def negated_distances(coords: np.ndarray) -> np.ndarray:
    p, n, _ = coords.shape
    dist = np.zeros((p, n, n), dtype=np.float64) # Creates p nxn matrices each for each objective
    for q in range(p):
        diff = coords[q, :, None, :] - coords[q, None, :, :]
        dist[q] = np.floor(np.sqrt(np.sum(diff ** 2, axis=-1)) + 0.5)
    return -dist
    
def generate_instance(problem: Problem, n: int, instance_id: int) -> Instance:
    """Ozpeynirci & Koksalan (2010) instances"""
    if problem.kind == "knapsack":
        # profits and weights are drawn from discrete Uniform[60, 100], capacity = half the total weight
        rng = np.random.default_rng(_knapsack_seed(n, instance_id))
        C = rng.integers(60, 101, size=(problem.p, n))
        w = rng.integers(60, 101, size=n)
        return Instance("knapsack", n, problem.p, instance_id,C=C, weights=w, capacity=int(np.floor(0.5 * w.sum())))
    # TSP: p independent sets of integer coordinates on [0, 999]
    rng = np.random.default_rng(1000 * n + instance_id)
    coords = rng.integers(0, 1000, size=(problem.p, n, 2))
    return Instance("tsp", n, problem.p, instance_id, coords=coords, dist=negated_distances(coords))
    
def train_instance(problem: Problem, n: int, k: int) -> Instance:
    return generate_instance(problem, n, TRAIN_ID_START + k)

def val_instance(problem: Problem, n: int, k: int) -> Instance:
    return generate_instance(problem, n, VAL_ID_START + k)

def test_instance(problem: Problem, n: int, k: int) -> Instance:
    return generate_instance(problem, n, TEST_ID_START + k)
    
# =============================================================================
# WEIGHTED-SUM SOLVER
# =============================================================================
def gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    return env
    
def _tsp_subtour_callback(model, where):
    """subtour elimination. With every city at degree 2 (2 edges - one incomign and one outgoing, the edges always form loops. 
    If no loop may be smaller than n cities, the only possibility left is one loop through all n cities.
    During the solve, Gurobi can propose candidates that satisfy the degree constraints but contain subtours. 
    The callback sees each one at MIPSOL and rejects it by adding a constraint."""
    # where is a code number Gurobi passes in to say at which moment of the solve it is calling your function. Gurobi calls the callback many times, at different points
    if where != GRB.Callback.MIPSOL: # act only when Gurobi has found a new integer solution
        return
    n, xvars = model._n, model._xvars
    # if x01 = 0, x12 = 1, x13 = 1, x23 = 0 then adj{0:[], 1:[2,3], 2:[1], 3:[1]}
    adj = {i: [] for i in range(n)}
    for (i, j), var in xvars.items():
        if model.cbGetSolution(var) > 0.5: # solution values are floats such as 0.9999999
            adj[i].append(j)
            adj[j].append(i)
    # connected components of the selected edges (depth-first search)
    visited, components = set(), []
    for start in range(n):
        if start in visited:
            continue
        comp, stack = [], [start] #
        """ Each comp is one group of cities connected by one cycle
        if there were no subtours, only one single while loop would have been needed , meaning start = one city only.
        Other cities will get skipped because in that previous while loop they got added to visited"""
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            stack.extend(nb for nb in adj[node] if nb not in visited)
        components.append(comp)
    if len(components) <= 1:
        return
    for tour in components:
        if len(tour) < n:  
            # Meaning if n=5 and 3 cities for 0-1-2-0 (len(tour)=3, edges = 3 which is > |S|-1 i.e 3-1=2) its not permitted 
            # 0-1-2-3-4-0 is permitted because len(tour) = n so then check of if number of edges <= |S|-1 does not apply
            # a subset S of cities may use at most |S| - 1 edges among themselves
            model.cbLazy(gp.quicksum(
                xvars[min(tour[a], tour[b]), max(tour[a], tour[b])]
                for a in range(len(tour)) for b in range(a + 1, len(tour))
            ) <= len(tour) - 1)
            
class Solver:
    """find solution x s.t sum_q lam_q * f_q(x)  for all q objectives is maximised.
    solve(lam) returns the objective vector z = (f1, f2, .., fp) or None if Gurobi found no solution.
    n_solves and solver_time count every call (solver_time = Gurobi Runtime).
    """

    def __init__(self, problem: Problem, inst: Instance, time_limit: float = 5400.0):
        self.inst = inst
        self.time_limit = float(time_limit)
        self.env = gurobi_env()
        self.m = gp.Model(env=self.env)
        self.m.setParam("OutputFlag", 0)
        self.m.setParam("Threads", 1)              # same as every reference
        self.m.setParam("MIPGap", problem.mip_gap)
        self.n_solves = 0
        self.solver_time = 0.0
        if inst.kind == "knapsack":
            self._build_knapsack()
        else:
            self._build_tsp()
        # the variables as one list (and, for the TSP, the (i, j) index of every edge)
        self._vars = list(self.x.values())
        if inst.kind == "tsp":
            ij = np.array(list(self.x.keys()), dtype=np.int64)
            self._edge_i, self._edge_j = ij[:, 0], ij[:, 1]
            
    def _build_knapsack(self):
        n = self.inst.n
        self.x = self.m.addVars(n, vtype=GRB.BINARY, name="x")
        self.m.addConstr(gp.quicksum(int(self.inst.weights[j]) * self.x[j] for j in range(n)) <= self.inst.capacity, name="capacity")
        self.m.ModelSense = GRB.MAXIMIZE
        self.m.update()
        
    def _build_tsp(self):
        n = self.inst.n
        self.m.setParam("LazyConstraints", 1)
        self.x = {}
        for i in range(n):
            for j in range(i + 1, n):
                self.x[i, j] = self.m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")  # Select the edge connecting city i and city j
        self.m.update()
        # Every city is entered once and left once. So no city can be skipped
        for i in range(n):  # every city has exactly two tour edges
            self.m.addConstr(gp.quicksum(self.x[min(i, j), max(i, j)] for j in range(n) if j != i) == 2, name=f"deg_{i}")
        self.m.ModelSense = GRB.MAXIMIZE
        self.m._xvars, self.m._n = self.x, n
        self.m.update()
        
    def solve(self, lam) -> Optional[np.ndarray]:
        lam = np.asarray(lam, dtype=np.float64)
        inst = self.inst
        if inst.kind == "knapsack":
            coefs = lam @ inst.C                      
            self.m.setAttr("Obj", self._vars, coefs.tolist())         
        else:
            combined = np.zeros((inst.n, inst.n)) 
            # for p=3, dist matrices will have 3 nxn matrices dist[0],dist[1],dist[2] each with entries [i,j] for an edge
            # combined[i,j] = lam_1 * dist[0][i,j] + lam_2 * dist[1][i,j] + lam_3 * dist[2][i,j]. That is it contains the objective values.
            for q in range(inst.p):
                combined += lam[q] * inst.dist[q]
            self.m.setAttr("Obj", self._vars, combined[self._edge_i, self._edge_j].tolist())
        self.m.Params.TimeLimit = self.time_limit
        self.m.reset(1)   
        if inst.kind == "knapsack":
            self.m.optimize()
        else:
            self.m.optimize(_tsp_subtour_callback)
        self.n_solves += 1
        self.solver_time += self.m.Runtime
        if self.m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or self.m.SolCount == 0:
            return None
        if inst.kind == "knapsack":
            x = np.array([round(v) for v in self.m.getAttr("X", self._vars)], dtype=np.float64)
            return inst.C @ x # z_vec = f1,f2,f3
        return self._tour_objectives()
        
    def _tour_objectives(self) -> np.ndarray:
        """Walk the selected edges from city 0 and sum the p distance matrices."""
        n = self.inst.n
        adj = {i: [] for i in range(n)}
        # self.x.keys() are the edge labels, in the order the variables were created (i < j) like [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        for (i, j), val in zip(self.x.keys(), self.m.getAttr("X", self._vars)):
            if val > 0.5: # If teh edge (i,j) is selected
                adj[i].append(j)
                adj[j].append(i)
        tour, prev, cur = [0], -1, 0
        for _ in range(n - 1):
            for nb in adj[cur]: # nb is the city pointing from curr to the next city
                if nb != prev:
                    tour.append(nb)
                    prev, cur = cur, nb
                    break
        z = np.zeros(self.inst.p)
        for q in range(self.inst.p):
            z[q] = sum(self.inst.dist[q, tour[k], tour[(k + 1) % n]] for k in range(n))
        return z #f1,f2,f3
        
    def close(self):
        self.m.dispose()
        self.env.dispose()
        
# =============================================================================
# 4. SCALE AND ExA DUMMY POINTS
# =============================================================================

def z_max(inst: Instance) -> float:
    """Largest possible |objective value|, exactly as ExA computes it.

    knapsack: max_q sum_j |C_qj|      (KS_P3/v8.py)
    TSP:      n * largest distance     (TSP_P3/tsp_exa_v2.py)
    Also used to normalise every geometric feature, so that features are
    comparable across instance sizes.
    """
    if inst.kind == "knapsack":
        return float(np.abs(inst.C).sum(axis=1).max())
    return float(inst.n * max(np.abs(inst.dist[q]).max() for q in range(inst.p)))


def exa_dummies(inst: Instance) -> list:
    """ExA's p artificial points m^q that start the search (p=3 only).
    knapsack: m^q = M e_q                        (v8.py)
    TSP:      m^q = M e_q - 2 z_max (1,...,1)    (tsp_exa_v2.py; shifted because TSP objective values are negative)
    with M = 1e5 * z_max, i.e. far outside the objective space, so the very
    first stage (the triangle of the dummies) lies above every real point.
    """
    zm = z_max(inst)
    M = 1e5 * zm
    eye = np.eye(inst.p)
    if inst.kind == "knapsack":
        return [M * eye[q] for q in range(inst.p)]
    return [M * eye[q] - 2.0 * zm * np.ones(inst.p) for q in range(inst.p)]
    
# For M,0,0 and 0,M,0 and 0,0,M
# For TSP M- 2.0 * zm~M, - 2.0 * zm,- 2.0 * zm and - 2.0 * zm,M- 2.0 * zm~M,- 2.0 * zm and - 2.0 * zm,- 2.0 * zm,M- 2.0 * zm~M why because
# since coeff are negative M,0,0 will be better than any real point in all objectives, We wany 1 component of the dummy to be better in 1 objective
# and other 2 components to be worse. To make it worse we use -2*z_max