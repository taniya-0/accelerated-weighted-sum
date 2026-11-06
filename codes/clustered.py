"""
clustered.py -- random (R), clustered (C) and random-clustered (RC) instances
============================================================================
  R   every value uniform on the grid (the training distribution)
  C   S ~ U{3,...,8} points are placed uniformly as cluster seeds; every other
      point is drawn uniformly and accepted with probability
      min(1, sum_s exp(-d(point, seed_s) / DECAY)), so points pile up around
      the seeds with an exponentially decaying density
  RC  half of the points as in R, the other half as in C

Every objective gets its OWN clusters (own S, own seeds), so the objectives stay independent of each other, as in the training data:

TSP: applied to the city coordinates of every objective separately, on the
training grid {0,...,999}^2, with DECAY = 40 .

Knapsack: the profit vector of every objective (c_q1, ..., c_qn) and the weight vector (w_1, ..., w_n) are each clustered on their own, as n points on the training
range {60,...,100} , with DECAY_KS = 2. In 1-D an accepted value lies at a typical distance of about DECAY from its seed: 2-4 of a range of 41, i.e.
clusters of the same relative tightness as the TSP (40-80 of 1000).Both the DRL evaluation (drl_eval_clustered.py) and SSA (eval_common.make_instance)
call generate(), so they see identical instances: same seeds, same instance ids.
"""
from __future__ import annotations

import numpy as np

SCENARIOS = ("clustered", "random_clustered")
SCENARIO_INDEX = {"clustered": 6, "random_clustered": 7}   # after the six OOD scenarios
GRID_TSP = (0, 999)
GRID_KS = (60, 100)
DECAY_TSP = 40.0
DECAY_KS = 2.0
SEEDS_RANGE = (3, 8)


def _clustered_points(rng, m: int, dim: int, low: int, high: int, decay: float) -> np.ndarray:
    """m integer points in {low..high} dim: S uniform seeds, the rest attracted to them."""
    if m <= 0:
        return np.zeros((0, dim), dtype=np.int64)
    S = int(min(m, rng.integers(SEEDS_RANGE[0], SEEDS_RANGE[1] + 1)))
    seeds = rng.integers(low, high + 1, size=(S, dim)).astype(np.float64)
    out = [seeds]                                       # the seeds are points of the instance too
    need = m - S
    while need > 0:                                     # rejection sampling around the seeds
        cand = rng.integers(low, high + 1, size=(max(64 * need, 20_000), dim)).astype(np.float64)
        d = np.sqrt(((cand[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2))
        prob = np.minimum(1.0, np.exp(-d / decay).sum(axis=1))
        keep = cand[rng.random(len(cand)) < prob][:need]
        out.append(keep)
        need -= len(keep)
    return np.vstack(out).astype(np.int64)


def _points(rng, scenario: str, m: int, dim: int, low: int, high: int, decay: float) -> np.ndarray:
    """m points of one scenario (C or RC), in random order."""
    if scenario == "clustered":
        pts = _clustered_points(rng, m, dim, low, high, decay)
    elif scenario == "random_clustered":
        half = m // 2
        uniform = rng.integers(low, high + 1, size=(half, dim))
        pts = np.vstack([uniform, _clustered_points(rng, m - half, dim, low, high, decay)])
    else:
        raise ValueError(scenario)
    return pts[rng.permutation(m)]                      # no ordering by cluster


def instance_id(p: int, scenario: str, trial: int) -> int:
    return 90_000_000 + int(p) * 1_000_000 + SCENARIO_INDEX[scenario] * 10_000 + int(trial)


def generate(kind: str, p: int, n: int, trial: int, scenario: str, base_seed: int) -> dict:
    """Instance data. kind: 'knapsack' -> C, weights, capacity; 'tsp' -> coords, dist (negated)."""
    ss = np.random.SeedSequence([int(base_seed), int(p), int(n), int(trial), SCENARIO_INDEX[scenario]])
    rng = np.random.default_rng(ss)
    if kind == "knapsack":
        # p profit vectors and one weight vector, each with its own clusters (1-D values)
        C = np.stack([_points(rng, scenario, n, 1, *GRID_KS, DECAY_KS)[:, 0] for _ in range(p)])
        weights = _points(rng, scenario, n, 1, *GRID_KS, DECAY_KS)[:, 0]
        return {"C": C, "weights": weights, "capacity": int(np.floor(0.5 * weights.sum()))}
    # TSP: p sets of city coordinates, each with its own clusters (2-D points)
    coords = np.stack([_points(rng, scenario, n, 2, *GRID_TSP, DECAY_TSP) for _ in range(p)])
    dist = np.zeros((p, n, n))
    for q in range(p):
        diff = coords[q, :, None, :] - coords[q, None, :, :]
        dist[q] = np.floor(np.sqrt((diff ** 2).sum(axis=-1)) + 0.5)
    return {"coords": coords, "dist": -dist}
