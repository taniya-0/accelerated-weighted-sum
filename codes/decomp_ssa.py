"""
decomp_ssa.py -- weight-space decomposition with the weights ExA and SSA solved
==============================================================================
For one test instance:
  * regions   the exact weight-space decomposition of the ESN set Y_E: the
              region of y in Y_E is {lambda in the simplex : lambda.y >= lambda.y'
              for every y' in Y_E}, a convex polygon (p=3) or interval (p=2),
              computed from Y_E alone (no grid, no extra solves). Black lines =
              region borders.
  * dots      every weight the exact algorithm (CAN / ExA, run to completion)
              solved that returned a NEW ESN point, coloured by the region of
              that point (one dot per ESN point).
  * circles   the weights SSA (calibrated tau) solved that returned a NEW ESN
              point (--all-ssa-weights: every SSA solve). If an SSA weight
              returned the same point as a dot and lies within --snap-tol of it,
              the dot itself is circled (cleaner picture); otherwise the circle
              is drawn at the SSA weight. So a dot without a circle is an ESN
              point SSA did not find.

Usage:
  python decomp_ssa.py --problem ks_p3 --sizes 50 100
  python decomp_ssa.py --problem tsp_p3 --sizes 15 --k 2
  python decomp_ssa.py --problem ks_p2 --sizes 50          # p=2: a strip over lambda_1
Outputs (SSA/results/decomposition/), per instance:
  decomposition_ssa_<problem>_n<n>_id<id>.png
  decomposition_ssa_<problem>_n<n>_id<id>.csv   every dot and SSA weight drawn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import hsv_to_rgb            # noqa: E402

import eval_common as ec                            # noqa: E402
from ssa import Scorer, load_tau, point_key, run_exact, run_ssa   # noqa: E402
from problems import PROBLEMS, Problem, test_instance             # noqa: E402

OUT_DIR = ec.HERE / "results" / "decomposition"
SNAP_TOL = 0.05          # SSA weight within this distance of a dot (same point) -> circle the dot
VERTEX_TOL = 1e-7


# =============================================================================
# THE EXACT WEIGHT-SPACE DECOMPOSITION
# =============================================================================

def _clip(poly: list, a: np.ndarray, c: float) -> list:
    """Part of a convex polygon (list of 2-D vertices) where a.x + c >= 0."""
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = float(a @ P + c), float(a @ Q + c)
        if fp >= 0:
            out.append(P)
        if (fp >= 0) != (fq >= 0):
            out.append(P + fp / (fp - fq) * (Q - P))
    return out


def regions_3d(points) -> list:
    """Region polygon of every point in the (lambda1, lambda2) plane, lambda3 = 1 - lambda1 - lambda2."""
    Y = np.asarray(points, dtype=np.float64)
    triangle = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    regions = []
    for i, y in enumerate(Y):
        poly = list(triangle)
        for j, other in enumerate(Y):
            if j == i or not poly:
                continue
            d = y - other                                   # lambda.d >= 0
            a = np.array([d[0] - d[2], d[1] - d[2]])
            scale = max(np.abs(a).max(), abs(d[2]), 1e-12)
            poly = _clip(poly, a / scale, d[2] / scale)
        poly = [P for i, P in enumerate(poly) if np.linalg.norm(P - poly[i - 1]) > VERTEX_TOL]
        regions.append(np.asarray(poly) if len(poly) >= 3 else np.zeros((0, 2)))
    return regions


def regions_2d(points) -> list:
    """Interval [lo, hi] of lambda1 (lambda2 = 1 - lambda1) on which each point is optimal."""
    Y = np.asarray(points, dtype=np.float64)
    out = []
    for i, y in enumerate(Y):
        lo, hi = 0.0, 1.0
        for j, other in enumerate(Y):
            if j == i:
                continue
            d = y - other                                   # lambda1 (d1 - d2) + d2 >= 0
            slope, c = d[0] - d[1], d[1]
            if abs(slope) < 1e-12:
                continue
            root = -c / slope
            if slope > 0:
                lo = max(lo, root)
            else:
                hi = min(hi, root)
        out.append((lo, hi))
    return out


def adjacency(regions, p: int) -> list:
    """Neighbouring regions (sharing a border), used to give neighbours distinct colours."""
    n = len(regions)
    nb = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if p == 2:
                touch = min(abs(regions[i][1] - regions[j][0]), abs(regions[j][1] - regions[i][0])) < VERTEX_TOL
            else:
                A, B = regions[i], regions[j]
                touch = (len(A) and len(B) and
                         int((np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) < VERTEX_TOL).any(axis=1).sum()) >= 2)
            if touch:
                nb[i].add(j)
                nb[j].add(i)
    return nb


def region_colors(neighbours: list) -> np.ndarray:
    """One distinct colour per region; each region takes the free colour farthest from its neighbours'."""
    n = len(neighbours)
    if n == 1:
        return np.array([[0.5, 0.5, 0.5]])
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    sat = np.where(np.arange(n) % 2 == 0, 0.80, 0.60)
    val = np.where(np.arange(n) % 3 == 0, 0.95, 0.80)
    palette = hsv_to_rgb(np.column_stack([hues, sat, val]))
    free = list(range(n))
    colors = np.zeros((n, 3))
    done = set()
    for r in sorted(range(n), key=lambda r: -len(neighbours[r])):
        used = [colors[q] for q in neighbours[r] if q in done]
        if used:
            dist = [min(np.linalg.norm(palette[c] - u) for u in used) for c in free]
            pick = free[int(np.argmax(dist))]
        else:
            pick = free[0]
        free.remove(pick)
        colors[r] = palette[pick]
        done.add(r)
    return colors


# =============================================================================
# DOTS (exact algorithm) AND CIRCLES (SSA)
# =============================================================================

def weight_table(problem: Problem, inst, scorer: Scorer, tau: float, snap_tol: float):
    """Exact run + SSA run -> (ESN points, dots dataframe, SSA dataframe)."""
    exact = run_exact(problem, inst, record=True)
    truth = exact.points
    region_of = {point_key(z): i for i, z in enumerate(truth)}
    lam_cols = [f"lambda{q + 1}" for q in range(problem.p)]

    dots, seen = [], set()
    for rec in exact.trace:
        k = None if rec.z is None else point_key(rec.z)
        if k in region_of and k not in seen:              # this solve found a new ESN point
            dots.append({"method": "exact", "step": rec.step, "region": region_of[k],
                         **dict(zip(lam_cols, rec.lam))})
        if k is not None:
            seen.add(k)
    dots = pd.DataFrame(dots)
    dot_lam = dots[lam_cols].to_numpy()

    ssa = run_ssa(problem, inst, scorer, tau, record=True)
    true_keys, seen, rows = set(region_of), set(), []
    for rec in ssa.trace:
        k = None if rec.z is None else point_key(rec.z)
        row = {"method": "ssa", "step": rec.step, "region": region_of.get(k, -1),
               "new_esn": bool(k in true_keys and k not in seen), "score": rec.score,
               **dict(zip(lam_cols, rec.lam))}
        if k is not None:
            seen.add(k)
        # circle the dot of the same point if this weight is close to it
        same = np.nonzero(dots["region"].to_numpy() == row["region"])[0] if row["region"] >= 0 else []
        row["circled_dot"] = -1
        if len(same):
            d = np.linalg.norm(dot_lam[same] - np.asarray(rec.lam), axis=1)
            if d.min() <= snap_tol:
                row["circled_dot"] = int(same[int(d.argmin())])
        rows.append(row)
    return truth, dots, pd.DataFrame(rows), exact, ssa


# =============================================================================
# DRAWING
# =============================================================================

def _circle_xy(dots: pd.DataFrame, ssa: pd.DataFrame, problem: Problem, all_weights: bool) -> np.ndarray:
    """Where the circles go: on the snapped dots, else at the SSA weight (duplicates removed)."""
    if not all_weights:
        ssa = ssa[ssa["new_esn"]]
    cols = ["lambda1", "lambda2"] if problem.p == 3 else ["lambda1"]
    on_dot = dots.iloc[sorted(set(ssa.loc[ssa["circled_dot"] >= 0, "circled_dot"]))][cols].to_numpy()
    off = ssa.loc[ssa["circled_dot"] < 0, cols].to_numpy()
    xy = np.vstack([on_dot, off]) if len(off) else on_dot
    return np.unique(np.round(xy, 9), axis=0) if len(xy) else xy


def _style_simplex(ax):
    ax.plot([1, 0], [0, 1], color="black", linewidth=1.5, zorder=2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax.set_ylabel(r"$\lambda_2$", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_borders_3d(ax, regions):
    def on_triangle(P, Q):
        return any(abs(P[i]) < 1e-9 and abs(Q[i]) < 1e-9 for i in (0, 1)) or \
            (abs(P.sum() - 1) < 1e-9 and abs(Q.sum() - 1) < 1e-9)
    for poly in regions:
        for i in range(len(poly)):
            P, Q = poly[i], poly[(i + 1) % len(poly)]
            if not on_triangle(P, Q):
                ax.plot([P[0], Q[0]], [P[1], Q[1]], color="black", linewidth=0.7, zorder=1)


def plot(problem: Problem, regions, colors, dots: pd.DataFrame, ssa: pd.DataFrame, path: Path,
         all_weights: bool = False):
    circles = _circle_xy(dots, ssa, problem, all_weights)
    dot_colors = colors[dots["region"].to_numpy()]
    if problem.p == 3:
        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_borders_3d(ax, regions)
        ax.scatter(dots["lambda1"], dots["lambda2"], s=60, c=dot_colors, edgecolors="none", zorder=3)
        ax.scatter(circles[:, 0], circles[:, 1], s=190, facecolors="none", edgecolors="black",
                   linewidths=1.8, zorder=4, label="SSA-selected weight")
        _style_simplex(ax)
        ax.legend(loc="upper right", frameon=False, fontsize=12, handletextpad=0.3)
    else:
        fig, ax = plt.subplots(figsize=(9, 1.6))
        ax.plot([0, 1], [0, 0], color="black", linewidth=1.0, zorder=1)
        for lo, hi in regions:
            for b in (lo, hi):
                if 1e-9 < b < 1 - 1e-9:
                    ax.plot([b, b], [-0.35, 0.35], color="black", linewidth=0.7, zorder=1)
        ax.scatter(dots["lambda1"], np.zeros(len(dots)), s=60, c=dot_colors, edgecolors="none", zorder=3)
        ax.scatter(circles[:, 0], np.zeros(len(circles)), s=190, facecolors="none", edgecolors="black",
                   linewidths=1.8, zorder=4, label="SSA-selected weight")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel(r"$\lambda_1$  ($\lambda_2 = 1 - \lambda_1$)", fontsize=12)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.95), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Weight-space decomposition with the ExA and SSA weights.")
    ap.add_argument("--problem", choices=list(PROBLEMS), required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=0, help="test instance index")
    ap.add_argument("--tau", type=float, default=None, help="default: the calibrated tau")
    ap.add_argument("--snap-tol", type=float, default=SNAP_TOL,
                    help="an SSA weight this close to a dot (same point) circles the dot")
    ap.add_argument("--all-ssa-weights", action="store_true",
                    help="circle every SSA solve, not only those that found a new ESN point")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    problem = PROBLEMS[a.problem]
    scorer = Scorer.load(problem)
    tau = load_tau(problem) if a.tau is None else float(a.tau)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for n in a.sizes or problem.sizes:
        inst = test_instance(problem, n, a.k)
        truth, dots, ssa, exact, ssa_res = weight_table(problem, inst, scorer, tau, a.snap_tol)
        regions = regions_3d(truth) if problem.p == 3 else regions_2d(truth)
        colors = region_colors(adjacency(regions, problem.p))
        stem = f"decomposition_ssa_{problem.key}_n{n}_id{inst.instance_id}"
        plot(problem, regions, colors, dots, ssa, a.out_dir / f"{stem}.png", a.all_ssa_weights)
        pd.concat([dots, ssa], ignore_index=True).to_csv(a.out_dir / f"{stem}.csv", index=False)
        print(f"{problem.key} n={n} k={a.k}: |Y_E|={len(truth)} {problem.exact_name} solves={exact.n_solves} | "
              f"SSA (tau={tau}) solves={ssa_res.n_solves} new ESN={int(ssa['new_esn'].sum())} "
              f"circled dots={ssa.loc[ssa['new_esn'] & (ssa['circled_dot'] >= 0), 'circled_dot'].nunique()} "
              f"off-dot new-ESN weights={int((ssa['new_esn'] & (ssa['circled_dot'] < 0)).sum())} "
              f"-> {a.out_dir / stem}.png")


if __name__ == "__main__":
    main()
