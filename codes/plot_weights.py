"""
plot_weights.py -- the weights ExA, SSA and the uniform heuristic solve on one instance
============================================================================================
One diagram per method, on the exact weight-space decomposition of the instance
(black lines: borders between the regions of the ESN points, see decomp_ssa.py):
  * filled dot, coloured by region: a solve that returned a NEW ESN point
  * hollow grey circle:             a solve that returned nothing new
                                    (an ESN point already found, or a non-ESN point)

  ExA      the exact algorithm (CAN for p=2, ExA for p=3), run here to completion
  SSA      the weights SSA solved in eval_main.py (results/main/<problem>_ssa_weights.csv)
  H        |Y_E| uniformly spread weights (as in eval_main.py), solved here

Usage (after eval_main.py):
  python plot_weights.py --problem ks_p3 --n 100 --k 0
Outputs (results/weights/): weights_<method>_<problem>_n<n>_k<k>.png and .pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

import eval_common as ec                            # noqa: E402
from decomp_ssa import _draw_borders_3d, _style_simplex, adjacency, region_colors, regions_2d, regions_3d  # noqa: E402
from ssa import point_key, run_exact                # noqa: E402
from problems import PROBLEMS, test_instance        # noqa: E402

OUT_DIR = ec.HERE / "results" / "weights"
NOTHING_NEW = "#9a9893"


def classify(weights, points, truth_keys, region_of):
    """One row per solve: weight, region of the returned point, and whether it was a new ESN point."""
    seen, rows = set(), []
    for lam, z in zip(weights, points):
        k = None if z is None else point_key(z)
        new = k is not None and k in truth_keys and k not in seen
        if k is not None:
            seen.add(k)
        rows.append({"lam": np.asarray(lam, dtype=float), "region": region_of.get(k, -1), "new": new})
    return rows


def ssa_rows(csv: Path, problem, n, instance_id, truth_keys, region_of):
    df = pd.read_csv(csv)
    df = df[(df["n"] == n) & (df["instance_id"] == instance_id)].sort_values("selection_step")
    if df.empty:
        raise SystemExit(f"no SSA weights for n={n}, instance {instance_id} in {csv}; run eval_main.py first")
    lam_cols = [f"lambda{q + 1}" for q in range(problem.p)]
    z_cols = [f"z{q + 1}" for q in range(problem.p)]
    weights = df[lam_cols].to_numpy(dtype=float)
    points = [None if np.isnan(r).any() else r for r in df[z_cols].to_numpy(dtype=float)]
    return classify(weights, points, truth_keys, region_of)


def draw(problem, regions, colors, rows, title, path):
    new = [r for r in rows if r["new"]]
    old = [r for r in rows if not r["new"]]
    if problem.p == 3:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _draw_borders_3d(ax, regions)
        if old:
            L = np.array([r["lam"] for r in old])
            ax.scatter(L[:, 0], L[:, 1], s=34, facecolors="none", edgecolors=NOTHING_NEW, linewidths=1.0,
                       zorder=3, label=f"no new ESN point ({len(old)})")
        if new:
            L = np.array([r["lam"] for r in new])
            ax.scatter(L[:, 0], L[:, 1], s=46, c=colors[[r["region"] for r in new]], edgecolors="none",
                       zorder=4, label=f"new ESN point ({len(new)})")
        _style_simplex(ax)
    else:
        fig, ax = plt.subplots(figsize=(9, 1.8))
        ax.plot([0, 1], [0, 0], color="black", linewidth=1.0, zorder=1)
        for lo, hi in regions:
            for b in (lo, hi):
                if 1e-9 < b < 1 - 1e-9:
                    ax.plot([b, b], [-0.35, 0.35], color="black", linewidth=0.7, zorder=1)
        if old:
            L = np.array([r["lam"] for r in old])
            ax.scatter(L[:, 0], np.zeros(len(L)), s=34, facecolors="none", edgecolors=NOTHING_NEW,
                       linewidths=1.0, zorder=3, label=f"no new ESN point ({len(old)})")
        if new:
            L = np.array([r["lam"] for r in new])
            ax.scatter(L[:, 0], np.zeros(len(L)), s=46, c=colors[[r["region"] for r in new]],
                       edgecolors="none", zorder=4, label=f"new ESN point ({len(new)})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel(r"$\lambda_1$", fontsize=12)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="#3d3d3a",
                      markeredgecolor="none", label=f"new ESN point, coloured by its region ({len(new)})"),
               Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="none",
                      markeredgecolor=NOTHING_NEW, label=f"no new ESN point ({len(old)})")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Weights solved by ExA, SSA and the uniform heuristic on one instance.")
    ap.add_argument("--problem", choices=list(PROBLEMS), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=0, help="test instance index")
    ap.add_argument("--weights-csv", type=Path, default=None,
                    help="default: results/main/<problem>_ssa_weights.csv")
    ap.add_argument("--name", default="SSA", help="name of the learned method in the titles")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    problem = PROBLEMS[a.problem]
    inst = test_instance(problem, a.n, a.k)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    exact = run_exact(problem, inst, record=True)
    truth = exact.points
    region_of = {point_key(z): i for i, z in enumerate(truth)}
    truth_keys = set(region_of)
    regions = regions_3d(truth) if problem.p == 3 else regions_2d(truth)
    colors = region_colors(adjacency(regions, problem.p))
    m = len(truth)

    rows_exa = classify([r.lam for r in exact.trace], [r.z for r in exact.trace], truth_keys, region_of)
    csv = a.weights_csv or ec.HERE / "results" / "main" / f"{problem.key}_ssa_weights.csv"
    rows_ssa = ssa_rows(csv, problem, a.n, inst.instance_id, truth_keys, region_of)
    W = ec.uniform_weights(problem.p, m)
    rows_h = classify(W, ec.solve_weights(problem, inst, W).returned, truth_keys, region_of)

    stem = f"{problem.key}_n{a.n}_k{a.k}"
    for key, label, rows in (("exa", problem.exact_name, rows_exa), ("ssa", a.name, rows_ssa),
                             ("uniform", "Uniform heuristic", rows_h)):
        found = sum(r["new"] for r in rows)
        title = f"{label}: {len(rows)} solves, {found} of {m} ESN points"
        draw(problem, regions, colors, rows, title, a.out_dir / f"weights_{key}_{stem}")
        print(f"{title} -> {a.out_dir / f'weights_{key}_{stem}'}.png/.pdf")


if __name__ == "__main__":
    main()
