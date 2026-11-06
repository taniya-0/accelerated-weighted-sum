"""
eval_main.py -- main results table (SSA vs Exact vs Heuristic)
===============================================================
For every problem, size n and test instance k = 0..9 (the SAME instances as
the DRL eval scripts: make_test_instance(n, k)):

  E (Exact)     CAN (p=2) / ExA (p=3) run to completion. Its ESN set is the
                ground truth (see eval_common.exact_truth).
  SSA           SSAv3 (learned facet search on the convex hull, ssa.py) with
                the calibrated tau.
  H (Heuristic) B = |Y_E| weights spread uniformly over the simplex (continuous,
                eval_common.uniform_weights), each solved once.

Metrics (pooled over the N instances of a size, i.e. sum / sum):
  m                 true ESN points identified (per instance)
  Recall            m_method / m_E
  Precision         true points in the method's output / size of its output
  k                 Gurobi weighted-sum solves
  SP-Ratio          k / m (solves per true ESN point found)
  t                 wall-clock time per instance (Gurobi model built before the clock)
  t-Ratio           t_method / t_E
  TPP               1000 * t / m  (milliseconds per true ESN point found)

Outputs (SSAv3/results/main/):
  <problem>_per_instance.csv   one row per instance (also the ExA-with-Rule-2 numbers, v8_*)
  <problem>_by_n.csv           the pooled metrics per n
  <problem>_table.tex          the table rows, ready to paste
  <problem>_ssa_weights.csv    every weight SSA solved (for decomp_ssa.py, plot_weights.py)

Usage:
  python eval_main.py --problem ks_p2
  python eval_main.py --problem all
  python eval_main.py --problem tsp_p3 --sizes 5 10 --instances 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eval_common as ec
from ssa import Scorer, load_tau, point_key, run_ssa
from problems import PROBLEMS, Problem, test_instance

OUT_DIR = ec.HERE / "results" / "main"
PROBLEM_TITLE = {"ks_p2": r"Knapsack, $p=2$", "ks_p3": r"Knapsack, $p=3$",
                 "tsp_p2": r"TSP, $p=2$", "tsp_p3": r"TSP, $p=3$"}


# =============================================================================
# ONE INSTANCE
# =============================================================================

def evaluate_instance(problem: Problem, inst, scorer: Scorer, tau: float, *,
                      with_v8: bool, trial: int = 0):
    """All three methods on one instance. Returns (row dict, list of SSA weight rows)."""
    row = {"problem": problem.key, "scenario": inst.scenario, "n": inst.n, "k": trial,
           "instance_id": inst.instance_id}

    # --- E: ground truth (exact algorithm, same solver as SSA) ---
    truth = ec.exact_truth(problem, inst)
    true_pts = truth.points
    row.update(exact_m=len(true_pts), exact_k=truth.n_solves, exact_t=truth.time,
               exact_complete=truth.complete)

    # --- ExA with Rule 2 (= v8.py / tsp_exa_v2.py: same points and solves) NOT USED IGNORE THIS---
    if with_v8:
        v8 = ec.reference_v8(problem, inst)
        row.update(v8_m=v8["n_esn"], v8_k=v8["n_solves"], v8_t=v8["time"],
                   v8_points_all_true=len(ec.matched_indices(v8["points"], true_pts)) == len(v8["points"]))

    # --- SSA ---
    res = run_ssa(problem, inst, scorer, tau, record=True)
    hit = ec.matched_indices(res.points, true_pts)
    row.update(ssa_out=len(res.points), ssa_matched=len(hit),
               ssa_k=res.n_solves, ssa_t=res.total_time, ssa_solver_t=res.solver_time,
               ssa_hv=ec.hv_fraction([true_pts[i] for i in hit], truth), ssa_stopped_by=res.stopped_by)

    # --- H: |Y_E| uniform weights, each solved once ---
    B = len(true_pts)
    h = ec.solve_weights(problem, inst, ec.uniform_weights(problem.p, B))
    t0 = time.perf_counter()
    h_out = ec.output_set(h.returned)
    h_time = h.time + (time.perf_counter() - t0)            # solves + building the output set
    h_hit = ec.matched_indices(h_out, true_pts)
    row.update(h_budget=B, h_out=len(h_out), h_matched=len(h_hit),
               h_k=h.solves, h_t=h_time, h_hv=ec.hv_fraction([true_pts[i] for i in h_hit], truth))

    # --- every weight SSA solved, with the point it returned ---
    true_keys = {point_key(z) for z in true_pts}
    seen, weight_rows = set(), []
    for rec in res.trace:
        k = None if rec.z is None else point_key(rec.z)
        wr = {"problem": problem.key, "scenario": inst.scenario, "n": inst.n, "instance_id": inst.instance_id,
              "selection_step": rec.step, "score": rec.score, "forced": rec.forced,
              "new_true_esn": bool(k is not None and k in true_keys and k not in seen)}
        if k is not None:
            seen.add(k)
        for q in range(problem.p):
            wr[f"lambda{q + 1}"] = float(rec.lam[q])
            wr[f"z{q + 1}"] = float(rec.z[q]) if rec.z is not None else np.nan
        weight_rows.append(wr)
    return row, weight_rows


# =============================================================================
# POOLED METRICS
# =============================================================================

def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per (problem, scenario, n), exactly as in the paper tables."""
    out = []
    for (prob, scen, n), g in detail.groupby(["problem", "scenario", "n"], sort=True):
        N = len(g)
        r = {"problem": prob, "scenario": scen, "n": n, "instances": N,
             "m_SSA": g["ssa_matched"].sum() / N, "m_E": g["exact_m"].sum() / N, "m_H": g["h_matched"].sum() / N,
             "recall_SSA": ec.pooled(g, "ssa_matched", "exact_m", 100), "recall_H": ec.pooled(g, "h_matched", "exact_m", 100),
             "precision_SSA": ec.pooled(g, "ssa_matched", "ssa_out", 100), "precision_H": ec.pooled(g, "h_matched", "h_out", 100),
             "k_SSA": g["ssa_k"].sum() / N, "k_E": g["exact_k"].sum() / N, "k_H": g["h_k"].sum() / N,
             "sp_SSA": ec.pooled(g, "ssa_k", "ssa_matched"), "sp_E": ec.pooled(g, "exact_k", "exact_m"),
             "sp_H": ec.pooled(g, "h_k", "h_matched"),
             "t_SSA": g["ssa_t"].sum() / N, "t_E": g["exact_t"].sum() / N, "t_H": g["h_t"].sum() / N,
             "t_ratio_SSA": ec.pooled(g, "ssa_t", "exact_t"), "t_ratio_H": ec.pooled(g, "h_t", "exact_t"),
             "tpp_SSA": ec.pooled(g, "ssa_t", "ssa_matched", 1000), "tpp_E": ec.pooled(g, "exact_t", "exact_m", 1000),
             "tpp_H": ec.pooled(g, "h_t", "h_matched", 1000),
             "hv_SSA": g["ssa_hv"].mean(), "hv_H": g["h_hv"].mean()}
        if "v8_m" in g:   # ExA with Rule 2, as v8.py
            r.update(m_v8=g["v8_m"].sum() / N, k_v8=g["v8_k"].sum() / N, t_v8=g["v8_t"].sum() / N)
        out.append(r)
    return pd.DataFrame(out)


def latex_rows(summary: pd.DataFrame, problem: Problem) -> str:
    """The rows of the paper table for one problem (sizes as columns)."""
    s = summary.sort_values("n")
    L = "SSA"

    def row(label, cells):
        return f"{label} & " + " & ".join(cells) + r" \\"

    def f(x, d):
        return "--" if pd.isna(x) else f"{x:.{d}f}"

    lines = [r"\midrule",
             row(rf"\textbf{{{PROBLEM_TITLE[problem.key]}}}", [rf"\textbf{{{int(n)}}}" for n in s["n"]]),
             r"\midrule",
             row(rf"$m$ ({L}/E/H)", [f"{f(a,1)}/{f(b,1)}/{f(c,1)}" for a, b, c in zip(s.m_SSA, s.m_E, s.m_H)]),
             row(rf"Recall (\%) ({L}/H)", [f"{f(a,1)}/{f(b,1)}" for a, b in zip(s.recall_SSA, s.recall_H)]),
             row(rf"Precision (\%) ({L}/H)", [f"{f(a,1)}/{f(b,1)}" for a, b in zip(s.precision_SSA, s.precision_H)]),
             row(rf"$k$ ({L}/E/H)", [f"{f(a,1)}/{f(b,1)}/{f(c,1)}" for a, b, c in zip(s.k_SSA, s.k_E, s.k_H)]),
             row(rf"SP-Ratio ({L}/E/H)", [f"{f(a,2)}/{f(b,2)}/{f(c,2)}" for a, b, c in zip(s.sp_SSA, s.sp_E, s.sp_H)]),
             row(rf"$t$ (s) ({L}/E/H)", [f"{f(a,4)}/{f(b,4)}/{f(c,4)}" for a, b, c in zip(s.t_SSA, s.t_E, s.t_H)]),
             row(rf"$t$-Ratio ({L}/H)", [f"{f(a,3)}/{f(b,3)}" for a, b in zip(s.t_ratio_SSA, s.t_ratio_H)]),
             row(rf"TPP (ms) ({L}/E/H)", [f"{f(a,2)}/{f(b,2)}/{f(c,2)}" for a, b, c in zip(s.tpp_SSA, s.tpp_E, s.tpp_H)])]
    return "\n".join(lines)


def print_summary(summary: pd.DataFrame, problem: Problem):
    bench = problem.exact_name
    print(f"\n=== {problem.key}: SSA vs {bench} (E) vs uniform heuristic (H) ===")
    print(f"{'n':>4} {'m SSA/E/H':>17} {'recall SSA/H':>12} {'prec SSA/H':>12} {'k SSA/E/H':>19} "
          f"{'SP SSA/E/H':>16} {'t(s) SSA/E/H':>26} {'tR SSA/H':>12}")
    for _, r in summary.sort_values("n").iterrows():
        print(f"{int(r.n):>4} {r.m_SSA:>5.1f}/{r.m_E:>5.1f}/{r.m_H:>5.1f} "
              f"{r.recall_SSA:>5.1f}/{r.recall_H:>5.1f} {r.precision_SSA:>5.1f}/{r.precision_H:>5.1f} "
              f"{r.k_SSA:>6.1f}/{r.k_E:>6.1f}/{r.k_H:>5.1f} {r.sp_SSA:>5.2f}/{r.sp_E:>4.2f}/{r.sp_H:>4.2f} "
              f"{r.t_SSA:>8.4f}/{r.t_E:>8.4f}/{r.t_H:>8.4f} {r.t_ratio_SSA:>5.3f}/{r.t_ratio_H:>5.3f}")
        if "m_v8" in r and not pd.isna(r.m_v8):
            print(f"{'':>4} ExA with Rule 2 (v8.py): m={r.m_v8:.1f} k={r.k_v8:.1f} t={r.t_v8:.4f}s")


# =============================================================================
# MAIN
# =============================================================================

def run_problem(problem: Problem, sizes, n_instances: int, tau, with_v8: bool, out_dir: Path):
    scorer = Scorer.load(problem)
    tau = load_tau(problem) if tau is None else float(tau)
    print(f"\n########## {problem.key}: tau = {tau} ##########", flush=True)
    rows, weights = [], []
    for n in sizes:
        for k in range(n_instances):
            row, w = evaluate_instance(problem, test_instance(problem, n, k), scorer, tau, with_v8=with_v8, trial=k)
            row["tau"] = tau
            rows.append(row)
            weights += w
            print(f"  n={n} k={k}: |Y_E|={row['exact_m']} | "
                  f"SSA true={row['ssa_matched']}/{row['ssa_out']} solves={row['ssa_k']} t={row['ssa_t']:.2f}s "
                  f"({row['ssa_stopped_by']}) | E solves={row['exact_k']} t={row['exact_t']:.2f}s | "
                  f"H true={row['h_matched']} solves={row['h_k']}"
                  + (f" | v8 |Y_E|={row['v8_m']} solves={row['v8_k']}" if with_v8 else ""), flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / f"{problem.key}_per_instance.csv", index=False)
    summary.to_csv(out_dir / f"{problem.key}_by_n.csv", index=False)
    pd.DataFrame(weights).to_csv(out_dir / f"{problem.key}_ssa_weights.csv", index=False)
    tex = latex_rows(summary, problem)
    (out_dir / f"{problem.key}_table.tex").write_text(tex + "\n")
    print_summary(summary, problem)
    print("\n" + tex)


def main():
    ap = argparse.ArgumentParser(description="Main table: SSA vs CAN/ExA vs uniform heuristic.")
    ap.add_argument("--problem", choices=["all", *PROBLEMS], default="all")
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--tau", type=float, default=None, help="default: the calibrated tau")
    ap.add_argument("--no-v8", action="store_true", help="skip the ExA-with-Rule-2 (v8.py) run")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(PROBLEMS) if a.problem == "all" else [a.problem]
    for key in keys:
        problem = PROBLEMS[key]
        run_problem(problem, a.sizes or problem.sizes, a.instances, a.tau, not a.no_v8, a.out_dir)
    print(f"\nResults in {a.out_dir}")


if __name__ == "__main__":
    main()
