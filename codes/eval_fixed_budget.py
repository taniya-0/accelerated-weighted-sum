"""
eval_fixed_budget.py -- same number of solves Q for every method (+ distribution shift)
=====================================================================================
Five methods get the same budget of Q weighted-sum solves:

  E    CAN / ExA from scratch, force-stopped after Q solves.
  H    Q weights spread uniformly over the simplex, each solved once.
  HB   the weight space split in bisection order (extremes, 1/2, 1/4 and 3/4,...) until Q weights are solved.
  A    largest-area heuristic (the SSAv3 facet search, largest open facetfirst), stopped after Q solves.
  SSA  stopped after Q solves (tau-stop off).

The budget Q is the target of eval_fixed_target.py for the same problem, size and scenario: Q = max(2, floor(0.75 * mean |Y_E|)) over the fixed-target
instances. It is read from results/fixed_target/<problem>_fixed_target_summary.csv when eval_fixed_target.py has been run, otherwise computed the same way
(eval_common.fixed_target_q: exact runs on the fixed-target instances).

Every point returned by a solve is matched against the full ESN set: found = distinct true ESN points,
recall = found / |Y_E|. Matching time is excluded. 10 instances per size and scenario (seed series eval_common.BUDGET_SEED).

Distribution shift: --scenarios all runs every scenario (each with the Q of its
own fixed-target race); the summary then also gives, per method,
Delta_max = ID value - worst shifted value (percentage points of recall and HV).

How to run:
  python eval_fixed_target.py --problem ks_p2      # first: the budget Q is read from its summary
  python eval_fixed_budget.py --problem ks_p2
  python eval_fixed_budget.py --problem all --scenarios all
  python eval_fixed_budget.py --latex > fixed_budget_table.tex    # paper table (training distribution)
Outputs (SSAv3/results/fixed_budget/): <problem>_fixed_budget_detail.csv, _summary.csv, _shift.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import eval_common as ec
from problems import PROBLEMS, Problem
from ssa import Scorer

OUT_DIR = ec.HERE / "results" / "fixed_budget"
TARGET_DIR = ec.HERE / "results" / "fixed_target"
METHODS = ("E", "H", "HB", "A", "SSA")


# =============================================================================
# THE BUDGET Q (= the fixed-target Q)
# =============================================================================

def budget_q(problem: Problem, n: int, scenario: str, target_csv: Path, target_instances: int) -> int:
    if target_csv.is_file():
        s = pd.read_csv(target_csv)
        row = s[(s["scenario"] == scenario) & (s["n"] == n)]
        if len(row):
            return int(row["Q"].iloc[0])
    print(f"  Q for n={n} [{scenario}] not in {target_csv.name}: computing it as eval_fixed_target.py does", flush=True)
    return ec.fixed_target_q(problem, n, scenario, target_instances)


# =============================================================================
# ONE INSTANCE, ALL METHODS
# =============================================================================

def run_instance(problem: Problem, inst, trial: int, Q: int, scorer: Scorer) -> list:
    truth = ec.exact_truth(problem, inst)
    rows = []

    def row(method, tracker, solves, time_s):
        found = len(tracker.matched)
        return {"problem": problem.key, "scenario": inst.scenario, "n": inst.n, "trial": trial,
                "instance_id": inst.instance_id, "Q": Q, "method": method, "exact_m": len(truth.points),
                "solves": int(solves), "time": float(time_s), "found": found,
                "recall": found / max(len(truth.points), 1),
                "hv_fraction": ec.hv_fraction(tracker.matched, truth),
                "spurious": len(tracker.spurious)}

    # E (ExA / CAN), A and SSA (facet search), stopped after Q solves
    for method in ("E", "A", "SSA"):
        tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
        res = ec.make_search(method, problem, inst, scorer, max_solves=Q,
                             on_solve=ec.observe_only(tr)).run(postprocess=False)
        rows.append(row(method, tr, res.n_solves, res.search_time - tr.check_time))

    # H: Q uniform weights
    tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
    w = ec.solve_weights(problem, inst, ec.uniform_weights(problem.p, Q), on_solve=ec.observe_only(tr))
    rows.append(row("H", tr, w.solves, w.time - w.check_time))

    # HB: bisection order until Q weights are solved
    tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
    w = ec.solve_weights(problem, inst, ec.bisection_weights(problem.p)[:Q], on_solve=ec.observe_only(tr))
    rows.append(row("HB", tr, w.solves, w.time - w.check_time))
    return rows


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["problem", "scenario", "n", "Q", "method"]
    return (detail.groupby(keys, sort=True)
            .agg(instances=("trial", "count"), exact_m=("exact_m", "mean"), solves=("solves", "mean"),
                 found=("found", "mean"), recall=("recall", "mean"), hv_fraction=("hv_fraction", "mean"),
                 time=("time", "mean"), spurious=("spurious", "mean"))
            .reset_index())


def shift_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Delta_max per method and size: ID value - worst shifted value (percentage points)."""
    out = []
    for (prob, n, method), g in summary.groupby(["problem", "n", "method"]):
        idv = g[g["scenario"] == "id"]
        shifted = g[g["scenario"] != "id"]
        if idv.empty or shifted.empty:
            continue
        worst_r = shifted.loc[shifted["recall"].idxmin()]
        worst_h = shifted.loc[shifted["hv_fraction"].idxmin()]
        out.append({"problem": prob, "n": n, "method": method,
                    "recall_id": 100 * idv["recall"].iloc[0], "recall_worst": 100 * worst_r["recall"],
                    "recall_worst_scenario": worst_r["scenario"],
                    "delta_max_recall_pp": 100 * (idv["recall"].iloc[0] - worst_r["recall"]),
                    "hv_id": 100 * idv["hv_fraction"].iloc[0], "hv_worst": 100 * worst_h["hv_fraction"],
                    "hv_worst_scenario": worst_h["scenario"],
                    "delta_max_hv_pp": 100 * (idv["hv_fraction"].iloc[0] - worst_h["hv_fraction"])})
    return pd.DataFrame(out)


def print_table(summary: pd.DataFrame):
    for (prob, scen), s in summary.groupby(["problem", "scenario"], sort=False):
        print(f"\n=== fixed budget: {prob} [{scen}]  (found / HV: " + "/".join(METHODS) + ") ===")
        for (n, Q), g in s.groupby(["n", "Q"]):
            m = {r.method: r for r in g.itertuples()}
            if not all(k in m for k in METHODS):
                continue
            print(f"{n:>4} Q={Q:>4} |Y_E|={m['E'].exact_m:>6.1f} | found "
                  + "/".join(f"{m[k].found:.1f}" for k in METHODS) + " | HV "
                  + "/".join(f"{m[k].hv_fraction:.3f}" for k in METHODS))


# =============================================================================
# MAIN
# =============================================================================

def run_problem(problem: Problem, sizes, scenarios, n_instances: int, target_instances: int, out_dir: Path):
    target_csv = TARGET_DIR / f"{problem.key}_fixed_target_summary.csv"
    scorer = Scorer.load(problem)
    rows = []
    for scen in scenarios:
        for n in sizes:
            Q = budget_q(problem, n, scen, target_csv, target_instances)
            print(f"\n{problem.key} [{scen}] n={n}: budget Q={Q} (fixed-target Q)", flush=True)
            for t in range(n_instances):
                inst = ec.make_instance(problem, n, t, scen, ec.BUDGET_SEED)
                r = run_instance(problem, inst, t, Q, scorer)
                rows += r
                print(f"  trial {t}: |Y_E|={r[0]['exact_m']} | " + " | ".join(
                    f"{x['method']} found={x['found']} hv={x['hv_fraction']:.3f}" for x in r), flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / f"{problem.key}_fixed_budget_detail.csv", index=False)
    summary.to_csv(out_dir / f"{problem.key}_fixed_budget_summary.csv", index=False)
    print_table(summary)
    if len(scenarios) > 1:
        shift = shift_table(summary)
        shift.to_csv(out_dir / f"{problem.key}_fixed_budget_shift.csv", index=False)
        print(f"\n=== distribution shift: {problem.key} (Delta_max = ID - worst shifted, percentage points) ===")
        print(shift.to_string(index=False, float_format=lambda v: f"{v:.1f}"))


# =============================================================================
# LATEX: the paper table (training distribution), knapsack | TSP
# =============================================================================

def latex_rows(out_dir: Path, key: str) -> list:
    path = out_dir / f"{key}_fixed_budget_summary.csv"
    if not path.is_file():
        return []
    s = pd.read_csv(path)
    s = s[s["scenario"] == "id"]
    rows = []
    for (n, Q), g in s.groupby(["n", "Q"], sort=True):
        m = {r.method: r for r in g.itertuples()}
        found = ec.bold_best([ec.fmt(m[k].found) if k in m else "--" for k in METHODS], "max")
        #hv = ec.bold_best([ec.fmt(100 * m[k].hv_fraction) if k in m else "--" for k in METHODS], "max")
        hv = ec.bold_best([ec.fmt(ec.hv_gap(m[k].hv_fraction), ec.HV_GAP_DECIMALS) if k in m else "--" for k in METHODS], "min")
        rows.append([f"{n}", f"{Q}"] + found + hv)
    return rows


def latex_table(out_dir: Path) -> str:
    nc = 12
    head = r"$n$ & $Q$ & E & H & HB & A & SSA & E & H & HB & A & SSA"
    lines = [r"\begin{table*}[!ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{1.9pt}",
             r"\caption{Fixed budget of $Q$ solves: reference ESN points found and their HV gap; "
             + ec.HV_GAP_DEF + r". E: CAN/ExA stopped after $Q$ solves; H: $Q$ uniform weights; HB: bisection until $Q$ "
             r"weights are solved; A: largest-area heuristic stopped after $Q$ solves; SSA: stopped after $Q$ "
             r"solves ($\tau=0$). Most points and lowest HV gap in bold.}",
             r"\label{tab:budget_id}",
             r"\begin{tabular}{@{}rrrrrrrrrrrr@{\hspace{6pt}}rrrrrrrrrrrr@{}}", r"\toprule",
             r"\multicolumn{12}{c}{\textbf{Knapsack}} & \multicolumn{12}{c}{\textbf{TSP}} \\",
             r"\cmidrule(r{6pt}){1-12}\cmidrule{13-24}",
             r" & & \multicolumn{5}{c}{ESN points found} & \multicolumn{5}{c}{HV gap (\%)} &  & & "
             r"\multicolumn{5}{c}{ESN points found} & \multicolumn{5}{c}{HV gap (\%)} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){15-19}\cmidrule(lr){20-24}",
             head + " & " + head + r" \\", r"\midrule"]
    for p in (2, 3):
        lines.append(rf"\multicolumn{{12}}{{l}}{{\textit{{$p={p}$}}}} & \multicolumn{{12}}{{l}}{{\textit{{$p={p}$}}}} \\")
        lines += ec.side_by_side(latex_rows(out_dir, f"ks_p{p}"), latex_rows(out_dir, f"tsp_p{p}"), nc)
        if p == 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Fixed budget: E / H / HB / A / SSA.")
    ap.add_argument("--problem", choices=["all", *PROBLEMS], default="all")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="default: training sizes")
    ap.add_argument("--scenarios", nargs="+", default=["id"],
                    help="'all' = id, the shifted distributions and the clustered placements")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--target-instances", type=int, default=ec.TARGET_INSTANCES,
                    help="instances used to compute Q when the fixed-target summary is missing")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--latex", action="store_true", help="print the paper table from the summaries and exit")
    a = ap.parse_args()
    if a.latex:
        print(latex_table(a.out_dir))
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for key in (list(PROBLEMS) if a.problem == "all" else [a.problem]):
        problem = PROBLEMS[key]
        scenarios = ec.all_scenarios(problem) if a.scenarios == ["all"] else a.scenarios
        run_problem(problem, a.sizes or list(problem.sizes), scenarios, a.instances, a.target_instances, a.out_dir)
    print(f"\nResults in {a.out_dir}")


if __name__ == "__main__":
    main()
