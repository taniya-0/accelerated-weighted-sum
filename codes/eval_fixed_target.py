"""
eval_fixed_target.py -- how many solves does each method need to find Q true ESN points?
=======================================================================================
For every instance the exact algorithm is first run to completion; its ESN set Y_E is the reference set. Then, from scratch:

  E   CAN / ExA in its own (depth-first) order, stopped as soon as its solves have returned Q distinct points of Y_E.
  SSA with tau = 0 (no early stop), stopped as soon as Q distinct points of Y_E are found.
  A   always solving the open facet with the largest triangle area (p=3) / segment length (p=2) next, stopped at Q points (see eval_common.AreaScorer).
  HB  bisection weights (extremes, 1/2, 1/4, 3/4, ...) until Q reference points are found, or failure once the bisection levels are used up
      (eval_common.MAX_BISECTION_WEIGHTS).
  H   exactly Q weights spread uniformly over the simplex, each solved once. how many reference points do these Q solves return? ("Found")

Q = max(2, floor(0.75 * mean |Y_E|)) of the size, measured on the same instances(eval_common.target_q); eval_fixed_budget.py uses the same Q as its budget.
Instances with |Y_E| < Q are excluded for all methods. Matching time is excluded from every time; SSA time includes the scorer. 10 instances per size from the
fixed-target seed series (eval_common.TARGET_SEED); on the training distribution also sizes not seen in training (dagger in the table).


How to run:
  python eval_fixed_target.py --problem ks_p2                     # training distribution
  python eval_fixed_target.py --problem all --scenarios all       # + shifted and clustered data
  python eval_fixed_target.py --latex > fixed_target_table.tex    # paper table (training distribution)
Outputs (SSA/results/fixed_target/): <problem>_fixed_target_detail.csv, _summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import eval_common as ec
from problems import PROBLEMS, Problem
from ssa import Scorer

OUT_DIR = ec.HERE / "results" / "fixed_target"
SEARCH_METHODS = ("E", "SSA", "A")                  # ExA / CAN, and the facet search in two orders
METHODS = (*SEARCH_METHODS, "HB", "H")


def race_row(method, problem, inst, trial, Q, truth, tracker, solves, time_s, reached):
    """One result row; HV is that of the reference points found.
    One result row: method, Q, reached?, solves, time, found, HV fraction"""
    return {"problem": problem.key, "scenario": inst.scenario, "n": inst.n, "trial": trial,
            "instance_id": inst.instance_id, "unseen": inst.n not in problem.sizes, "method": method,
            "Q": Q, "exact_m": len(truth.points), "eligible": True,
            "reached": bool(reached), "solves": int(solves), "time": float(time_s),
            "found": len(tracker.matched), "hv_fraction": ec.hv_fraction(tracker.matched, truth)}


def run_instance(problem: Problem, inst, trial: int, Q: int, truth, scorer: Scorer) -> list:
    """All methods on one eligible instance. Matching time is subtracted from every time.
    postprocess = False because a race needs no final output filtering, because the tracker already counted the truth points found."""
    rows = []

    # E (ExA / CAN), SSA and A (facet search, scorer / largest-area order), raced to Q
    for method in SEARCH_METHODS:
        tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
        res = ec.make_search(method, problem, inst, scorer, on_solve=ec.stop_at_target(tr, Q)).run(postprocess=False)
        rows.append(race_row(method, problem, inst, trial, Q, truth, tr, res.n_solves, res.search_time - tr.check_time, res.stopped_by == "target"))

    # HB: bisection order until Q reference points are found
    tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
    w = ec.solve_weights(problem, inst, ec.bisection_weights(problem.p), on_solve=ec.stop_at_target(tr, Q))
    rows.append(race_row("HB", problem, inst, trial, Q, truth, tr, w.solves, w.time - w.check_time, len(tr.matched) >= Q))

    # H: exactly Q uniform weights, all solved
    tr = ec.RaceTracker(truth.points, ec.MATCH_TOL)
    w = ec.solve_weights(problem, inst, ec.uniform_weights(problem.p, Q), on_solve=ec.observe_only(tr))
    rows.append(race_row("H", problem, inst, trial, Q, truth, tr, w.solves, w.time - w.check_time, len(tr.matched) >= Q))
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Means per (scenario, n, method). Solves, time and HV of E/SSA/A/HB: over the instances that reached Q."""
    out = []
    keys = ["problem", "scenario", "n", "unseen", "Q", "method"]
    for key, g in detail[detail["eligible"]].groupby(keys, sort=True):
        reached = g[g["reached"]] if key[-1] != "H" else g
        r = dict(zip(keys, key))
        r.update(instances=len(g), exact_m=g["exact_m"].mean(), success_rate=g["reached"].mean(),
                 solves=reached["solves"].mean(), time=reached["time"].mean(), found=g["found"].mean(),
                 hv_fraction=reached["hv_fraction"].mean())
        out.append(r)
    return pd.DataFrame(out)


def print_table(summary: pd.DataFrame):
    for (prob, scen), s in summary.groupby(["problem", "scenario"], sort=False):
        print(f"\n=== fixed target: {prob} [{scen}]  (solves to Q: E/SSA/A/HB; H = found with Q solves) ===")
        for (n, unseen, Q), g in s.groupby(["n", "unseen", "Q"], sort=True):
            m = {r.method: r for r in g.itertuples()}
            if not all(k in m for k in METHODS):
                continue
            print(f"{n:>4}{'+' if unseen else ' '} Q={Q:>4} |Y_E|={m['E'].exact_m:>6.1f} | solves "
                  + "/".join(f"{m[k].solves:.1f}" for k in ("E", "SSA", "A", "HB"))
                  + f" | H found {m['H'].found:.1f} | HV " + "/".join(f"{m[k].hv_fraction:.3f}" for k in METHODS)
                  + f" | HB success {m['HB'].success_rate:.0%}")
    print("  (+ = size not used in training)")


def run_problem(problem: Problem, plan: dict, n_instances: int, out_dir: Path):
    """plan: {scenario: sizes}."""
    scorer = Scorer.load(problem)
    rows = []
    for scen, sizes in plan.items():
        for n in sizes:
            # pass 1: reference set of every instance, and Q
            insts = [ec.make_instance(problem, n, t, scen, ec.TARGET_SEED) for t in range(n_instances)]
            truths = [ec.exact_truth(problem, inst) for inst in insts]
            mean_m = float(np.mean([len(t.points) for t in truths]))
            Q = ec.target_q(mean_m)
            print(f"\n{problem.key} [{scen}] n={n}{'+' if n not in problem.sizes else ''}: "
                  f"mean |Y_E|={mean_m:.1f} -> Q={Q}", flush=True)
            # pass 2: every method on every eligible instance
            for t, (inst, truth) in enumerate(zip(insts, truths)):
                if not truth.complete or len(truth.points) < Q:
                    rows += [{"problem": problem.key, "scenario": scen, "n": n, "trial": t,
                              "instance_id": inst.instance_id, "unseen": n not in problem.sizes,
                              "method": m, "Q": Q, "exact_m": len(truth.points), "eligible": False}
                             for m in METHODS]
                    print(f"  trial {t}: |Y_E|={len(truth.points)} < Q, excluded")
                    continue
                r = run_instance(problem, inst, t, Q, truth, scorer)
                rows += r
                print("  trial " + str(t) + ": " + " | ".join(
                    f"{x['method']} solves={x['solves']} found={x['found']}" for x in r), flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / f"{problem.key}_fixed_target_detail.csv", index=False)
    summary.to_csv(out_dir / f"{problem.key}_fixed_target_summary.csv", index=False)
    print_table(summary)


# =============================================================================
# LATEX: the paper table (training distribution), knapsack | TSP
# =============================================================================

def latex_rows(out_dir: Path, key: str) -> list:
    path = out_dir / f"{key}_fixed_target_summary.csv"
    if not path.is_file():
        return []
    s = pd.read_csv(path)
    s = s[s["scenario"] == "id"]
    rows = []
    for (n, Q), g in s.groupby(["n", "Q"], sort=True):
        m = {r.method: r for r in g.itertuples()}
        get = lambda k, col: getattr(m[k], col) if k in m else None   # noqa: E731
        solves = ec.bold_best([ec.fmt(get(k, "solves"), 0) for k in ("E", "HB", "A", "SSA")], "min")
        hv = ec.bold_best([ec.fmt(None if get(k, "hv_fraction") is None else ec.hv_gap(get(k, "hv_fraction")),
                                  ec.HV_GAP_DECIMALS) for k in ("E", "H", "HB", "A", "SSA")], "min")
        mark = r"$^\dagger$" if n not in PROBLEMS[key].sizes else ""
        rows.append([f"{n}{mark}", f"{Q}"] + solves + hv)
    return rows


def latex_table(out_dir: Path) -> str:
    nc = 11
    head = r"$n$ & $Q$ & E & HB & A & SSA & E & H & HB & A & SSA"
    lines = [r"\begin{table*}[!ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{Fixed target: solves E, HB, A and SSA ($\tau=0$) need to find $Q$ reference ESN points "
             r"(HB: mean over the instances on which it reached $Q$), and HV gap of the points found (H: of $Q$ "
             r"uniform weights); " + ec.HV_GAP_DEF + r". E: CAN/ExA; H: uniform weights; HB: bisection; A: "
             r"largest-area heuristic. $^\dagger$: size not used in training. Fewest solves and lowest HV gap in "
             r"bold.}",
             r"\label{tab:fixed_target}",
             r"\begin{tabular}{@{}rrrrrrrrrrr@{\hspace{8pt}}rrrrrrrrrrr@{}}", r"\toprule",
             r"\multicolumn{11}{c}{\textbf{Knapsack}} & \multicolumn{11}{c}{\textbf{TSP}} \\",
             r"\cmidrule(r{8pt}){1-11}\cmidrule{12-22}",
             r" & & \multicolumn{4}{c}{Solves to $Q$} & \multicolumn{5}{c}{HV gap (\%)} & & & "
             r"\multicolumn{4}{c}{Solves to $Q$} & \multicolumn{5}{c}{HV gap (\%)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr{8pt}){7-11}\cmidrule(lr){14-17}\cmidrule(l){18-22}",
             head + " & " + head + r" \\", r"\midrule"]
    for p in (2, 3):
        lines.append(rf"\multicolumn{{11}}{{l}}{{\textit{{$p={p}$}}}} & \multicolumn{{11}}{{l}}{{\textit{{$p={p}$}}}} \\")
        lines += ec.side_by_side(latex_rows(out_dir, f"ks_p{p}"), latex_rows(out_dir, f"tsp_p{p}"), nc)
        if p == 2:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Fixed-target race: E / SSA / A / HB / H.")
    ap.add_argument("--problem", choices=["all", *PROBLEMS], default="all")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="default: training + unseen sizes on id, training sizes on the other scenarios")
    ap.add_argument("--scenarios", nargs="+", default=["id"],
                    help="'all' = id, the shifted distributions and the clustered placements")
    ap.add_argument("--instances", type=int, default=ec.TARGET_INSTANCES)
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
        plan = {s: a.sizes or (sorted(set(problem.sizes) | set(problem.extra_sizes)) if s == "id"
                               else list(problem.sizes)) for s in scenarios}
        run_problem(problem, plan, a.instances, a.out_dir)
    print(f"\nResults in {a.out_dir}")


if __name__ == "__main__":
    main()
