"""
eval_ood.py -- SSA (with calibrated tau) under distribution shift
========================================================================
The scorer and tau were fitted on the training distribution only (knapsack:
profits/weights ~ U[60,100]; TSP: coordinates on [0,999]). Here the same model
is run, unchanged, on shifted value distributions (eval_common.KS_SCENARIOS / TSP_SCENARIOS) and clustered  (clustered.py)
with 10 instances per size and scenario (seed series eval_common.TARGET_SEED;
the same instances as the DRL runs ood_eval_v6.py / drl_eval_clustered.py).
For every instance: Exact E (ground truth), SSA and the uniform heuristic H,
scored as in the main table (eval_main.evaluate_instance).

Outputs (SSAv3/results/ood/)
  <problem>_ood_detail.csv       one row per instance
  <problem>_ood_summary.csv      pooled metrics per scenario and size (as the main table)
  <problem>_ood_robustness.csv   per size: ID value, worst shifted value and Delta_max = ID - worst (SP: worst - ID) of SSA recall, precision, SP-ratio, HV and of H


How to run:
  python eval_ood.py --problem all                         # every scenario
  python eval_ood.py --problem ks_p3 --scenarios clustered # id + clustered + random_clustered only
  python eval_ood.py --latex shift > ood_table.tex
  python eval_ood.py --latex clustered --drl-summary ../logs/clustered_ood/ood_summary.csv > clustered_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import clustered as cl
import eval_common as ec
from eval_main import evaluate_instance, summarize
from problems import PROBLEMS, Problem
from ssa import Scorer, load_tau

OUT_DIR = ec.HERE / "results" / "ood"
SCENARIO_SETS = {"all": "every scenario", "shift": "id + the five shifted value distributions",
                 "clustered": "id + clustered + random_clustered"}


def scenarios_of(problem: Problem, choice: list) -> list:
    if choice == ["all"]:
        return ec.all_scenarios(problem)
    if choice == ["shift"]:
        return ec.scenario_names(problem)
    if choice == ["clustered"]:
        return ["id", *cl.SCENARIOS]
    return choice


def robustness(summary: pd.DataFrame) -> pd.DataFrame:
    """ID value, worst shifted value and Delta_max of the SSA (and H) metrics, per size."""
    metrics = {"recall_SSA": "higher", "precision_SSA": "higher", "hv_SSA": "higher", "sp_SSA": "lower",
               "recall_H": "higher", "sp_H": "lower"}
    out = []
    for (prob, n), g in summary.groupby(["problem", "n"]):
        idv, shifted = g[g["scenario"] == "id"], g[g["scenario"] != "id"]
        if idv.empty or shifted.empty:
            continue
        r = {"problem": prob, "n": n}
        for col, better in metrics.items():
            worst = shifted.loc[shifted[col].idxmin() if better == "higher" else shifted[col].idxmax()]
            r[f"{col}_id"] = idv[col].iloc[0]
            r[f"{col}_worst"] = worst[col]
            r[f"{col}_worst_scenario"] = worst["scenario"]
            r[f"{col}_delta_max"] = (idv[col].iloc[0] - worst[col]) if better == "higher" else (worst[col] - idv[col].iloc[0])
        out.append(r)
    return pd.DataFrame(out)


def run_problem(problem: Problem, sizes, scenarios, n_instances: int, tau, out_dir: Path):
    scorer = Scorer.load(problem)
    tau = load_tau(problem) if tau is None else float(tau)
    rows = []
    for scen in scenarios:
        for n in sizes:
            print(f"\n{problem.key} [{scen}] n={n} (tau={tau})", flush=True)
            for t in range(n_instances):
                inst = ec.make_instance(problem, n, t, scen, ec.TARGET_SEED)
                row, _ = evaluate_instance(problem, inst, scorer, tau, with_v8=False, trial=t)
                row["tau"] = tau
                rows.append(row)
                print(f"  trial {t}: |Y_E|={row['exact_m']} | SSA true={row['ssa_matched']}/{row['ssa_out']} "
                      f"solves={row['ssa_k']} (E {row['exact_k']}) | H true={row['h_matched']}", flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(out_dir / f"{problem.key}_ood_detail.csv", index=False)
    summary.to_csv(out_dir / f"{problem.key}_ood_summary.csv", index=False)
    cols = ["scenario", "n", "m_E", "recall_SSA", "precision_SSA", "sp_SSA", "sp_E", "hv_SSA", "recall_H", "t_ratio_SSA"]
    print(f"\n=== {problem.key}: SSA under distribution shift ===")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    rob = robustness(summary)
    if not rob.empty:
        rob.to_csv(out_dir / f"{problem.key}_ood_robustness.csv", index=False)
        show = ["n"] + [c for c in rob.columns if c.endswith(("_id", "_worst", "_delta_max"))
                        and c.startswith(("recall_SSA", "sp_SSA", "hv_SSA"))]
        print(f"\n=== {problem.key}: Delta_max (ID vs worst shifted scenario) ===")
        print(rob[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# =============================================================================
# LATEX
# =============================================================================

def _detail(out_dir: Path, key: str):
    path = out_dir / f"{key}_ood_detail.csv"
    return pd.read_csv(path) if path.is_file() else None


def pooled_by_scenario(d: pd.DataFrame) -> pd.DataFrame:
    """SSA recall, precision, SP-ratio and HV of every scenario, pooled over all sizes and instances."""
    g = d.groupby("scenario").agg(mE=("exact_m", "sum"), m=("ssa_matched", "sum"), out=("ssa_out", "sum"),
                                  k=("ssa_k", "sum"), hv=("ssa_hv", "mean"))
    return pd.DataFrame({"recall": 100 * g.m / g.mE, "precision": 100 * g.m / g.out, "sp": g.k / g.m, "hv": 100 * g.hv})


def shift_rows(out_dir: Path, kind: str) -> list:
    """Rows of one half (knapsack or TSP) of the shift table: scenario, shift, then recall/SP/HV for p=2 and p=3."""
    key = "ks" if kind == "knapsack" else "tsp"
    P = {p: _detail(out_dir, f"{key}_p{p}") for p in (2, 3)}
    order = list(ec.SCENARIO_LABEL[kind])
    present = [s for s in order if any(P[p] is not None and (P[p]["scenario"] == s).any() for p in (2, 3))]
    pooled = {p: pooled_by_scenario(P[p]) if P[p] is not None else None for p in (2, 3)}
    rows = []
    for scen in present:
        cells = [ec.SCENARIO_LABEL[kind][scen], ec.SCENARIO_SHIFT[kind][scen]]
        for p in (2, 3):
            r = pooled[p].loc[scen] if pooled[p] is not None and scen in pooled[p].index else None
            cells += ["--"] * 3 if r is None else [ec.fmt(r.recall), ec.fmt(r.sp, 2), ec.fmt(ec.hv_gap(r.hv / 100), ec.HV_GAP_DECIMALS)]   # HV gap
        rows.append(cells)
    return rows


def shift_table(out_dir: Path) -> str:
    nc = 8
    head = r"Scenario & Shift & Rec. & SP & Gap & Rec. & SP & Gap"
    lines = [r"\begin{table*}[!ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{1.6pt}",
             r"\caption{SSA under distribution shift (same model and $\tau^*$), pooled over all sizes, 10 instances "
             r"per size and scenario: recall and HV (\%), and SP-ratio. Shift: change with respect to the training "
             r"distribution.}",
             r"\label{tab:ood_scenarios}",
             r"\begin{tabular}{@{}llrrrrrr@{\hspace{6pt}}llrrrrrr@{}}", r"\toprule",
             r"\multicolumn{8}{c}{\textbf{Knapsack}} & \multicolumn{8}{c}{\textbf{TSP}} \\",
             r"\cmidrule(r{6pt}){1-8}\cmidrule{9-16}",
             r" & & \multicolumn{3}{c}{$p=2$} & \multicolumn{3}{c}{$p=3$} & & & \multicolumn{3}{c}{$p=2$} & "
             r"\multicolumn{3}{c}{$p=3$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){11-13}\cmidrule(l){14-16}",
             head + " & " + head + r" \\", r"\midrule"]
    lines += ec.side_by_side(shift_rows(out_dir, "knapsack"), shift_rows(out_dir, "tsp"), nc)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def per_instance_means(d: pd.DataFrame) -> pd.DataFrame:
    """Mean per-instance recall (%) and SP-ratio of E, SSA and H per scenario, averaged over the sizes n > 5.

    Mean per instance (not pooled) so that the numbers compare with the DRL
    evaluation, which reports mean_recall and mean_solves_per_matched.
    """
    d = d[d["n"] > 5].copy()
    d["rec_SSA"] = 100 * d["ssa_matched"] / d["exact_m"]
    d["rec_H"] = 100 * d["h_matched"] / d["exact_m"]
    d["sp_E"] = d["exact_k"] / d["exact_m"].clip(lower=1)
    d["sp_SSA"] = d["ssa_k"] / d["ssa_matched"].clip(lower=1)
    d["sp_H"] = d["h_k"] / d["h_matched"].clip(lower=1)
    cols = ["rec_SSA", "rec_H", "sp_E", "sp_SSA", "sp_H"]
    return d.groupby(["scenario", "n"])[cols].mean().groupby("scenario").mean()


def drl_means(path: Path, key: str) -> pd.DataFrame:
    """DRL mean recall (%) and solves per matched point per scenario, models at their training size, n > 5."""
    r = pd.read_csv(path)
    r = r[r["model_name"].str.replace(r"_n\d+$", "", regex=True) == key]
    r = r[r["model_name"].str.extract(r"_n(\d+)$")[0].astype(int) == r["n"]]
    r = r[r["n"] > 5]
    return r.groupby("scenario").agg(rec_DRL=("mean_recall", "mean"), sp_DRL=("mean_solves_per_matched", "mean")) \
        .assign(rec_DRL=lambda x: 100 * x.rec_DRL)


def clustered_table(out_dir: Path, drl_summary) -> str:
    with_drl = drl_summary is not None
    methods = ["E", "SSA", *(["DRL"] if with_drl else []), "H"]
    k = len(methods)
    title = {"ks_p2": r"KP, $p=2$", "ks_p3": r"KP, $p=3$", "tsp_p2": r"TSP, $p=2$", "tsp_p3": r"TSP, $p=3$"}
    lines = [r"\begin{table}[!ht]", r"\centering",
             r"\caption{Random (R), random-clustered (RC) and clustered (C) instances. Means over the training sizes "
             r"(10 instances each). Recall: mean per-instance recall (\%). SP: solves per ESN point found (mean per "
             r"instance). E: CAN ($p=2$) / ExA ($p=3$) run to completion, which defines the reference set. H solves "
             r"$|Y_E|$ uniformly spread weights. Best recall (without E) and lowest SP in bold.}",
             r"\label{tab:clustered}", r"\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{ll{'r' * (2 * k)}}}", r"\toprule",
             rf" & & \multicolumn{{{k}}}{{c}}{{Recall (\%)}} & \multicolumn{{{k}}}{{c}}{{SP}} \\",
             rf"\cmidrule(lr){{3-{2 + k}}}\cmidrule(lr){{{3 + k}-{2 + 2 * k}}}",
             "Problem & Data & " + " & ".join(methods + methods) + r" \\"]
    for key in ("ks_p2", "ks_p3", "tsp_p2", "tsp_p3"):
        d = _detail(out_dir, key)
        if d is None:
            continue
        M = per_instance_means(d)
        if M.empty:                                    # no size n > 5 in the detail file
            continue
        D = drl_means(drl_summary, key) if with_drl else None
        lines.append(r"\midrule")
        for i, (scen, short) in enumerate((("id", "R"), ("random_clustered", "RC"), ("clustered", "C"))):
            if scen not in M.index:
                continue
            m = M.loc[scen]
            rec = {"E": 100.0, "SSA": m.rec_SSA, "H": m.rec_H}
            sp = {"E": m.sp_E, "SSA": m.sp_SSA, "H": m.sp_H}
            if with_drl:
                rec["DRL"] = D.loc[scen, "rec_DRL"] if scen in D.index else float("nan")
                sp["DRL"] = D.loc[scen, "sp_DRL"] if scen in D.index else float("nan")
            rec_cells = ["100.0"] + ec.bold_best([ec.fmt(rec[x]) for x in methods[1:]], "max")
            sp_cells = ec.bold_best([ec.fmt(sp[x], 2) for x in methods], "min")
            lines.append((title[key] if i == 0 else "") + f" & {short} & " + " & ".join(rec_cells + sp_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="SSA under distribution shift (shifted values and clustered placements).")
    ap.add_argument("--problem", choices=["all", *PROBLEMS], default="all")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="default: training sizes")
    ap.add_argument("--scenarios", nargs="+", default=["all"],
                    help="all | shift | clustered | explicit scenario names (" +
                         "; ".join(f"{k}: {v}" for k, v in SCENARIO_SETS.items()) + ")")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--tau", type=float, default=None, help="default: the calibrated tau")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--latex", choices=["shift", "clustered"], default=None,
                    help="print a paper table from the detail files and exit")
    ap.add_argument("--drl-summary", type=Path, default=None,
                    help="--latex clustered: ood_summary.csv of drl_eval_clustered.py (adds the DRL columns)")
    a = ap.parse_args()
    if a.latex == "shift":
        print(shift_table(a.out_dir))
        return
    if a.latex == "clustered":
        print(clustered_table(a.out_dir, a.drl_summary))
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for key in (list(PROBLEMS) if a.problem == "all" else [a.problem]):
        problem = PROBLEMS[key]
        run_problem(problem, a.sizes or list(problem.sizes), scenarios_of(problem, a.scenarios),
                    a.instances, a.tau, a.out_dir)
    print(f"\nResults in {a.out_dir}")


if __name__ == "__main__":
    main()
