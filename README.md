# Learning to Accelerate the Weighted Sum Method (SSA)

Code for the paper *Learning to Accelerate the Weighted Sum Method for Multi-Objective Combinatorial Optimization* (Shaji, Sobhanan, and Charkhgard).

The **Stage Search Algorithm (SSA)** finds the extreme supported nondominated (ESN) points of bi- and tri-objective integer programs by solving weighted-sum problems. It keeps the convex hull of the points found so far, scores every open facet of the hull with a small neural network, solves the most promising facet next, and stops when no facet scores above a calibrated threshold. It is compared with exact algorithms (CAN for two objectives, ExA for three) and three heuristics on multi-objective knapsack and traveling salesman instances.

## Files

| File | What it does |
|---|---|
| `problems.py` | Problem definitions (knapsack, TSP), instance generation, Gurobi weighted-sum solver |
| `exa.py` | Exact benchmarks: CAN (p = 2) and ExA (p = 3); stage features; scorer network |
| `ssa.py` | SSA itself; training data collection, training, threshold calibration |
| `clustered.py` | Clustered (C) and random-clustered (RC) instances for the distribution-shift tests |
| `eval_common.py` | Shared evaluation code: instances, heuristics (H, HB, A), metrics, hypervolume |
| `eval_main.py` | Main comparison: SSA vs. E vs. H (Table 3) |
| `eval_ood.py` | SSA under distribution shift (Table 4) |
| `eval_fixed_target.py` | Solves needed to find a fixed number of ESN points (Table 5) |
| `eval_fixed_budget.py` | ESN points found with a fixed number of solves (Table 6) |
| `decomp_ssa.py`, `plot_weights.py` | Weight-space figure (Figure 1) |
| `runs/<problem>/` | Trained scorers (`scorer.npz`) and thresholds (`calibration.json`) |

The four problems are `ks_p2`, `ks_p3`, `tsp_p2`, `tsp_p3` (knapsack / TSP with 2 or 3 objectives). Every script accepts `--problem all` or a single problem.

## Requirements

- Python 3.13
- Gurobi 13 (`gurobipy==13.0.2`) with a valid license. The size-limited license installed with
  `pip install gurobipy` may suffice for the instance sizes in the paper; otherwise use a (free
  academic) Gurobi license.
- NumPy, SciPy, pandas, Matplotlib
- PyTorch 2.10 (only needed to retrain the scorers)

## Reproducing the paper

All instances are generated from fixed random seeds, so every run produces the same instances.

**Step 1 – Train (optional).** The trained scorers and thresholds are included in `runs/`. To retrain them (collect training data, train the network, calibrate the threshold):

```bash
python ssa.py pipeline --problem all
```

**Step 2 – Main comparison (Table 3).**

```bash
python eval_main.py --problem all --no-v8
```

Output: `results/main/<problem>_by_n.csv` and `results/main/<problem>_table.tex`.

**Step 3 – Weight-space figure (Figure 1)** (needs Step 2):

```bash
python plot_weights.py --problem ks_p3 --n 100 --k 0
```

**Step 4 – Distribution shift (Table 4).**

```bash
python eval_ood.py --problem all
python eval_ood.py --latex shift > ood_table.tex
```

**Step 5 – Fixed target (Table 5).**

```bash
python eval_fixed_target.py --problem all
python eval_fixed_target.py --latex > fixed_target_table.tex
```

**Step 6 – Fixed budget (Table 6)** (needs Step 5, which sets the budget):

```bash
python eval_fixed_budget.py --problem all
python eval_fixed_budget.py --latex > fixed_budget_table.tex
```
