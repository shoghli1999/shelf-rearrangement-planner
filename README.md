# Rearranging objects in a shelf with a fixed-base robot arm

Code for my M.Sc. thesis in Data Mining at Tarbiat Modares University, Tehran (Faculty of Mathematical Sciences, supervisor Dr. Fatemeh Zahra Saberifar, 2026):
*Rearranging Rigid Objects Using a Fixed-Base Robot Arm*.

## The problem

A robot arm has to move several cans in a shelf from where they are to a goal arrangement. The shelf is only open at the front, so the arm can't lift one can over another, and its links sweep through the space in front of whatever it reaches for. On a table this problem has been studied a lot. In a shelf like this it is much harder, because almost every move blocks or frees other moves.

## How CDG-LB works

First, for every pose and approach direction I compute the region the three-link arm sweeps through. Each arm constraint then becomes a precedence between two objects (A has to move before B). Together they form what I call a confined dependency graph.

With a single approach option, the graph tells in O(n²) time whether the cans can be moved monotonically, meaning each can moves only once. I prove this in the thesis.

For the general case, a constraint-programming model (Google OR-Tools CP-SAT) decides which cans need a temporary buffer spot, the order of the moves and the approach for each move. Its optimum is a lower bound on the length of any valid plan, and if the model has no solution, no plan exists.

The plan is then executed with lazy buffer allocation inside the shelf, local repair, re-planning and some post-processing.

## Results

All numbers come from `results/`. The 520 evaluation instances were not used while I developed the method.

Tidy goal arrangement, 20 instances per size, 60 s time limit, compared with PERTS:

| Cans | Solved by CDG-LB | Solved by PERTS | Motion-planning queries (CDG-LB / PERTS) | Plan length (CDG-LB / PERTS) |
|---:|---:|---:|---:|---:|
| 12 | 100% | 100% | 515 / 1,705 | 13.8 / 14.7 |
| 15 | 100% | 90% | 1,824 / 41,231 | 19.0 / 21.9 |
| 18 | 100% | 15% | 12,959 / 169,220 | 28.5 / 29.0 |
| 21 | 95% | 0% | 33,488 / 189,950 | 43.1 / – |
| 24 | 45% | 0% | 39,821 / 163,764 | 61.4 / – |

On the 18 fifteen-can instances that both methods solved, CDG-LB's plans were about 14% shorter (18.9 vs. 21.9 actions, Wilcoxon p < 0.001) and needed about 15 times fewer motion-planning queries (1,666 vs. 25,794).

On a small shelf where A* can find the optimal plan (80 instances), CDG-LB found the optimum 62 times and PERTS 41 times, and my lower bound was equal to the optimum 69 times.

The O(n²) monotonicity test agreed with complete search on all 160 test instances, and all 1,113 plans from all runs passed an independent step-by-step geometric check (`verify`). What makes an instance hard is mostly the free space left in the shelf, together with the number of cans.

## Files

| File | What it does |
|---|---|
| `rearrangement.py` | Shelf, arm and can model; swept areas and blocking bitsets; all planners (CDG, CDG-LB, mRS, DFS_DP, CIRS, PERTS, TRLB-table, A*); the independent plan checker `verify` |
| `experiments.py` | Builds the test instances and runs experiments E0 to E7; each run is one JSON line in `results/<exp>.jsonl` |
| `figures.py` | Reads `results/`, writes the thesis figures to `figures/` and LaTeX tables to `tables/`, and prints every number quoted in the thesis |
| `results/` | Raw results of one complete run of the final code |

## Running it

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python rearrangement.py            # quick demo on one instance
python experiments.py              # all experiments (about two hours on 2 cores)
python experiments.py E2 E4        # only some experiments
python figures.py                  # figures, tables and a printed summary from results/
```

```python
import rearrangement as R

inst = R.make_instance(n=15, seed=1, lying_ratio=0.3, goal_style="rows")
res = R.cdg_lb(inst, time_limit=60)
print(res.success, res.n_actions, res.lower_bound)
print(R.verify(inst, res.actions))       # (True, 'ok')
```

| Experiment | Setup |
|---|---|
| E0 | Model check: planner masks against an instant-by-instant check with a 2 mm step, 6 shelf settings |
| E1 | Monotone test: K=1 for n=6..16, K=10 for n=6..20 |
| E2 | Tidy goal: n=6..24, 20 instances per n, 60 s limit |
| E3 | Messy goal: n=6..21, 10 instances per n |
| E4 | Optimality: n=3..6 on a 0.5 × 0.3 m shelf, compared with A* |
| E5 | Ablation on the first 10 instances of E2 for n=12 and 18 |
| E6 | Shelf conditions: pillars, share of lying cans, shelf width |
| E7 | Sensitivity: grid step and number of approach options, on separate development instances |

Every instance comes from a fixed random seed, and CP-SAT runs with one worker and a fixed seed. Because the solver works under time limits, run times and, in rare cases, plans can differ slightly on another machine.

The figures are labelled in Persian, as in the thesis, and use the Vazirmatn font in `fonts/` (SIL Open Font License, see `fonts/OFL.txt`).

## Tech

Python · NumPy · SciPy · Google OR-Tools (CP-SAT) · NetworkX · matplotlib
