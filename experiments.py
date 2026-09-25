"""
Experiments of the thesis. Every run is written as one line of JSON in
results/<experiment>.jsonl so that the figures script can read them.

  E0  check of the swept-area model against a fine instant-by-instant check (Section 7.2)
  E1  monotone test (Section 7.3)
  E2  tidy goal, CDG-LB against PERTS (Section 7.4)
  E3  messy goal (Section 7.4)
  E4  small instances against the optimum found by A* (Section 7.5)
  E5  ablation of the parts of CDG-LB (Section 7.6)
  E6  pillars, lying cans and shelf width (Section 7.7)
  E7  sensitivity to the grid step and to the number of approach options,
      on separate development instances (Section 7.8)

Seeds: the evaluation instances (E0 to E6) use seeds that were never used
while the method was being developed. E7 uses its own development seeds.

Usage:  python experiments.py            (all experiments, about two hours on 2 cores)
        python experiments.py E2 E4      (only some of them)
"""

import os
import sys
import json
import time
import random
from multiprocessing import Pool

import numpy as np

import rearrangement as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PILLARS = [(0.30, 0.20), (0.50, 0.28), (0.70, 0.20)]
LIMIT = 60.0          # seconds per run
EVAL = 20000          # seed offset of the evaluation instances
DEV = 90000           # seed offset of the development instances (E7)


def record(exp, inst, res, **extra):
    # "lying" holds the number of lying cans; jobs that set the share of lying
    # cans overwrite it with that share (a float) through `extra`
    ok = R.verify(inst, res.actions)[0] if res.success else False
    row = dict(exp=exp, n=inst.n, method=res.method, success=res.success and ok,
               time=round(res.time, 4), calls=res.calls,
               actions=res.n_actions if res.success else None,
               buffers=R.count_buffers(inst, res.actions) if res.success else None,
               lb=res.lower_bound, note=res.note, infeasible=res.infeasible,
               verified=ok, density=round(inst.density(), 4), prep=round(inst.prep_time, 3),
               poses=inst.P, lying=sum(k == "lie" for k in inst.kinds))
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# E0: every precomputed mask against the fine check
# --------------------------------------------------------------------------
def model_check(inst, samples, rng):
    """Compare planner masks with fine_motion for a sample of poses.

    unsafe = the fine check sees contact but the mask says free (must be 0)
    extra  = the mask says blocked but there is no real contact (caution)
    """
    sh = inst.shelf
    out = dict(motions=0, valid_both=0, valid_only_planner=0, valid_only_fine=0,
               pairs=0, unsafe=0, extra=0)
    for p in rng.sample(range(inst.P), min(samples, inst.P)):
        for k, opt in enumerate(sh.options):
            out["motions"] += 1
            fm = R.fine_motion(sh, inst.poses[p], opt)
            planner_ok = k in inst.valid[p]
            if fm is None or not planner_ok:
                out["valid_only_planner"] += planner_ok and fm is None
                out["valid_only_fine"] += (fm is not None) and not planner_ok
                continue
            out["valid_both"] += 1
            la, lb, ca, cb = fm
            touch = (R.seg_dist(ca, cb, inst.qa, inst.qb).min(0) < 2 * R.CAN_R - R.TOUCH) | \
                    (R.seg_dist(la, lb, inst.qa, inst.qb).min(0) < sh.link_r + R.CAN_R - R.TOUCH)
            touch[p] = False                      # the moved can itself
            mask = np.array([bool(inst.mask[p][k] >> q & 1) for q in range(inst.P)])
            mask[p] = False
            out["pairs"] += inst.P - 1
            out["unsafe"] += int(np.sum(touch & ~mask))
            out["extra"] += int(np.sum(mask & ~touch))
    return out


# --------------------------------------------------------------------------
# One job = one instance, all methods of the experiment on it
# --------------------------------------------------------------------------
def job(args):
    exp, params = args
    rows = []
    try:
        if exp == "E0":
            inst = R.make_instance(params["n"], params["seed"], lying_ratio=params["lying"],
                                   width=params["width"], pillars=PILLARS if params["pillars"] else ())
            row = model_check(inst, 120, random.Random(params["seed"]))
            row.update(exp=exp, poses=inst.P, **params)
            return [row]

        if exp == "E1":
            n, seed, options, kind = params["n"], params["seed"], params["options"], params["kind"]
            if kind == "feasible":
                inst = R.make_monotone_instance(n, seed, options=options)
            else:
                inst = R.make_hard_infeasible(n, seed, options=options)
                if inst is None:
                    return rows
            for m in ["CDG", "CIRS", "DFS_DP", "mRS"]:
                res = R.run_monotone(inst, m, time_limit=10.0)
                row = record(exp, inst, res, seed=seed, options=options, kind=kind)
                row["success"] = res.success
                row["decision"] = res.note
                if m == "CDG":
                    # the graph test is very fast, so time it over many repetitions
                    reps = 200 if options == "single" else 5
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        R.cdg_monotone(inst, inst.start, time.perf_counter() + 30)
                    row["time"] = (time.perf_counter() - t0) / reps
                rows.append(row)
            return rows

        n, seed = params["n"], params["seed"]
        inst = R.make_instance(n, seed, lying_ratio=params.get("lying", 0.3),
                               width=params.get("width", 1.0), depth=params.get("depth", 0.4),
                               pillars=PILLARS if params.get("pillars") else (),
                               goal_style=params.get("style", "rows"),
                               options=params.get("options", "multi"),
                               grid=params.get("grid", 0.06))
        best = None
        for m in params["methods"]:
            if m == "A*":
                # look only for a plan shorter than the one CDG-LB found
                res = R.astar(inst, time_limit=LIMIT, upper=best)
            else:
                res = R.solve(inst, m, time_limit=LIMIT, seed=seed)
                if res.success and m == "CDG-LB":
                    best = res.n_actions
            rows.append(record(exp, inst, res, seed=seed,
                               **{k: v for k, v in params.items() if k not in ("n", "seed", "methods")}))
    except Exception as err:          # keep the batch running, but note the problem
        rows.append(dict(exp=exp, error=repr(err), **{k: v for k, v in params.items() if k != "methods"}))
    return rows


def plan(exp):
    jobs = []
    both = ["CDG-LB", "PERTS"]
    if exp == "E0":
        for s, (lying, width, pillars) in enumerate([(0.0, 1.0, False), (0.3, 1.0, False), (0.6, 1.0, False),
                                                     (0.3, 0.8, False), (0.3, 1.1, False), (0.3, 1.0, True)]):
            jobs.append(dict(n=12, seed=EVAL + 90 + s, lying=lying, width=width, pillars=pillars))
    elif exp == "E1":
        for options, ns in (("single", [6, 8, 10, 12, 14, 16]), ("multi", [6, 8, 10, 12, 14, 16, 18, 20])):
            for n in ns:
                for s in range(6):
                    for kind in ("feasible", "infeasible"):
                        jobs.append(dict(n=n, seed=EVAL + 1000 * n + s, options=options, kind=kind))
    elif exp == "E2":
        for n in [6, 9, 12, 15, 18, 21, 24]:
            for s in range(20):
                jobs.append(dict(n=n, seed=EVAL + 100 * n + s, style="rows", methods=both))
    elif exp == "E3":
        for n in [6, 9, 12, 15, 18, 21]:
            for s in range(10):
                jobs.append(dict(n=n, seed=EVAL + 5000 + 100 * n + s, style="random", methods=both))
    elif exp == "E4":
        for n in [3, 4, 5, 6]:
            for s in range(20):
                jobs.append(dict(n=n, seed=EVAL + 9000 + 100 * n + s, style="random", width=0.5, depth=0.3,
                                 methods=["CDG-LB", "A*", "PERTS"]))
    elif exp == "E5":
        # the same instances as the first ten of E2
        for n in [12, 18]:
            for s in range(10):
                jobs.append(dict(n=n, seed=EVAL + 100 * n + s, style="rows",
                                 methods=["CDG-LB/no-smoothing", "CDG-LB/no-dwell",
                                          "CDG-LB/random-buffer", "TRLB-table"]))
    elif exp == "E6":
        for n in [8, 12, 16]:
            for s in range(10):
                jobs.append(dict(n=n, seed=EVAL + 7000 + 100 * n + s, style="rows", pillars=True, methods=both))
        # lying share and width change on the n = 15 instances of E2 (seed kept)
        for lying in [0.0, 0.6]:
            for s in range(10):
                jobs.append(dict(n=15, seed=EVAL + 1500 + s, style="rows", lying=lying, methods=both))
        for width in [0.8, 0.9, 1.1]:
            for s in range(10):
                jobs.append(dict(n=15, seed=EVAL + 1500 + s, style="rows", width=width, methods=both))
    elif exp == "E7":
        for n in [15, 18]:
            for s in range(10):
                seed = DEV + 100 * n + s
                for grid in [0.04, 0.06, 0.08]:
                    jobs.append(dict(n=n, seed=seed, style="rows", grid=grid, options="multi", methods=["CDG-LB"]))
                for options in ["front", "three"]:
                    jobs.append(dict(n=n, seed=seed, style="rows", grid=0.06, options=options, methods=["CDG-LB"]))
    return [(exp, p) for p in jobs]


def main(exps):
    os.makedirs(OUT, exist_ok=True)
    for exp in exps:
        jobs = plan(exp)
        path = os.path.join(OUT, f"{exp}.jsonl")
        t0 = time.time()
        with open(path, "w") as f, Pool(2) as pool:
            for rows in pool.imap_unordered(job, jobs):
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.flush()
        print(f"{exp}: {len(jobs)} instances in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["E0", "E4", "E2", "E5", "E3", "E6", "E1", "E7"])
