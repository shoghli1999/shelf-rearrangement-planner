"""
Figures and LaTeX tables of the thesis.

Reads results/*.jsonl written by experiments.py and writes
  figures/fig_*.pdf
  tables/tab_*.tex

Usage:  python figures.py
"""

import os
import json
import math
import random
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FuncFormatter
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
from scipy import stats

import rearrangement as R

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")

# ---------------------------------------------------------------------------
# Persian text in matplotlib
# ---------------------------------------------------------------------------
FONT_FILE = os.path.join(HERE, "fonts", "Vazirmatn-Regular.ttf")
font_manager.fontManager.addfont(FONT_FILE)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_FILE).get_name()
plt.rcParams["font.size"] = 8.5
plt.rcParams["legend.fontsize"] = 7.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

FA_DIGITS = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def fa(text):
    """Shape Persian text so matplotlib draws it correctly."""
    return get_display(arabic_reshaper.reshape(text))


def fa_num(x, _=None):
    if abs(x - round(x)) < 1e-9:
        s = str(int(round(x)))
    else:
        s = f"{x:g}"
    return s.translate(FA_DIGITS)


COLORS = {"CDG-LB": "#1b6ca8", "PERTS": "#d1495b", "TRLB-table": "#8f8f8f",
          "CDG": "#1b6ca8", "CIRS": "#edae49", "DFS_DP": "#66a182", "mRS": "#d1495b",
          "A*": "#00798c", "CDG-LB/no-dwell": "#6d9dc5", "CDG-LB/random-buffer": "#a3c4dc"}
MARK = {"CDG-LB": "o", "PERTS": "s", "CDG": "o", "CIRS": "^", "DFS_DP": "D", "mRS": "s", "A*": "x"}


def load(name):
    path = os.path.join(RES, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "error" not in r]


def mean(v):
    return st.mean(v) if v else float("nan")


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_can(ax, pose, color, alpha=1.0, dashed=False, label=None, lw=1.0):
    a0, a1 = pose.axis()
    style = dict(color=color, alpha=alpha, zorder=3)
    if dashed:
        style = dict(fill=False, edgecolor=color, linestyle="--", linewidth=lw, zorder=2)
    if not pose.lying:
        ax.add_patch(Circle(a0, R.CAN_R, **style))
    else:
        d = a1 - a0
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d) * R.CAN_R
        ang = np.linspace(0, math.pi, 12)
        ex = d / np.linalg.norm(d)
        cap1 = [a1 + R.CAN_R * (math.cos(t - math.pi / 2) * ex + math.sin(t - math.pi / 2) * nrm / R.CAN_R) for t in ang]
        cap0 = [a0 - R.CAN_R * (math.cos(t - math.pi / 2) * ex + math.sin(t - math.pi / 2) * nrm / R.CAN_R) for t in ang]
        ax.add_patch(Polygon(np.array(cap1 + cap0), closed=True, **style))
    if label is not None:
        c = (a0 + a1) / 2
        ax.text(c[0], c[1], label, ha="center", va="center", fontsize=7,
                color="white" if not dashed else color, zorder=4)


def draw_shelf(ax, shelf, arm=True):
    w, d = shelf.width, shelf.depth
    ax.plot([0, 0, w, w], [0, d, d, 0], color="k", lw=3, solid_capstyle="butt")
    ax.plot([0, w], [0, 0], color="k", lw=0.6, ls=":")
    for p in shelf.pillars:
        ax.add_patch(Circle(p, R.PILLAR_R, color="k", zorder=3))
    if arm:
        ax.plot(*shelf.base, marker="s", color="k", ms=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, w + 0.08)
    ax.set_ylim(-0.5 if arm else -0.06, d + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def arm_pose_lines(shelf, pose, option):
    """Joint positions of the arm when it holds a can at `pose`."""
    phi, elbow = option
    u = np.array([math.sin(phi), math.cos(phi)])
    a0, a1 = pose.axis()
    dirv = (a1 - a0) / (2 * R.HALF_AXIS) if pose.lying else np.zeros(2)
    reach = R.CAN_R + (R.HALF_AXIS * abs(dirv @ u) if pose.lying else 0.0)
    c = np.array([pose.x, pose.y])
    tip = c - reach * u
    wr = tip - shelf.hand * u
    dv = wr - shelf.base
    cos2 = (dv @ dv - shelf.l1 ** 2 - shelf.l2 ** 2) / (2 * shelf.l1 * shelf.l2)
    q2 = elbow * math.acos(max(-1, min(1, cos2)))
    q1 = math.atan2(dv[1], dv[0]) - math.atan2(shelf.l2 * math.sin(q2), shelf.l1 + shelf.l2 * math.cos(q2))
    j = shelf.base + shelf.l1 * np.array([math.cos(q1), math.sin(q1)])
    return np.array([shelf.base, j, wr, tip])


# ---------------------------------------------------------------------------
# Figure: the model (shelf, arm, sweep, blocked grid poses)
# ---------------------------------------------------------------------------
def fig_model():
    inst = R.make_instance(3, 5, lying_ratio=0.0)
    sh = inst.shelf
    target = R.Pose(0.30, 0.30)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # options 4, 1 and 8 are (0, +1), (-30, -1) and (+30, +1); words avoid sign problems in RTL text
    titles = ["ورود مستقیم، آرنج راست", "ورود مایل به چپ (۳۰ درجه)، آرنج چپ",
              "ورود مایل به راست (۳۰ درجه)، آرنج راست"]
    for ax, k, title in zip(axs, [4, 1, 8], titles):
        draw_shelf(ax, sh)
        sw = sh.sweep(target, sh.options[k])
        if sw is None:
            ax.set_title(fa(title + " (نامعتبر)"), fontsize=8)
            continue
        hit = R.blocked_by_sweep(sw, inst.qa, inst.qb, sh.link_r)
        for q in range(inst.cand_first, inst.P):
            c = inst.poses[q]
            if c.lying:
                continue
            ax.add_patch(Circle((c.x, c.y), 0.007, color="#d1495b" if hit[q] else "0.75", zorder=2))
        arm_a, arm_b, can_a, can_b = sw
        for a, b in zip(arm_a, arm_b):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#1b6ca8", alpha=0.08, lw=4, solid_capstyle="round")
        pts = arm_pose_lines(sh, target, sh.options[k])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color="#333333", lw=2, ms=3, zorder=5)
        draw_can(ax, target, "#2e8b57")
        ax.set_title(fa(title), fontsize=8)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_model.pdf")


# ---------------------------------------------------------------------------
# Figure: a small instance and its confined dependency graph
# ---------------------------------------------------------------------------
def fig_cdg_example():
    # search a small single-option instance whose graph has a cycle
    for seed in range(220, 400):
        inst = R.make_instance(6, 9000 + seed, lying_ratio=0.3, width=0.6, depth=0.3,
                               options="single", goal_style="random")
        edges, stuck = [], False
        for i in range(inst.n):
            b, a, s = R.dependency_edges(inst, inst.start, i, 0, 0)
            stuck |= s
            edges += [(j, i, "s") for j in b] + [(i, j, "g") for j in a]
        G = nx.DiGraph()
        G.add_nodes_from(range(inst.n))
        G.add_edges_from([(u, v) for u, v, _ in edges])
        if not stuck and not nx.is_directed_acyclic_graph(G) and 4 <= len(edges) <= 9:
            break
    sh = inst.shelf
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    draw_shelf(ax, sh, arm=True)
    cyc = nx.find_cycle(G)
    i0 = cyc[0][1]
    sw = sh.sweep(inst.poses[inst.start[i0]], sh.options[0])
    for a, b in zip(sw[0], sw[1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#edae49", alpha=0.10, lw=4, solid_capstyle="round")
    ax.plot([sw[2][2][0], sw[2][3][0]], [sw[2][2][1], sw[2][3][1]], color="#edae49", lw=5, alpha=0.3)
    cmap = plt.get_cmap("tab10")
    for i in range(inst.n):
        draw_can(ax, inst.poses[inst.start[i]], cmap(i), label=str(i + 1).translate(FA_DIGITS))
        draw_can(ax, inst.poses[inst.goal[i]], cmap(i), dashed=True, label=str(i + 1).translate(FA_DIGITS), lw=1.4)
    ax.set_ylim(-0.16, sh.depth + 0.04)
    ax.set_title(fa("چیدمان اولیه (پررنگ) و هدف (خط‌چین)"), fontsize=9)

    ax = axs[1]
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[cmap(i) for i in G.nodes], node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: str(i + 1).translate(FA_DIGITS) for i in G.nodes},
                            font_color="white", font_size=8, font_family=plt.rcParams["font.family"])
    s_edges = [(u, v) for u, v, t in edges if t == "s"]
    g_edges = [(u, v) for u, v, t in edges if t == "g"]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=s_edges, edge_color="#1b6ca8", arrows=True,
                           arrowsize=11, width=1.2, node_size=300, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=g_edges, edge_color="#d1495b", arrows=True,
                           arrowsize=11, width=1.2, node_size=300, style="dashed", connectionstyle="arc3,rad=0.1")
    ax.plot([], [], color="#1b6ca8", label=fa("جای اولیه در مسیر"))
    ax.plot([], [], color="#d1495b", ls="--", label=fa("جای هدف در مسیر"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=7.5)
    ax.set_title(fa("گراف وابستگی فضای محدود"), fontsize=9)
    ax.axis("off")
    save(fig, "fig_cdg_example.pdf")
    return inst


# ---------------------------------------------------------------------------
# Figure: snapshots of a solved instance
# ---------------------------------------------------------------------------
def fig_snapshots():
    for seed in range(20):
        inst = R.make_instance(10, 7000 + seed, lying_ratio=0.3, goal_style="rows")
        res = R.cdg_lb(inst, 30, seed)
        nb = R.count_buffers(inst, res.actions)
        if res.success and 2 <= nb <= 4 and res.n_actions == res.lower_bound:
            break
    acts = res.actions
    steps = sorted(set([0, len(acts)] + [int(round(len(acts) * f)) for f in (0.2, 0.4, 0.6, 0.8)]))
    rows = (len(steps) + 1) // 2
    fig, grid = plt.subplots(rows, 2, figsize=(6.4, 1.45 * rows))
    # Persian order: the first panel is at the top right
    axs = [grid[k // 2, 1 - k % 2] for k in range(len(steps))]
    for k in range(len(steps), 2 * rows):
        grid[k // 2, 1 - k % 2].axis("off")
    cmap = plt.get_cmap("tab10")
    pos = list(inst.start)
    k = 0
    for ax, st_ in zip(axs, steps):
        while k < st_:
            i, _, to = acts[k]
            pos[i] = to
            k += 1
        draw_shelf(ax, inst.shelf, arm=False)
        for i in range(inst.n):
            draw_can(ax, inst.poses[inst.goal[i]], cmap(i), dashed=True, lw=0.8)
            p = inst.poses[pos[i]]
            in_buffer = pos[i] not in (inst.start[i], inst.goal[i])
            draw_can(ax, p, cmap(i), label=str(i + 1).translate(FA_DIGITS))
            if in_buffer:
                ax.add_patch(Circle((p.x, p.y), R.CAN_R + 0.012, fill=False, ec="k", lw=1.2, zorder=5))
        title = "چیدمان اولیه" if st_ == 0 else f"پس از {str(st_).translate(FA_DIGITS)} اقدام"
        ax.set_title(fa(title), fontsize=8.5, pad=2)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    save(fig, "fig_snapshots.pdf")
    print("snapshots: actions", res.n_actions, "lower bound", res.lower_bound, "buffers", nb)
    return inst, res


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """95% Wilson interval of a success rate, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (mid - half), 100 * (mid + half)


def mean_ci(v):
    """Mean and half width of the 95% t interval."""
    if len(v) < 2:
        return mean(v), 0.0
    return st.mean(v), stats.t.ppf(0.975, len(v) - 1) * st.stdev(v) / math.sqrt(len(v))


def paired_test(a, b):
    """Wilcoxon signed-rank test on paired values. Returns (pairs, p value)."""
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 5 or all(x == 0 for x in d):
        return len(d), float("nan")
    return len(d), stats.wilcoxon(a, b).pvalue


def pairs(rows, key, m1="CDG-LB", m2="PERTS", n=None):
    """Values of `key` for instances that both methods solved."""
    a = {r["seed"]: r for r in rows if r["method"] == m1 and r["success"] and (n is None or r["n"] == n)}
    b = {r["seed"]: r for r in rows if r["method"] == m2 and r["success"] and (n is None or r["n"] == n)}
    common = sorted(set(a) & set(b))
    return [a[s][key] for s in common], [b[s][key] for s in common]


# ---------------------------------------------------------------------------
# Result figures
# ---------------------------------------------------------------------------
def by_n(rows, method, key, only_success=False):
    groups = defaultdict(list)
    for r in rows:
        if r["method"] == method and (r["success"] or not only_success):
            groups[r["n"]].append(r[key])
    return groups


def fig_main(rows, name):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for m in ["CDG-LB", "PERTS"]:
        g = by_n(rows, m, "success")
        ns = sorted(g)
        rate = [100 * mean(g[n]) for n in ns]
        lo_hi = [wilson(sum(g[n]), len(g[n])) for n in ns]
        axs[0].errorbar(ns, rate, yerr=[[r - lo for r, (lo, _) in zip(rate, lo_hi)],
                                        [hi - r for r, (_, hi) in zip(rate, lo_hi)]],
                        marker=MARK[m], color=COLORS[m], label=m, capsize=2, ms=4, lw=1.2)
        # run times are skewed, so the median and the quartiles are shown
        g = by_n(rows, m, "time")
        q = [np.percentile(g[n], [25, 50, 75]) for n in ns]
        axs[1].errorbar(ns, [b for _, b, _ in q], yerr=[[b - a for a, b, _ in q], [c - b for _, b, c in q]],
                        marker=MARK[m], color=COLORS[m], label=m, capsize=2, ms=4, lw=1.2)
        g = by_n(rows, m, "actions", only_success=True)
        ns2 = [n for n in sorted(g) if g[n]]
        mc = [mean_ci(g[n]) for n in ns2]
        axs[2].errorbar(ns2, [x for x, _ in mc], yerr=[h for _, h in mc], marker=MARK[m],
                        color=COLORS[m], label=m, capsize=2, ms=4, lw=1.2)
    lb = defaultdict(list)
    for r in rows:
        if r["method"] == "CDG-LB" and r["lb"] > 0:
            lb[r["n"]].append(r["lb"])
    ns = sorted(lb)
    axs[2].plot(ns, [mean(lb[n]) for n in ns], ls="--", color="k", label=fa("کران پایین"))
    axs[0].set_ylabel(fa("نرخ موفقیت (درصد)"))
    axs[1].set_ylabel(fa("میانه‌ی زمان (ثانیه)"))
    axs[1].set_yscale("log")
    axs[2].set_ylabel(fa("طول طرح"))
    all_n = sorted(set(r["n"] for r in rows))
    for ax in axs:
        ax.set_xlabel(fa("تعداد قوطی‌ها"))
        ax.set_xticks(all_n)
        ax.xaxis.set_major_formatter(FuncFormatter(fa_num))
        ax.grid(alpha=0.3)
    axs[0].yaxis.set_major_formatter(FuncFormatter(fa_num))
    axs[1].yaxis.set_major_formatter(FuncFormatter(fa_num))
    axs[2].yaxis.set_major_formatter(FuncFormatter(fa_num))
    axs[0].set_ylim(-3, 103)
    axs[2].legend(frameon=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, name)


def fig_mono(rows):
    """Motion-planner queries needed to decide monotonicity."""
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, options in zip(axs, ["single", "multi"]):
        sub = [r for r in rows if r["options"] == options]
        for m in ["CIRS", "DFS_DP", "mRS"]:
            for kind, ls in (("feasible", "-"), ("infeasible", "--")):
                g = defaultdict(list)
                for r in sub:
                    if r["method"] == m and r["kind"] == kind and r["decision"] != "timeout":
                        g[r["n"]].append(max(r["calls"], 1))
                ns = sorted(g)
                ax.plot(ns, [mean(g[n]) for n in ns], marker=MARK[m], color=COLORS[m], ls=ls, ms=4, lw=1.2,
                        label=m.replace("DFS_DP", "DFS-DP") if kind == "feasible" else None)
        ax.set_yscale("log")
        ax.set_xticks(sorted(set(r["n"] for r in sub)))
        ax.xaxis.set_major_formatter(FuncFormatter(fa_num))
        ax.set_xlabel(fa("تعداد قوطی‌ها"))
        ax.set_ylabel(fa("پرس‌وجوی برنامه‌ریز حرکت"))
        ax.grid(alpha=0.3)
        ax.set_title(fa("یک گزینه‌ی ورود" if options == "single" else "ده گزینه‌ی ورود"), fontsize=9)
    axs[0].plot([], [], color="0.4", ls="--", label=fa("غیریکنوا"))
    axs[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_mono_calls.pdf")


def fig_variants(groups, labels):
    """groups: list of {method: rows}. Horizontal bars, first condition on top."""
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True)
    y = np.arange(len(groups))[::-1]
    h = 0.38
    for k, m in enumerate(["CDG-LB", "PERTS"]):
        succ = [100 * mean([r["success"] for r in g[m]]) for g in groups]
        axs[0].barh(y + (0.5 - k) * h, succ, h, color=COLORS[m], label=m)
        acts = [mean([r["actions"] for r in g[m] if r["success"]]) for g in groups]
        axs[1].barh(y + (0.5 - k) * h, [0 if math.isnan(a) else a for a in acts], h, color=COLORS[m], label=m)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([fa(l) for l in labels])
    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(fa_num))
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(axis="y", length=0)
    axs[0].set_xlabel(fa("نرخ موفقیت (درصد)"))
    axs[1].set_xlabel(fa("طول طرح (موارد موفق)"))
    axs[0].set_xlim(0, 105)
    axs[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_variants.pdf")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def fmt(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{d}f}".replace(".", "٫")


def fmt_t(v):
    # short times get one more digit so they do not show as zero
    if v is not None and not (isinstance(v, float) and math.isnan(v)) and v < 0.1:
        return fmt(v, 3)
    return fmt(v, 2)


def write(name, text):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def two(a, b):
    """Header cell on two lines, so that wide tables need less shrinking."""
    return r"\begin{tabular}[b]{@{}c@{}}" + a + r"\\" + b + r"\end{tabular}"


def table(name, label, caption, cols, head, body, resize=True):
    inner = rf"""\begin{{tabular}}{{{cols}}}
\toprule
{head}
\midrule
{body}
\bottomrule
\end{{tabular}}"""
    if resize:
        inner = r"\fittable{%" + "\n" + inner + "}"
    text = rf"""\begin{{table}}[!htb]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
{inner}
\end{{table}}
"""
    write(name, text)


def table_model(e0):
    lines = []
    for r in sorted(e0, key=lambda r: (r["pillars"], r["width"] != 1.0, r["width"], r["lying"])):
        cond = []
        if r["pillars"]:
            cond.append("ستون ثابت")
        cond.append(f"خوابیده {fmt(100 * r['lying'], 0)}\\%")
        cond.append(f"عرض {fmt(r['width'], 1)}")
        lines.append(" & ".join(["، ".join(cond), str(r["poses"]), str(r["motions"]), str(r["valid_both"]),
                                 str(r["valid_only_planner"] + r["valid_only_fine"]),
                                 f"{r['pairs']:,}".replace(",", "٬"), str(r["unsafe"]),
                                 fmt(100 * r["extra"] / r["pairs"], 2)]) + r" \\")
    table("tab_model.tex", "tab:model", "مقایسه‌ی مدل جاروب با بررسی لحظه‌به‌لحظه‌ی مستقل.",
          "lccccccc",
          r"شرایط & $|\cP|$ & حرکت & معتبر & " + two("اختلاف", "اعتبار") + " & " + two("زوج", "بررسی‌شده")
          + r" & ناامن & " + two("محافظه‌کارانه", r"(\%)") + r" \\",
          "\n".join(lines))


def table_main(rows, name, caption, label):
    lines = []
    for n in sorted(set(r["n"] for r in rows)):
        c = [r for r in rows if r["n"] == n and r["method"] == "CDG-LB"]
        p = [r for r in rows if r["n"] == n and r["method"] == "PERTS"]
        cs = [r for r in c if r["success"]]
        ps = [r for r in p if r["success"]]
        lbs = [r["lb"] for r in c if r["lb"] > 0]
        gap = mean([(r["actions"] - r["lb"]) / r["lb"] * 100 for r in cs if r["lb"] > 0])
        lines.append(" & ".join([
            f"{n}", fmt(mean([r["density"] for r in c]), 2),
            fmt(100 * mean([r['success'] for r in c]), 0), fmt(100 * mean([r['success'] for r in p]), 0),
            fmt_t(mean([r['time'] for r in c])), fmt_t(mean([r['time'] for r in p])),
            fmt(mean([r['calls'] for r in c]), 0), fmt(mean([r['calls'] for r in p]), 0),
            fmt(mean(lbs), 1),
            fmt(mean([r['actions'] for r in cs]), 1), fmt(mean([r['actions'] for r in ps]), 1),
            fmt(mean([r['buffers'] for r in cs]), 1), fmt(gap, 1),
        ]) + r" \\")
    head = (r" & & \multicolumn{2}{c}{موفقیت (\%)} & \multicolumn{2}{c}{زمان (ثانیه)} & "
            r"\multicolumn{2}{c}{پرس‌وجو} & & \multicolumn{2}{c}{طول طرح} & & \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){10-11}" "\n"
            r"$n$ & تراکم & \CDGLB & \lr{PERTS} & \CDGLB & \lr{PERTS} & \CDGLB & \lr{PERTS} & $\LB$ & "
            r"\CDGLB & \lr{PERTS} & بافر & " + two(r"فاصله تا $\LB$", r"(\%)") + r" \\")
    table(name, label, caption, "ccccccccccccc", head, "\n".join(lines))


def table_mono(rows):
    lines = []
    for options, label in (("single", "۱"), ("multi", "۱۰")):
        sub = [r for r in rows if r["options"] == options]
        for n in sorted(set(r["n"] for r in sub)):
            cells = [label, str(n), str(len({r["seed"] for r in sub if r["n"] == n and r["kind"] == "infeasible"}))]
            for m in ["CDG", "CIRS", "DFS_DP", "mRS"]:
                rr = [r for r in sub if r["n"] == n and r["method"] == m]
                to = sum(r["decision"] == "timeout" for r in rr)
                t = mean([r["time"] * 1000 for r in rr if r["decision"] != "timeout"])
                cells.append(fmt(t, 3 if (m == "CDG" and options == "single") else 1) + (f" ({to})" if to else ""))
                if m != "CDG":
                    cells.append(fmt(mean([r["calls"] for r in rr if r["decision"] != "timeout"]), 0))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    head = (r" & & & \CDG & \multicolumn{2}{c}{\lr{CIRS}} & \multicolumn{2}{c}{\lr{DFS\textsubscript{DP}}} & "
            r"\multicolumn{2}{c}{\lr{mRS}} \\" "\n"
            r"\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" "\n"
            r"$K$ & $n$ & غیریکنوا & زمان & زمان & پرس‌وجو & زمان & پرس‌وجو & زمان & پرس‌وجو \\")
    table("tab_mono.tex", "tab:mono",
          "زمان میانگین تصمیم یکنوایی (میلی‌ثانیه) و تعداد پرس‌وجو روی شش نمونه‌ی یکنوا و تعداد درج‌شده نمونه‌ی غیریکنوا؛ عدد داخل پرانتز، اجراهای رسیده به سقف ده ثانیه است.",
          "cccccccccc", head, "\n".join(lines[:-1]))


def optimum_rows(rows):
    """For each small instance: CDG-LB result, optimum (if known), PERTS result."""
    out = []
    for s in sorted({r["seed"] for r in rows}):
        by = {r["method"]: r for r in rows if r["seed"] == s}
        c, a, p = by.get("CDG-LB"), by.get("A*"), by.get("PERTS")
        opt = None
        if c and c["success"] and a:
            if a["note"] == "no shorter plan":
                opt = c["actions"]
            elif a["success"]:
                opt = a["actions"]
        out.append((c, a, p, opt))
    return out


def table_small(rows):
    lines = []
    for n in sorted(set(r["n"] for r in rows)):
        info = optimum_rows([r for r in rows if r["n"] == n])
        known = [(c, p, o) for c, a, p, o in info if o is not None]
        c_all = [c for c, a, p, o in info]
        lines.append(" & ".join([
            str(n), fmt(mean([c["density"] for c in c_all]), 2),
            f"{sum(c['success'] for c in c_all)}/{len(c_all)}",
            f"{len(known)}/{len(c_all)}", fmt(mean([o for _, _, o in known]), 1),
            f"{sum(c['actions'] == o for c, _, o in known)}/{len(known)}",
            fmt(mean([c['actions'] - o for c, _, o in known]), 2),
            fmt(mean([p['actions'] - o for _, p, o in known if p and p['success']]), 2),
            f"{sum(c['lb'] == o for c, _, o in known)}/{len(known)}",
            fmt_t(mean([a["time"] for c, a, p, o in info if a])),
            fmt_t(mean([c["time"] for c in c_all])),
        ]) + r" \\")
    head = " & ".join([r"$n$", "تراکم", two("حل", r"\CDGLB"), two("بهینه‌ی", "معلوم"), two("میانگین", "بهینه"),
                       two(r"\CDGLB", "بهینه"), two("اضافه‌ی", r"\CDGLB"), two("اضافه‌ی", r"\lr{PERTS}"),
                       r"$\LB=\mathrm{OPT}$", two("زمان", r"\lr{A*}"), two("زمان", r"\CDGLB")]) + r" \\"
    table("tab_small.tex", "tab:small", "مقایسه با جواب بهینه در قفسه‌ی کوچک به عرض 0٫5 و عمق 0٫3 متر (بیست نمونه برای هر $n$).",
          "ccccccccccc", head, "\n".join(lines))


def table_ablation(e2, e5):
    names = [("CDG-LB", "روش کامل"), ("CDG-LB/no-smoothing", "بدون هموارسازی"),
             ("CDG-LB/no-dwell", "بدون مرحله‌ی دوم"),
             ("CDG-LB/random-buffer", "بافر تصادفی"),
             ("TRLB-table", "گراف رومیزی")]
    lines = []
    for n in [12, 18]:
        seeds = set(r["seed"] for r in e5 if r["n"] == n)
        for m, label in names:
            src = e2 if m == "CDG-LB" else e5
            rr = [r for r in src if r["n"] == n and r["method"] == m and r["seed"] in seeds]
            ok = [r for r in rr if r["success"]]
            lines.append(" & ".join([
                str(n), label, fmt(100 * mean([r["success"] for r in rr]), 0),
                fmt_t(mean([r["time"] for r in rr])),
                fmt(mean([r["actions"] for r in ok]), 1),
                fmt(mean([r["buffers"] for r in ok]), 1),
                fmt(mean([r["calls"] for r in rr]), 0),
            ]) + r" \\")
        lines.append(r"\midrule")
    table("tab_ablation.tex", "tab:ablation", "اثر هر بخش از روش (ده نمونه‌ی هدف مرتب برای هر $n$).",
          "ccccccc", r"$n$ & نسخه & موفقیت (\%) & زمان (ثانیه) & طول طرح & بافر & پرس‌وجو \\",
          "\n".join(lines[:-1]), resize=False)


def lying_share(r):
    """Share of lying cans set by the job, or None.

    Every row stores the number of lying cans under "lying"; the jobs that
    change the share overwrite it with the share itself, which is a float."""
    v = r.get("lying")
    return v if isinstance(v, float) else None


def variant_groups(e2, e6):
    """Rows of each condition in Experiment 6, with the matching E2 rows as reference."""
    seeds = {r["seed"] for r in e6 if lying_share(r) is not None or "width" in r}
    base15 = [r for r in e2 if r["n"] == 15 and r["seed"] in seeds]
    out = []
    for n in [8, 12, 16]:
        out.append((f"ستون ثابت، $n={n}$", f"ستون، {str(n).translate(FA_DIGITS)} قوطی",
                    [r for r in e6 if r.get("pillars") and r["n"] == n]))
    for lying, lab in [(0.0, "۰"), (0.3, "۳۰"), (0.6, "۶۰")]:
        rr = base15 if lying == 0.3 else [r for r in e6 if lying_share(r) == lying]
        out.append((f"خوابیده {lab}\\%", f"خوابیده {lab}٪", rr))
    for width in [0.8, 0.9, 1.0, 1.1]:
        rr = base15 if width == 1.0 else [r for r in e6 if abs(r.get("width", -1) - width) < 1e-9]
        out.append((f"عرض {fmt(width, 1)} متر", f"عرض {str(width).translate(FA_DIGITS)}", rr))
    return out


def table_variants(groups):
    lines = []
    for k, (label, _, rr) in enumerate(groups):
        cells = [label, fmt(mean([r["density"] for r in rr]), 2)]
        for m in ["CDG-LB", "PERTS"]:
            x = [r for r in rr if r["method"] == m]
            ok = [r for r in x if r["success"]]
            cells += [fmt(100 * mean([r["success"] for r in x]), 0), fmt_t(mean([r["time"] for r in x])),
                      fmt(mean([r["actions"] for r in ok]), 1)]
        cells.append(fmt(mean([r["buffers"] for r in rr if r["method"] == "CDG-LB" and r["success"]]), 1))
        lines.append(" & ".join(cells) + r" \\")
        if k in (2, 5):
            lines.append(r"\midrule")
    head = (r" & & \multicolumn{3}{c}{\CDGLB} & \multicolumn{3}{c}{\lr{PERTS}} & \\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}" "\n"
            + two("شرایط", r"($n=15$ مگر ذکر شود)") + " & تراکم & " + two("موفقیت", r"(\%)")
            + " & زمان & طول & " + two("موفقیت", r"(\%)") + r" & زمان & طول & بافر \\")
    table("tab_variants.tex", "tab:variants", "نتایج در شرایط مختلف قفسه (ده نمونه برای هر ردیف).",
          "lcccccccc", head, "\n".join(lines))


def table_sensitivity(e7):
    rows_out = []
    settings = [(0.04, "multi", 10), (0.06, "multi", 10), (0.08, "multi", 10),
                (0.06, "three", 6), (0.06, "front", 2)]
    for n in [15, 18]:
        for grid, options, k in settings:
            rr = [r for r in e7 if r["n"] == n and abs(r["grid"] - grid) < 1e-9 and r["options"] == options]
            ok = [r for r in rr if r["success"]]
            gaps = [(r["actions"] - r["lb"]) / r["lb"] * 100 for r in ok if r["lb"] > 0]
            rows_out.append(" & ".join([
                str(n), fmt(100 * grid, 0), str(k), str(round(mean([r["poses"] for r in rr]))),
                fmt(mean([r["prep"] for r in rr]), 1),
                fmt(100 * mean([r["success"] for r in rr]), 0), fmt_t(mean([r["time"] for r in rr])),
                fmt(mean([r["actions"] for r in ok]), 1), fmt(mean(gaps), 1),
            ]) + r" \\")
        rows_out.append(r"\midrule")
    head = " & ".join([r"$n$", two("گام شبکه", "(سانتی‌متر)"), r"$K$", r"$|\cP|$", two("پیش‌محاسبه", "(ثانیه)"),
                       two("موفقیت", r"(\%)"), two("زمان", "(ثانیه)"), "طول طرح",
                       two(r"فاصله تا $\LB$", r"(\%)")]) + r" \\"
    table("tab_sensitivity.tex", "tab:sensitivity",
          "حساسیت به گام شبکه و تعداد گزینه‌های ورود (ده نمونه‌ی توسعه در هر ردیف؛ پیش‌فرض: گام ۶ و $K=10$).",
          "ccccccccc", head, "\n".join(rows_out[:-1]))


# ---------------------------------------------------------------------------
# Numbers quoted in the text
# ---------------------------------------------------------------------------
def summary(ex):
    runs = [r for rows in ex.values() for r in rows if "method" in r]
    planners = [r for r in runs if r["exp"] != "E1"]
    ok = [r for r in planners if r["success"]]
    print("runs:", len(runs), "planner runs:", len(planners), "with a plan:", len(ok),
          "all verified:", all(r["verified"] for r in ok))
    solved_but_rejected = [r for rows in ex.values() for r in rows
                           if "method" in r and r["exp"] != "E1" and r["actions"] is None and r["note"].startswith("rounds")]
    print("plans rejected by the checker:", len(solved_but_rejected))
    inst = set()
    for r in runs:
        # E7 runs the same development instances with several settings
        options = "multi" if r["exp"] == "E7" else r.get("options", "multi")
        key = (r["exp"] == "E7", r["n"], r["seed"], options, r.get("kind"), lying_share(r),
               r.get("width", 1.0), r.get("pillars", False), r.get("style"))
        inst.add(key)
    dev = [k for k in inst if k[0]]
    print("distinct instances: evaluation", len(inst) - len(dev), "development", len(dev),
          "| runs: evaluation", sum(r["exp"] != "E7" for r in runs), "development",
          sum(r["exp"] == "E7" for r in runs), "monotonicity-test runs", sum(r["exp"] == "E1" for r in runs))
    e0 = ex["E0"]
    if e0:
        print("E0 pairs", sum(r["pairs"] for r in e0), "unsafe", sum(r["unsafe"] for r in e0),
              "extra %", round(100 * sum(r["extra"] for r in e0) / sum(r["pairs"] for r in e0), 3),
              "validity mismatch", sum(r["valid_only_planner"] + r["valid_only_fine"] for r in e0))
    e1 = ex["E1"]
    if e1:
        agree = defaultdict(set)
        for r in e1:
            if r["decision"] != "timeout":
                agree[(r["n"], r["seed"], r["options"], r["kind"])].add(r["decision"])
        print("E1 instances", len(agree), "decisions agree:", all(len(v) == 1 for v in agree.values()),
              "single:", sum(k[2] == "single" for k in agree),
              "timeouts:", {m: sum(r["decision"] == "timeout" for r in e1 if r["method"] == m) for m in ["CDG", "CIRS", "DFS_DP", "mRS"]})
        for options in ["single", "multi"]:
            t = [r["time"] * 1000 for r in e1 if r["method"] == "CDG" and r["options"] == options]
            print("  CDG", options, "ms min/mean/max", round(min(t), 4), round(mean(t), 3), round(max(t), 3))
    for name in ["E2", "E3"]:
        rows = ex[name]
        for n in sorted(set(r["n"] for r in rows)):
            c = [r for r in rows if r["n"] == n and r["method"] == "CDG-LB"]
            p = [r for r in rows if r["n"] == n and r["method"] == "PERTS"]
            a1, a2 = pairs(rows, "actions", n=n)
            q1, q2 = pairs(rows, "calls", n=n)
            np_, pa = paired_test(a1, a2)
            _, pq = paired_test(q1, q2)
            sc = sum(r["success"] for r in c)
            sp = sum(r["success"] for r in p)
            print(name, n, "succ", sc, sp, "/", len(c), "wilson", [round(x) for x in wilson(sc, len(c))],
                  [round(x) for x in wilson(sp, len(p))],
                  "time", fmt(mean([r["time"] for r in c]), 2), fmt(mean([r["time"] for r in p]), 2),
                  "calls", fmt(mean([r["calls"] for r in c]), 0), fmt(mean([r["calls"] for r in p]), 0),
                  "| paired", np_, "act", fmt(mean(a1), 1), fmt(mean(a2), 1), "p=%.3g" % pa,
                  "calls", fmt(mean(q1), 0), fmt(mean(q2), 0), "p=%.3g" % pq,
                  "| lb", fmt(mean([r["lb"] for r in c if r["lb"] > 0]), 1),
                  "gap%", fmt(mean([(r["actions"] - r["lb"]) / r["lb"] * 100 for r in c if r["success"] and r["lb"] > 0]), 1),
                  "infeasible", sum(r["infeasible"] for r in c))
    e4 = ex["E4"]
    if e4:
        info = optimum_rows(e4)
        known = [(c, p, o) for c, a, p, o in info if o is not None]
        print("E4 instances", len(info), "optimum known", len(known),
              "CDG optimal", sum(c["actions"] == o for c, _, o in known),
              "LB tight", sum(c["lb"] == o for c, _, o in known),
              "max extra", max(c["actions"] - o for c, _, o in known),
              "A* limits", sum(1 for c, a, p, o in info if a and a["note"] == "limit"),
              "PERTS solved", sum(1 for c, a, p, o in info if p and p["success"]),
              "PERTS optimal", sum(1 for c, p, o in known if p and p["success"] and p["actions"] == o))
    for label, key in (("E5", "E5"), ("E6", "E6"), ("E7", "E7")):
        print(label, "rows", len(ex[key]))
    if ex["E7"]:
        print("E7 instances reported infeasible:", sum(r["infeasible"] for r in ex["E7"]))


def main():
    ex = {name: load(name) for name in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]}
    fig_model()
    fig_cdg_example()
    fig_snapshots()
    if ex["E0"]:
        table_model(ex["E0"])
    if ex["E1"]:
        fig_mono(ex["E1"])
        table_mono(ex["E1"])
    if ex["E2"]:
        fig_main(ex["E2"], "fig_main_rows.pdf")
        table_main(ex["E2"], "tab_main_rows.tex",
                   "هدف مرتب: میانگین بیست نمونه برای هر $n$؛ طول طرح و بافر تنها برای موارد موفق.",
                   "tab:main-rows")
    if ex["E3"]:
        table_main(ex["E3"], "tab_main_random.tex",
                   "هدف نامرتب: میانگین ده نمونه برای هر $n$.", "tab:main-random")
    if ex["E4"]:
        table_small(ex["E4"])
    if ex["E5"] and ex["E2"]:
        table_ablation(ex["E2"], ex["E5"])
    if ex["E6"] and ex["E2"]:
        groups = variant_groups(ex["E2"], ex["E6"])
        table_variants(groups)
        fig_variants([{m: [r for r in rr if r["method"] == m] for m in ["CDG-LB", "PERTS"]} for _, _, rr in groups],
                     [short for _, short, _ in groups])
    if ex["E7"]:
        table_sensitivity(ex["E7"])
    summary(ex)


if __name__ == "__main__":
    main()
