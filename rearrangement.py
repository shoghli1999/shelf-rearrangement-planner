"""
Rearranging cans inside a shelf with a fixed-base robot arm.

This file holds everything the experiments need:
  - the shelf / arm / can model (top view, 2D),
  - the swept-area collision test for one reach-in motion,
  - the planners compared in the thesis (mRS, DFS_DP, CIRS, PERTS,
    tabletop dependency graph, A*, and the proposed CDG-LB),
  - an independent plan checker.

Units are metres and radians.
"""

import math
import time
import heapq
import random
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model


# ----------------------------------------------------------------------------
# Constants of the physical setup
# ----------------------------------------------------------------------------
CAN_R = 0.035           # can radius
TOUCH = 1e-7            # two shapes closer than this count as touching
CAN_LEN = 0.12          # can length (used when the can is lying down)
HALF_AXIS = CAN_LEN / 2 - CAN_R   # half length of the capsule axis of a lying can
GAP = 0.002             # minimum clearance between two cans
PILLAR_R = 0.03         # radius of fixed pillars (static obstacles)
STEP = 0.01             # sampling step along a reach-in motion


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def seg_dist(a0, a1, b0, b1):
    """Distances between every segment a0a1 (N,2) and every segment b0b1 (M,2).

    Returns an (N, M) array. Segments of zero length (points) are allowed.
    Based on the closest-point method in Ericson, Real-Time Collision Detection.
    """
    a0 = np.asarray(a0, float)
    a1 = np.asarray(a1, float)
    b0 = np.asarray(b0, float)
    b1 = np.asarray(b1, float)
    d1x, d1y = (a1[:, 0] - a0[:, 0])[:, None], (a1[:, 1] - a0[:, 1])[:, None]
    d2x, d2y = (b1[:, 0] - b0[:, 0])[None, :], (b1[:, 1] - b0[:, 1])[None, :]
    rx = a0[:, 0][:, None] - b0[:, 0][None, :]
    ry = a0[:, 1][:, None] - b0[:, 1][None, :]
    a = d1x * d1x + d1y * d1y
    e = d2x * d2x + d2y * d2y
    f = d2x * rx + d2y * ry
    c = d1x * rx + d1y * ry
    b = d1x * d2x + d1y * d2y
    eps = 1e-12
    a, e = np.broadcast_to(a, f.shape), np.broadcast_to(e, f.shape)
    a_ok = a > eps
    e_ok = e > eps
    safe_a = np.where(a_ok, a, 1.0)
    safe_e = np.where(e_ok, e, 1.0)

    denom = a * e - b * b
    good = denom > eps
    s = np.where(good, np.clip((b * f - c * e) / np.where(good, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / safe_e
    s = np.where(t < 0, np.clip(-c / safe_a, 0, 1), np.where(t > 1, np.clip((b - c) / safe_a, 0, 1), s))
    t = np.clip(t, 0, 1)

    # degenerate cases
    s = np.where(~a_ok, 0.0, s)
    t = np.where(~a_ok & e_ok, np.clip(f / safe_e, 0, 1), t)
    t = np.where(~e_ok, 0.0, t)
    s = np.where(~e_ok & a_ok, np.clip(-c / safe_a, 0, 1), s)

    dx = rx + d1x * s - d2x * t
    dy = ry + d1y * s - d2y * t
    return np.sqrt(dx * dx + dy * dy)


@dataclass
class Pose:
    """Top-view pose of a can. A lying can is a capsule, an upright can a disc."""
    x: float
    y: float
    th: float = 0.0
    lying: bool = False

    def axis(self):
        if not self.lying:
            p = np.array([self.x, self.y])
            return p, p
        d = HALF_AXIS * np.array([math.cos(self.th), math.sin(self.th)])
        c = np.array([self.x, self.y])
        return c - d, c + d

    def highest_y(self):
        return self.y + CAN_R + (HALF_AXIS * abs(math.sin(self.th)) if self.lying else 0.0)


# ----------------------------------------------------------------------------
# Shelf and arm
# ----------------------------------------------------------------------------
class Shelf:
    """A shelf open only at the front (y = 0). Closed on the left, right, back and top.

    The robot base is fixed in front of the shelf. The arm is a planar chain of
    three links: upper arm, forearm and hand. The hand enters the shelf along a
    straight line with a fixed direction (the approach option).
    """

    def __init__(self, width=1.0, depth=0.4, pillars=(), options="multi"):
        self.width = width
        self.depth = depth
        self.base = np.array([width / 2, -0.45])
        # links are made long enough to reach the far corners of the shelf
        need = math.hypot(width / 2 - CAN_R, depth - CAN_R + 0.45) - 0.155 + 0.02
        scale = max(1.0, need / 0.85)
        self.l1, self.l2, self.hand = 0.45 * scale, 0.40 * scale, 0.12
        self.link_r = 0.03
        self.pillars = np.array(pillars, float).reshape(-1, 2)
        # approach options: entry angle (degrees) and elbow branch
        angles, elbows = {
            "single": ([0], [1]),
            "front": ([0], [1, -1]),
            "three": ([-15, 0, 15], [1, -1]),
            "multi": ([-30, -15, 0, 15, 30], [1, -1]),
        }[options]
        self.options = [(math.radians(a), e) for a in angles for e in elbows]
        w, d = width, depth
        self.wall_a = np.array([[0, 0], [w, 0], [0, d]], float)
        self.wall_b = np.array([[0, d], [w, d], [w, d]], float)

    # -- a single reach-in motion ------------------------------------------
    def sweep(self, pose, option):
        """Segments swept by the arm and by the held can when the can goes from
        outside the shelf to `pose` along approach `option`.

        Returns (arm_a, arm_b, can_a, can_b) or None if the motion is not
        possible (no IK, or the arm/can hits a wall or a pillar).
        """
        phi, elbow = option
        u = np.array([math.sin(phi), math.cos(phi)])     # direction into the shelf
        a0, a1 = pose.axis()
        c = np.array([pose.x, pose.y])
        dirv = (a1 - a0) / (2 * HALF_AXIS) if pose.lying else np.zeros(2)
        reach = CAN_R + (HALF_AXIS * abs(dirv @ u) if pose.lying else 0.0)

        # travel until the whole can (its highest point too) has left the shelf
        travel = (pose.highest_y() + 0.005) / u[1]
        m = max(2, int(math.ceil(travel / STEP)) + 1)
        ts = np.linspace(0.0, travel, m)
        centers = c[None, :] - ts[:, None] * u[None, :]
        tips = centers - reach * u
        wrists = tips - self.hand * u

        # inverse kinematics of the first two links
        dv = wrists - self.base
        dist2 = np.sum(dv * dv, 1)
        cos2 = (dist2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if np.any(np.abs(cos2) > 1.0):
            return None
        q2 = elbow * np.arccos(cos2)
        q1 = np.arctan2(dv[:, 1], dv[:, 0]) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        elbows = self.base + self.l1 * np.stack([np.cos(q1), np.sin(q1)], 1)

        # arm segments: upper arm and forearm sampled, hand is one straight segment
        arm_a = np.vstack([np.repeat(self.base[None], m, 0), elbows, wrists[-1:]])
        arm_b = np.vstack([elbows, wrists, tips[:1]])
        # can segments: the two ends of the can axis moving along u, plus the can at both ends
        shift = travel * u
        can_a = np.array([a0, a1, a0, a0 - shift])
        can_b = np.array([a0 - shift, a1 - shift, a1, a1 - shift])

        # static checks: walls and pillars
        if np.any(seg_dist(arm_a, arm_b, self.wall_a, self.wall_b) < self.link_r):
            return None
        if np.any(seg_dist(can_a, can_b, self.wall_a, self.wall_b) < CAN_R - 1e-9):
            return None
        if len(self.pillars):
            if np.any(seg_dist(arm_a, arm_b, self.pillars, self.pillars) < self.link_r + PILLAR_R):
                return None
            if np.any(seg_dist(can_a, can_b, self.pillars, self.pillars) < CAN_R + PILLAR_R):
                return None

        # only arm pieces that come close to the shelf can touch a can
        near = np.maximum(arm_a[:, 1], arm_b[:, 1]) > -(CAN_R + self.link_r)
        return arm_a[near], arm_b[near], can_a, can_b

    def inside(self, pose):
        a0, a1 = pose.axis()
        for p in (a0, a1):
            if not (CAN_R <= p[0] <= self.width - CAN_R and CAN_R <= p[1] <= self.depth - CAN_R):
                return False
        if len(self.pillars):
            d = seg_dist(np.array([a0]), np.array([a1]), self.pillars, self.pillars)
            if np.any(d < CAN_R + PILLAR_R + GAP):
                return False
        return True


def fine_motion(shelf, pose, option, step=0.002):
    """The arm and the held can at many single instants of one reach-in motion.

    Unlike Shelf.sweep, nothing is merged into a swept area: every instant is
    kept as it is, with a 2 mm step. Returns (link_a, link_b, can_a, can_b)
    stacked over all instants, or None if the motion is not possible.
    """
    phi, elbow = option
    u = np.array([math.sin(phi), math.cos(phi)])
    a0, a1 = pose.axis()
    c = np.array([pose.x, pose.y])
    dirv = (a1 - a0) / (2 * HALF_AXIS) if pose.lying else np.zeros(2)
    reach = CAN_R + (HALF_AXIS * abs(dirv @ u) if pose.lying else 0.0)
    ts = np.arange(0.0, (pose.highest_y() + 0.005) / u[1] + step, step)
    tips = c[None] - ts[:, None] * u[None] - reach * u
    wrists = tips - shelf.hand * u
    dv = wrists - shelf.base
    cos2 = (np.sum(dv * dv, 1) - shelf.l1 ** 2 - shelf.l2 ** 2) / (2 * shelf.l1 * shelf.l2)
    if np.any(np.abs(cos2) > 1.0):
        return None
    q2 = elbow * np.arccos(cos2)
    q1 = np.arctan2(dv[:, 1], dv[:, 0]) - np.arctan2(shelf.l2 * np.sin(q2), shelf.l1 + shelf.l2 * np.cos(q2))
    elbows = shelf.base + shelf.l1 * np.stack([np.cos(q1), np.sin(q1)], 1)
    m = len(ts)
    link_a = np.vstack([np.repeat(shelf.base[None], m, 0), elbows, wrists])
    link_b = np.vstack([elbows, wrists, tips])
    can_a = a0[None] - ts[:, None] * u[None]
    can_b = a1[None] - ts[:, None] * u[None]
    walls_a, walls_b = shelf.wall_a, shelf.wall_b
    if len(shelf.pillars):
        walls_a = np.vstack([walls_a, shelf.pillars])
        walls_b = np.vstack([walls_b, shelf.pillars])
    pr = np.r_[np.zeros(3), np.full(len(shelf.pillars), PILLAR_R)]
    if np.any(seg_dist(link_a, link_b, walls_a, walls_b) < shelf.link_r + pr[None, :]):
        return None
    if np.any(seg_dist(can_a, can_b, walls_a, walls_b) < CAN_R + pr[None, :] - 1e-9):
        return None
    return link_a, link_b, can_a, can_b


def blocked_by_sweep(sw, qa, qb, link_r):
    """Which cans (axes qa->qb) touch the swept area `sw`. Returns a bool array."""
    arm_a, arm_b, can_a, can_b = sw
    pts = np.vstack([arm_a, arm_b, can_a, can_b])
    reach = max(2 * CAN_R, link_r + CAN_R) + GAP + HALF_AXIS
    lo = pts.min(0) - reach
    hi = pts.max(0) + reach
    mid = (qa + qb) / 2
    near = np.nonzero(np.all((mid >= lo) & (mid <= hi), 1))[0]
    hit = np.zeros(len(qa), bool)
    if len(near) == 0:
        return hit
    na, nb = qa[near], qb[near]
    # the same safety gap is kept for the held can and for the arm
    h = seg_dist(can_a, can_b, na, nb).min(0) < 2 * CAN_R + GAP
    if len(arm_a):
        h |= seg_dist(arm_a, arm_b, na, nb).min(0) < link_r + CAN_R + GAP
    hit[near] = h
    return hit


# ----------------------------------------------------------------------------
# Problem instance with precomputed blocking masks
# ----------------------------------------------------------------------------
def bits_of(bool_array):
    v = 0
    for q in np.nonzero(bool_array)[0]:
        v |= 1 << int(q)
    return v


class Instance:
    """Start/goal arrangements plus a table of all poses the planners may use.

    Pose ids: 0..n-1 are starts, n..2n-1 are goals, the rest are candidate
    buffer poses. For every pose p and approach option k we store
    mask[p][k], the set of poses (as a bit mask) that block this motion.
    """

    def __init__(self, shelf, starts, goals, kinds, grid=0.06):
        t0 = time.perf_counter()
        self.shelf = shelf
        self.n = len(starts)
        self.kinds = kinds
        poses = list(starts) + list(goals)
        self.cand_first = len(poses)
        if "up" in kinds:
            poses += self._grid(False, grid)
        if "lie" in kinds:
            poses += self._grid(True, grid)
        self.poses = poses
        self.P = len(poses)
        axes = [p.axis() for p in poses]
        self.qa = np.array([a for a, _ in axes])
        self.qb = np.array([b for _, b in axes])
        self.bit = [1 << q for q in range(self.P)]

        # footprint overlaps
        dd = seg_dist(self.qa, self.qb, self.qa, self.qb)
        ov = dd < 2 * CAN_R + GAP
        self.overlap = [bits_of(ov[p]) for p in range(self.P)]

        # blocking masks for each pose and option
        K = len(shelf.options)
        self.valid = [[] for _ in range(self.P)]
        self.mask = [[0] * K for _ in range(self.P)]
        for p in range(self.P):
            for k, opt in enumerate(shelf.options):
                sw = shelf.sweep(poses[p], opt)
                if sw is None:
                    continue
                self.valid[p].append(k)
                self.mask[p][k] = bits_of(blocked_by_sweep(sw, self.qa, self.qb, shelf.link_r))

        # candidate buffers for each kind: statically reachable grid poses plus starts and goals
        self.cands = {}
        for kind in set(kinds):
            lying = kind == "lie"
            ids = [q for q in range(self.cand_first, self.P)
                   if poses[q].lying == lying and self.valid[q]]
            ids += [q for q in range(2 * self.n) if poses[q].lying == lying]
            self.cands[kind] = ids

        # beta score: how many candidate poses become unreachable if a can sits at q
        all_c = sorted(set(c for ids in self.cands.values() for c in ids))
        blocks_all = {}
        for c in all_c:
            m = -1
            for k in self.valid[c]:
                m &= self.mask[c][k]
            blocks_all[c] = m
        self.beta = {}
        for q in all_c:
            self.beta[q] = sum(1 for c in all_c if c != q and blocks_all[c] >> q & 1)
        self.prep_time = time.perf_counter() - t0
        self.calls = 0

    def swap_goals(self, i, j):
        """Copy of this instance where objects i and j exchange their goals.
        The pose table stays the same, so the masks only need two bits swapped."""
        a, b = self.n + i, self.n + j
        new = object.__new__(Instance)
        new.__dict__.update(self.__dict__)

        def sw(v):
            if (v >> a & 1) != (v >> b & 1):
                v ^= (1 << a) | (1 << b)
            return v

        new.poses = list(self.poses)
        new.poses[a], new.poses[b] = self.poses[b], self.poses[a]
        new.qa, new.qb = self.qa.copy(), self.qb.copy()
        new.qa[[a, b]] = self.qa[[b, a]]
        new.qb[[a, b]] = self.qb[[b, a]]
        new.overlap = [sw(v) for v in self.overlap]
        new.overlap[a], new.overlap[b] = new.overlap[b], new.overlap[a]
        new.mask = [[sw(v) for v in row] for row in self.mask]
        new.mask[a], new.mask[b] = new.mask[b], new.mask[a]
        new.valid = list(self.valid)
        new.valid[a], new.valid[b] = self.valid[b], self.valid[a]
        new.beta = dict(self.beta)
        new.beta[a], new.beta[b] = self.beta.get(b), self.beta.get(a)
        new.calls = 0
        return new

    def _grid(self, lying, step):
        sh = self.shelf
        out = []
        if not lying:
            for oy, ox in ((0.0, 0.0), (step / 2, step / 2)):
                y = CAN_R + 0.005 + oy
                while y <= sh.depth - CAN_R:
                    x = CAN_R + 0.005 + ox
                    while x <= sh.width - CAN_R:
                        p = Pose(x, y)
                        if sh.inside(p):
                            out.append(p)
                        x += step
                    y += step
        else:
            for th in (0.0, math.pi / 2):
                sx = CAN_LEN + 0.01 if th == 0.0 else step
                sy = step if th == 0.0 else CAN_LEN + 0.01
                y = CAN_R + 0.005 + (0 if th == 0.0 else HALF_AXIS)
                while y <= sh.depth - CAN_R:
                    x = CAN_R + 0.005 + (HALF_AXIS if th == 0.0 else 0)
                    while x <= sh.width - CAN_R:
                        p = Pose(x, y, th, True)
                        if sh.inside(p):
                            out.append(p)
                        x += sx
                    y += sy
        return out

    # -- motion query ---------------------------------------------------------
    def free(self, p, occ):
        for k in self.valid[p]:
            if self.mask[p][k] & occ == 0:
                return True
        return False

    def can_move(self, frm, to, occ):
        """One motion-planner call: pick at `frm` and place at `to` with the
        other cans occupying the poses in `occ`."""
        self.calls += 1
        return self.free(frm, occ) and self.free(to, occ)

    def occ(self, pos, skip=None):
        v = 0
        for j, p in enumerate(pos):
            if j != skip and p is not None:
                v |= self.bit[p]
        return v

    @property
    def start(self):
        return list(range(self.n))

    @property
    def goal(self):
        return list(range(self.n, 2 * self.n))

    def density(self):
        area = self.n * math.pi * CAN_R ** 2
        area += sum(1 for k in self.kinds if k == "lie") * (2 * HALF_AXIS) * 2 * CAN_R
        return area / (self.shelf.width * self.shelf.depth)


def reachable(shelf, pose):
    return any(shelf.sweep(pose, o) is not None for o in shelf.options)


def random_arrangement(shelf, kinds, rng, tries=20000):
    placed = []
    for kind in kinds:
        for _ in range(tries):
            lying = kind == "lie"
            p = Pose(rng.uniform(0, shelf.width), rng.uniform(0, shelf.depth),
                     rng.uniform(0, math.pi) if lying else 0.0, lying)
            if not shelf.inside(p):
                continue
            if placed:
                a0, a1 = p.axis()
                qa = np.array([q.axis()[0] for q in placed])
                qb = np.array([q.axis()[1] for q in placed])
                if seg_dist(np.array([a0]), np.array([a1]), qa, qb).min() < 2 * CAN_R + 2 * GAP:
                    continue
            if not reachable(shelf, p):
                continue
            placed.append(p)
            break
        else:
            return None
    return placed


def organized_arrangement(shelf, kinds, rng):
    """Tidy goal: cans stand in rows, filled from the back of the shelf to the
    front and from left to right. Lying cans lie along the rows."""
    order = list(range(len(kinds)))
    rng.shuffle(order)
    poses = [None] * len(kinds)
    pitch = 2 * CAN_R + 0.01
    y = shelf.depth - CAN_R - 0.005
    x = 0.005
    for i in order:
        lying = kinds[i] == "lie"
        w = CAN_LEN if lying else 2 * CAN_R
        while True:
            if x + w > shelf.width - 0.005:
                y -= pitch
                x = 0.005
            if y < CAN_R:
                return None
            p = Pose(x + w / 2, y, 0.0, lying)
            x += w + 0.01
            if shelf.inside(p) and reachable(shelf, p):
                poses[i] = p
                break
    return poses


def make_instance(n, seed, lying_ratio=0.3, width=1.0, depth=0.4, options="multi",
                  pillars=(), goal_style="rows", grid=0.06):
    """Random instance. The start is always messy (random positions and
    angles); the goal is either tidy rows or another messy arrangement."""
    rng = random.Random(seed)
    shelf = Shelf(width, depth, pillars, options)
    while True:
        kinds = ["lie" if rng.random() < lying_ratio else "up" for _ in range(n)]
        s = random_arrangement(shelf, kinds, rng)
        if not s:
            continue
        if goal_style == "rows":
            g = organized_arrangement(shelf, kinds, rng)
        else:
            g = random_arrangement(shelf, kinds, rng)
        if g:
            return Instance(shelf, s, g, kinds, grid)


def make_monotone_instance(n, seed, lying_ratio=0.3, width=1.0, depth=0.4, options="multi", grid=0.06):
    """Instance that is solvable with one move per object. The goal is built by
    really moving the cans one by one, in a random order, to free grid poses."""
    rng = random.Random(seed)
    shelf = Shelf(width, depth, (), options)
    while True:
        kinds = ["lie" if rng.random() < lying_ratio else "up" for _ in range(n)]
        s = random_arrangement(shelf, kinds, rng)
        if not s:
            continue
        tmp = Instance(shelf, s, s, kinds, grid)       # only used for its pose table
        cur = list(range(n))
        left = list(range(n))
        while left:
            occ_all = tmp.occ(cur)
            # a goal pose that would close the last free approach of a waiting can
            # is avoided when possible
            placed = tmp.occ([cur[j] if j not in left else None for j in range(n)])
            locks = {}
            for j in left:
                m = -1
                for k in tmp.valid[cur[j]]:
                    if tmp.mask[cur[j]][k] & placed == 0:
                        m &= tmp.mask[cur[j]][k]
                locks[j] = m
            moves, safe = [], []
            for i in left:
                occ = occ_all & ~tmp.bit[cur[i]]
                if not tmp.free(cur[i], occ):
                    continue
                for c in tmp.cands[kinds[i]]:
                    if c >= tmp.cand_first and tmp.overlap[c] & occ == 0 and tmp.free(c, occ):
                        moves.append((i, c))
                        if not any(locks[j] >> c & 1 for j in left if j != i):
                            safe.append((i, c))
            if not moves:
                break
            i, c = rng.choice(safe or moves)
            cur[i] = c
            left.remove(i)
        if not left:
            goals = [tmp.poses[cur[i]] for i in range(n)]
            return Instance(shelf, s, goals, kinds, grid)


def make_hard_infeasible(n, seed, options="multi", tries=30):
    """A monotone instance where the goals of two cans are swapped until the
    instance is no longer monotone. Most cans can still move freely, so a
    search over orders has to look at many branches before giving up."""
    inst = make_monotone_instance(n, seed, options=options)
    rng = random.Random(seed + 7919)
    for _ in range(tries):
        i, j = rng.sample(range(n), 2)
        if inst.kinds[i] != inst.kinds[j]:
            continue
        cand = inst.swap_goals(i, j)
        st, _ = cdg_monotone(cand, cand.start, time.perf_counter() + 30)
        if st == "fail":
            return cand
        inst = cand if rng.random() < 0.5 else inst
    return None


# ----------------------------------------------------------------------------
# Result record
# ----------------------------------------------------------------------------
@dataclass
class Result:
    method: str
    success: bool
    time: float
    calls: int
    actions: list = field(default_factory=list)   # (object, from pose id, to pose id)
    note: str = ""
    lower_bound: int = -1
    infeasible: bool = False

    @property
    def n_actions(self):
        return len(self.actions)


def count_buffers(inst, actions):
    """Number of moves that end somewhere other than the object's goal."""
    return sum(1 for i, _, to in actions if to != inst.n + i)


# ----------------------------------------------------------------------------
# Monotone solvers: mRS, DFS_DP and CIRS
# ----------------------------------------------------------------------------
def monotone_search(inst, pos0, deadline, memo=True, prune=False, tree=None):
    """Depth-first search where every object moves at most once, straight to its goal.

    memo=False, prune=False  -> mRS (plain backtracking over orders)
    memo=True,  prune=False  -> DFS_DP (remember arrangements already expanded)
    memo=True,  prune=True   -> CIRS (also skip moves that leave some object stuck)

    Returns (status, order) with status in {"ok", "fail", "timeout"}.
    If `tree` is a dict, every arrangement reached is stored in it as
    tree[arrangement] = (parent arrangement, action).
    """
    n = inst.n
    goal = inst.goal
    todo = [i for i in range(n) if pos0[i] != goal[i]]
    m = len(todo)
    full = (1 << m) - 1

    base_occ = inst.occ([pos0[i] if pos0[i] == goal[i] else None for i in range(n)])
    start_bits = [inst.bit[pos0[i]] for i in todo]
    goal_bits = [inst.bit[goal[i]] for i in todo]

    # for pruning: masks of each option restricted to goal poses
    goal_all = 0
    for g in goal:
        goal_all |= inst.bit[g]
    pick_masks = [[inst.mask[pos0[i]][k] & goal_all for k in inst.valid[pos0[i]]] for i in todo]
    place_masks = [[inst.mask[goal[i]][k] & goal_all for k in inst.valid[goal[i]]] for i in todo]

    def stuck(u, at_goal_bits):
        if all(mm & at_goal_bits for mm in pick_masks[u]):
            return True
        return all(mm & at_goal_bits for mm in place_masks[u])

    seen = set()
    order = []
    timed_out = [False]

    def arrangement(done):
        return tuple(goal[todo[b]] if done >> b & 1 else pos0[todo[b]] for b in range(m))

    def full_arr(done):
        arr = list(pos0)
        for b in range(m):
            if done >> b & 1:
                arr[todo[b]] = goal[todo[b]]
        return tuple(arr)

    def rec(done, occ, at_goal):
        if done == full:
            return True
        if time.perf_counter() > deadline:
            timed_out[0] = True
            return False
        if memo:
            if done in seen:
                return False
            seen.add(done)
        for b in range(m):
            if done >> b & 1:
                continue
            new_goal = at_goal | goal_bits[b]
            if prune:
                bad = False
                for u in range(m):
                    if u != b and not done >> u & 1 and stuck(u, new_goal):
                        bad = True
                        break
                if bad:
                    continue
            i = todo[b]
            others = occ & ~start_bits[b]
            if inst.can_move(pos0[i], goal[i], others):
                nd = done | (1 << b)
                if tree is not None:
                    child = full_arr(nd)
                    if child not in tree:
                        tree[child] = (full_arr(done), (i, pos0[i], goal[i]))
                order.append(i)
                if rec(nd, others | goal_bits[b], new_goal):
                    return True
                order.pop()
                if timed_out[0]:
                    return False
        return False

    ok = rec(0, base_occ | sum_bits(start_bits), base_occ)
    if ok:
        return "ok", order
    return ("timeout" if timed_out[0] else "fail"), []


def sum_bits(bs):
    v = 0
    for b in bs:
        v |= b
    return v


def run_monotone(inst, method, time_limit=30.0):
    inst.calls = 0
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    if method == "mRS":
        st, order = monotone_search(inst, inst.start, deadline, memo=False)
    elif method == "DFS_DP":
        st, order = monotone_search(inst, inst.start, deadline, memo=True)
    elif method == "CIRS":
        st, order = monotone_search(inst, inst.start, deadline, memo=True, prune=True)
    elif method == "CDG":
        st, order = cdg_monotone(inst, inst.start, deadline)
    else:
        raise ValueError(method)
    acts = [(i, i, inst.n + i) for i in order]
    return Result(method, st == "ok", time.perf_counter() - t0, inst.calls, acts,
                  note=st, infeasible=(st == "fail"))


# ----------------------------------------------------------------------------
# Proposed: confined dependency graph (CDG)
# ----------------------------------------------------------------------------
def dependency_edges(inst, pos0, i, pick_k, place_k):
    """Precedence edges caused by moving object i with the given options.

    Returns (before, after, stuck): objects that must move before i, objects
    that must move after i, and True if some object blocks i both at its
    current and at its goal pose (then i cannot move without a buffer).
    """
    goal = inst.goal
    S = inst.mask[pos0[i]][pick_k] | inst.mask[goal[i]][place_k]
    before, after = [], []
    for j in range(inst.n):
        if j == i:
            continue
        at_start = S & inst.bit[pos0[j]]
        at_goal = S & inst.bit[goal[j]]
        if pos0[j] == goal[j]:
            if at_start:
                return before, after, True
            continue
        if at_start and at_goal:
            return before, after, True
        if at_start:
            before.append(j)
        if at_goal:
            after.append(j)
    return before, after, False


def topo_order(n, nodes, edges):
    indeg = {v: 0 for v in nodes}
    out = {v: [] for v in nodes}
    for a, b in edges:
        out[a].append(b)
        indeg[b] += 1
    ready = [v for v in nodes if indeg[v] == 0]
    order = []
    while ready:
        v = ready.pop()
        order.append(v)
        for w in out[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
    return order if len(order) == len(nodes) else None


def cdg_monotone(inst, pos0, deadline):
    """Monotone feasibility with the dependency graph.

    With a single approach option this is a plain graph test (Theorem 1):
    build the edges, look for a cycle, return a topological order.
    With several options the choice of option per object is solved exactly
    with the CP-SAT model (no buffers allowed).
    """
    goal = inst.goal
    todo = [i for i in range(inst.n) if pos0[i] != goal[i]]
    if len(inst.shelf.options) == 1:
        edges = []
        for i in todo:
            if not inst.valid[pos0[i]] or not inst.valid[goal[i]]:
                return "fail", []
            before, after, stuck = dependency_edges(inst, pos0, i, 0, 0)
            if stuck:
                return "fail", []
            edges += [(j, i) for j in before] + [(i, j) for j in after]
        order = topo_order(inst.n, todo, edges)
        return ("ok", order) if order is not None else ("fail", [])
    plan = primitive_plan(inst, pos0, buffers=False,
                          time_limit=max(0.1, deadline - time.perf_counter()))
    if plan["status"] == "infeasible":
        return "fail", []
    if plan["status"] == "unknown":
        return "timeout", []
    return "ok", [i for _, kind, i, _ in plan["events"]]


def primitive_plan(inst, pos0, buffers=True, sweep="arm", time_limit=10.0,
                   dwell=True, seed=0, weights=None):
    """Exact CP-SAT model of the rearrangement with 'lazy' buffers.

    Every object gets a leave time t_out and an arrive time t_in. If it is
    not buffered the two are equal (one direct move). Buffer positions are
    left open, so the optimum is a lower bound on the true number of actions.

    sweep="arm"   : blocking comes from the arm and can sweeps (confined shelf)
    sweep="table" : only footprint overlaps count (tabletop assumption)

    Returns a dict with status in {"optimal", "feasible", "infeasible",
    "unknown"}, the lower bound on actions and the ordered events
    (time, kind, object, option) with kind in {"go", "out", "in"}.
    """
    n = inst.n
    goal = inst.goal
    T = 2 * n + 1
    moving = [pos0[i] != goal[i] for i in range(n)]
    md = cp_model.CpModel()
    t_out, t_in, x, pick, place = [], [], [], [], []

    def opts(p):
        return inst.valid[p] if sweep == "arm" else [0]

    def smask(p, k):
        return inst.mask[p][k] if sweep == "arm" else inst.overlap[p]

    for i in range(n):
        a = md.NewIntVar(0, T, f"out{i}")
        b = md.NewIntVar(0, T, f"in{i}")
        xi = md.NewBoolVar(f"x{i}") if buffers else md.NewConstant(0)
        t_out.append(a)
        t_in.append(b)
        x.append(xi)
        md.Add(b == a).OnlyEnforceIf(xi.Not())
        if buffers:
            md.Add(b >= a + 1).OnlyEnforceIf(xi)
        po, pl = opts(pos0[i]), opts(goal[i])
        if moving[i] and (not po or not pl):
            return {"status": "infeasible", "events": [], "lb": -1}
        if not moving[i] and (not po or not pl):
            md.Add(xi == 0)
        yi = {k: md.NewBoolVar("") for k in po}
        zi = {k: md.NewBoolVar("") for k in pl}
        active = 1 if moving[i] else xi
        md.Add(sum(yi.values()) == active)
        md.Add(sum(zi.values()) == active)
        pick.append(yi)
        place.append(zi)

    for i in range(n):
        for events, pose, t_ev in ((pick[i], pos0[i], t_out[i]), (place[i], goal[i], t_in[i])):
            for k, lit in events.items():
                S = smask(pose, k)
                for j in range(n):
                    if j == i:
                        continue
                    if S & inst.bit[pos0[j]]:
                        md.Add(t_out[j] + 1 <= t_ev).OnlyEnforceIf(lit)
                    if S & inst.bit[goal[j]]:
                        md.Add(t_ev + 1 <= t_in[j]).OnlyEnforceIf(lit)

    running = None
    if buffers and dwell:
        running = md.NewIntVar(0, n, "running")
        boxes = []
        for i in range(n):
            size = md.NewIntVar(0, T, "")
            boxes.append(md.NewOptionalIntervalVar(t_out[i], size, t_in[i], x[i], ""))
        md.AddCumulative(boxes, [1] * n, running)

    extra = sum(x[i] if moving[i] else 2 * x[i] for i in range(n))
    base_moves = sum(moving)
    md.Minimize(extra)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, time_limit - (min(0.02 * n, time_limit * 0.3) if dwell else 0.0))
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    st = solver.Solve(md)
    if st == cp_model.INFEASIBLE:
        return {"status": "infeasible", "events": [], "lb": -1}
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "unknown", "events": [], "lb": base_moves + int(math.ceil(solver.BestObjectiveBound() - 1e-6))}
    status = "optimal" if st == cp_model.OPTIMAL else "feasible"
    best = int(round(solver.ObjectiveValue()))
    lb = base_moves + int(math.ceil(solver.BestObjectiveBound() - 1e-6))

    if dwell and buffers:
        # second stage: same number of actions, buffers kept as short as possible
        for i in range(n):
            md.AddHint(t_out[i], solver.Value(t_out[i]))
            md.AddHint(t_in[i], solver.Value(t_in[i]))
            md.AddHint(x[i], solver.Value(x[i]))
        md.Add(extra <= best)
        # fewest cans parked at the same time, then the shortest waiting times
        w = weights or [1] * n
        md.Minimize(4 * n * running + sum(w[i] * (t_in[i] - t_out[i]) for i in range(n)))
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = max(0.05, min(0.02 * n, time_limit * 0.3))
        s2.parameters.num_workers = 1
        s2.parameters.random_seed = seed
        st2 = s2.Solve(md)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solver = s2

    events = []
    for i in range(n):
        xi = solver.Value(x[i])
        k_pick = next((k for k, l in pick[i].items() if solver.Value(l)), None)
        k_place = next((k for k, l in place[i].items() if solver.Value(l)), None)
        if moving[i] and not xi:
            events.append((solver.Value(t_out[i]), "go", i, (k_pick, k_place)))
        elif xi:
            events.append((solver.Value(t_out[i]), "out", i, k_pick))
            events.append((solver.Value(t_in[i]), "in", i, k_place))
    events.sort(key=lambda e: (e[0], 1 if e[1] == "out" else 0))
    parked = peak = 0
    for e in events:
        parked += (e[1] == "out") - (e[1] == "in")
        peak = max(peak, parked)
    return {"status": status, "events": events, "lb": lb, "actions": base_moves + best,
            "running": peak}


# ----------------------------------------------------------------------------
# Proposed: lazy buffer allocation inside the shelf
# ----------------------------------------------------------------------------
def execute_events(inst, pos, events, rng, buffer_rule="beta", deadline=None):
    """Carry out a primitive plan and choose buffer poses on the way.

    Buffers: for a buffered object we look at the events that happen while it
    waits (its window) and prefer a pose that blocks none of them. Objects
    whose buffer is not chosen yet are treated as absent in this look-ahead.

    Repair: the primitive plan already respects start and goal poses, so an
    event can only be blocked by objects that sit in buffers. Such an object
    is sent to its goal early if that harms nothing, otherwise to another
    buffer. If no repair is found the caller re-plans.

    Returns (finished, pos, actions).
    """
    goal = inst.goal
    pos = list(pos)
    actions = []
    in_index = {e[2]: idx for idx, e in enumerate(events) if e[1] == "in"}
    waiting = set()          # objects currently parked in a buffer
    last = len(events) - 1

    def conflicts(sim, first, stop):
        bad = 0
        for _, kind, k, _ in events[first:stop + 1]:
            occ = inst.occ(sim, skip=k)
            inst.calls += 1
            if kind == "go":
                ok = inst.free(sim[k], occ) and inst.free(goal[k], occ)
                sim[k] = goal[k]
            elif kind == "out":
                ok = inst.free(sim[k], occ)
                sim[k] = None
            elif sim[k] == goal[k]:
                ok = True
            else:
                ok = (sim[k] is None or inst.free(sim[k], occ)) and inst.free(goal[k], occ)
                sim[k] = goal[k]
            bad += not ok
        return bad

    def sorted_cands(i):
        occ = inst.occ(pos, skip=i)
        cands = [c for c in inst.cands[inst.kinds[i]] if c != pos[i] and inst.overlap[c] & occ == 0]
        if buffer_rule == "beta":
            px, py = inst.poses[pos[i]].x, inst.poses[pos[i]].y
            cands.sort(key=lambda c: (inst.beta[c], math.hypot(inst.poses[c].x - px, inst.poses[c].y - py)))
        else:
            rng.shuffle(cands)
        return cands, occ

    def find_buffer(i, first, stop, avoid=0):
        """Best buffer for object i whose waiting window is events[first..stop]."""
        cands, occ = sorted_cands(i)
        best, best_bad = None, None
        for c in cands:
            if deadline and time.perf_counter() > deadline:
                break
            if inst.bit[c] & avoid or c == goal[i]:
                continue
            if not inst.can_move(pos[i], c, occ):
                continue
            sim = list(pos)
            sim[i] = c
            bad = conflicts(sim, first, stop) if stop >= first else 0
            if best is None or bad < best_bad:
                best, best_bad = c, bad
            if bad == 0:
                break
        return best, best_bad

    def blockers(k, side_poses):
        """Smallest set of parked objects that blocks the motion of k, and the
        swept area of the options that give it."""
        total, area = set(), 0
        for p in side_poses:
            best = None
            for o in inst.valid[p]:
                m = inst.mask[p][o]
                hit = [j for j in range(inst.n) if j != k and pos[j] is not None and inst.bit[pos[j]] & m]
                if any(j not in waiting for j in hit):
                    continue
                if best is None or len(hit) < len(best[0]):
                    best = (hit, m)
            if best is None:
                return None, 0
            total |= set(best[0])
            area |= best[1]
        return total, area

    def park(j, first, area):
        """Move parked object j out of `area`: to its goal if harmless, else to a new buffer."""
        occ = inst.occ(pos, skip=j)
        g = goal[j]
        if not inst.bit[g] & area and inst.overlap[g] & occ == 0 and inst.can_move(pos[j], g, occ):
            sim = list(pos)
            sim[j] = g
            if conflicts(sim, first, last) == 0:
                actions.append((j, pos[j], g))
                pos[j] = g
                waiting.discard(j)
                return True
        c, bad = find_buffer(j, first, in_index[j], avoid=area)
        if c is None or bad:
            return False
        actions.append((j, pos[j], c))
        pos[j] = c
        return True

    for idx, (_, kind, i, _) in enumerate(events):
        if deadline and time.perf_counter() > deadline:
            return False, pos, actions
        if kind == "in" and pos[i] == goal[i]:
            continue                       # already sent home during a repair
        occ = inst.occ(pos, skip=i)
        if kind in ("go", "in"):
            if not inst.can_move(pos[i], goal[i], occ):
                block, area = blockers(i, (pos[i], goal[i]))
                if not block:
                    return False, pos, actions
                for j in block:
                    if not park(j, idx, area):
                        return False, pos, actions
                occ = inst.occ(pos, skip=i)
                if not inst.can_move(pos[i], goal[i], occ):
                    return False, pos, actions
            actions.append((i, pos[i], goal[i]))
            pos[i] = goal[i]
            waiting.discard(i)
            continue

        # kind == "out": choose a buffer pose
        c, _ = find_buffer(i, idx + 1, in_index[i])
        if c is None:
            return False, pos, actions
        actions.append((i, pos[i], c))
        pos[i] = c
        waiting.add(i)
    return True, pos, actions


def random_perturbation(inst, pos, rng, tries=200):
    """Move one random object to a random reachable free candidate pose."""
    n = inst.n
    for _ in range(tries):
        i = rng.randrange(n)
        cands = inst.cands[inst.kinds[i]]
        c = cands[rng.randrange(len(cands))]
        occ = inst.occ(pos, skip=i)
        if c == pos[i] or inst.overlap[c] & occ:
            continue
        if inst.can_move(pos[i], c, occ):
            return (i, pos[i], c)
    return None


def replay_ok(inst, pos, actions):
    """True if `actions` can be executed from arrangement `pos` and end at the goal."""
    pos = list(pos)
    for i, frm, to in actions:
        if pos[i] != frm:
            return False
        occ = inst.occ(pos, skip=i)
        if inst.overlap[to] & occ or not inst.can_move(frm, to, occ):
            return False
        pos[i] = to
    return pos == inst.goal


def shorten_plan(inst, actions, deadline):
    """Plan smoothing: two moves of the same object are joined into one move,
    either at the time of the first move or at the time of the second, and a
    round trip (p -> ... -> p) is removed, whenever the plan stays valid."""
    acts = list(actions)
    t = 0
    pos = list(inst.start)          # arrangement before action t
    while t < len(acts) and time.perf_counter() < deadline:
        i, p, q = acts[t]
        u = next((k for k in range(t + 1, len(acts)) if acts[k][0] == i), None)
        changed = False
        if u is not None:
            r = acts[u][2]
            middle, tail = acts[t + 1:u], acts[u + 1:]
            direct = [(i, p, r)] if p != r else []
            for new_part in (middle + direct, direct + middle):
                if replay_ok(inst, pos, new_part + tail):
                    acts = acts[:t] + new_part + tail
                    changed = True
                    break
        if not changed:
            pos[i] = q
            t += 1
    return acts


def shortcut_plan(inst, actions, deadline, rng):
    """Replace the tail of a plan by a shorter one when possible.

    For every arrangement reached along the plan (earliest first), the exact
    model gives a lower bound for the rest of the work. Only where this bound
    is smaller than the length of the remaining tail do we plan again from
    that arrangement; a new tail is kept if it is valid and shorter.
    """
    acts = list(actions)
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        t = 0
        pos = list(inst.start)
        while t < len(acts) and time.perf_counter() < deadline:
            left = len(acts) - t
            away = sum(1 for i in range(inst.n) if pos[i] != inst.goal[i])
            if away < left:
                plan = primitive_plan(inst, pos, time_limit=min(0.5, max(0.05, deadline - time.perf_counter())),
                                      seed=rng.randrange(1000),
                                      weights=[rng.randint(1, 3) for _ in range(inst.n)])
                if plan["status"] in ("optimal", "feasible") and plan["lb"] < left:
                    ok, _, tail = execute_events(inst, pos, plan["events"], rng, "beta", deadline)
                    if ok and len(tail) < left and replay_ok(inst, pos, tail):
                        acts = acts[:t] + tail
                        improved = True
            if t == len(acts):
                break
            i, _, q = acts[t]
            pos[i] = q
            t += 1
    return acts


def cdg_lb(inst, time_limit=30.0, seed=0, buffer_rule="beta", dwell=True,
           sweep="arm", name="CDG-LB", smooth=True):
    """Proposed planner (CDG-LB).

    1. Solve the primitive plan on the confined dependency graph (CP-SAT).
    2. Execute it, choosing buffer poses lazily inside the shelf.
    3. If an event becomes impossible, re-plan from the arrangement reached.
    """
    inst.calls = 0
    rng = random.Random(seed)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    pos = inst.start
    actions = []
    lb = -1
    rounds = 0
    seen = {tuple(pos)}
    while time.perf_counter() < deadline:
        left = deadline - time.perf_counter()
        weights = None if rounds == 0 else [rng.randint(1, 3) for _ in range(inst.n)]
        plan = primitive_plan(inst, pos, sweep=sweep, time_limit=min(left, 5.0),
                              dwell=dwell, seed=seed + rounds, weights=weights)
        rounds += 1
        if rounds == 1 and sweep == "arm" and plan["status"] == "optimal":
            lb = plan["lb"]
        if plan["status"] == "infeasible" and sweep == "arm":
            return Result(name, False, time.perf_counter() - t0, inst.calls, actions,
                          note="proved infeasible", lower_bound=lb, infeasible=True)
        moved = False
        if plan["status"] in ("optimal", "feasible"):
            ok, new_pos, done = execute_events(inst, pos, plan["events"], rng, buffer_rule, deadline)
            if done:
                actions += done
                pos = new_pos
                moved = True
            if ok:
                raw = len(actions)
                if smooth:
                    # post-processing: merge moves, then look for shorter tails;
                    # it may use at most half of the time spent so far (0.5 to 10 s)
                    spent = time.perf_counter() - t0
                    stop = min(deadline, time.perf_counter() + min(10.0, max(0.5, 0.5 * spent)))
                    actions = shorten_plan(inst, actions, stop)
                    if len(actions) > lb:
                        actions = shortcut_plan(inst, actions, stop, rng)
                        actions = shorten_plan(inst, actions, stop)
                return Result(name, True, time.perf_counter() - t0, inst.calls, actions,
                              note=f"rounds={rounds} raw={raw}", lower_bound=lb)
        if not moved or tuple(pos) in seen:
            mv = random_perturbation(inst, pos, rng)
            if mv:
                actions.append(mv)
                pos = list(pos)
                pos[mv[0]] = mv[2]
        seen.add(tuple(pos))
    return Result(name, False, time.perf_counter() - t0, inst.calls, actions,
                  note="timeout", lower_bound=lb)


def tabletop_lb(inst, time_limit=30.0, seed=0):
    """TRLB-style planner that builds its primitive plan from the tabletop
    dependency graph (footprint overlaps only), ignoring the arm."""
    return cdg_lb(inst, time_limit, seed, sweep="table", name="TRLB-table")


# ----------------------------------------------------------------------------
# Baseline: PERTS with CIRS as local solver
# ----------------------------------------------------------------------------
def perts(inst, time_limit=30.0, seed=0):
    inst.calls = 0
    rng = random.Random(seed)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    goal = tuple(inst.goal)
    root = tuple(inst.start)
    tree = {root: (None, None)}
    level = {root: 0}

    def grow(node):
        before = set(tree)
        st, _ = monotone_search(inst, list(node), deadline, memo=True, prune=True, tree=tree)
        for a in tree:
            if a not in before:
                level[a] = level[tree[a][0]]
        return goal in tree

    found = grow(root)
    cur_level = 0
    attempts = 0
    while not found and time.perf_counter() < deadline:
        pool = [a for a, l in level.items() if l == cur_level]
        node = pool[rng.randrange(len(pool))]
        mv = random_perturbation(inst, list(node), rng, tries=20)
        attempts += 1
        if mv:
            child = list(node)
            child[mv[0]] = mv[2]
            child = tuple(child)
            if child not in tree:
                tree[child] = (node, mv)
                level[child] = cur_level + 1
                found = grow(child)
        if attempts >= 2 * len(pool) + 5 and any(l == cur_level + 1 for l in level.values()):
            cur_level += 1
            attempts = 0
    el = time.perf_counter() - t0
    if not found:
        return Result("PERTS", False, el, inst.calls, note="timeout")
    acts = []
    a = goal
    while tree[a][0] is not None:
        acts.append(tree[a][1])
        a = tree[a][0]
    acts.reverse()
    return Result("PERTS", True, el, inst.calls, acts)


# ----------------------------------------------------------------------------
# Baseline: optimal search (A*) over start, goal and grid poses
# ----------------------------------------------------------------------------
def astar(inst, time_limit=60.0, cand_stride=1, upper=None, max_nodes=1_500_000):
    """Best-first search over arrangements. Buffers are the grid poses (every
    `cand_stride`-th one) plus all start and goal poses.

    If `upper` is given, only plans shorter than `upper` are searched for.
    "optimal" means the search finished: the returned plan is the shortest
    one, or (note "no shorter plan") no plan shorter than `upper` exists.
    """
    inst.calls = 0
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    n = inst.n
    goal = tuple(inst.goal)
    start = tuple(inst.start)
    dests = {}
    for kind, ids in inst.cands.items():
        grid = [c for c in ids if c >= inst.cand_first][::cand_stride]
        dests[kind] = grid + [c for c in ids if c < inst.cand_first]
    bound = upper if upper is not None else 1 << 30

    def h(s):
        return sum(1 for i in range(n) if s[i] != goal[i])

    heap = [(h(start), 0, start)]
    parent = {start: (None, None)}
    gbest = {start: 0}
    while heap:
        if time.perf_counter() > deadline or len(gbest) > max_nodes:
            return Result("A*", False, time.perf_counter() - t0, inst.calls, note="limit")
        f, g, s = heapq.heappop(heap)
        g = -g
        if g > gbest.get(s, 1 << 30):
            continue
        if s == goal:
            acts = []
            while parent[s][0] is not None:
                acts.append(parent[s][1])
                s = parent[s][0]
            acts.reverse()
            return Result("A*", True, time.perf_counter() - t0, inst.calls, acts, note="optimal")
        for i in range(n):
            occ = inst.occ(s, skip=i)
            if not inst.free(s[i], occ):
                continue
            for c in [goal[i]] + dests[inst.kinds[i]]:
                if c == s[i]:
                    continue
                if inst.overlap[c] & occ:
                    continue
                t = list(s)
                t[i] = c
                t = tuple(t)
                if g + 1 >= gbest.get(t, 1 << 30) or g + 1 + h(t) >= bound:
                    continue
                if not inst.can_move(s[i], c, occ):
                    continue
                gbest[t] = g + 1
                parent[t] = (s, (i, s[i], c))
                heapq.heappush(heap, (g + 1 + h(t), -(g + 1), t))
    note = "no shorter plan" if upper is not None else "no plan"
    return Result("A*", False, time.perf_counter() - t0, inst.calls, note=note, infeasible=upper is None)


# ----------------------------------------------------------------------------
# Independent plan checker (recomputes all geometry from the poses)
# ----------------------------------------------------------------------------
def verify(inst, actions):
    """Replay a plan and check every move with fresh geometry.

    This check does not use the pose table, the bit masks or the swept
    areas of the planner. For each move it looks at the arm and the can at
    single instants 2 mm apart (fine_motion) and looks for real contact
    (no safety gap; shapes that only touch are accepted). Checks: the object is where the action says, the new
    pose is inside the shelf and touches no other can, and the pick and the
    place each have at least one approach option without contact.
    """
    sh = inst.shelf
    cur = [inst.poses[i] for i in range(inst.n)]
    ids = list(range(inst.n))
    for i, frm, to in actions:
        if ids[i] != frm:
            return False, "object not at the expected pose"
        others = [cur[j] for j in range(inst.n) if j != i]
        qa = np.array([o.axis()[0] for o in others]).reshape(-1, 2)
        qb = np.array([o.axis()[1] for o in others]).reshape(-1, 2)
        new = inst.poses[to]
        if not sh.inside(new):
            return False, "pose outside the shelf"
        a0, a1 = new.axis()
        if len(others) and seg_dist(np.array([a0]), np.array([a1]), qa, qb).min() < 2 * CAN_R - TOUCH:
            return False, "overlap at the new pose"
        for pose in (cur[i], new):
            ok = False
            for opt in sh.options:
                fm = fine_motion(sh, pose, opt)
                if fm is None:
                    continue
                link_a, link_b, can_a, can_b = fm
                if not len(others) or (
                        seg_dist(can_a, can_b, qa, qb).min() >= 2 * CAN_R - TOUCH and
                        seg_dist(link_a, link_b, qa, qb).min() >= sh.link_r + CAN_R - TOUCH):
                    ok = True
                    break
            if not ok:
                return False, "no free approach"
        cur[i] = new
        ids[i] = to
    if ids != inst.goal:
        return False, "goal not reached"
    return True, "ok"


def solve(inst, method, time_limit=30.0, seed=0):
    if method == "CDG-LB":
        return cdg_lb(inst, time_limit, seed)
    if method == "CDG-LB/no-smoothing":
        return cdg_lb(inst, time_limit, seed, smooth=False, name=method)
    if method == "CDG-LB/no-dwell":
        return cdg_lb(inst, time_limit, seed, dwell=False, name=method)
    if method == "CDG-LB/random-buffer":
        return cdg_lb(inst, time_limit, seed, buffer_rule="random", name=method)
    if method == "TRLB-table":
        return tabletop_lb(inst, time_limit, seed)
    if method == "PERTS":
        return perts(inst, time_limit, seed)
    if method == "A*":
        return astar(inst, time_limit)
    raise ValueError(method)


if __name__ == "__main__":
    inst = make_instance(12, 1)
    print("poses", inst.P, "prep", round(inst.prep_time, 2), "density", round(inst.density(), 3))
    for m in ["CDG-LB", "PERTS", "TRLB-table"]:
        r = solve(inst, m, 20)
        print(m, r.success, round(r.time, 2), r.n_actions, count_buffers(inst, r.actions),
              r.calls, r.note, r.lower_bound, verify(inst, r.actions) if r.success else "")
