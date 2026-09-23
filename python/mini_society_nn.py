#!/usr/bin/env python3
"""
Mini Society -- Neural Edition (Python port)
============================================

A faithful Python/NumPy port of the JavaScript engine in ``app/mini-society-nn.html``.

Every agent owns TWO tiny MLPs, trained online with Monte-Carlo REINFORCE on the
agent's OWN payoffs -- no hand-coded strategies, no imitation:

* move policy:  9 inputs -> hidden tanh -> sigmoid P(cooperate)
* accept policy: 5 inputs -> hidden tanh -> sigmoid P(accept invite)

Reproducibility: the RNG is a bit-identical port of the JavaScript ``mulberry32``,
so ``seed=7`` here produces exactly the same run as ``seed=7`` in the HTML version.

Dependencies: numpy only (matplotlib is optional, used just for --plot).

Quick start
-----------
    python mini_society_nn.py                      # Reality preset, 400 rounds
    python mini_society_nn.py --preset "No honor"  # the norm-free collapse
    python mini_society_nn.py --preset "High temptation" --seed 11 --plot

As a library::

    from mini_society_nn import NNSociety, base_config, PRESETS
    soc = NNSociety(base_config(PRESETS["Reality"], seed=7))
    soc.run(400)
    print(soc.summary())
"""

import argparse
import csv
import math

import numpy as np

# ---------------------------------------------------------------------------
# RNG: bit-identical port of JavaScript mulberry32, so seeds match the HTML app
# ---------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF


def _s32(x: int) -> int:
    x &= _MASK32
    return x - 0x100000000 if x & 0x80000000 else x


def _u32(x: int) -> int:
    return x & _MASK32


def _imul(x: int, y: int) -> int:
    return _s32(_u32(x) * _u32(y))


class Mulberry32:
    """Deterministic RNG matching the browser version exactly."""

    def __init__(self, seed: int):
        self.a = _u32(seed) or 1

    def random(self) -> float:
        a = _s32(self.a + 0x6D2B79F5)
        t = _imul(a ^ (_u32(a) >> 15), 1 | a)
        t = _s32(t + _imul(t ^ (_u32(t) >> 7), 61 | t)) ^ t
        self.a = _u32(a)
        return _u32(t ^ (_u32(t) >> 14)) / 4294967296.0

    def randint(self, n: int) -> int:
        """Uniform int in [0, n)."""
        return int(self.random() * n)

    def shuffle(self, xs: list) -> None:
        """In-place Fisher-Yates, same draw order as the JS version."""
        for i in range(len(xs) - 1, 0, -1):
            j = self.randint(i + 1)
            xs[i], xs[j] = xs[j], xs[i]


# ---------------------------------------------------------------------------
# Tiny MLP: n_in -> n_hid (tanh) -> 1 (sigmoid), with a REINFORCE update
# ---------------------------------------------------------------------------

class TinyNN:
    def __init__(self, n_in: int, n_hid: int, rng: Mulberry32):
        self.n_in, self.n_hid = n_in, n_hid
        s1 = math.sqrt(1.0 / n_in)
        s2 = math.sqrt(1.0 / n_hid)
        # Draw order matches the JS constructor: W1 row-major, then W2.
        self.W1 = (np.array([rng.random() for _ in range(n_hid * n_in)])
                   .reshape(n_hid, n_in) * 2.0 - 1.0) * s1
        self.b1 = np.zeros(n_hid)
        self.W2 = (np.array([rng.random() for _ in range(n_hid)]) * 2.0 - 1.0) * s2
        self.b2 = 0.0

    def forward(self, x) -> tuple:
        """Returns (p, h, x): cooperation/accept probability, hidden state, input."""
        x = np.asarray(x, dtype=float)
        h = np.tanh(self.W1 @ x + self.b1)
        z = float(self.W2 @ h + self.b2)
        p = 1.0 / (1.0 + math.exp(-z))
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return p, h, x

    def prob(self, x) -> float:
        return self.forward(x)[0]

    def reinforce(self, cache, action: int, adv: float, lr: float, beta: float):
        """Gradient ASCENT on J = adv * log pi(a|x) + beta * H(pi). action in {0,1}."""
        p, h, x = cache
        dz2 = adv * (action - p) + beta * math.log((1.0 - p) / p) * p * (1.0 - p)
        dh = self.W2 * dz2                      # computed BEFORE the W2 update
        self.W2 += lr * dz2 * h
        self.b2 += lr * dz2
        dz1 = dh * (1.0 - h * h)
        self.W1 += lr * np.outer(dz1, x)
        self.b1 += lr * dz1


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    __slots__ = ("id", "move_net", "acc_net", "b_move", "v_move", "b_acc", "v_acc",
                 "wealth", "games", "coop_acts", "recent", "mem", "gossip")

    def __init__(self, id_: int, rng: Mulberry32, hidden: int):
        self.id = id_
        self.move_net = TinyNN(9, hidden, rng)
        self.acc_net = TinyNN(5, hidden, rng)
        self.b_move, self.v_move = 0.0, 1.0   # payoff baseline + running variance
        self.b_acc, self.v_acc = 0.0, 1.0
        self.wealth = 0.0
        self.games = 0
        self.coop_acts = 0
        self.recent = []                       # last <=20 own moves (1=C, 0=D)
        self.mem = {}                          # partner_id -> [(my, th, round), ...]
        self.gossip = {}                       # partner_id -> [(their_move, round), ...]


def _pd_payoff(ma: str, mb: str, P: dict) -> float:
    if ma == "C":
        return P["R"] if mb == "C" else P["S"]
    return P["T"] if mb == "C" else P["P"]


# ---------------------------------------------------------------------------
# The society
# ---------------------------------------------------------------------------

class NNSociety:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.cfg["payoffs"] = dict(cfg["payoffs"])
        self.rng = Mulberry32(self.cfg["seed"])
        self.round = 0
        self.agents = [Agent(i, self.rng, self.cfg["hidden"])
                       for i in range(self.cfg["population"])]

        # Founding prior: diverse newborn temperaments (some trusting, some
        # cynical). Without behavioral variation no brain could ever learn
        # "cooperate with cooperators", because cooperators wouldn't exist yet.
        div = self.cfg.get("init_diversity", 1.2)
        for a in self.agents:
            a.move_net.b2 = self.cfg.get("init_coop", 0.0) + (self.rng.random() * 2 - 1) * div
        # Youthful openness: newborn accept-brains start out trusting.
        it = self.cfg.get("init_trust", 1.0)
        if it:
            for a in self.agents:
                a.acc_net.b2 += it
        # Selective-trust prior: newborn accept-brains are wired to favor the
        # trustworthy. Fully plastic -- experience can overwrite it.
        tp = self.cfg.get("trust_prior", 1.5)
        if tp:
            for a in self.agents:
                net, k = a.acc_net, 2.0
                net.W1[0] = np.array([k, 0.0, 0.0, 0.0, 0.0])
                net.b1[0] = -k * 0.5
                net.W2[0] = tp

        self.stats = {"coop": [], "accept": [], "games": [], "wealth": [],
                      "entropy": [], "pT1": [], "pT0": []}
        self.recent_games = []
        self.events = [(0, f"{self.cfg['population']} blank-slate minds are born. "
                           "Nobody knows anything yet.")]
        self.total_invites = 0
        self.total_accepts = 0
        self.total_games = 0
        self.total_gossip = 0

    # -- perception ------------------------------------------------------
    def belief(self, s: Agent, o: Agent) -> float:
        """Trust in o: weighted blend of direct memory and gossip (0..1)."""
        M, c, w = self.cfg["memory"], 0.0, 0.0
        if M > 0:
            d = s.mem.get(o.id)
            if d:
                for my, th, r in d[-M:]:
                    if th == "C":
                        c += 1.0
                    w += 1.0
            g = s.gossip.get(o.id)
            if g:
                t = self.cfg["gossip_trust"]
                for mv, r in g[-M:]:
                    if mv == "C":
                        c += t
                    w += t
        return 0.5 if w == 0 else c / w

    # -- features --------------------------------------------------------
    def move_feats(self, a: Agent, b: Agent, am_inviter: bool):
        P, hist = self.cfg["payoffs"], a.mem.get(b.id)
        last = hist[-1] if hist else None
        rc = rn = grudge = 0
        if hist:
            for my, th, r in hist[-5:]:
                rn += 1
                if th == "C":
                    rc += 1
            for my, th, r in hist:
                if th == "D":
                    grudge = 1
                    break
        denom = P["T"] - P["S"]
        avg = a.wealth / a.games if a.games else 0.0
        return [
            self.belief(a, b),
            (last[1] == "C") - (last[1] == "D") if last else 0,   # their last move
            (last[0] == "C") - (last[0] == "D") if last else 0,   # my last move
            rc / rn if rn else 0.5,                              # their recent coop rate
            min(1.0, len(hist) / 10) if hist else 0.0,            # familiarity
            min(1.0, max(0.0, (avg - P["S"]) / denom)) if denom > 1e-9 else 0.5,
            1 if am_inviter else -1,
            self.round / self.cfg["rounds"],
            grudge,                                              # 1 if they EVER betrayed me
        ]

    def acc_feats(self, b: Agent, a: Agent):
        P, hist = self.cfg["payoffs"], b.mem.get(a.id)
        rc = rn = 0
        if hist:
            for my, th, r in hist[-5:]:
                rn += 1
                if th == "C":
                    rc += 1
        denom = P["T"] - P["S"]
        avg = b.wealth / b.games if b.games else 0.0
        return [
            self.belief(b, a),
            rc / rn if rn else 0.5,
            min(1.0, len(hist) / 10) if hist else 0.0,
            min(1.0, max(0.0, (avg - P["S"]) / denom)) if denom > 1e-9 else 0.5,
            self.round / self.cfg["rounds"],
        ]

    # -- social structure -------------------------------------------------
    def choose_partner(self, a: Agent):
        cfg, n, rng = self.cfg, len(self.agents), self.rng
        if n < 2:
            return None
        loc = cfg.get("locality", 0.7)
        nb = cfg.get("neighbors", 8)
        if rng.random() < loc:
            pool = []
            for k in range(1, nb // 2 + 1):
                pool.append(self.agents[(a.id + k) % n])
                pool.append(self.agents[(a.id - k) % n])
        else:
            pool = self.agents[:]
        if rng.random() < cfg["epsilon"]:
            b = pool[rng.randint(len(pool))]
            return None if b.id == a.id else b
        rng.shuffle(pool)
        best, best_b = None, -1.0
        for b in pool:
            if b.id == a.id:
                continue
            bl = self.belief(a, b)
            if bl > best_b:
                best_b, best = bl, b
        return best

    # -- learning ----------------------------------------------------------
    def _remember(self, a: Agent, b: Agent, my: str, th: str):
        arr = a.mem.get(b.id)
        if arr is None:
            arr = a.mem[b.id] = []
        arr.append((my, th, self.round))
        if len(arr) > 60:
            del arr[0]

    def _learn_meeting(self, a: Agent, steps: list, gamma: float):
        """Credit each game-step with the discounted payoff stream it caused."""
        G, rets = 0.0, [0.0] * len(steps)
        for i in range(len(steps) - 1, -1, -1):
            G = steps[i]["r"] + gamma * G
            rets[i] = G
        bd = self.cfg["base_decay"]
        for i, s in enumerate(steps):
            d = rets[i] - a.b_move
            a.v_move += (1 - bd) * (d * d - a.v_move)
            adv = d / (math.sqrt(max(1e-6, a.v_move)) + 1.0)
            a.b_move += bd * d
            a.move_net.reinforce(s["cache"], s["act"], adv, self.cfg["lr"], self.cfg["beta"])

    def _learn_acc(self, b: Agent, cache, act: int, G: float):
        """Social openness learns SLOWLY -- trust, once given, is stubborn."""
        lr_a = self.cfg.get("lr_acc", 0.008)
        bd = self.cfg["base_decay"] * 0.25
        d = G - b.b_acc
        b.v_acc += (1 - bd) * (d * d - b.v_acc)
        adv = d / (math.sqrt(max(1e-6, b.v_acc)) + 1.0)
        b.b_acc += bd * d
        b.acc_net.reinforce(cache, act, adv, lr_a, self.cfg["beta"])

    # -- main loop ---------------------------------------------------------
    def step(self):
        cfg, rng, P = self.cfg, self.rng, self.cfg["payoffs"]
        m, gamma = cfg["meeting"], cfg["gamma"]
        self.round += 1
        order = self.agents[:]
        rng.shuffle(order)
        games = []
        coop_acts = acts = invites = accepts = 0
        for a in order:
            for _ in range(cfg["invites"]):
                b = self.choose_partner(a)
                if b is None:
                    continue
                invites += 1
                cache_acc = b.acc_net.forward(self.acc_feats(b, a))
                accepted = 1 if rng.random() < cache_acc[0] else 0
                if not accepted:
                    # Rejecting isn't free: loneliness keeps society mingling.
                    self._learn_acc(b, cache_acc, 0, -(cfg["reject_cost"] or 0))
                    continue
                accepts += 1
                # Relationship bonding: strangers play for low stakes; old
                # friends play for real. Discovering cooperators is cheap,
                # sustaining cooperation is lucrative.
                bonding = cfg.get("bonding", 0.85)
                steps_a, steps_b = [], []
                for gi in range(m):
                    h_ab, h_ba = a.mem.get(b.id), b.mem.get(a.id)
                    fam = min(h_ab and min(1.0, len(h_ab) / 12) or 0.0,
                              h_ba and min(1.0, len(h_ba) / 12) or 0.0)
                    stake = (1 - bonding) + bonding * fam
                    c_a = a.move_net.forward(self.move_feats(a, b, True))
                    c_b = b.move_net.forward(self.move_feats(b, a, False))
                    ma = "C" if rng.random() < c_a[0] else "D"
                    mb = "C" if rng.random() < c_b[0] else "D"
                    # Retaliation reflex (the one hard-wired norm): you cannot
                    # cooperate with someone who JUST betrayed you.
                    if cfg.get("reflex", 1):
                        if gi > 0 and steps_b[gi - 1]["act"] == 0:
                            ma = "D"
                        if gi > 0 and steps_a[gi - 1]["act"] == 0:
                            mb = "D"
                    pa, pb = _pd_payoff(ma, mb, P) * stake, _pd_payoff(mb, ma, P) * stake
                    lam = cfg.get("fairness", 0) or 0   # inequity aversion
                    ineq = abs(pa - pb)
                    ua, ub = pa - lam * ineq, pb - lam * ineq
                    steps_a.append({"cache": c_a, "act": 1 if ma == "C" else 0, "r": ua})
                    steps_b.append({"cache": c_b, "act": 1 if mb == "C" else 0, "r": ub})
                    self._remember(a, b, ma, mb)
                    self._remember(b, a, mb, ma)
                    if ma == "C":
                        coop_acts += 1
                        a.coop_acts += 1
                    if mb == "C":
                        coop_acts += 1
                        b.coop_acts += 1
                    acts += 2
                    a.wealth += pa
                    b.wealth += pb
                    a.games += 1
                    b.games += 1
                    a.recent.append(1 if ma == "C" else 0)
                    if len(a.recent) > 20:
                        del a.recent[0]
                    b.recent.append(1 if mb == "C" else 0)
                    if len(b.recent) > 20:
                        del b.recent[0]
                    games.append((a.id, b.id, ma, mb))
                self._learn_meeting(a, steps_a, gamma)
                self._learn_meeting(b, steps_b, gamma)
                G = 0.0
                for s in reversed(steps_b):
                    G = s["r"] + gamma * G
                self._learn_acc(b, cache_acc, 1, G)

        # Gossip: intel is a club good -- shared only with people you trust.
        goss = 0
        gc = cfg.get("gossip_circle", 0.55)
        for a in self.agents:
            if rng.random() < cfg["gossip"]:
                keys = list(a.mem.keys())
                if keys:
                    jid = keys[rng.randint(len(keys))]
                    recs = a.mem[jid]
                    th = recs[-1][1]
                    c = self.agents[rng.randint(len(self.agents))]
                    if c.id != a.id and c.id != jid and self.belief(a, c) >= gc:
                        arr = c.gossip.get(jid)
                        if arr is None:
                            arr = c.gossip[jid] = []
                        arr.append((th, self.round))
                        if len(arr) > 60:
                            del arr[0]
                        goss += 1

        # Cultural transmission (optional): blend toward the most successful
        # person you know, with a little mutation noise.
        if cfg.get("social_rate", 0) > 0:
            for a in self.agents:
                if rng.random() < cfg["social_rate"]:
                    known = {a.id} | set(a.mem) | set(a.gossip)
                    best = a
                    for oid in known:
                        o = self.agents[oid]
                        if o.wealth > best.wealth:
                            best = o
                    if best is not a and best.move_net.n_hid == a.move_net.n_hid:
                        tau = 0.5
                        A, B = a.move_net, best.move_net
                        A.W1[:] = ((1 - tau) * A.W1 + tau * B.W1
                                   + (np.array([rng.random() for _ in range(A.W1.size)])
                                      .reshape(A.W1.shape) * 2 - 1) * 0.05)
                        A.b1[:] = ((1 - tau) * A.b1 + tau * B.b1
                                   + (np.array([rng.random() for _ in range(A.n_hid)]) * 2 - 1) * 0.05)
                        A.W2[:] = ((1 - tau) * A.W2 + tau * B.W2
                                   + (np.array([rng.random() for _ in range(A.n_hid)]) * 2 - 1) * 0.05)
                        A.b2 = (1 - tau) * A.b2 + tau * B.b2 + (rng.random() * 2 - 1) * 0.05
                        if rng.random() < 0.1:
                            self.events.append(
                                (self.round, f"#{a.id} adopted the ways of thriving #{best.id}"))

        # -- stats ---------------------------------------------------------
        self.recent_games = games
        self.total_invites += invites
        self.total_accepts += accepts
        self.total_games += len(games)
        self.total_gossip += goss
        self.stats["coop"].append(coop_acts / acts if acts else 0.0)
        self.stats["accept"].append(accepts / invites if invites else 0.0)
        self.stats["games"].append(len(games))
        self.stats["wealth"].append(sum(a.wealth for a in self.agents) / len(self.agents))
        rt = self.round / cfg["rounds"]
        ent = pt1 = pt0 = 0.0
        for a in self.agents:
            pn = a.move_net.prob([0.5, 0, 0, 0.5, 0, 0.5, 1, rt, 0])
            ent += -(pn * math.log(pn) + (1 - pn) * math.log(1 - pn))
            pt1 += a.move_net.prob([1, 1, 0, 1, 1, 0.5, 1, rt, 0])
            pt0 += a.move_net.prob([0, -1, 0, 0, 1, 0.5, 1, rt, 1])
        n = len(self.agents)
        self.stats["entropy"].append(ent / n)
        self.stats["pT1"].append(pt1 / n)
        self.stats["pT0"].append(pt0 / n)
        if len(self.events) > 400:
            del self.events[:len(self.events) - 400]

    def run(self, n: int, progress: bool = False, finalize: bool = True):
        """Run n rounds. progress=True prints a status line every 10%.

        finalize=False skips the end-of-run milestone detection -- useful when
        driving the loop yourself in chunks (e.g. a Streamlit progress loop).
        """
        target = self.round + n
        mark = max(1, n // 10)
        done = 0
        while self.round < target:
            self.step()
            done += 1
            if progress and done % mark == 0:
                s = self.summary(50)
                print(f"  round {self.round}/{self.cfg['rounds']}  "
                      f"coop={s['coop']:.1%} accept={s['accept']:.1%}", flush=True)
        if finalize:
            self._milestones()
        return self

    def _milestones(self):
        s = self.summary(50)
        r = self.round
        if r > 60 and s["coop"] >= 0.75:
            self.events.append((r, f"Cooperation is flourishing -- "
                                   f"{s['coop']:.0%} of moves are cooperative."))
        if r > 60 and s["coop"] <= 0.25:
            self.events.append((r, f"Trust has collapsed -- only {s['coop']:.0%} cooperation."))
        if r > 60 and s["cond_gap"] > 0.5:
            self.events.append((r, "The minds have learned conditional trust -- cooperate "
                                   "with the trustworthy, punish the shady."))

    def summary(self, tail: int = 50) -> dict:
        def avg(key):
            d = self.stats[key][-tail:]
            return sum(d) / len(d) if d else 0.0

        return {"rounds": self.round, "coop": avg("coop"), "accept": avg("accept"),
                "wealth": avg("wealth"), "entropy": avg("entropy"),
                "pT1": avg("pT1"), "pT0": avg("pT0"),
                "cond_gap": avg("pT1") - avg("pT0"),
                "total_games": self.total_games, "total_gossip": self.total_gossip}


# ---------------------------------------------------------------------------
# Configs and presets (same defaults as the HTML app)
# ---------------------------------------------------------------------------

def base_config(overrides: dict | None = None, **kw) -> dict:
    cfg = dict(
        seed=7, population=60, rounds=400, invites=3,
        payoffs={"T": 5, "R": 3, "P": -3, "S": -4},
        memory=10, gossip=0.3, gossip_trust=0.5, gossip_circle=0.55,
        bonding=0.85, locality=0.7, neighbors=8, epsilon=0.15,
        lr=0.005, lr_acc=0.004, beta=0.01, base_decay=0.02, hidden=16,
        meeting=4, gamma=0.95, fairness=0.0,
        init_coop=0.0, init_diversity=1.2, init_trust=1.0,
        trust_prior=1.5, reject_cost=0.3, reflex=1, social_rate=0.0,
    )
    for src in (overrides, kw):
        if src:
            cfg.update(src)
    return cfg


PRESETS = {
    "Reality": {},
    "Classic PD lab": {"payoffs": {"T": 5, "R": 3, "P": 1, "S": 0},
                       "meeting": 1, "gamma": 0.9, "gossip_circle": 0, "reflex": 0},
    "No honor": {"reflex": 0},
    "Long relationships": {"meeting": 8},
    "Open gossip": {"gossip_circle": 0},
    "Bitter world": {"fairness": 0.5},
    "Amnesia": {"memory": 0, "gossip": 0},
    "High temptation": {"payoffs": {"T": 9, "R": 3, "P": -3, "S": -4}},
    "Panopticon": {"gossip": 0.9, "gossip_trust": 0.8, "memory": 20},
    "Slow learners": {"lr": 0.005},
    "Hot curiosity": {"beta": 0.05},
}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _print_summary(s: dict):
    print(f"\nrounds:        {s['rounds']}")
    print(f"cooperation:   {s['coop']:.1%}  (last-50-round avg)")
    print(f"acceptance:    {s['accept']:.1%}")
    print(f"mean wealth:   {s['wealth']:.0f}")
    print(f"P(coop|trusted):   {s['pT1']:.1%}   P(coop|distrusted): {s['pT0']:.1%}")
    print(f"policy entropy:{s['entropy']:.3f}")
    print(f"total games:   {s['total_games']:,}   gossip msgs: {s['total_gossip']:,}")


def _plot(soc: NNSociety):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots (pip install matplotlib).")
        return
    st = soc.stats
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Mini Society -- Neural Edition  (seed {soc.cfg['seed']})")
    ax[0, 0].plot(st["coop"]); ax[0, 0].set_title("Cooperation rate"); ax[0, 0].set_ylim(0, 1)
    ax[0, 1].plot(st["wealth"]); ax[0, 1].set_title("Mean wealth")
    ax[1, 0].plot(st["pT1"], label="trusted"); ax[1, 0].plot(st["pT0"], label="distrusted")
    ax[1, 0].set_title("Learned P(cooperate)"); ax[1, 0].set_ylim(0, 1); ax[1, 0].legend()
    ax[1, 1].plot(st["entropy"]); ax[1, 1].set_title("Policy entropy")
    fig.tight_layout()
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mini Society -- Neural Edition (Python)")
    ap.add_argument("--preset", default="Reality", choices=list(PRESETS),
                    help="scenario preset (default: Reality)")
    ap.add_argument("--rounds", type=int, default=None, help="override rounds")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--population", type=int, default=None, help="override population")
    ap.add_argument("--plot", action="store_true", help="show matplotlib charts at the end")
    ap.add_argument("--csv", default=None, metavar="FILE", help="save per-round stats CSV")
    ap.add_argument("--quiet", action="store_true", help="no progress output")
    args = ap.parse_args(argv)

    over = dict(PRESETS[args.preset])
    if args.rounds is not None:
        over["rounds"] = args.rounds
    if args.seed is not None:
        over["seed"] = args.seed
    if args.population is not None:
        over["population"] = args.population
    soc = NNSociety(base_config(over))

    print(f"preset={args.preset}  seed={soc.cfg['seed']}  "
          f"population={soc.cfg['population']}  rounds={soc.cfg['rounds']}")
    soc.run(soc.cfg["rounds"], progress=not args.quiet)
    for r, t in soc.events[1:]:
        print(f"  [round {r}] {t}")
    _print_summary(soc.summary())

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "coop_rate", "accept_rate", "games", "mean_wealth",
                        "entropy", "p_coop_trusted", "p_coop_distrusted"])
            for i in range(soc.round):
                w.writerow([i + 1, f"{soc.stats['coop'][i]:.4f}",
                            f"{soc.stats['accept'][i]:.4f}", soc.stats["games"][i],
                            f"{soc.stats['wealth'][i]:.2f}",
                            f"{soc.stats['entropy'][i]:.4f}",
                            f"{soc.stats['pT1'][i]:.4f}", f"{soc.stats['pT0'][i]:.4f}"])
        print(f"saved {args.csv}")
    if args.plot:
        _plot(soc)


if __name__ == "__main__":
    main()
