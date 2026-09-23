"""Mini Society -- Neural Edition: interactive Streamlit lab.

Run with:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

A friendlier front-end for mini_society_nn.py: pick a scenario preset, tweak the
key parameters, watch cooperation evolve live, then inspect individual agents'
learned policy curves, the society diary, and the final relationship network.
"""
import csv
import io

import numpy as np
import streamlit as st

from mini_society_nn import NNSociety, base_config, PRESETS

st.set_page_config(page_title="Mini Society -- Neural Edition", layout="wide")
st.title("Mini Society -- Neural Edition")
st.caption("Every agent owns two tiny neural nets, trained online with REINFORCE on "
           "its OWN payoffs. No hand-coded strategies -- watch norms emerge (or collapse).")

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------------
# Sidebar: preset + overrides
# --------------------------------------------------------------------------

def _apply_preset():
    """Seed the override widgets with the newly chosen preset's values."""
    b = base_config(PRESETS[st.session_state.preset_name])
    st.session_state.seed = b["seed"]
    st.session_state.population = b["population"]
    st.session_state.rounds = b["rounds"]
    st.session_state.invites = b["invites"]
    st.session_state.temptation = b["payoffs"]["T"]
    st.session_state.memory = b["memory"]
    st.session_state.gossip = b["gossip"]


if "preset_name" not in st.session_state:
    st.session_state.preset_name = "Reality"
    _apply_preset()
    st.session_state.soc = None
    st.session_state.run_label = ""

st.sidebar.header("Simulation")
st.sidebar.selectbox("Scenario preset", list(PRESETS), key="preset_name",
                     on_change=_apply_preset,
                     help="Starting point. The overrides below replace the preset's values.")
run_btn = st.sidebar.button("Run simulation", type="primary", use_container_width=True)

st.sidebar.subheader("Overrides")
st.sidebar.number_input("Random seed", key="seed", step=1,
                        help="Same seed = same run, also in the HTML version.")
st.sidebar.slider("Population", 4, 200, key="population")
st.sidebar.slider("Rounds", 20, 1000, key="rounds", step=20)
st.sidebar.slider("Invites per agent per round", 1, 6, key="invites")
st.sidebar.slider("Temptation payoff T", 3, 12, key="temptation",
                  help="Reward for betraying a cooperator. Higher = harder to sustain trust.")
st.sidebar.slider("Memory length", 0, 30, key="memory",
                  help="How many past encounters each agent remembers per partner.")
st.sidebar.slider("Gossip probability", 0.0, 1.0, key="gossip", step=0.05)
st.sidebar.selectbox("Retaliation reflex", ["Preset default", "On", "Off"], key="reflex_mode",
                     help="The one hard-wired norm: never cooperate with someone who JUST betrayed you.")


def _build_cfg():
    name = st.session_state.preset_name
    cfg = base_config(PRESETS[name])
    cfg["seed"] = int(st.session_state.seed)
    cfg["population"] = int(st.session_state.population)
    cfg["rounds"] = int(st.session_state.rounds)
    cfg["invites"] = int(st.session_state.invites)
    cfg["payoffs"] = dict(cfg["payoffs"])
    cfg["payoffs"]["T"] = int(st.session_state.temptation)
    cfg["memory"] = int(st.session_state.memory)
    cfg["gossip"] = float(st.session_state.gossip)
    mode = st.session_state.reflex_mode
    if mode == "On":
        cfg["reflex"] = 1
    elif mode == "Off":
        cfg["reflex"] = 0
    return cfg, name


# --------------------------------------------------------------------------
# Run loop with live progress
# --------------------------------------------------------------------------

if run_btn:
    cfg, name = _build_cfg()
    total = cfg["rounds"]
    soc = NNSociety(cfg)
    prog = st.progress(0, text=f"Round 0 / {total}")
    live = st.empty()
    chunk = max(1, total // 40)
    while soc.round < total:
        soc.run(min(chunk, total - soc.round), finalize=False)
        prog.progress(soc.round / total, text=f"Round {soc.round} / {total}")
        live.line_chart({"cooperation": soc.stats["coop"]}, height=180)
    soc._milestones()
    prog.empty()
    live.empty()
    st.session_state.soc = soc
    st.session_state.run_label = name


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def _csv_bytes(soc: NNSociety) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["round", "coop_rate", "accept_rate", "games", "mean_wealth",
                "entropy", "p_coop_trusted", "p_coop_distrusted"])
    for i in range(soc.round):
        w.writerow([i + 1, f"{soc.stats['coop'][i]:.4f}", f"{soc.stats['accept'][i]:.4f}",
                    soc.stats["games"][i], f"{soc.stats['wealth'][i]:.2f}",
                    f"{soc.stats['entropy'][i]:.4f}",
                    f"{soc.stats['pT1'][i]:.4f}", f"{soc.stats['pT0'][i]:.4f}"])
    return buf.getvalue().encode()


def _overview_fig(soc):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    st_ = soc.stats
    ax[0, 0].plot(st_["coop"])
    ax[0, 0].set_title("Cooperation rate")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].plot(st_["wealth"])
    ax[0, 1].set_title("Mean wealth")
    ax[1, 0].plot(st_["pT1"], label="P(coop | trusted)")
    ax[1, 0].plot(st_["pT0"], label="P(coop | distrusted)")
    ax[1, 0].set_title("Learned conditional trust")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend()
    ax[1, 1].plot(st_["entropy"])
    ax[1, 1].set_title("Policy entropy (decisiveness)")
    fig.tight_layout()
    return fig


def _network_fig(soc):
    n = len(soc.agents)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(ang), np.sin(ang)
    coop = [sum(a.recent) / len(a.recent) if a.recent else 0.5 for a in soc.agents]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    seen = set()
    for ai, bi, _ma, _mb in soc.recent_games:
        key = (min(ai, bi), max(ai, bi))
        if key in seen:
            continue
        seen.add(key)
        ax.plot([xs[ai], xs[bi]], [ys[ai], ys[bi]], color="gray", alpha=0.15, lw=0.6)
    sc = ax.scatter(xs, ys, c=coop, cmap="RdYlGn", vmin=0, vmax=1, s=70,
                    zorder=3, edgecolors="black", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="recent cooperation rate")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Society network -- final round ({len(seen)} ties)")
    return fig


def _policy_curves(agent, rt):
    """P(cooperate) vs trust belief for a familiar partner: loyal vs after betrayal."""
    ts = np.linspace(0, 1, 21)
    loyal = [agent.move_net.prob([t, 1, 1, 0.8, 1.0, 0.5, 1, rt, 0]) for t in ts]
    betrayed = [agent.move_net.prob([t, -1, 0, 0.2, 1.0, 0.5, 1, rt, 1]) for t in ts]
    accept = [agent.acc_net.prob([t, 0.8, 1.0, 0.5, rt]) for t in ts]
    return ts, loyal, betrayed, accept


def render_results(soc: NNSociety, label: str):
    s = soc.summary()
    st.subheader(f"Results -- {label} (seed {soc.cfg['seed']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cooperation", f"{s['coop']:.1%}")
    c2.metric("Invite acceptance", f"{s['accept']:.1%}")
    c3.metric("Mean wealth", f"{s['wealth']:.0f}")
    c4.metric("Conditional-trust gap", f"{s['cond_gap']:+.1%}",
              help="P(cooperate|trusted) minus P(cooperate|distrusted)")

    st.download_button("Download per-round CSV", _csv_bytes(soc),
                       file_name="mini_society_run.csv", mime="text/csv")

    with st.expander("Effective configuration"):
        cfg = soc.cfg
        st.json({k: v for k, v in cfg.items() if k != "payoffs"} | {"payoffs": cfg["payoffs"]})

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Agent brains", "Society diary", "Network"])

    with tab1:
        if HAS_MPL:
            st.pyplot(_overview_fig(soc))
        else:
            st.line_chart({"cooperation": soc.stats["coop"]})
            st.line_chart({"P(coop|trusted)": soc.stats["pT1"],
                           "P(coop|distrusted)": soc.stats["pT0"]})
            st.line_chart({"mean wealth": soc.stats["wealth"]})
            st.line_chart({"policy entropy": soc.stats["entropy"]})
        st.caption(f"{s['total_games']:,} games played, {s['total_gossip']:,} gossip messages. "
                   f"Policy entropy near 0 = the minds have made up their minds.")

    with tab2:
        aid = st.selectbox("Inspect agent", [f"#{a.id}" for a in soc.agents], index=0)
        a = soc.agents[int(aid[1:])]
        rt = soc.round / soc.cfg["rounds"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Wealth", f"{a.wealth:.0f}")
        m2.metric("Cooperation rate", f"{a.coop_acts / a.games:.1%}" if a.games else "--")
        m3.metric("Partners known", len(a.mem))
        ts, loyal, betrayed, accept = _policy_curves(a, rt)
        st.line_chart({"move policy, loyal partner": loyal,
                       "move policy, after betrayal": betrayed},
                      x=ts, height=260)
        st.caption("Move policy: P(cooperate) vs trust belief (x-axis 0 to 1). "
                   "A learned conscience cooperates with the trustworthy and punishes betrayal.")
        st.line_chart({"accept policy": accept}, x=ts, height=200)
        st.caption("Accept policy: P(accept invite) vs trust in the inviter.")

    with tab3:
        for r, t in reversed(soc.events):
            st.write(f"**Round {r}:** {t}")
        st.caption(f"Invites: {soc.total_invites:,} ({soc.total_accepts / soc.total_invites:.1%} "
                   f"accepted)" if soc.total_invites else "No invites were sent.")

    with tab4:
        if HAS_MPL:
            st.pyplot(_network_fig(soc))
        else:
            st.info("Install matplotlib for the network view: pip install matplotlib")


soc = st.session_state.soc
if soc is None:
    st.info("Choose a scenario preset in the sidebar, tweak the overrides if you like, "
            "then hit **Run simulation**.")
    with st.expander("How this works"):
        st.markdown("""
- Each agent owns **two tiny neural nets**: a move policy (cooperate/defect) and an
  invite-accept policy, both trained online with **REINFORCE on the agent's own payoffs**.
- Agents choose partners by trust, remember past encounters, gossip only with people
  they trust, and play multi-game meetings where relationship bonding raises the stakes.
- The one hard-wired norm (toggleable): the **retaliation reflex** -- never cooperate
  with someone who *just* betrayed you. Everything else is learned from scratch.
- Try **"No honor"** to watch a society collapse, then **"High temptation"** to see how
  much greed trust can survive.
""")
else:
    render_results(soc, st.session_state.run_label)
