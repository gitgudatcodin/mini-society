# Mini Society — Neural Edition

A prisoner's-dilemma society lab where **every agent is a self-interested neural network**,
not a hand-coded strategy. Each agent owns two tiny MLPs (hand-rolled in vanilla JS,
zero dependencies) trained online with Monte-Carlo REINFORCE **on its own payoff**:

- **Move policy** — 9 inputs → 16 tanh hidden units → sigmoid (cooperate / betray)
- **Invite-acceptance policy** — 5 inputs → 16 tanh hidden units → sigmoid

Just open the HTML file in a browser — no build step, no server, no dependencies.

## Contents

- `app/mini-society-nn.html` — the Neural Edition simulator (31 tunable sliders, live
  network visualization, charts, brain inspector, CSV export, scenario presets)
- `app/mini-society.html` — the original rule-based version (ALLC, TFT, GTFT, PAVLOV,
  GRIM, ALLD, RAND + memory, gossip, imitation) for comparison
- `python/mini_society_nn.py` — **Python/NumPy port of the neural engine** (numpy only,
  no other dependencies). Bit-identical RNG to the browser version, so the same seed
  gives the same run. Verified against the JavaScript engine: all per-round stats
  match to <1e-9 over a 20-round test run. See `python/` usage below.
- `python/streamlit_app.py` — interactive Streamlit lab: presets, parameter overrides,
  live progress, agent brain inspector, society diary, network view, CSV download.
  Run with `pip install -r python/requirements.txt && streamlit run python/streamlit_app.py`
- `python/requirements.txt` — numpy, streamlit, matplotlib
- `paper/mini-society-paper.pdf` — 18-page writeup: model spec, experiments, findings
- `paper/paper-draft.md` — paper source (Markdown)
- `paper/render.py`, `paper/make_figures.py` — paper build scripts
- `paper/figures/` — the five result figures

## Run it (browser)

```bash
# no install needed — just open it
open app/mini-society-nn.html
```

Click **Run**, watch the society evolve. Click any node to inspect that agent's learned
policy curves (P(cooperate) vs. trust).

## Run it (Python)

```bash
pip install numpy          # the only dependency (matplotlib optional, for --plot)
cd python
python mini_society_nn.py                       # Reality preset, 400 rounds
python mini_society_nn.py --preset "No honor"   # watch trust collapse
python mini_society_nn.py --preset "High temptation" --seed 11 --plot
python mini_society_nn.py --rounds 100 --population 20 --csv run.csv
```

Or use it as a library:

```python
from mini_society_nn import NNSociety, base_config, PRESETS

soc = NNSociety(base_config(PRESETS["Reality"], seed=7))
soc.run(400)
print(soc.summary())
# {'rounds': 400, 'coop': 0.999, 'accept': 0.989, ...}
```

### Interactive Streamlit lab

For a point-and-click version of the above (no HTML/JS needed):

```bash
pip install -r python/requirements.txt
streamlit run python/streamlit_app.py
```

Pick a scenario preset in the sidebar, tweak the key parameters (seed, population,
temptation payoff, memory, gossip, reflex), and hit **Run simulation**. You get a live
cooperation curve while it runs, then four tabs: Overview charts, per-agent **brain
inspector** (each mind's learned P(cooperate) vs. trust curve), the society diary, and
the final relationship network — plus one-click CSV download of the per-round stats.

## Key findings (from the paper)

- Blank-slate selfish neural RL converges to mutual betrayal (~0–1% cooperation) on
  every seed — intelligence finds the Nash equilibrium. Same social infrastructure
  (memory + gossip + reputation) sustains ~90%+ cooperation with hand-coded conditional
  strategies but ~0% with selfish neural learners. **The minds matter more than the world.**
- With a retaliation-reflex norm as scaffolding: ~99.9% cooperation across 5 seeds.
- Lifting the reflex after 1,000 rounds preserves 99.9% — but lifting at 200 rounds
  drops to ~71–74%. The scaffold has a critical period.
- Headline result: the reflex is scaffolding for **ostracism, not punishment**. After
  convergence, agents never retaliate in-game (~100% cooperation even against injected
  defectors) — they simply stop inviting them (4–5% of games), and invaders end up
  ~25% poorer. The sustaining threat is *"no one will play with you."*

## Model defaults (documented in the paper)

- Population 60, 400 rounds, 3 invitations/round, meeting length 4, γ = 0.95
- Payoffs: T=5, R=3, P=−3, S=−4
- Memory 10, gossip 0.3, gossip trust 0.5, bonding 0.85, locality 0.7
- Move LR 0.005, accept LR 0.004, entropy β=0.01
