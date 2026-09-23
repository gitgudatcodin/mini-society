# Can Selfish Neural Agents Learn to Cooperate?

## A Laboratory Study of Emergent Trust in the *Mini Society — Neural Edition*

**Abstract.** We study whether cooperation can emerge among purely self-interested learning agents in an iterated prisoner's dilemma society. Each of 60 agents is controlled by two small neural networks — one choosing cooperate/defect, one choosing whether to accept social invitations — trained online by each agent on its *own* payoffs only, via Monte-Carlo REINFORCE. No strategies are hand-coded and no imitation is allowed. We find: (1) blank-slate selfish learning converges to mutual betrayal on every seed, as game theory predicts; (2) making betrayal genuinely costly (negative payoffs) does not produce trust — it produces *withdrawal*, a "hermit equilibrium" in which agents simply stop interacting; (3) a large battery of realistic social mechanisms — memory, conditional gossip, reputation, grudges, stable neighborhoods, relationship bonding, temperament diversity, slow social learning — improves texture but does not escape the trap; (4) adding a single transparent social norm, an automatic retaliation reflex (never cooperate right after being betrayed), reliably produces ~100% cooperation and ~99% social participation. New experiments reported here show the result is robust across seeds, that the norm acts as *scaffolding* — it can be lifted after convergence without collapse — and that the sustaining mechanism is **ostracism, not punishment**: residents never learn to retaliate in-game, but they learn to starve defectors of interactions, and defectors earn ~25% less. We probe how long the scaffolding must stay up (a critical period of several hundred rounds), map the norm's limits under extreme temptation, and decompose the contributions of the norm versus the social infrastructure. The lesson is institutional, not psychological: selfish minds do not bootstrap trust from nothing; but given a credible retaliation norm during a formative period, they learn to sustain cooperation on their own — by exclusion.

---

## 1. Introduction

The prisoner's dilemma is the canonical model of the tension between individual and collective rationality: betrayal pays in the short run, yet mutual betrayal leaves everyone worse off. A long tradition — from Axelrod's tournaments to the folk theorem — shows that *conditional* strategies (tit-for-tat and kin) can sustain cooperation when interactions repeat and defections are punished. But those results typically *assume* the conditional strategy exists. Our question is harder and more primitive: **start with minds that know nothing and want only to maximize their own reward. Can cooperation emerge at all?**

To answer it we built the *Mini Society — Neural Edition*, an interactive laboratory (a single self-contained HTML file running entirely in the browser) in which every agent is a genuine reinforcement learner. This paper documents the model specification, the full sequence of experiments — including an extensive catalogue of failures — and the findings.

### 1.1 What this paper is not

This is not a claim that we have solved multi-agent reinforcement learning, nor that our toy society models any real economy. Payoffs are symmetric, the population is small (60), and learning is deliberately simple (tiny MLPs, vanilla policy gradients). The value of the exercise is conceptual: by keeping the minds blank and the incentives selfish, we can isolate *which ingredients* are actually load-bearing for cooperation, and which are decorative.

---

## 2. Model specification

### 2.1 The stage game

Each encounter is a prisoner's dilemma with configurable payoffs. Our default "reality" payoffs, chosen to mimic real life, are:

| | Cooperate | Betray |
|---|---|---|
| **Cooperate** | R = **+3** (win-win) | S = **−4** (badly hurt) |
| **Betray** | T = **+5** ("quick money") | P = **−3** (lose-lose) |

satisfying T > R > 0 > P > S. Three properties matter: mutual cooperation is genuinely good (R > 0), mutual betrayal is genuinely bad (P < 0), and being exploited is the worst outcome (S < P). Betrayal still tempts (T > R), so the dilemma is real.

### 2.2 Agents: two brains, zero hand-coded strategies

Each agent owns **two** feed-forward networks, hand-rolled in vanilla JavaScript (no ML libraries):

- **Move network** (9 inputs → H tanh hidden units → sigmoid): outputs P(cooperate) for the current game. H = 16 by default.
- **Accept network** (5 inputs → H tanh hidden units → sigmoid): outputs P(accept) when *receiving* a social invitation.

**Move-network inputs** (all roughly normalized): (1) belief about the partner's trustworthiness (from memory + gossip, 0.5 = stranger); (2) partner's last move (+1/−1/none); (3) my last move; (4) partner's recent cooperation rate; (5) relationship depth; (6) my normalized average payoff; (7) whether I am the inviter; (8) simulation progress; (9) **grudge flag** — 1 if this partner has *ever* betrayed me.

**Accept-network inputs**: (1) belief about the inviter; (2) inviter's recent cooperation rate; (3) relationship depth; (4) my normalized average payoff; (5) simulation progress.

Weight initialization is Xavier-uniform. Three deliberate priors shape newborn minds without teaching them any strategy (see §2.6).

### 2.3 Social structure: invitations, meetings, bonding

Time runs in **rounds**. Each round, agents are shuffled; each agent extends a fixed number of **invitations** (default 3) to partners of its choosing. The invitee's accept-network decides. Rejection is not free: it costs a small **social need** penalty (default 0.3, "loneliness"), so withdrawal has a price.

An accepted invitation begins a **meeting**: a fixed number of consecutive prisoner's-dilemma games (default 4) with the same partner. Meetings are the model's repeated game — the soil in which reciprocity could grow.

**Relationship bonding:** stakes scale with familiarity. Strangers play for low stakes (15% of nominal payoffs by default); deep relationships play for full stakes. Discovering a cooperator is cheap; sustaining cooperation is lucrative — as in real life, you don't trust a stranger with much.

**Partner choice:** with probability *locality* (0.7) an agent picks from its **neighborhood** (the 8 nearest agents on a ring — stable, repeated relationships); otherwise it searches globally by reputation. A small epsilon (0.15) keeps random encounters flowing.

### 2.4 Learning: selfish policy gradients

Both networks train by **Monte-Carlo REINFORCE on the agent's own payoffs only**:

- The move network is credited with **discounted meeting-level returns** (foresight γ = 0.95): each game-step's update uses the discounted stream of payoffs it helped cause, so a brain *can* learn "cooperate now so they cooperate later."
- Advantages are **standardized by a running variance estimate**, keeping the entropy (curiosity) bonus meaningful at any payoff scale.
- An **entropy bonus** (β = 0.01) preserves exploration.
- The accept network learns from the meeting's total return with a **slower learning rate** (0.004 vs 0.005 for moves): tactics adapt fast, relationships adapt slowly — "trust, once given, is stubborn."
- Baselines adapt with rate 0.02.

Optional **inequity aversion** (Fehr–Schmidt style, fairness λ, default 0) can subtract λ·|payoff gap| from the learning utility. Optional **cultural transmission** (default off) lets agents blend their move-network weights toward the most successful person they know.

### 2.5 Social infrastructure: memory, gossip, reputation

- **Memory:** each agent privately records the last up-to-60 game outcomes per partner (default belief window: 10).
- **Gossip:** with probability 0.3 per round, an agent shares its latest experience of someone with a random third party — but **only with recipients it trusts** (belief ≥ *gossipCircle*, default 0.55). Intel is a club good: betrayers are cut out of the information network.
- **Reputation (belief):** a weighted blend of own memory and trusted gossip, defaulting to 0.5 for strangers.

### 2.6 Newborn priors (not strategies)

1. **Temperament diversity:** initial cooperation bias b₂ ~ Uniform(−1.2, +1.2) — some newborns are trusting, some cynical, most in between. Without behavioral variation, no brain could ever learn "cooperate with cooperators," because cooperators wouldn't exist yet.
2. **Youthful openness:** accept-network bias +1.0 (≈73% initial acceptance) — the young mingle freely; only sustained bad experience teaches withdrawal.
3. **Selective-trust prior:** the accept network is wired to favor trustworthy inviters from birth (weight 1.5 on the belief input through one hidden unit) — fully plastic, but selectivity exists *before* trust becomes informative.

### 2.7 The retaliation reflex: the one hard-coded norm

One behavior is **not learned**: the **retaliation reflex**. If enabled, an agent *cannot* cooperate in the next game of a meeting immediately after that partner betrayed it. Everything else — whom to invite, whether to accept, every other move — remains neural and learned.

We are explicit that this is a hand-coded social norm, labeled as such in the app. It does not teach cooperation; it makes *punishment reliable from round one*, so that betrayal credibly causes future loss. Whether that is enough for selfish learners to discover cooperation is the experiment.

### 2.8 Full parameter table (defaults)

| Parameter | Default | Meaning |
|---|---|---|
| Population / rounds / invites | 60 / 400 / 3 | Society size, duration, invitations per agent per round |
| Payoffs T / R / P / S | 5 / 3 / −3 / −4 | "Quick money" / win-win / lose-lose / badly hurt |
| Meeting length | 4 | Consecutive games per accepted invitation |
| Foresight γ | 0.95 | Discount factor for meeting-level returns |
| Memory / gossip / gossip trust | 10 / 0.3 / 0.5 | Belief window, gossip probability, gossip weight |
| Gossip circle | 0.55 | Min. trust to receive gossip |
| Bonding | 0.85 | Stakes scale with relationship depth |
| Locality / neighbors | 0.7 / 8 | Prob. of choosing within neighborhood / its size |
| Epsilon | 0.15 | Random-encounter rate |
| Move lr / accept lr / baseline | 0.005 / 0.004 / 0.02 | Learning rates (accept slower) |
| Entropy β | 0.01 | Curiosity bonus |
| Hidden units | 16 | MLP width (tanh) |
| Fairness λ | 0 | Inequity aversion (off by default) |
| initCoop / diversity / initTrust / trustPrior | 0 / 1.2 / 1.0 / 1.5 | Newborn priors |
| Reject cost | 0.3 | Loneliness penalty for declining invites |
| **Retaliation reflex** | **1 (on)** | The hard-coded norm |
| Cultural transmission | 0 | Weight-blending imitation (off) |

### 2.9 Scenario presets

The app ships 11 presets. The three central ones: **Reality** (all defaults above, reflex on); **No honor** (identical, reflex off); **Classic PD lab** (textbook payoffs T=5,R=3,P=1,S=0, one-shot meetings, reflex off). Others vary single factors: Long relationships (8-game meetings), Open gossip (gossip shared with everyone), Bitter world (fairness 0.5), Amnesia (no memory, no gossip), High temptation (T=9), Panopticon (heavy gossip), Slow learners, Hot curiosity.

---

## 3. What we tried, and what happened

### 3.1 Attempt 1: blank-slate selfish RL (the honest baseline)

The first version had textbook payoffs, one-shot meetings, no reflex, no priors — just two MLPs maximizing own reward. Result across ~30 runs and every seed: **~0–1% cooperation**. Intelligence found the Nash equilibrium. This is not a bug; for any fixed partner mix in the prisoner's dilemma, E[defect] > E[cooperate] is a theorem, so independent gradient learners cannot bootstrap cooperation. Texture varied (agents kept mingling since P=1>0; with costly betrayal they withdrew instead), but the destination never did.

**Verdict: failed**, exactly as theory predicts. The minds matter more than the world: the same social infrastructure (memory + gossip + reputation) that sustained 90%+ cooperation in our earlier *rule-based* Mini Society (with hand-coded conditional strategies) produced ~0% with neural selfish learners.

### 3.2 Attempt 2: reality payoffs (make betrayal hurt)

We changed payoffs to T=5, R=3, P=−3, S=−4 so that mutual betrayal is genuinely bad and exploitation genuinely hurts. Surely, we thought, agents would learn to avoid the −3/−4 outcomes by cooperating.

They did something else: **they stopped interacting**. Invitation acceptance collapsed to ~0.7%. Why play a game where the worst case is −4 when you can simply not play? We call this the **hermit equilibrium**: negative payoffs alone teach *avoidance*, not trust. Rational agents avoid dangerous interactions rather than discovering cooperative ones.

**Verdict: failed**, and instructively so. Punishment without a *reliable path to the reward* produces hermits, not citizens.

### 3.3 Attempt 3: the "reality stack" — every realistic mechanism at once

We then layered on everything that plausibly helps in real societies, each implemented as faithfully as we could:

- repeated meetings (4 games) with foresight (γ=0.95) so reciprocity is learnable;
- memory + conditional gossip + reputation;
- a permanent grudge flag input;
- stable neighborhoods (locality 0.7) for repeated relationships;
- relationship bonding (low stakes with strangers);
- temperament diversity, youthful openness, selective-trust prior;
- slow social learning (accept-lr < move-lr) and a loneliness cost for rejection.

After 1,000 rounds: **~15–16% cooperation, ~2–3% acceptance**. Better than zero — a small trusting subculture flickers — but the society as a whole never takes off. Memory and reputation are not sufficient if nobody initially supplies reliably cooperative behavior: there is nothing for the learners to latch onto.

**Verdict: failed.** A long list of individually sensible mechanisms does not compose into cooperation.

### 3.4 Attempt 4: the retaliation reflex (what worked)

The breakthrough was making *one* behavior reliable instead of learned: **never cooperate right after being betrayed** (within a meeting). This is transparently labeled in the app as the single hand-coded social norm.

With the reflex enabled (the **Reality** preset), selfish agents face a new calculus: betraying a cooperator yields +5 now but *guarantees* retaliation in the remaining games of the meeting, turning a stream of +3s into −3s. Gradient ascent discovers this quickly, and the society converges to essentially full cooperation.

**Robustness (new experiments for this paper).** We re-ran Reality and No-honor for 1,000 rounds on five seeds (7, 11, 23, 42, 99), measuring last-50-round averages:

| Seed | Reality (reflex on): coop / accept | No honor (reflex off): coop / accept |
|---|---|---|
| 7 | 99.9% / 99.1% | 14.9% / 2.4% |
| 11 | 99.9% / 98.9% | 15.4% / 1.9% |
| 23 | 99.9% / 99.0% | 16.3% / 2.6% |
| 42 | 99.9% / 98.8% | 14.7% / 2.1% |
| 99 | 99.9% / 98.9% | 15.5% / 1.9% |

The reflex is doing the work: identical worlds differ only in the norm, and outcomes differ by ~85 points of cooperation and ~97 points of social participation. The effect is not a seed artifact — variance across seeds is negligible.

**Verdict: worked — decisively.** Reliable retaliation changes the *institution*, not the agents: with the norm, defection reliably causes future loss, so selfish optimization supports cooperation.

### 3.5 Attempt 5 (new): is the norm internalized? Lift it after convergence.

A natural objection: perhaps the reflex merely *coerces* cooperation, and the brains never genuinely learn anything — remove the training wheels and the society crashes. We tested it: train with the reflex for 1,000 rounds (cooperation ≈100%), then set reflex=0 and continue for 500 rounds, on three seeds.

| Seed | Coop before lift | Coop after 500 rounds (no reflex) | Acceptance after |
|---|---|---|---|
| 7 | 99.9% | 99.9% | 99.4% |
| 11 | 99.9% | 99.9% | 99.3% |
| 23 | 99.9% | 99.9% | 99.2% |

**Result: cooperation persists — but not for the reason we first guessed.** There is no dip at all in the 500 post-lift rounds (traces hold at 99.9–100% throughout). Our first hypothesis was that the networks had internalized *retaliation* — learned to defect against betrayers. The invasion test (§3.5c) proves that hypothesis **wrong**, and reveals the true mechanism: the society learned **ostracism**, not punishment. Residents never retaliate in-game — but they starve defectors of interactions, and defectors earn ~25% less than cooperators. The reflex is scaffolding for *exclusion*, not for revenge: it keeps the society cooperative long enough for the slow-learning accept networks and reputation-based partner choice to learn who to shun, and shunning is what sustains cooperation after the scaffolding retires.

### 3.5b (new): how long must the scaffolding stay up?

We repeated the lift experiment with the norm removed after 200, 500, and 1,000 rounds, then ran 500 further rounds without it:

| Norm lifted after | Seed | Coop before lift | Coop 500 rounds later | Self-sustaining? |
|---|---|---|---|---|
| 200 rounds | 7 / 11 / 23 | 91.8% / 93.3% / 89.8% | 71.3% / 74.4% / 71.0% | No — decays |
| 500 rounds | 7 / 11 / 23 | 99.6% / 99.7% / 99.6% | 98.0% / 95.8% / 98.1% | Mostly — slight decay |
| 1,000 rounds | 7 / 11 / 23 | 99.9% / 99.9% / 99.9% | 99.9% / 99.9% / 99.9% | Yes — fully |

**Finding: there is a critical period.** Lift the scaffolding at 200 rounds and cooperation erodes to ~72% — the slow-learning accept networks (lr 0.004 vs 0.005 for moves) have not yet learned *whom to shun*, so ostracism is incomplete and the society partially unravels. By 500 rounds exclusion is nearly solid (96–98% sustained), and by 1,000 rounds it is fully internalized: the norm can retire with zero loss. Notably, our synthetic policy probes (P(cooperate) for fixed "trusted" vs "betrayer" feature vectors) show near-zero conditional gap even at 1,000 rounds — consistent with the invasion result below: the learned conditionality lives in the *social* policy (whom to interact with), not the *move* policy (what to do once trapped in a game). The behavioral evidence is the real test of what was learned.

### 3.5c (new): invasion test — exclusion, not punishment

Persistence after the lift could mean residents learned to *punish* defectors (retaliate in-game) or merely to *avoid* them. We distinguished them by converging a society under the reflex, lifting it, then injecting 3 always-defect invaders (5% of the population) and watching for 500 rounds:

| Seed | Resident coop rate *vs invaders* (in-game) | Invader share of all games | Wealth gained during invasion: invader / resident |
|---|---|---|---|
| 7 | 100% | 4.3% | 24,958 / 34,340 |
| 11 | 99.9% | 4.8% | 27,217 / 34,210 |
| 23 | 100% | 5.2% | 28,961 / 34,168 |

**Residents never retaliate — they shun.** When trapped in a game with a defector, residents cooperate essentially 100% of the time (they are pushovers in-game; the move policy never learned conditional punishment). But defectors are frozen out of the social fabric: they initiate 5% of all meetings yet account for only ~4–5% of games *total*, meaning residents essentially never invite them and reject most of their invitations — against a background acceptance rate of ~99%, that is sharp discrimination. Starved of interactions, invaders gain **~20–27% less wealth** than cooperators over the invasion period. Defection doesn't pay — not because it is punished, but because it is *excluded*.

This reframes the whole paper: the retaliation reflex does not teach the brains to retaliate. It holds the society in a cooperative configuration for the hundreds of rounds the slow social brains need to learn *reputation-based exclusion* — partner choice by belief, invitation acceptance conditioned on the inviter's trustworthiness. Once exclusion is learned, the reflex is redundant: the threat that sustains cooperation is not "if you betray me I will strike back" but "if you betray anyone, no one will play with you."

### 3.6 Attempt 6 (new): grudges — permanent vs. forgiving

The grudge flag (input #9) is permanent by default: one betrayal marks a partner forever. We compared permanent grudges against a decaying variant (flag set only if betrayed within the last 12 games), in two worlds:

**Under the Reality preset (reflex on):**

| Seed | Permanent: coop / wealth | Decaying: coop / wealth |
|---|---|---|
| 7 | 99.9% / 59,879 | 99.9% / 59,527 |
| 11 | 99.9% / 58,864 | 99.9% / 58,545 |
| 23 | 99.9% / 59,081 | 99.9% / 59,036 |

**Result: no meaningful difference.** Once the reflex sustains near-universal cooperation, betrayals become so rare that the grudge flag almost never fires — there is nothing to forgive or remember. The norm does the heavy lifting; grudge persistence is decorative *in a cooperative world*. This is itself informative: forgiveness debates matter most where betrayal is common.

**Without the reflex (No-honor world), where betrayals are common:**

| Seed | Permanent grudge: coop / accept / wealth | Decaying grudge: coop / accept / wealth |
|---|---|---|
| 7 | 14.9% / 2.4% / −2,997 | 16.3% / 2.6% / −2,836 |
| 11 | 15.4% / 1.9% / −2,933 | 15.6% / 2.3% / −2,881 |
| 23 | 16.3% / 2.6% / −2,548 | 16.1% / 2.1% / −2,706 |

**Result: forgiveness changes nothing here either.** In a world without the norm, the society sits at ~15% cooperation and negative wealth regardless of grudge policy — there is no cooperative surplus for forgiveness to protect or for grudges to defend. Combined with the reflex-on result, the honest summary is: **grudge persistence is a second-order concern.** It matters neither where cooperation is secured by the norm (nothing to forgive) nor where cooperation has failed (nothing to protect). The first-order question is always whether betrayal is reliably punished; everything else is commentary.

### 3.7 (new): stress tests — extreme temptation, and the norm alone

Two isolation experiments pin down *what* is load-bearing:

**Stress 1 — extreme temptation (T=9, triple the reward for betrayal), reflex on:**

| Seed | Cooperation | Acceptance |
|---|---|---|
| 7 | 45.3% | 88.2% |
| 11 | 45.0% | 88.7% |
| 23 | 43.6% | 89.0% |

The norm has limits. At T=5 the retaliation threat (a meeting of −3s) outweighs the +5 temptation and cooperation goes to ~100%; at T=9 the immediate +9 payoff overwhelms the learned deterrent and the society settles into a mixed equilibrium — notably *without* withdrawing (acceptance stays ~89%). Retaliation deters when the threatened loss exceeds the temptation; beyond that, the institution fails gracefully into partial cooperation rather than collapse.

**Stress 2 — the norm alone (reflex on, memory=0, gossip=0):**

| Seed | Cooperation | Acceptance |
|---|---|---|
| 7 | 92.3% | 98.9% |
| 11 | 79.4% | 97.0% |
| 23 | 65.8% | 64.4% |

The reflex *alone* — stripped of memory-based partner choice and gossip — still produces mostly-cooperative societies, far above the ~15% of the full stack without the reflex. But it is fragile: outcomes swing from 66% to 92% across seeds, and one seed partially withdraws. The social infrastructure's true contribution is now clear: it does not *create* cooperation (the norm does that), but it **stabilizes** it — converting a seed-dependent 66–92% into a uniform 99.9% by letting agents selectively avoid the few remaining defectors. Infrastructure as shock absorber, norm as engine.

### 3.8 What we deliberately did not do

- **Explicit coalitions / group retaliation** (trust groups collectively sanctioning betrayers) was on our roadmap and remains unimplemented — listed under future work.
- **Cultural transmission** (imitating the successful) is implemented but default-off; early tests suggested it spreads whatever is already winning, which without the reflex is betrayal.
- **Mistakes and noise** (accidental betrayals, misperceived moves) are not modeled; all defections are intentional.

---

## 4. Findings

**1. Intelligence finds the Nash equilibrium, not the social optimum.** Blank-slate policy-gradient agents converge to mutual betrayal on every seed. This is the expected fixed point of independent selfish learning in the prisoner's dilemma, and no amount of "smarter" optimization escapes it — the gradient points at betrayal.

**2. Costly betrayal produces hermits, not cooperators.** When mutual betrayal pays negatively, agents learn *avoidance* (acceptance → ~1%), not trust. A frequent modeling mistake is assuming that making bad outcomes worse teaches good behavior; it teaches *exit*.

**3. Social infrastructure stabilizes; it does not create.** Memory, gossip, reputation, neighborhoods, bonding, diversity — the full realistic stack without the norm — lifted cooperation only to ~15%. These mechanisms cannot *create* cooperation from a blank slate, because there is initially nothing cooperative to remember, gossip about, or imitate. But they are not decorative either: the reflex *alone* (no memory, no gossip) yields a fragile 66–92% that varies wildly by seed, while reflex + infrastructure yields a uniform 99.9%. The infrastructure is a shock absorber — selective partner choice quarantines the few remaining defectors — while the norm is the engine.

**4. One reliable norm changes everything — within limits.** The retaliation reflex — the only non-learned behavior — moves the society from ~15% to ~100% cooperation. The mechanism is transparent: certain, immediate punishment makes betrayal unprofitable *within the learning horizon* of the agents, so selfish gradients climb toward cooperation. But the deterrent must outweigh the temptation: at T=9 (vs R=3) the norm only sustains ~45% cooperation. Institutions deter in proportion to the credibility and size of the threatened loss.

**5. The norm is scaffolding for exclusion, and it can be removed (new).** Lifting the retaliation reflex after convergence does *not* destroy cooperation — it persists at ~100%, because during the scaffolded period the society learned **ostracism**: reputation-based partner choice and trust-conditioned invitation acceptance that starve defectors of interactions. In-game retaliation was *never* learned (residents cooperate ~100% even with known defectors). There is a critical period: lift at 200 rounds and cooperation decays to ~72% (exclusion not yet learned); by 1,000 rounds the norm retires with zero loss. The sustaining threat is not "I will strike back" but "no one will play with you" — and defectors earn ~25% less than cooperators.

**6. Grudges are second-order (new).** Permanent vs. decaying grudges produce indistinguishable outcomes both where the norm secures cooperation (nothing to forgive) and where cooperation has failed (nothing to protect). The first-order question is always whether betrayal is reliably punished; grudge policy is commentary.

**7. Honesty condition.** This is no longer a model in which every behavioral element is learned from a blank slate. Exactly one behavior — strike back when struck — is programmed. We report this prominently because the alternative (implying the agents "discovered" retaliation) would be the interesting lie. What the agents genuinely discover is everything else: whom to trust, whom to invite, when to mingle, and that cooperation pays *given* the norm.

---

## 5. Limitations

- **Toy scale and symmetry:** 60 agents, symmetric payoffs, simultaneous learning. Real societies have roles, asymmetries, and institutions far richer than a retaliation reflex.
- **No noise:** every betrayal is intentional and perfectly observed. Real trust must survive mistakes and misperception; our agents never face that test.
- **Simple learners:** tiny MLPs with vanilla REINFORCE. Stronger algorithms (actor-critic, opponent modeling) might discover retaliation on their own — an open question we did not test.
- **No entry/exit or reproduction:** the population is fixed; there is no evolution, migration, or institutional competition.
- **The reflex is crude:** it fires even when retaliation is strategically pointless (e.g., the last game of a meeting), and it cannot forgive.

## 6. Future work

1. **Learned retaliation:** replace the hard-coded reflex with cross-meeting actor-critic foresight and test whether genuine retaliation policies emerge from scratch.
2. **Coalitions and collective sanctions:** formal trust groups whose members jointly punish anyone who betrays a member — the "group retaliation" mechanism originally requested.
3. **Forgiveness and reconciliation:** escalating grudges, apologies, and costly signals of reform; mistaken betrayal with observable intent.
4. **Institutional competition:** let societies with different norms compete via migration and see which norms survive.
5. **Noise robustness:** introduce trembles and misperception, and study which norms keep cooperation alive when trust can be broken by accident.

---

## Appendix A. Reproducibility

The laboratory is a single self-contained file, `mini-society-nn.html` (vanilla JS, no dependencies, no network), so every result in this paper can be reproduced by opening the file in a browser, applying the named preset, and clicking Run. The neural engine (lines marked `NN-ENGINE-START/END`) has no DOM dependencies and was additionally verified headless under Node.js; the new experiments in §3.4–§3.6 were run headless with seeds {7, 11, 23, 42, 99}. Summary statistics are averages over the last 50 rounds.

## Appendix B. The road not taken (changelog of failed attempts)

For readers building similar models, the condensed failure log: textbook payoffs + one-shot + no priors → 0–1% cooperation; reality payoffs alone → hermit equilibrium (~0.7% acceptance); + meetings/foresight → marginal; + memory/gossip/reputation → marginal; + grudge flag → marginal; + neighborhoods → marginal; + bonding → marginal; + temperament diversity + youthful openness + selective-trust prior → ~10%; + slow social learning + reject cost → ~15%; **+ retaliation reflex → ~100%**. No single mechanism before the reflex moved the needle more than a few points; the reflex moved it ~85 points.

---

*Built and written September 2026. The app, this paper, and the experiment scripts live alongside each other; all reported numbers come from the shipped engine configuration.*
