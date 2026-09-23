#!/usr/bin/env python3
"""Generate paper figures from experiment JSON files."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/tmp/nnexp'
OUT = '/home/hatch/workspace/mini-society-paper/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

# --- Figure 1: robustness bar chart (Reality vs No honor, 5 seeds) ---
rob = json.load(open(f'{D}/robust.json'))
fig, ax = plt.subplots(figsize=(7, 3.6))
seeds = [str(r['seed']) for r in rob['reality']]
x = np.arange(len(seeds)); w = 0.35
rc = [r['coop'] * 100 for r in rob['reality']]
nc = [r['coop'] * 100 for r in rob['nohonor']]
ax.bar(x - w/2, rc, w, label='Reality (reflex on)', color='#16a34a')
ax.bar(x + w/2, nc, w, label='No honor (reflex off)', color='#dc2626')
ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in seeds])
ax.set_ylabel('Cooperation rate, last 50 rounds (%)')
ax.set_title('The retaliation norm is doing the work: 5 seeds, 1,000 rounds each')
ax.legend(frameon=False)
ax.set_ylim(0, 110)
for i, v in enumerate(rc): ax.text(i - w/2, v + 1.5, f'{v:.1f}%', ha='center', fontsize=8)
for i, v in enumerate(nc): ax.text(i + w/2, v + 1.5, f'{v:.1f}%', ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1-robustness.png', dpi=150); plt.close(fig)

# --- Figure 2: internalization traces ---
intern = json.load(open(f'{D}/internalize.json'))
fig, ax = plt.subplots(figsize=(7, 3.6))
xs = [(i + 1) * 50 for i in range(10)]
for run in intern['runs']:
    ax.plot(xs, [c * 100 for c in run['trace']], marker='o', ms=3,
            label=f"seed {run['seed']} (was {run['beforeCoop']*100:.0f}% at lift)")
ax.axhline(100, ls='--', c='gray', lw=1)
ax.set_xlabel('Rounds after the norm is lifted')
ax.set_ylabel('Cooperation rate (%)')
ax.set_title('Lifting the retaliation norm after convergence: cooperation collapses')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(0, 105)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2-internalize.png', dpi=150); plt.close(fig)

# --- Figure 3: grudge permanent vs decaying ---
gr = json.load(open(f'{D}/grudge.json'))
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
for ax, key, title in zip(axes, ['coop', 'wealth'],
                          ['Cooperation rate (last 50 rounds)', 'Mean wealth per agent']):
    vals = {k: [r[key] for r in gr[k]] for k in ['permanent', 'decay12']}
    x = np.arange(3); w = 0.35
    if key == 'coop': vals = {k: [v * 100 for v in vs] for k, vs in vals.items()}
    ax.bar(x - w/2, vals['permanent'], w, label='Permanent grudge', color='#7c3aed')
    ax.bar(x + w/2, vals['decay12'], w, label='Decaying grudge (12 games)', color='#f59e0b')
    ax.set_xticks(x); ax.set_xticklabels([f"seed {r['seed']}" for r in gr['permanent']])
    ax.set_title(title, fontsize=10)
    if key == 'coop': ax.set_ylim(0, 110)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Permanent vs. forgiving grudges (Reality preset, 1,000 rounds)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3-grudge.png', dpi=150); plt.close(fig)

# --- Figure 4: scaffold critical period ---
sc = json.load(open(f'{D}/scaffold.json'))
fig, ax = plt.subplots(figsize=(7, 3.6))
groups = {}
for r in sc['runs']:
    groups.setdefault(r['liftAt'], []).append(r)
xs = sorted(groups)
before = [np.mean([r['before']['coop'] for r in groups[x]]) * 100 for x in xs]
after = [np.mean([r['after']['coop'] for r in groups[x]]) * 100 for x in xs]
x = np.arange(len(xs)); w = 0.35
ax.bar(x - w/2, before, w, label='At lift (reflex on)', color='#0ea5e9')
ax.bar(x + w/2, after, w, label='500 rounds after lift (reflex off)', color='#f97316')
ax.set_xticks(x); ax.set_xticklabels([f'{n} rounds' for n in xs])
ax.set_ylabel('Cooperation rate (%)')
ax.set_title('Critical period: the norm can only retire once exclusion is learned')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(0, 110)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4-scaffold.png', dpi=150); plt.close(fig)

# --- Figure 5: invasion — ostracism, not punishment ---
iv = json.load(open(f'{D}/invasion2.json'))
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
seeds = [str(r['seed']) for r in iv['runs']]
x = np.arange(len(seeds)); w = 0.35
wi = [r['wealthGainInvader'] for r in iv['runs']]
wr = [r['wealthGainResident'] for r in iv['runs']]
axes[0].bar(x - w/2, wi, w, label='Invaders (always defect)', color='#dc2626')
axes[0].bar(x + w/2, wr, w, label='Residents', color='#16a34a')
axes[0].set_xticks(x); axes[0].set_xticklabels([f'seed {s}' for s in seeds])
axes[0].set_title('Wealth gained during invasion', fontsize=10)
axes[0].legend(frameon=False, fontsize=8)
share = [r['invaderGameShare'] * 100 for r in iv['runs']]
axes[1].bar(x, share, 0.6, color='#7c3aed')
axes[1].axhline(9.8, ls='--', c='gray', lw=1)
axes[1].text(2.1, 10.3, 'random-pairing baseline ~9.8%', fontsize=7, color='gray')
axes[1].set_xticks(x); axes[1].set_xticklabels([f'seed {s}' for s in seeds])
axes[1].set_title('Invader share of games (%)', fontsize=10)
fig.suptitle('Invaders are not punished — they are shunned (residents cooperate ~100% vs invaders in-game)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig5-invasion.png', dpi=150); plt.close(fig)

print('figures written to', OUT)
