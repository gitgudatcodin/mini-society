#!/usr/bin/env python3
"""Render paper-draft.md + figures into a styled HTML file, then PDF via chromium."""
import base64, markdown, os, re

BASE = '/home/hatch/workspace/mini-society-paper'
FIGS = {
    'fig1-robustness.png': 'Figure 1. The retaliation norm is doing the work: cooperation rate (last 50 of 1,000 rounds) for the Reality preset (reflex on) vs. No honor (reflex off), across five random seeds.',
    'fig2-internalize.png': 'Figure 2. Lifting the norm after 1,000 rounds of convergence: cooperation holds at ~100% for 500 further rounds with no reflex, on all three seeds.',
    'fig3-grudge.png': 'Figure 3. Permanent vs. decaying grudges under the Reality preset: no meaningful difference in cooperation or wealth — a null result.',
    'fig4-scaffold.png': 'Figure 4. Critical period: lifting the norm after 200 rounds → cooperation decays to ~72%; after 500 → ~97%; after 1,000 → fully self-sustaining.',
    'fig5-invasion.png': 'Figure 5. Invasion test: 5% always-defect invaders are not punished in-game (residents cooperate ~100% with them) but are shunned — they play few games and gain ~25% less wealth.',
}
# figure placement: insert after the section that first discusses the experiment
PLACEMENT = {
    'fig1-robustness.png': '### 3.4',
    'fig2-internalize.png': '### 3.5b',
    'fig4-scaffold.png': '### 3.5c',
    'fig5-invasion.png': '### 3.6',
    'fig3-grudge.png': '## 4. Findings',
}

md = open(f'{BASE}/paper-draft.md').read()
for fname, anchor in PLACEMENT.items():
    data = base64.b64encode(open(f'{BASE}/figures/{fname}', 'rb').read()).decode()
    fig_html = (f'\n\n<figure><img src="data:image/png;base64,{data}" alt="{fname}">'
                f'<figcaption>{FIGS[fname]}</figcaption></figure>\n\n')
    md = md.replace(anchor, fig_html + anchor, 1)

html_body = markdown.markdown(md, extensions=['tables', 'fenced_code'])

CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:760px;margin:0 auto;padding:48px 28px;color:#1a1a1a;line-height:1.65;font-size:16.5px}
h1{font-size:30px;line-height:1.25;margin-bottom:6px}
h2{font-size:22px;margin-top:44px;border-bottom:1px solid #ddd;padding-bottom:6px}
h3{font-size:18px;margin-top:32px}
table{border-collapse:collapse;margin:18px 0;width:100%;font-size:14px}
th,td{border:1px solid #ccc;padding:7px 10px;text-align:left}
th{background:#f4f4f4}
figure{margin:28px 0;text-align:center}
figure img{max-width:100%;border:1px solid #e5e5e5;border-radius:6px}
figcaption{font-size:13px;color:#555;margin-top:8px;font-style:italic;max-width:640px;margin-left:auto;margin-right:auto}
code{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:14px}
hr{margin:40px 0;border:none;border-top:1px solid #ddd}
@media print{body{padding:0;font-size:12pt}h2{page-break-after:avoid}figure{page-break-inside:avoid}}
"""

title = "Can Selfish Neural Agents Learn to Cooperate?"
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>{html_body}
<hr><p style="font-size:13px;color:#666">Built and written September 2026. All reported numbers come from the shipped engine
configuration (<code>mini-society-nn.html</code>), verified headless under Node.js; experiment scripts and figures
are archived with this paper.</p></body></html>"""
open(f'{BASE}/paper.html', 'w').write(html)
print('wrote paper.html', len(html), 'bytes')
