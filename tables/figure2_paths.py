"""Figure 2: period parameters of selected variables in the random parameters models reported in Table 3 (adaptive fused,
partially constrained) and in the separate period models (fully unconstrained, with 95% intervals)."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import rp_fit, decision, RESULTS, SUBSETS, OUT

T_ = {s: SUBSETS[s]['T'] for s in SUBSETS}
def periods(name, T):
    m = re.search(r'\[(\d{4})(?:-(\d{2,4}))?\]$', name)
    if not m: return list(range(T))
    a = int(m.group(1)); b = m.group(2); b = a if b is None else (int(b) if len(b) == 4 else 2000 + int(b))
    yrs = range(a, b + 1)
    return sorted({(y - 2016) // 2 for y in yrs}) if T == 4 else [y - 2016 for y in yrs]
def base(n): return re.sub(r' \[[^\]]*\]$', '', n.replace('mean:', ''))
def model_paths(sub, app):
    T = T_[sub]; F = rp_fit(sub, decision(sub)['choice'][app])
    P = {}
    for n, b, s in zip(F['names'], F['th'], F['se']):
        if n.startswith(('SD:', 'HM:', 'HV:')): continue
        v = base(n); P.setdefault(v, [(0.0, None)] * T)
        for q in periods(n, T): P[v][q] = (float(b), float(s))
    return P
def uncon_paths(sub):
    T = T_[sub]; P = {}
    for q in range(T):
        e = pd.read_csv(RESULTS / 'rp' / sub / f'uncon_p{q}_estimates.csv')
        for n, b, s in zip(e.parameter, e.estimate, e.se):
            if n.startswith(('SD:', 'HM:', 'HV:')): continue
            P.setdefault(base(n), [None] * T)[q] = (float(b), float(s))
    return P

SEL = {'S1': [('out_of_control', 'Loss of vehicle control'), ('drunk_driving', 'Drunk driving'),
               ('main_carriageway', 'Main carriageway'), ('dark_no_lighting', 'Dark, no street lighting')],
       'S2': [('Northern', 'Northern region'), ('Southern', 'Southern region'), ('frontage_road', 'Frontage road'),
              ('rear_end_same_lane', 'Rear-end, same lane')],
       'S3': [('falling_asleep', 'Driver fell asleep'), ('speeding', 'Speeding'), ('off_right', 'Ran off road to the right'),
              ('dark_no_lighting', 'Dark, no street lighting')],
       'S4': [('barrier_median', 'Barrier median'), ('sideswipe_parallel', 'Sideswipe'),
              ('car_>1', 'Two or more passenger cars'), ('dark_no_lighting', 'Dark, no street lighting')]}
ROW = {'S1': 'S1: single motorcycle', 'S2': 'S2: motorcycle + other vehicle', 'S3': 'S3: single vehicle',
       'S4': 'S4: multi-vehicle'}
C_FU, C_PC, C_UN, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#52514e', '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelcolor': INK, 'text.color': INK})
fig, axes = plt.subplots(4, 4, figsize=(7.2, 8.6))
export = []
for r, sub in enumerate(['S1', 'S2', 'S3', 'S4']):
    T = T_[sub]; Fu, Pc, Un = model_paths(sub, 'fused'), model_paths(sub, 'pc'), uncon_paths(sub)
    labs = ['16–17', '18–19', '20–21', '22–23'] if T == 4 else [f"'{y % 100:02d}" for y in range(2016, 2024)]
    for c, (v, title) in enumerate(SEL[sub]):
        ax = axes[r, c]; x = np.arange(T)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        for name, P, col, ls, lw in [('PC', Pc, C_PC, (0, (3, 1.6)), 1.4), ('Fused', Fu, C_FU, '-', 2.0)]:
            yv = np.array([b for b, _ in P.get(v, [(0.0, None)] * T)])
            ax.step(np.r_[x - 0.5, T - 0.5], np.r_[yv, yv[-1]], where='post', color=col, ls=ls, lw=lw, zorder=3 if name == 'Fused' else 2)
            export += [dict(subset=sub, variable=v, model=name, period=labs[p], estimate=yv[p]) for p in range(T)]
        u = Un.get(v, [None] * T)
        for p in range(T):
            if u[p] is None:
                ax.plot(p, 0, 'o', ms=3.6, mfc='white', mec=C_UN, mew=0.8, zorder=4)
                export.append(dict(subset=sub, variable=v, model='Unconstrained', period=labs[p], estimate=0.0, note='not in model'))
            else:
                b, s = u[p]; ax.errorbar(p, b, yerr=1.96 * s, fmt='o', ms=3.6, color=C_UN, elinewidth=0.8, capsize=1.6, zorder=4)
                export.append(dict(subset=sub, variable=v, model='Unconstrained', period=labs[p], estimate=b, se=s))
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.2 if T == 8 else 6.4); ax.set_xlim(-0.6, T - 0.4)
        ax.set_title(title, fontsize=7.2, color=INK, pad=3)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color=GRID, lw=0.4, alpha=0.6); ax.set_axisbelow(True)
        if c == 0: ax.set_ylabel(ROW[sub], fontsize=7.2, fontweight='bold')
leg = [Line2D([], [], color=C_FU, lw=2.0, label='Adaptive fused (90% rule)'),
       Line2D([], [], color=C_PC, lw=1.4, ls=(0, (3, 1.6)), label='Partially constrained (LR tests, 90%)'),
       Line2D([], [], color=C_UN, marker='o', ls='', ms=3.6, label='Fully unconstrained (95% CI)'),
       Line2D([], [], color=C_UN, marker='o', mfc='white', ls='', ms=3.6, label='Not in that period\'s model')]
fig.legend(handles=leg, loc='upper center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.998))
fig.tight_layout(rect=(0, 0, 1, 0.962), h_pad=1.2, w_pad=0.8)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'Figure_2_parameter_paths.{ext}', dpi=400 if ext == 'png' else None)
pd.DataFrame(export).to_csv(OUT / 'Figure_2_parameter_paths_data.csv', index=False)
print('saved', OUT / 'Figure_2_parameter_paths.pdf')
