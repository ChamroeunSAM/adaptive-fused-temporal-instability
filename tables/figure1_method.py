"""Figure 1: the adaptive fused formulation (schematic). (a) level and step variables for an indicator over four periods;
(b) how the steps build the period parameters before and after the penalty. Illustrative values, no data."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import OUT
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
C_SEL, C_SMALL, C_LARGE, C_SPL, C_UN = '#2a78d6', '#8fb8ea', '#8c8b86', '#eb6834', '#3b3a37'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'text.color': INK, 'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(7.2, 3.4))
axa = fig.add_axes([0.01, 0.06, 0.47, 0.82]); axb = fig.add_axes([0.585, 0.24, 0.40, 0.64])

# ---- (a) design table
axa.set_axis_off(); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
cols = [0.0, 0.17, 0.26, 0.37, 0.50, 0.63, 0.77]
hdr = ['Period', '$t$', '$x$', r'$x\,D_2$', r'$x\,D_3$', r'$x\,D_4$', r'Parameter $\beta(t)$']
rows = [('2016–17', 1, [1, 0, 0, 0], r'$\alpha$'), ('2018–19', 2, [1, 1, 0, 0], r'$\alpha+\gamma_2$'),
        ('2020–21', 3, [1, 1, 1, 0], r'$\alpha+\gamma_2+\gamma_3$'), ('2022–23', 4, [1, 1, 1, 1], r'$\alpha+\gamma_2+\gamma_3+\gamma_4$')]
y0, dy = 0.86, 0.105
fig.text(0.01, 0.975, '(a) Level and step variables for an indicator $x = 1$', fontsize=7.6, fontweight='bold', va='top')
for c, h in zip(cols, hdr): axa.text(c, y0, h, fontweight='bold', va='center', ha='left')
axa.plot([0, 1], [y0 + dy * 0.5] * 2, color=INK, lw=0.8); axa.plot([0, 1], [y0 - dy * 0.5] * 2, color=INK, lw=0.5)
for i, (p, tt, v, par) in enumerate(rows):
    yy = y0 - dy * (i + 1)
    axa.text(cols[0], yy, p, va='center'); axa.text(cols[1], yy, str(tt), va='center')
    for c, val in zip(cols[2:6], v): axa.text(c + 0.02, yy, str(val), va='center', color=INK if val else MUTED)
    axa.text(cols[6], yy, par, va='center')
axa.plot([0, 1], [y0 - dy * 4.5] * 2, color=INK, lw=0.8)
axa.text(0.0, y0 - dy * 5.35, r'$D_s = 1[t \geq s]$: step indicator, so $\gamma_s = \beta(s) - \beta(s-1)$, the change between'
         '\nadjacent periods. The adaptive penalty shrinks small steps to exactly zero;'
         '\n' r'with $\gamma_3 = \gamma_4 = 0$, periods 2–4 share one parameter (panel b).', va='top', color=MUTED, fontsize=6.9, linespacing=1.45)

# ---- (b) schematic: how the steps of (a) build beta(t), before and after the penalty
T = 4; x = np.arange(1, T + 1)
un = np.array([-0.60, 0.20, 0.33, 0.50])          # unpenalised: every step estimated
fu = np.array([-0.56, 0.34, 0.34, 0.34])          # adaptive fused: gamma_3 = gamma_4 = 0
def step(P, **kw): axb.step(np.r_[x - 0.5, T + 0.5], np.r_[P, P[-1]], where='post', **kw)
step(un, color=C_LARGE, lw=1.3, ls=(0, (4, 1.8)), zorder=2)
step(fu, color=C_SEL, lw=2.4, zorder=3)
ar = dict(arrowstyle='-|>', mutation_scale=7, lw=0.9, shrinkA=0, shrinkB=0)
# alpha (level)
axb.annotate('', xy=(0.62, un[0]), xytext=(0.62, 0), arrowprops=dict(ar, color=C_UN))
axb.text(0.68, un[0] / 2, r'$\alpha$', va='center', fontsize=8)
# unpenalised steps
for s_, (a, b) in enumerate(zip(un[:-1], un[1:]), start=2):
    xx = s_ - 0.5 + 0.06
    axb.annotate('', xy=(xx, b), xytext=(xx, a), arrowprops=dict(ar, color=C_LARGE))
    if s_ > 2: axb.text(xx + 0.06, (a + b) / 2, rf'$\gamma_{s_}$', va='center', color=MUTED, fontsize=7.5)
# fused step
axb.annotate('', xy=(1.5 - 0.06, fu[1]), xytext=(1.5 - 0.06, fu[0]), arrowprops=dict(ar, color=C_SEL, lw=1.3))
axb.text(1.5 - 0.13, (fu[0] + fu[1]) / 2, r'$\gamma_2$', va='center', ha='right', color=C_SEL, fontsize=8.5, fontweight='bold')
axb.text(3.3, -0.06, r'small steps: large $w_{ks}$,' '\n' r'shrunk to $\gamma_3=\gamma_4=0$', ha='center', va='top', fontsize=6.8, color=MUTED)
axb.text(1.62, 0.39, r'periods 2–4 share $\alpha+\gamma_2$', ha='left', va='bottom', fontsize=6.8, color=C_SEL)
axb.set_xticks(x); axb.set_xticklabels([f'$t$ = {v}' for v in x]); axb.set_xlim(0.45, T + 0.55); axb.set_ylim(-0.8, 0.75)
axb.set_yticks([]); axb.axhline(0, color=GRID, lw=0.8, zorder=0)
axb.set_ylabel(r'Parameter $\beta(t)$', color=INK)
for sp in ['top', 'right']: axb.spines[sp].set_visible(False)
fig.text(0.535, 0.975, r'(b) The steps of (a) before and after the penalty', fontsize=7.6, fontweight='bold', va='top')
leg = [Line2D([], [], color=C_LARGE, lw=1.3, ls=(0, (4, 1.8)), label=r'Unpenalised ($\lambda = 0$): every step estimated'),
       Line2D([], [], color=C_SEL, lw=2.4, label=r'Adaptive fused (selected $\lambda$): one break')]
fig.legend(handles=leg, loc='lower center', ncol=1, frameon=False, fontsize=6.8, bbox_to_anchor=(0.765, 0.0), handlelength=2.4)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'Figure_1_adaptive_fused_illustration.{ext}', dpi=400 if ext == 'png' else None)
print('saved')
