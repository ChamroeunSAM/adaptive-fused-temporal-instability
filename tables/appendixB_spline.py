"""Tables B1-B4: spline-indicator fixed-parameter binary logit (Shabab et al., 2024) at the 90% level, one table per subset.
Re-estimates the Stage-1 spline model, checks k and LL against results/stage1, and reports the implied period parameters
beta_k(t) with delta-method t-statistics."""
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from _common import load_subset, stage1, styles, save, fm, desc, CATS, ORDER, REFTXT, SUBSETS
from temporal import f_spline
from stage1 import spline_model

order = ORDER
YRS = [str(y) for y in range(2016, 2024)]
SUBS = [('S1', 'B1', 'single-motorcycle crashes', ['2016–2017', '2018–2019', '2020–2021', '2022–2023']),
        ('S2', 'B2', 'crashes involving a motorcycle and another vehicle type', YRS),
        ('S3', 'B3', 'single-vehicle crashes without a motorcycle', YRS),
        ('S4', 'B4', 'multi-vehicle crashes without a motorcycle', YRS)]
st = dict(c=ParagraphStyle('c', fontName='Helvetica', fontSize=6.7, leading=8.0),
          sd=ParagraphStyle('d', fontName='Helvetica-Oblique', fontSize=6.5, leading=7.8, leftIndent=7),
          h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=6.9, leading=8.2, alignment=1),
          sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=6.9, leading=8.3),
          sub=ParagraphStyle('u', fontName='Helvetica-Oblique', fontSize=6.7, leading=8.1),
          t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.0, leading=11.2),
          v=ParagraphStyle('v', fontName='Helvetica', fontSize=6.7, leading=8.0, alignment=1),
          n=ParagraphStyle('n', fontName='Helvetica', fontSize=6.7, leading=8.2))
P = lambda s, k='c': Paragraph(s, st[k])

for SUB, tno, subtxt, LAB in SUBS:
    X, y, t, names, T = load_subset(SUB); K, N = X.shape[1], len(y); REF = REFTXT[SUB]
    S = spline_model(X, y, t, T); keep, b, cov, ll = S['keep'], S['b'], S['cov'], S['LL']; k_ = S['k']
    ref = stage1(SUB)['spline']
    assert k_ == ref['k'] and abs(ll - ref['LL']) < 1e-4, (SUB, k_, ref['k'], ll, ref['LL'])
    # implied period parameters and delta-method t-statistics
    tt = np.arange(T); pos = 0; beta, tb = np.zeros((K, T)), np.zeros((K, T))
    for k in range(K):
        idx = list(range(pos, pos + len(keep[k]))); pos += len(keep[k])
        G = np.array([f_spline(j)(tt) for j in keep[k]])          # terms x periods
        beta[k] = G.T @ b[idx]; V = G.T @ cov[np.ix_(idx, idx)] @ G; tb[k] = beta[k] / np.sqrt(np.diag(V))
    terms = lambda k: ', '.join(str(j) for j in keep[k])
    rowv = lambda k: [P(terms(k), 'v')] + [P(f'{fm(beta[k, s])} ({fm(tb[k, s], 2)})', 'v') for s in range(T)]
    data = [[P('Variable description', 'h'), P('Spline terms retained (<i>j</i>)', 'h')] + [P(l, 'h') for l in LAB]]; sty = []
    def sec(text, k='sec'):
        data.append([P(text, k)] + [''] * (T + 1)); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    sec('Constant [serious injury / fatal]'); data.append([P('Constant')] + rowv(0))
    for c in CATS + ['Other']:
        ks = sorted([k for k in range(1, K) if desc(names[k])[0] == c], key=lambda k: (order.get(names[k], 999), k))
        if not ks: continue
        sec(c, 'sub')
        for k in ks: data.append([P(f'{desc(names[k])[1]} (1 if yes, 0 otherwise)')] + rowv(k))
    q = y.mean(); LL0 = N * np.log(0.5); LLC = N * (q * np.log(q) + (1 - q) * np.log(1 - q))
    nb = sum(len(kp) == 1 for kp in keep)
    sec('Model statistics')
    for a, v in [('Number of observations', f'{N:,}'), ('Number of estimated spline coefficients', f'{k_}'),
                 ('Variables with the base term only (parameter proportional to the period index)', f'{nb} of {K}'),
                 ('Log-likelihood at zero, LL(0)', fm(LL0, 2)), ('Log-likelihood with constant only, LL(C)', fm(LLC, 2)),
                 ('Log-likelihood at convergence, LL(β)', fm(ll, 2)), ('McFadden pseudo ρ<super>2</super> = 1 − LL(β)/LL(0)', f'{1 - ll / LL0:.3f}'),
                 ('AIC / BIC', f'{-2 * ll + 2 * k_:,.1f} / {-2 * ll + np.log(N) * k_:,.1f}'),
                 ('LR test against the fixed-parameter separate-period models', f"χ<super>2</super> = {ref['LR']:.2f}, df = {ref['df']}, "
                  f"confidence of rejection = {ref['conf']:.3f}")]:
        data.append([P(a), P(v, 'v')] + [''] * T); sty.append(('SPAN', (1, len(data) - 1), (-1, len(data) - 1)))
    W = landscape(A4)[0] - 20 * mm; wd, wt = (0.26, 0.07) if T == 8 else (0.36, 0.10)
    tb_ = Table(data, colWidths=[W * wd, W * wt] + [W * (1 - wd - wt) / T] * T, repeatRows=1)
    tb_.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                             ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                             ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                             ('TOPPADDING', (0, 0), (-1, -1), 0.8), ('BOTTOMPADDING', (0, 0), (-1, -1), 0.8),
                             ('LEFTPADDING', (1, 0), (-1, -1), 2), ('RIGHTPADDING', (1, 0), (-1, -1), 2)] + sty))
    per = 'two-year period' if SUB == 'S1' else 'year'
    title = P(f'<b>Table {tno}.</b> Spline-indicator fixed-parameter binary logit model of crash injury severity (serious injury/fatal vs. '
              f'PDO/minor injury), {subtxt}, Thailand, 2016–2023: implied parameter for each {per}, estimated parameter (<i>t</i>-statistic).', 't')
    notes = P('<b>Notes.</b> A positive parameter increases the probability of a serious injury/fatal outcome. Reference categories: '
              f'{REF}. Following Shabab et al. (2024), each variable enters through spline terms <i>f<sub>j</sub></i>(τ) = max(τ − <i>j</i> + 1, 0), '
              f'<i>j</i> = 1, …, {T}, where τ = 1, …, {T} indexes the {per}s; the parameter in {per} τ is β(τ) = Σ<sub><i>j</i></sub> '
              '<i>b<sub>j</sub> f<sub>j</sub></i>(τ). The base term (<i>j</i> = 1) is always retained; the other terms were removed by '
              'backward elimination at the 90% level (|<i>t</i>| < 1.645, one term per variable per round). A variable with the base term '
              'only has a parameter proportional to the period index. The table reports the implied parameters β(τ); <i>t</i>-statistics '
              'by the delta method from the covariance of the spline coefficients.', 'n')
    save(f'Table_{tno}_{SUB}_spline', landscape(A4), [title, Spacer(1, 5), tb_, Spacer(1, 5), notes], f'{SUB} spline-indicator model',
         margins=(10, 10, 9, 9))
    big = [(names[k], LAB[s], round(beta[k, s], 2)) for k in range(K) for s in range(T) if abs(beta[k, s]) > 5]
    if big: print('  implied parameters larger than 5 in absolute value:', big)
