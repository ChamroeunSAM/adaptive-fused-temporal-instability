"""Tables C1-C4: fully unconstrained random parameters binary logit models with heterogeneity in means and variances,
estimated separately for each period (pruned models), one table per subset, 'parameter (t-statistic)' per period."""
import re, json, numpy as np, pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from _common import load_subset, rp_res, styles, save, fm, desc, CATS, ORDER, REFTXT, RESULTS

order = ORDER
YRS = [str(y) for y in range(2016, 2024)]
SUBS = {'S1': ('C1', 'single-motorcycle crashes', ['2016–2017', '2018–2019', '2020–2021', '2022–2023']),
        'S2': ('C2', 'crashes involving a motorcycle and another vehicle type', YRS),
        'S3': ('C3', 'single-vehicle crashes without a motorcycle', YRS),
        'S4': ('C4', 'multi-vehicle crashes without a motorcycle', YRS)}
base = lambda n: re.sub(r' \[[^\]]*\]$', '', n)
cell = lambda b, t: f'{fm(b)} ({fm(t, 2)})'
st = dict(c=ParagraphStyle('c', fontName='Helvetica', fontSize=6.7, leading=8.0),
          sd=ParagraphStyle('d', fontName='Helvetica-Oblique', fontSize=6.5, leading=7.8, leftIndent=7),
          h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=6.9, leading=8.2, alignment=1),
          sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=6.9, leading=8.3),
          sub=ParagraphStyle('u', fontName='Helvetica-Oblique', fontSize=6.7, leading=8.1),
          t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.0, leading=11.2),
          v=ParagraphStyle('v', fontName='Helvetica', fontSize=6.7, leading=8.0, alignment=1),
          n=ParagraphStyle('n', fontName='Helvetica', fontSize=6.7, leading=8.2))
P = lambda s, k='c': Paragraph(s, st[k])

for SUB, (tno, subtxt, LAB) in SUBS.items():
    T = len(LAB); X, y, t, names, _ = load_subset(SUB); REF = REFTXT[SUB]
    E = [pd.read_csv(RESULTS / 'rp' / SUB / f'uncon_p{p}_estimates.csv') for p in range(T)]
    R = [rp_res(SUB, f'uncon_p{p}') for p in range(T)]
    # per period: {key: (estimate, t)}; keys: ('b', var) main/mean, ('sd', var), ('hm', var, z), ('hv', var, z)
    V = []
    for e in E:
        d = {}
        for n, b, tv in zip(e.parameter, e.estimate, e.t):
            if n.startswith('SD:'): d[('sd', base(n[3:]))] = (abs(b), abs(tv))
            elif n.startswith(('HM:', 'HV:')):
                r, z = n[3:].split('|'); d[(n[:2].lower(), base(r), base(z))] = (b, tv)
            else: d[('b', base(n.replace('mean:', '')))] = (b, tv)
        V.append(d)
    keys = set().union(*V)
    for p in range(T): assert len(V[p]) == len(E[p]), (SUB, p)
    row = lambda k: [P(cell(*V[p][k]), 'v') if k in V[p] else '' for p in range(T)]
    data = [[P('Variable description', 'h')] + [P(l, 'h') for l in LAB]]; sty = []
    def sec(text, k='sec'):
        data.append([P(text, k)] + [''] * T); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    sec('Constant [serious injury / fatal]'); data.append([P('Constant')] + row(('b', 'constant')))
    mains = sorted({k[1] for k in keys if k[0] == 'b' and k[1] != 'constant'}, key=lambda v: (order.get(v, 999), v))
    for c in CATS + ['Other']:
        vs = [v for v in mains if desc(v)[0] == c]
        if not vs: continue
        sec(c, 'sub')
        for v in vs:
            data.append([P(f'{desc(v)[1]} (1 if yes, 0 otherwise)')] + row(('b', v)))
            if any(('sd', v) in d for d in V):
                data.append([P('(Standard deviation of parameter distribution)', 'sd')] + row(('sd', v)))
    for kind, title in [('hm', 'Heterogeneity in the means of random parameters'), ('hv', 'Heterogeneity in the variances of random parameters')]:
        ks = sorted({k for k in keys if k[0] == kind}, key=lambda k: (order.get(k[1], 999), order.get(k[2], 999)))
        if not ks: continue
        sec(title)
        for k in ks: data.append([P(f'{desc(k[1])[1]}: {desc(k[2])[1]}')] + row(k))
    sec('Model statistics')
    stats = []
    for p in range(T):
        yy = y[t == p]; N = len(yy); assert N == R[p]['N'], (SUB, p, N, R[p]['N'])
        q = yy.mean(); LL0 = N * np.log(0.5); LLC = N * (q * np.log(q) + (1 - q) * np.log(1 - q)); LL = R[p]['LL']
        stats.append(dict(N=f'{N:,}', k=str(R[p]['k']), LL0=fm(LL0, 2), LLC=fm(LLC, 2), LL=fm(LL, 2), rho=f'{1 - LL / LL0:.3f}',
                          rp=f"{len(R[p]['random'])} / {len(R[p]['HM'])} / {len(R[p]['HV'])}"))
    for lab, key in [('Number of observations', 'N'), ('Number of estimated parameters', 'k'),
                     ('Random parameters / heterogeneity in means / in variances', 'rp'),
                     ('Log-likelihood at zero, LL(0)', 'LL0'), ('Log-likelihood with constant only, LL(C)', 'LLC'),
                     ('Log-likelihood at convergence, LL(β)', 'LL'), ('McFadden pseudo ρ<super>2</super> = 1 − LL(β)/LL(0)', 'rho')]:
        data.append([P(lab)] + [P(s[key], 'v') for s in stats])
    W = landscape(A4)[0] - 20 * mm; wd = 0.27 if T == 8 else 0.40
    tb = Table(data, colWidths=[W * wd] + [W * (1 - wd) / T] * T, repeatRows=1)
    tb.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (1, 1), (-1, -1), 6.7),
                            ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                            ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                            ('TOPPADDING', (0, 0), (-1, -1), 0.8), ('BOTTOMPADDING', (0, 0), (-1, -1), 0.8),
                            ('LEFTPADDING', (1, 0), (-1, -1), 2), ('RIGHTPADDING', (1, 0), (-1, -1), 2)] + sty))
    per = 'two-year period' if SUB == 'S1' else 'year'
    title = P(f'<b>Table {tno}.</b> Fully unconstrained random parameters binary logit models with heterogeneity in means and variances '
              f'of crash injury severity (serious injury/fatal vs. PDO/minor injury), estimated separately for each {per}: {subtxt}, '
              'Thailand, 2016–2023. Estimated parameter (<i>t</i>-statistic).', 't')
    notes = P('<b>Notes.</b> A positive parameter increases the probability of a serious injury/fatal outcome. Reference categories: '
              f'{REF}. Each column is a separate model for one {per}. Each model allows at most two random parameters, two '
              'heterogeneity-in-means terms and one heterogeneity-in-variance term, searched with 100 Halton draws at the 90% level and '
              'confirmed with 2,000 draws; non-significant parameters were then removed, so all parameters except constants and the '
              'means of random parameters are statistically significant at the 90% level. A blank cell means the variable is not in '
              f'that {per}\'s model. Rows with a standard deviation give the mean of the random parameter in the {per}s where it is '
              f'random and a fixed parameter in the other {per}s. A variable used in a heterogeneity function enters only the random '
              f'parameter shown in that {per} and is removed from the main utility. Random parameters normally distributed; 2,000 '
              'Halton draws.', 'n')
    save(f'Table_{tno}_{SUB}_unconstrained', landscape(A4), [title, Spacer(1, 5), tb, Spacer(1, 5), notes],
         f'{SUB} fully unconstrained RPBLHMV', margins=(10, 10, 9, 9))
