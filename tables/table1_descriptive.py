"""Table 1: descriptive statistics of the four subsets (crashes and serious/fatal share by period; indicator means)."""
from _common import load_subset, styles, save, desc, CATS, ORDER, SUBS
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

LAB = {'S1': 'Single motorcycle', 'S2': 'Motorcycle and another vehicle', 'S3': 'Single vehicle, no motorcycle',
       'S4': 'Multi-vehicle, no motorcycle'}
D = {}
for s in SUBS:
    X, y, t, names, T = load_subset(s); D[s] = dict(X=X, y=y, t=t, names=names)
st = styles(7.2); P = lambda s, k='c': Paragraph(s, st[k])
data = [[P('Variable', 'h')] + [P(f'{s}: {LAB[s]}', 'h') for s in SUBS]]; sty = []
def sec(text, k='sec'):
    data.append([P(text, k)] + [''] * 4); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
cell = lambda n, p: P(f'{n:,} ({100 * p:.1f}%)', 'v')
sec('Number of crashes (percentage serious injury/fatal)')
r0 = len(data)
for yr in range(8):
    row = [P(str(2016 + yr))]
    for s in SUBS:
        d = D[s]
        if s == 'S1':
            if yr % 2 == 0:
                m = d['t'] == yr // 2; row.append(cell(int(m.sum()), d['y'][m].mean()))
                sty.append(('SPAN', (1, r0 + yr), (1, r0 + yr + 1)))
            else: row.append('')
        else:
            m = d['t'] == yr; row.append(cell(int(m.sum()), d['y'][m].mean()))
    data.append(row)
data.append([P('<b>All years, 2016–2023</b>')] + [P(f'<b>{len(D[s]["y"]):,} ({100 * D[s]["y"].mean():.1f}%)</b>', 'v') for s in SUBS])
sec('Explanatory variables, mean (standard deviation)')
allv = sorted({n for s in D for n in D[s]['names'] if n != 'constant'}, key=lambda v: (ORDER.get(v, 999), v))
for c in CATS + ['Other']:
    vs = [v for v in allv if desc(v)[0] == c]
    if not vs: continue
    sec(c, 'sub')
    for v in vs:
        row = [P(desc(v)[1])]
        for s in SUBS:
            d = D[s]
            if v in d['names']:
                xv = d['X'][:, d['names'].index(v)]; row.append(P(f"{xv.mean():.3f} ({xv.std(ddof=1):.3f})", 'v'))
            else: row.append(P('–', 'v'))
        data.append(row)
W = A4[0] - 24 * mm
tb = Table(data, colWidths=[W * 0.40] + [W * 0.15] * 4, repeatRows=1)
tb.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                        ('LINEABOVE', (0, r0 + 8), (-1, r0 + 8), 0.4, colors.black),
                        ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                        ('TOPPADDING', (0, 0), (-1, -1), 0.9), ('BOTTOMPADDING', (0, 0), (-1, -1), 0.9)] + sty))
title = P('<b>Table 1.</b> Descriptive statistics of the four crash subsets, Thailand, 2016–2023.', 't')
notes = P('<b>Notes.</b> Outcome: serious injury/fatal (1) vs. property damage only/minor injury (0). S1 is analysed in two-year '
          'periods (2016–2017, 2018–2019, 2020–2021, 2022–2023) because of its smaller sample; S2–S4 in yearly periods. All '
          'explanatory variables are indicators (1 if yes, 0 otherwise), so the mean is the share of crashes with the indicator; '
          'standard deviation is the sample standard deviation. – : not a candidate variable in that subset. Reference '
          'categories: Central region, daytime, no median (and one motorcycle in S1, one pickup truck in S3).', 'n')
save('Table_1_descriptive_statistics', A4, [title, Spacer(1, 5), tb, Spacer(1, 6), notes], 'Descriptive statistics')
