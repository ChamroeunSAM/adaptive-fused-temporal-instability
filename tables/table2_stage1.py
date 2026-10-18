"""Table 2: temporal approaches in the fixed-parameter binary logit (Stage 1)."""
import pandas as pd
from _common import stage1, styles, save, neg, SUBS, SUBLAB, OUT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

MOD = [('uncon', 'Fully unconstrained (separate-period logits)'), ('pc', 'Partially constrained (LR tests, 90%)'),
       ('spline', 'Spline indicator (90%)'), ('fused', 'Adaptive fused (90% rule)')]
st = styles(7.6); st.update(sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=7.8, leading=9.4),
          t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.4, leading=11.6),
          n=ParagraphStyle('n', fontName='Helvetica', fontSize=7.0, leading=8.6), c=ParagraphStyle('c', fontName='Helvetica', fontSize=7.6, leading=9.2),
          h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=7.6, leading=9.2, alignment=1))
P = lambda x, k='c': Paragraph(x, st[k])
hdr = ['Temporal approach', 'Number of parameters', 'Variables with temporally varying parameters',
       'Variables restricted to a proportional trend', 'LL(β)', 'AIC', 'BIC',
       'LR test against the fully unconstrained model, χ<super>2</super> (df)', 'Confidence of rejection']
data = [[P(h, 'h') for h in hdr]]; sty = []; rows = []
for sub in SUBS:
    d = stage1(sub)
    data.append([P(f'{SUBLAB[sub]}; N = {d["N"]:,}; {d["K"]} candidate variables (including the constant)', 'sec')] + [''] * 8)
    sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    best = min(d[a]['BIC'] for a, _ in MOD if a == 'uncon' or d[a]['conf'] < 0.90)
    for a, name in MOD:
        r = d[a]
        if r['BIC'] == best: sty.append(('FONTNAME', (6, len(data)), (6, len(data)), 'Helvetica-Bold'))
        lr = '–' if a == 'uncon' else neg(f"{r['LR']:.1f} ({r['df']})"); cf = '–' if a == 'uncon' else f"{r['conf']:.3f}"
        data.append([P(name), str(r['k']), f"{r['varying']} of {d['K']}" if a != 'uncon' else 'all',
                     str(r['proportional']) if a == 'spline' else '–', neg(f"{r['LL']:,.2f}"), f"{r['AIC']:,.1f}", f"{r['BIC']:,.1f}", lr, cf])
        rows.append(dict(subset=sub, model=name, k=r['k'], varying=r['varying'], LL=r['LL'], AIC=r['AIC'], BIC=r['BIC'],
                         LR=r.get('LR'), df=r.get('df'), conf=r.get('conf'), proportional=r.get('proportional'),
                         lam=r.get('lam')))
W = landscape(A4)[0] - 24 * mm
tb = Table(data, colWidths=[W * x for x in (0.23, 0.08, 0.12, 0.11, 0.09, 0.08, 0.08, 0.13, 0.08)], repeatRows=1)
tb.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 1), (-1, -1), 'CENTER'), ('FONTSIZE', (0, 0), (-1, -1), 7.6),
                        ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                        ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                        ('TOPPADDING', (0, 0), (-1, -1), 1.6), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6)] + sty))
title = P('<b>Table 2.</b> Comparison of temporal approaches in the fixed-parameter binary logit model of crash injury severity '
          '(serious injury/fatal vs. PDO/minor injury), Thailand, 2016–2023.', 't')
notes = P('<b>Notes.</b> Lowest BIC within each subset among models not rejected at the 90% level in bold. Fully unconstrained: a '
          'separate parameter for every variable in every period. Partially constrained: adjacent periods merged when a likelihood '
          'ratio test does not reject equal parameters at the 90% level. Spline indicator (Shabab et al., 2024): each variable enters '
          'through spline terms, the base term giving a parameter proportional to the period index; the other terms removed by '
          'backward elimination at the 90% level. Adaptive fused: sparsest penalty whose structure is not rejected at the 90% level '
          'against the fully unconstrained model (LR < χ<super>2</super><sub>0.90</sub>(df)). Variables with temporally varying '
          'parameters count variables (including the constant) whose parameter differs between at least two periods; for the '
          'spline every variable varies over time. LR test: χ<super>2</super> = −2[LL(β)<sub>model</sub> − LL(β)<sub>fully '
          'unconstrained</sub>], df = difference in the number of parameters; confidence of rejection below 0.90 means that the '
          'model is not rejected at the 90% level.', 'n')
save('Table_2_stage1_fixed_parameter', landscape(A4), [title, Spacer(1, 6), tb, Spacer(1, 6), notes], 'Stage-1 comparison')
pd.DataFrame(rows).to_csv(OUT / 'Table_2_stage1_fixed_parameter_values.csv', index=False)
