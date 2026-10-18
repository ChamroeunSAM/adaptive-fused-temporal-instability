"""Table 3: temporal approaches in the random parameters binary logit with heterogeneity in means and variances: fully
unconstrained (separate period models), partially constrained and adaptive fused, each at the stage chosen in Section 2.2."""
import re
import numpy as np
import pandas as pd
from _common import rp_res, rp_fit, decision, styles, save, neg, SUBS, SUBLAB, SUBSETS, STAGE_OF, OUT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer


def n_vary(sub, tag):
    nm = rp_fit(sub, tag)['names']
    return str(len({re.sub(r' \[.*', '', n.replace('mean:', '')) for n in nm if '[' in n and not n.startswith(('SD:', 'HM:', 'HV:'))}))


rows = []
for sub in SUBS:
    T = SUBSETS[sub]['T']; dec = decision(sub)
    U = [rp_res(sub, f'uncon_p{p}') for p in range(T)]
    llU, kU, N = sum(u['LL'] for u in U), sum(u['k'] for u in U), sum(u['N'] for u in U)
    rp = ' / '.join(str(sum(len(u[x]) for u in U)) for x in ['random', 'HM', 'HV'])
    rows.append(dict(sub=sub, N=N, model='Fully unconstrained (separate period models)<super>a</super>', tag='uncon', stage='–', k=kU,
                     rp_hm_hv=rp, vary='all', LL=llU, LR='–', df=None, conf=None))
    for app, name in [('pc', 'Partially constrained (LR tests, 90%)'), ('fused', 'Adaptive fused (90% rule)')]:
        tag = dec['choice'][app]; r = rp_res(sub, tag); m = dec['models'][tag]
        rows.append(dict(sub=sub, N=N, model=name, tag=tag, stage=STAGE_OF(tag), k=r['k'],
                         rp_hm_hv=f"{len(r['random'])} / {len(r['HM'])} / {len(r['HV'])}", vary=n_vary(sub, tag), LL=r['LL'],
                         LR=f"{m['LR']:.1f} ({m['df']})", df=m['df'], conf=m['conf']))
Tb = pd.DataFrame(rows)
Tb['AIC'] = -2 * Tb.LL + 2 * Tb.k; Tb['BIC'] = -2 * Tb.LL + np.log(Tb.N) * Tb.k
Tb.assign(model=Tb.model.str.replace(r'<super>.*?</super>', '', regex=True)).to_csv(OUT / 'Table_3_model_comparison_values.csv', index=False)

st = styles(7.6); st.update(sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=7.8, leading=9.4),
          t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.4, leading=11.6),
          n=ParagraphStyle('n', fontName='Helvetica', fontSize=7.0, leading=8.6), c=ParagraphStyle('c', fontName='Helvetica', fontSize=7.6, leading=9.2),
          h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=7.6, leading=9.2, alignment=1))
P = lambda x, k='c': Paragraph(x, st[k])
hdr = ['Temporal approach', 'Temporal structure selected in<super>b</super>', 'Number of parameters',
       'Random parameters / heterogeneity in means / in variances', 'Variables with temporally varying parameters', 'LL(β)', 'AIC', 'BIC',
       'LR test against the fully unconstrained model, χ<super>2</super> (df)<super>c</super>', 'Confidence of rejection']
data = [[P(h, 'h') for h in hdr]]; sty = []
for sub in SUBS:
    S = Tb[Tb['sub'] == sub]
    data.append([P(f'{SUBLAB[sub]}; N = {int(S.N.iloc[0]):,}', 'sec')] + [''] * 9); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    ok = S[(S.conf.isna()) | (S.conf < 0.90)]; best = ok.BIC.min()
    for _, r in S.iterrows():
        if r.BIC == best: sty.append(('FONTNAME', (7, len(data)), (7, len(data)), 'Helvetica-Bold'))
        data.append([P(r.model), P(r.stage), str(r.k), r.rp_hm_hv, r.vary, neg(f'{r.LL:,.2f}'), f'{r.AIC:,.1f}', f'{r.BIC:,.1f}',
                     neg(r.LR), '–' if r.conf is None or pd.isna(r.conf) else f'{r.conf:.3f}'])
W = landscape(A4)[0] - 24 * mm
t_ = Table(data, colWidths=[W * x for x in (0.20, 0.11, 0.07, 0.13, 0.09, 0.08, 0.07, 0.07, 0.11, 0.07)], repeatRows=1)
t_.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (2, 1), (-1, -1), 'CENTER'), ('FONTSIZE', (0, 0), (-1, -1), 7.6),
                        ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                        ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                        ('TOPPADDING', (0, 0), (-1, -1), 1.6), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6)] + sty))
title = P('<b>Table 3.</b> Comparison of temporal approaches for the random parameters binary logit model with heterogeneity in means '
          'and variances of crash injury severity (serious injury/fatal vs. PDO/minor injury), Thailand, 2016–2023.', 't')
notes = P('<b>Notes.</b> All models: random parameters normally distributed, 2,000 Halton draws; random parameters and heterogeneity '
          'terms searched with 100 draws and confirmed with 2,000 draws; non-significant parameters removed (90% level). Lowest BIC '
          'within each subset among models not rejected at the 90% level in bold. Partially constrained: adjacent periods merged when a '
          'likelihood ratio test does not reject equal parameters at the 90% level. Adaptive fused: sparsest penalty of the adaptive fused '
          'lasso whose structure is not rejected at the 90% level against the fully unconstrained model (LR < χ<super>2</super><sub>0.90</sub>(df)). '
          'Variables with temporally varying parameters count the variables (including the constant) whose parameter differs between '
          'at least two periods.<br/>'
          '<super>a</super> Separate random parameters models for each period (two-year periods for S1, yearly periods for S2–S4); parameter '
          'counts and LL(β) are summed over the period models, and AIC and BIC are computed from these sums with the total number of '
          'observations.<br/>'
          '<super>b</super> Each approach selects its temporal structure either in the fixed-parameter binary logit (Stage 1, Table 2) or '
          'inside the random parameters model; the stage with the lower BIC among models not rejected at the 90% level is reported. '
          'Both stages for both approaches, and every candidate structure estimated, are given in Table D1.<br/>'
          '<super>c</super> χ<super>2</super> = −2[LL(β)<sub>temporal model</sub> − Σ LL(β)<sub>period models</sub>], df = difference in the number of '
          'parameters. Because the period models are specified separately, the test is not strictly nested and χ<super>2</super> can '
          'be negative when the temporal model fits better than the period models with fewer parameters; such a model is not '
          'rejected. Confidence of rejection is the χ<super>2</super> cumulative probability; values below 0.90 mean that the model is '
          'not rejected at the 90% level.', 'n')
save('Table_3_model_comparison', landscape(A4), [title, Spacer(1, 6), t_, Spacer(1, 6), notes], 'Comparison of temporal approaches')
print(Tb.drop(columns=['model']).to_string(index=False))
