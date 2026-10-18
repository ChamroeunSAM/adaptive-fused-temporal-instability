"""Table D1: every temporal structure estimated as a random parameters model, for both approaches and both selection stages,
with the confidence of rejection at selection and after rebuilding/pruning (2,000 Halton draws)."""
import json
import numpy as np
import pandas as pd
from _common import decision, stage1, styles, save, neg, SUBS, SUBLAB, RESULTS, OUT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

rows = []
for sub in SUBS:
    dec = decision(sub); S1 = stage1(sub); M = dec['models']; N = dec['reference']['N']
    Sp = json.load(open(RESULTS / 'rp' / sub / 'pc_rp_structure.json', encoding='utf-8'))
    def add(app, stage, cand, lam, ksel, csel, tag):
        m = M[tag]
        rows.append(dict(sub=sub, subset=SUBLAB[sub], N=N, approach=app, stage=stage, candidate=cand, lam=lam, k_sel=ksel, conf_sel=csel,
                         tag=tag, k=m['k'], LL=m['LL'], AIC=m['AIC'], BIC=m['BIC'], LR=m['LR'], df=m['df'], conf=m['conf']))
    add('Partially constrained', 'Fixed-parameter logit', '–', None, S1['pc']['k'], S1['pc']['conf'], 'pc_s1')
    add('Partially constrained', 'Random parameters model', '–', None, Sp['k'], Sp['conf_ref'], 'pc_rp')
    add('Adaptive fused', 'Fixed-parameter logit', '–', S1['fused']['lam'], S1['fused']['k'], S1['fused']['conf'], 'fused_s1')
    for d in dec['fused_candidates_tried']:
        add('Adaptive fused', 'Random parameters model', str(d['candidate'] + 1), d['lam'], d['k_selection'], d['conf_selection'], d['tag'])
Tb = pd.DataFrame(rows)
def status(r):
    if r.tag == decision(r.sub)['choice']['pc' if r.approach.startswith('Partially') else 'fused']: return 'Reported (Table 3)'
    if r.conf >= 0.90: return 'Rejected' + (' → next candidate' if r.approach == 'Adaptive fused' and r.stage.startswith('Random') else '')
    return 'Not rejected'
Tb['status'] = [status(r) for r in Tb.itertuples()]
Tb.drop(columns=['sub']).to_csv(OUT / 'Table_D1_structure_candidates_values.csv', index=False)

st = dict(c=ParagraphStyle('c', fontName='Helvetica', fontSize=7.2, leading=8.7),
          h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=7.2, leading=8.7, alignment=1),
          sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=7.6, leading=9.2),
          t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.4, leading=11.6),
          n=ParagraphStyle('n', fontName='Helvetica', fontSize=6.9, leading=8.5))
P = lambda x, k='c': Paragraph(x, st[k])
def sci(v):
    e = int(np.floor(np.log10(v))); m = v / 10 ** e
    return f'{m:.2f} × 10<super>−{-e}</super>'
hdr = ['Temporal approach', 'Structure selected in', 'Candidate<super>a</super>', 'Penalty λ', 'Parameters at selection<super>b</super>',
       'Confidence of rejection at selection<super>b</super>', 'Parameters after rebuilding and pruning', 'LL(β)', 'AIC', 'BIC',
       'LR test against the fully unconstrained model, χ<super>2</super> (df)', 'Confidence of rejection', 'Outcome']
data = [[P(h, 'h') for h in hdr]]; sty = []
for sub in SUBS:
    S = Tb[Tb['sub'] == sub]
    data.append([P(f'{SUBLAB[sub]}; N = {int(S.N.iloc[0]):,}', 'sec')] + [''] * 12); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    for _, r in S.iterrows():
        if r.status.startswith('Reported'): sty.append(('FONTNAME', (9, len(data)), (9, len(data)), 'Helvetica-Bold'))
        data.append([P(r.approach), P(r.stage), r.candidate, '–' if r.lam is None or pd.isna(r.lam) else P(sci(r.lam)),
                     str(r.k_sel), f'{r.conf_sel:.3f}', str(r.k), neg(f'{r.LL:,.2f}'), f'{r.AIC:,.1f}', f'{r.BIC:,.1f}',
                     neg(f'{r.LR:.1f} ({r.df})'), f'{r.conf:.3f}', P(r.status)])
W = landscape(A4)[0] - 20 * mm
tb = Table(data, colWidths=[W * x for x in (0.10, 0.10, 0.05, 0.06, 0.06, 0.08, 0.08, 0.07, 0.06, 0.06, 0.09, 0.07, 0.12)], repeatRows=1)
tb.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (2, 1), (-2, -1), 'CENTER'), ('FONTSIZE', (0, 0), (-1, -1), 7.2),
                        ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black),
                        ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                        ('TOPPADDING', (0, 0), (-1, -1), 1.4), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.4)] + sty))
title = P('<b>Table D1.</b> Temporal structures estimated as random parameters binary logit models with heterogeneity in means and '
          'variances, by approach and selection stage, Thailand, 2016–2023.', 't')
notes = P('<b>Notes.</b> All models: 2,000 Halton draws; random parameters and heterogeneity terms searched with 100 draws and confirmed '
          'with 2,000 draws; non-significant parameters removed (90% level). The BIC of the reported model of each approach is in bold. '
          'Partially constrained: adjacent periods merged when a likelihood ratio test does not reject equal parameters at the 90% level, '
          'in the fixed-parameter logit or inside the random parameters model (random parameters and heterogeneity terms of the '
          'fixed-parameter-stage model held fixed during merging). Adaptive fused: sparsest penalty λ whose structure is not rejected at '
          'the 90% level (LR < χ<super>2</super><sub>0.90</sub>(df)), in the fixed-parameter logit or inside the random parameters model. '
          'Each structure is then rebuilt (search for random parameters and heterogeneity terms) and pruned, and tested against the pruned '
          'fully unconstrained (separate period) random parameters models. For each approach, the stage with the lower BIC among models '
          'not rejected at the 90% level is reported in Table 3.<br/>'
          '<super>a</super> Structures selected inside the random parameters model by the adaptive fused lasso are tried in order of '
          'sparsity (decreasing λ); a candidate rejected after rebuilding and pruning is replaced by the next one.<br/>'
          '<super>b</super> Before rebuilding and pruning. Fixed-parameter logit: parameters of the fixed-parameter model and confidence of '
          'rejection against the separate-period logits (Table 2). Random parameters model: parameters of the selected structure with the '
          'random parameters and heterogeneity terms held fixed, and confidence of rejection against the separate period random '
          'parameters models before pruning.', 'n')
save('Table_D1_structure_candidates', landscape(A4), [title, Spacer(1, 6), tb, Spacer(1, 6), notes], 'Structure candidates',
     margins=(10, 10, 10, 10))
