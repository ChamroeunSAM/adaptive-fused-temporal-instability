"""Tables 4-7 (adaptive fused) and A1-A4 (partially constrained): random parameters binary logit models with heterogeneity in
means and variances, the model chosen for each approach and subset (Table 3), with marginal effects simulated with the
2,000 estimation Halton draws.
usage: python tables/coef_tables.py [all | <subset> <fused|pc> <table number>]"""
import sys, json, numpy as np
from scipy.stats import norm
from scipy.special import expit
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from _common import (load_subset, rp_res, rp_fit, decision, styles, save, fm, desc, CATS, ORDER, REFTXT, OUT)
from temporal import group_design
from rpbl import radical_inverse, PRIMES

JOBS = [('S1', 'fused', '4'), ('S2', 'fused', '5'), ('S3', 'fused', '6'), ('S4', 'fused', '7'),
        ('S1', 'pc', 'A1'), ('S2', 'pc', 'A2'), ('S3', 'pc', 'A3'), ('S4', 'pc', 'A4')]
STAGE_KEY = lambda tag: {'fused_s1': 'fusedS1', 'pc_s1': 'pc', 'pc_rp': 'pcRP'}.get(tag, 'fused')


def make(SUB, APP, TNO):
    tag = decision(SUB)['choice'][APP]; STAGE = STAGE_KEY(tag)
    F = rp_fit(SUB, tag); RES = rp_res(SUB, tag); groups = F['groups']
    X, y, t, names, T = load_subset(SUB)
    A, cn, var = group_design(X, t, names, groups, T); assert cn == F['cn']
    spec, th, se, pn = F['spec'], np.array(F['th']), np.array(F['se']), F['names']
    rnd, hm, hv = spec['rnd'], [tuple(x) for x in spec['hm']], [tuple(x) for x in spec['hv']]
    moved = {c for _, c in hm} | {c for _, c in hv}; drop = set(spec.get('drop', []))
    fx = [j for j in range(A.shape[1]) if j not in rnd and j not in moved and j not in drop]
    N, R, J = len(y), RES['draws_final'], len(rnd)
    print(f'Table {TNO}: {SUB} {APP} -> {tag}, k={RES["k"]}, LL={RES["LL"]:.3f}', flush=True)
    # ---- simulation with the estimation draws (standard Halton, observation i uses points 10+i*R..10+i*R+R-1); the draws are
    #      generated chunk by chunk and every marginal effect is accumulated within the chunk (memory-light, 2,000 draws)
    na = len(fx) + J + len(hm); lin, sig, om = th[:na], th[na:na + J], th[na + J:]

    def draws(i0, i1):
        NU = np.empty((J, i1 - i0, R), np.float32)
        for j in range(J):
            NU[j] = norm.ppf(radical_inverse(PRIMES[j], 10 + i0 * R, (i1 - i0) * R)).astype(np.float32).reshape(i1 - i0, R)
        return NU

    def lin_part(Ac):
        D = np.hstack([Ac[:, fx], Ac[:, rnd]] + [(Ac[:, r] * Ac[:, q])[:, None] for r, q in hm])
        return D @ lin

    def rand_part(Ac, NUc):
        U = np.zeros((Ac.shape[0], R))
        for jj, r in enumerate(rnd):
            s_ = sig[jj] * np.ones(Ac.shape[0])
            for h, (rr, q) in enumerate(hv):
                if rr == r: s_ = s_ * np.exp(om[h] * Ac[:, q])
            U += (Ac[:, r] * s_)[:, None] * NUc[jj]
        return U

    gcols = {}
    for j, k in enumerate(var): gcols.setdefault(k, []).append(j)
    per_of = [np.unique(t[A[:, j] != 0]) for j in range(A.shape[1])]
    grp_of = {}
    for k in range(len(names)):
        for g in groups[k]:
            for j in gcols[k]:
                if set(per_of[j]) <= set(g) and len(per_of[j]): grp_of[j] = g
    MEJ = [j for j in range(A.shape[1]) if var[j] != 0 and j in grp_of and j not in moved and j not in drop]
    in_rand = {var[r] for r in rnd} | {var[q] for _, q in hv}
    p_all = np.zeros(N); ME_sum = {j: 0.0 for j in MEJ}; ME_n = {j: 0 for j in MEJ}
    CHK = 1000
    for i0 in range(0, N, CHK):
        i1 = min(i0 + CHK, N); NUc = draws(i0, i1); Ac = A[i0:i1]
        RU = rand_part(Ac, NUc)
        p_all[i0:i1] = expit(lin_part(Ac)[:, None] + RU).mean(1)
        for j in MEJ:
            k = var[j]; loc = np.where(np.isin(t[i0:i1], grp_of[j]))[0]
            if not len(loc): continue
            pr = []
            for v in (0.0, 1.0):
                Xm = X[i0 + loc].copy(); Xm[:, k] = v
                Am, _, _ = group_design(Xm, t[i0 + loc], names, groups, T)
                Ur = rand_part(Am, NUc[:, loc]) if k in in_rand else RU[loc]
                pr.append(expit(lin_part(Am)[:, None] + Ur).mean(1))
            ME_sum[j] += float((pr[1] - pr[0]).sum()); ME_n[j] += len(loc)
    LLchk = float(np.sum(np.log(np.where(y == 1, p_all, 1 - p_all))))
    print('  LL check', LLchk, RES['LL'], flush=True); assert abs(LLchk - RES['LL']) < 1e-3
    ME = {j: ME_sum[j] / ME_n[j] for j in MEJ if ME_n[j]}
    print('  marginal effects done', len(ME), flush=True)

    # ---- descriptions
    order = ORDER
    LAB = {4: ['2016–2017', '2018–2019', '2020–2021', '2022–2023'], 8: [str(v) for v in range(2016, 2024)]}
    def span(g):
        a, b = LAB[T][g[0]], LAB[T][g[-1]]
        return f'[{a.split("–")[0]}–{b.split("–")[-1]}]' if T == 4 else (f'[{a}]' if a == b else f'[{a}–{b}]')
    def dsc(k): return desc(names[k])
    def label(j): return f'{dsc(var[j])[1]} (1 if yes, 0 otherwise) {span(grp_of.get(j, list(range(T))))}'

    est = dict(zip(pn, th)); ses = dict(zip(pn, se))
    st = styles(7.3); st.update(cell=st['c'], hdr=st['h'], title=st['t'], ctr=st['v'], note=st['n'])
    P = lambda s, sty='cell': Paragraph(s, st[sty])

    data = [[P('Variable description', 'hdr'), P('Estimated parameter', 'hdr'), P('<i>t</i>-statistic', 'hdr'), P('Marginal effects', 'hdr'), ''],
            ['', '', '', P('PDO / minor injury', 'hdr'), P('Serious injury / fatal', 'hdr')]]
    sty = []
    def sec(text, s='sec'):
        data.append([P(text, s), '', '', '', '']); sty.append(('SPAN', (0, len(data) - 1), (-1, len(data) - 1)))
    def row(j):
        name = f'mean:{cn[j]}' if j in rnd else cn[j]
        b, s_ = est[name], ses[name]; me = ME.get(j)
        data.append([P(label(j)), fm(b), fm(b / s_, 2), '' if me is None else fm(-me, 4), '' if me is None else fm(me, 4)])
        if j in rnd:
            sd = abs(est[f'SD:{cn[j]}']); tsd = abs(est[f'SD:{cn[j]}'] / ses[f'SD:{cn[j]}'])
            above = norm.cdf(b / sd) * 100
            data.append([P(f'(Standard deviation of parameter distribution; {above:.1f}% above zero)', 'sd'), fm(sd), fm(tsd, 2), '', ''])

    sec('Constant [serious injury / fatal]')
    for j in range(A.shape[1]):
        if var[j] == 0 and j not in drop: data.append([P(f'Constant {span(grp_of.get(j, list(range(T))))}'), fm(est[cn[j]]), fm(est[cn[j]] / ses[cn[j]], 2), '', ''])

    main = [j for j in range(A.shape[1]) if var[j] != 0 and j not in moved and j not in drop]
    stable = [j for j in main if len(groups[var[j]]) == 1]; vary = [j for j in main if len(groups[var[j]]) > 1]
    for title, js in [(f'Variables producing the same parameter value across all periods {span(list(range(T)))}', stable),
                      ('Variables with temporally varying parameters', vary)]:
        if not js: continue
        sec(title)
        for c in CATS + ['Other']:
            s = sorted([j for j in js if dsc(var[j])[0] == c], key=lambda j: (order.get(names[var[j]], 999), j))
            if not s: continue
            sec(c, 'sub')
            for j in s: row(j)

    if hm:
        sec('Heterogeneity in the means of random parameters')
        for r, c in hm:
            nm = f'HM:{cn[r]}|{cn[c]}'
            data.append([P(f'{dsc(var[r])[1]} {span(grp_of[r])}: {dsc(var[c])[1]} {span(grp_of[c])}'), fm(est[nm]), fm(est[nm] / ses[nm], 2), '', ''])
    if hv:
        sec('Heterogeneity in the variances of random parameters')
        for r, c in hv:
            nm = f'HV:{cn[r]}|{cn[c]}'
            data.append([P(f'{dsc(var[r])[1]} {span(grp_of[r])}: {dsc(var[c])[1]} {span(grp_of[c])}'), fm(est[nm]), fm(est[nm] / ses[nm], 2), '', ''])

    LL = RES['LL']; k = RES['k']; LL0 = N * np.log(0.5); p = y.mean(); LLc = N * (p * np.log(p) + (1 - p) * np.log(1 - p))
    sec('Model statistics')
    for a, b in [('Number of observations', f'{N:,}'), ('Number of estimated parameters', f'{k}'),
                 ('Log-likelihood at zero, LL(0)', fm(LL0, 2)), ('Log-likelihood with constant only, LL(C)', fm(LLc, 2)),
                 ('Log-likelihood at convergence, LL(β)', fm(LL, 2)),
                 ('McFadden pseudo ρ<super>2</super> = 1 − LL(β)/LL(0)', f'{1 - LL / LL0:.3f}'),
                 ('AIC / BIC', f'{-2 * LL + 2 * k:,.1f} / {-2 * LL + np.log(N) * k:,.1f}'),
                 ('Random parameter distribution; simulation', 'Normal; 2,000 Halton draws')]:
        data.append([P(a), P(b, 'ctr'), '', '', '']); sty.append(('SPAN', (1, len(data) - 1), (-1, len(data) - 1)))

    W = A4[0] - 24 * mm
    t_ = Table(data, colWidths=[W * 0.56, W * 0.11, W * 0.09, W * 0.12, W * 0.12], repeatRows=2)
    t_.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 2), (-1, -1), 'CENTER'), ('FONTSIZE', (0, 0), (-1, -1), 7.3),
                            ('SPAN', (3, 0), (4, 0)), ('SPAN', (0, 0), (0, 1)), ('SPAN', (1, 0), (1, 1)), ('SPAN', (2, 0), (2, 1)),
                            ('LINEABOVE', (0, 0), (-1, 0), 0.9, colors.black), ('LINEBELOW', (0, 1), (-1, 1), 0.6, colors.black),
                            ('LINEBELOW', (3, 0), (4, 0), 0.4, colors.black), ('LINEBELOW', (0, -1), (-1, -1), 0.9, colors.black),
                            ('TOPPADDING', (0, 0), (-1, -1), 1.0), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.0)] + sty))
    SUBTXT = {'S1': 'single-motorcycle crashes, Thailand, 2016–2023 (two-year periods)',
              'S2': 'crashes involving a motorcycle and another vehicle type, Thailand, 2016–2023 (yearly periods)',
              'S3': 'single-vehicle crashes without a motorcycle, Thailand, 2016–2023 (yearly periods)',
              'S4': 'multi-vehicle crashes without a motorcycle, Thailand, 2016–2023 (yearly periods)'}[SUB]
    REF = REFTXT[SUB]
    APPN = dict(fused='adaptive fused', pc='partially constrained')[APP]
    title = P(f'<b>Table {TNO}.</b> Random parameters binary logit model with heterogeneity in means and variances of crash injury severity '
              f'(serious injury/fatal vs. PDO/minor injury), {SUBTXT}, '
              f'{APPN} temporal structure.', 'title')
    STRUCT = {'fusedS1': 'Temporal structure selected by the adaptive fused lasso in the fixed-parameter binary logit (Stage 1): the '
              'sparsest penalty whose structure is not rejected at the 90% level against the separate-period logits '
              '(LR < χ<super>2</super><sub>0.90</sub>(df)).',
              'fused': 'Temporal structure re-selected by the adaptive fused lasso inside the random parameters model: the sparsest penalty '
              'whose structure is not rejected at the 90% level against the fully unconstrained (separate period) random parameters models '
              '(LR < χ<super>2</super><sub>0.90</sub>(df)); candidate structures were rebuilt and pruned in order of sparsity until one was not rejected (Table D1).',
              'pc': 'Temporal structure from the partially constrained procedure in the fixed-parameter binary logit (Stage 1): adjacent periods '
              'merged when a likelihood ratio test does not reject equal parameters at the 90% level.',
              'pcRP': 'Temporal structure from the partially constrained procedure applied inside the random parameters model: adjacent periods '
              'merged when a likelihood ratio test (simulated log-likelihood, 2,000 Halton draws) does not reject equal parameters at the 90% level.'}[STAGE]
    STRUCT += (' The stage at which the structure was chosen (fixed-parameter or random parameters model) is the one with the lower BIC among '
               'models not rejected at the 90% level against the fully unconstrained models (Table 3; all candidates in Table D1).')
    notes = P('<b>Notes.</b> A positive parameter increases the probability of a serious injury/fatal outcome. Reference categories: '
              f'{REF}. Periods in brackets share one parameter value; for a variable with temporally varying parameters, periods not '
              f'shown have a parameter of zero. {STRUCT} Random parameters (at most eight), heterogeneity in means (at most three terms) '
              'and in variances (at most one term) were searched with 100 Halton draws at the 90% level and confirmed with 2,000 draws; '
              'non-significant parameters were then removed, so all parameters except constants and the means of random parameters are '
              'statistically significant at the 90% level (a parameter is kept random only while its standard deviation is significant). '
              'A variable used in a heterogeneity function enters only the random parameter shown, for crashes in the shared periods, and '
              'is removed from the main utility. Marginal effects are average changes in the simulated unconditional probability (2,000 '
              "Halton draws) when the indicator switches from 0 to 1, averaged over crashes in the parameter's periods; for random "
              'parameters they include the heterogeneity terms. PDO/minor = −(serious/fatal). Percentage above zero evaluated at the '
              'heterogeneity variables equal to zero.', 'note')
    name = f'Table_{TNO}_{SUB}_RPBLHMV_{"adaptive_fused" if APP == "fused" else "partially_constrained"}'
    save(name, A4, [title, Spacer(1, 5), t_, Spacer(1, 6), notes], f'{SUB} RPBLHMV {APP} estimates', margins=(12, 12, 10, 10))
    json.dump({cn[j]: v for j, v in ME.items()}, open(OUT / f'{name}_marginal_effects.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    jobs = JOBS if len(sys.argv) < 2 or sys.argv[1] == 'all' else [tuple(sys.argv[1:4])]
    for job in jobs: make(*job)
