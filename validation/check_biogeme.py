"""Biogeme checks 3 and 4 (Appendix E). Biogeme is given the same standard Halton draws as the Python estimator (as a
custom generator), so the two log-likelihoods must agree exactly at the same parameter values.
Part 1: Biogeme evaluates the log-likelihood and its gradient at the Python estimates (validation/output/python_check<n>.csv).
Part 2 (--estimate): Biogeme estimates the model itself from the fixed-parameter logit with SDs of 0.1 and heterogeneity
terms of 0, independently of the Python solution, and the estimates and Hessian-based standard errors are compared.

usage: python validation/check_biogeme.py [3 4] [--estimate]          (Biogeme 3.3.5; part 2 takes 10-20 minutes per check)
writes validation/output/biogeme_check<n>.json and biogeme_check<n>.csv
"""
import sys, json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from biogeme.database import Database
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp
from biogeme.draws import RandomNumberGeneratorTuple
from check_data import SPEC, OUTDIR, check_data_1to3, check_data_4
from rpbl import halton_normal
from temporal import newton

checks = [int(a) for a in sys.argv[1:] if a.isdigit()] or [3, 4]
ESTIMATE = '--estimate' in sys.argv


def run(c):
    S = SPEC[f'check{c}']; R = S['draws']
    py = pd.read_csv(OUTDIR / f'python_check{c}.csv').set_index('parameter'); LLpy = json.load(open(OUTDIR / f'python_check{c}.json'))['LL']
    if c == 3:
        df = check_data_1to3(); FX, RD = S['fixed'], S['random']; y = df.sev.to_numpy(float)
        data = df[list(dict.fromkeys(FX + RD + [z for _, z in S['het_mean']] + [w for _, w in S['het_variance']]))].astype(float)
        cols = {n: n for n in data.columns}
    else:
        A, y, cn = check_data_4(); RD = S['random']; FX = [n for n in cn if n not in RD]
        data = pd.DataFrame({f'c{j}': A[:, j] for j in range(A.shape[1])}); cols = {n: f'c{j}' for j, n in enumerate(cn)}
    data['sev'] = y; N = len(y)
    nu = halton_normal(N, R, len(RD)).astype(float)
    gens = {f'HAL{d}': RandomNumberGeneratorTuple(generator=(lambda dd: (lambda n, r: nu[dd][:n, :r]))(d), description=f'prime {d}')
            for d in range(len(RD))}
    b0, _, _ = newton(sp.csr_matrix(data[[cols[n] for n in FX + RD]].to_numpy(float)), y)
    start = dict(zip(FX + RD, b0))
    bn = lambda n: 'b_' + cols[n]                                 # Biogeme parameter names
    V = 0; pymap = {}
    for n in FX: V = V + Beta(bn(n), float(start[n]), None, None, 0) * Variable(cols[n]); pymap[bn(n)] = n
    for d, r in enumerate(RD):
        mu = Beta('mu_' + cols[r], float(start[r]), None, None, 0); pymap['mu_' + cols[r]] = r
        for r_, z in S.get('het_mean', []):
            if r_ == r: mu = mu + Beta(f'hm_{cols[r]}_{cols[z]}', 0, None, None, 0) * Variable(cols[z]); pymap[f'hm_{cols[r]}_{cols[z]}'] = f'HM:{r}|{z}'
        sd = Beta('sd_' + cols[r], 0.1, None, None, 0); pymap['sd_' + cols[r]] = f'SD:{r}'
        for r_, w in S.get('het_variance', []):
            if r_ == r: sd = sd * exp(Beta(f'hv_{cols[r]}_{cols[w]}', 0, None, None, 0) * Variable(cols[w])); pymap[f'hv_{cols[r]}_{cols[w]}'] = f'HV:{r}|{w}'
        V = V + (mu + sd * Draws(f'nu{d}', f'HAL{d}')) * Variable(cols[r])
    yv = Variable('sev'); P = yv / (1 + exp(-V)) + (1 - yv) / (1 + exp(V))
    bg = BIOGEME(Database(f'check{c}', data), log(MonteCarlo(P)), random_number_generators=gens, number_of_draws=R,
                 generate_html=False, generate_yaml=False, save_iterations=False, analytical_hessian_mode='chunked',
                 hessian_parameter_block_size=4, hessian_observation_batch_size=2000)
    bg.model_name = f'check{c}'
    pyv = {k: float(py.loc[v, 'estimate']) for k, v in pymap.items()}
    t0 = time.time(); r = bg.function_evaluator.evaluate(the_betas=pyv, gradient=True, hessian=False, bhhh=False)
    g = dict(zip(bg.free_betas_names, np.asarray(r.gradient)))
    out = dict(check=c, LL_python=LLpy, LL_biogeme_at_python_estimates=float(r.function), difference=float(r.function) - LLpy,
               max_abs_biogeme_gradient_at_python_estimates=float(max(abs(v) for v in g.values())), seconds=round(time.time() - t0, 1))
    print(out, flush=True)
    if ESTIMATE:
        from biogeme.results_processing import EstimateVarianceCovariance as EVC
        t1 = time.time(); res = bg.estimate(); bv = res.get_beta_values(); names = list(pymap)
        Tb = pd.DataFrame(dict(parameter=[pymap[k] for k in names], python_estimate=[pyv[k] for k in names],
                               python_se=[float(py.loc[pymap[k], 'se']) for k in names], biogeme_estimate=[float(bv[k]) for k in names],
                               biogeme_se=[float(res.get_parameter_std_err(k, EVC.RAO_CRAMER)) for k in names]))
        sd = Tb.parameter.str.startswith('SD:'); Tb.loc[sd, ['python_estimate', 'biogeme_estimate']] = Tb.loc[sd, ['python_estimate', 'biogeme_estimate']].abs()
        Tb.to_csv(OUTDIR / f'biogeme_check{c}.csv', index=False, float_format='%.6f')
        out.update(LL_biogeme_estimated=float(res.final_log_likelihood),
                   max_abs_coef_diff=float((Tb.biogeme_estimate - Tb.python_estimate).abs().max()),
                   max_abs_se_diff=float((Tb.biogeme_se - Tb.python_se).abs().max()),
                   max_rel_se_diff=float(((Tb.biogeme_se - Tb.python_se).abs() / Tb.python_se).max()),
                   estimation_minutes=round((time.time() - t1) / 60, 1))
        print(out, flush=True)
    json.dump(out, open(OUTDIR / f'biogeme_check{c}.json', 'w'), indent=1)


for c in checks: run(c)
