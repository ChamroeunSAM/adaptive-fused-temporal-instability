"""Python estimates of the four verification models (Appendix E), with the estimator used in the paper (src/rpbl.py).

  check 1: fixed-parameter binary logit (43 parameters)
  check 2: random parameters binary logit, new_year and Northeastern normal (45 parameters), 500 Halton draws
  check 3: check 2 + heterogeneity in the mean of new_year (Eastern) and in the variances of both random parameters
           (speeding 2018-2023) (48 parameters), 500 Halton draws
  check 4: random parameters binary logit on the S1 partially constrained structure, eight random parameters,
           1,000 Halton draws
Random parameters models are estimated from the fixed-parameter logit with SD starts 0.1, 0.5 and 1.0; the best is kept.

usage: python validation/estimate_python.py [1 2 3 4] [--backend numpy|torch]
writes validation/output/python_check<n>.csv (parameter, estimate, se) and python_check<n>.json (LL, k, N)
"""
import sys, json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from check_data import SPEC, OUTDIR, check_data_1to3, check_data_4
from temporal import newton

backend = 'torch' if '--backend' in sys.argv and sys.argv[sys.argv.index('--backend') + 1] == 'torch' else 'numpy'
if backend == 'torch': from rpbl_torch import RPBLTorch as RPBL
else: from rpbl import RPBL
checks = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4]


def fit_rp(Xf, Xr, y, HM, HV, Z, R, names_f, names_r, names_z):
    b, llfix, _ = newton(sp.csr_matrix(np.hstack([Xf, Xr])), y); best = None
    for sd0 in (0.1, 0.5, 1.0):
        m = RPBL(Xf, Xr, y, HM=HM, HV=HV, Z=Z, R=R, names_f=names_f, names_r=names_r, names_z=names_z)
        m.fit(m.start(b, sd0=sd0)); print(f'  SD start {sd0}: LL = {m.LL:.4f}', flush=True)
        if best is None or m.LL > best.LL: best = m
    best.hessian_se()
    T = best.table(); sd = T.parameter.str.startswith('SD:'); T.loc[sd, 'estimate'] = T.loc[sd, 'estimate'].abs()
    return best, T


for c in checks:
    t0 = time.time(); S = SPEC[f'check{c}']
    if c in (1, 2, 3):
        df = check_data_1to3(); y = df.sev.to_numpy(float)
    if c == 1:
        cols = S['fixed']; b, ll, cov = newton(sp.csr_matrix(df[cols].to_numpy(float)), y)
        T = pd.DataFrame(dict(parameter=cols, estimate=b, se=np.sqrt(np.diag(cov)))); k, N = len(cols), len(y)
    elif c in (2, 3):
        FX, RD = S['fixed'], S['random']
        ZN = list(dict.fromkeys([z for _, z in S['het_mean']] + [w for _, w in S['het_variance']]))
        Z = df[ZN].to_numpy(float) if ZN else None
        HM = [(RD.index(r), ZN.index(z)) for r, z in S['het_mean']]; HV = [(RD.index(r), ZN.index(w)) for r, w in S['het_variance']]
        m, T = fit_rp(df[FX].to_numpy(float), df[RD].to_numpy(float), y, HM, HV, Z, S['draws'], FX, RD, ZN)
        ll, k, N = m.LL, len(m.th), len(y)
    else:
        A, y, cn = check_data_4(); rnd = [cn.index(r) for r in S['random']]; fx = [j for j in range(A.shape[1]) if j not in rnd]
        m, T = fit_rp(A[:, fx], A[:, rnd], y, [], [], A, S['draws'], [cn[j] for j in fx], S['random'], cn)
        ll, k, N = m.LL, len(m.th), len(y)
    T.parameter = T.parameter.str.replace('mean:', '', regex=False)
    T.to_csv(OUTDIR / f'python_check{c}.csv', index=False)
    json.dump(dict(check=c, model=S['model'], LL=float(ll), k=int(k), N=int(N), draws=S.get('draws'), backend=backend,
                   seconds=round(time.time() - t0, 1)), open(OUTDIR / f'python_check{c}.json', 'w'), indent=1)
    print(f'check {c}: LL = {ll:.4f}, k = {k}  ({time.time() - t0:.0f} s)', flush=True)
