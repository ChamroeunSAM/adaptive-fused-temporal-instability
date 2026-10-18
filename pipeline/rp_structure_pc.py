"""Partially constrained structure re-selected INSIDE the random parameters model (Section 2.2, random parameters stage).

Model during selection: every variable enters the utility with period-specific coefficients; the random parameters and
heterogeneity terms of the random parameters model estimated on the Stage-1 partially constrained structure are kept as
they are (random terms add sigma * nu * x over their Stage-1 period span; their means are carried by the period-specific
coefficients). Merging rule (as in Stage 1, but with the simulated log-likelihood at 2,000 Halton draws): in each round,
for every variable the adjacent pair of period groups whose merging reduces the log-likelihood least is merged if
LR = 2[LL(current) - LL(merged)] < chi2_0.90(1) = 2.706; the model is re-estimated and merging continues until no pair of
any variable can be merged. Each restricted model is fitted from the averaged start by Newton steps with the information
matrix of the current model (numerical derivative of the analytic gradient), and by BFGS from three starts if that does
not converge.

usage: python pipeline/rp_structure_pc.py <subset> [--backend torch|numpy]
needs  results/stage1/struct_<subset>_pc.json, results/rp/<subset>/pc_s1_fit.json, results/rp/<subset>/uncon_p*_search.json
writes results/rp/<subset>/pc_rp_structure.json and pc_rp_structure.log
"""
import os, sys, json, time, argparse
import _paths  # noqa: F401
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import chi2
from settings import SUBSETS, RESULTS, DRAWS_FINAL, CONF, subset_file
from temporal import newton, group_design

ap = argparse.ArgumentParser(); ap.add_argument('subset'); ap.add_argument('--backend', default='torch', choices=['torch', 'numpy'])
ap.add_argument('--out'); a = ap.parse_args()
if a.backend == 'torch':
    from rpbl_torch import RPBLTorch as RPBL, DEVICE
else:
    from rpbl import RPBL
    DEVICE = 'cpu (numpy)'
SUB = a.subset; T = SUBSETS[SUB]['T']
OUT = os.path.abspath(a.out or os.path.join(RESULTS, 'rp', SUB))
R = int(os.environ.get('AFTI_DRAWS_FINAL', DRAWS_FINAL)); CRIT = chi2.ppf(CONF, 1)
LOG = open(os.path.join(OUT, 'pc_rp_structure.log'), 'a', encoding='utf-8')
def log(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

Zd = np.load(subset_file(SUB), allow_pickle=True)
X, y, t, names = Zd['X'], Zd['y'], Zd['t'], [str(n) for n in Zd['names']]; K, N = X.shape[1], len(y)
t0 = time.time()

# ---- Stage-1 PC structure and its random parameters model (random / heterogeneity terms kept fixed) ----
groups1 = json.load(open(os.path.join(RESULTS, 'stage1', f'struct_{SUB}_pc.json'), encoding='utf-8'))['groups']
A3, cn3, var3 = group_design(X, t, names, groups1, T)
F = json.load(open(os.path.join(OUT, 'pc_s1_fit.json'), encoding='utf-8')); assert F['cn'] == cn3, 'starting structure mismatch'
rnd = list(F['spec']['rnd']); hm = [tuple(x) for x in F['spec']['hm']]; hv = [tuple(x) for x in F['spec']['hv']]
J, nHM, nHV = len(rnd), len(hm), len(hv)
HMm = [(rnd.index(r), c) for r, c in hm]; HVm = [(rnd.index(r), c) for r, c in hv]
log(f'== PC structure in the RP model {SUB} [LR merges at {CONF:.0%}, {R} draws] [{a.backend}, {DEVICE}]: N={N}, K={K}, T={T}; '
    f'random={[cn3[j] for j in rnd]}, HM={[(cn3[r], cn3[c]) for r, c in hm]}, HV={[(cn3[r], cn3[c]) for r, c in hv]}')

U = [json.load(open(os.path.join(OUT, f'uncon_p{p}_search.json'), encoding='utf-8')) for p in range(T)]
llRef, kRef = sum(u['LL'] for u in U), sum(u['k'] for u in U)
log(f'separate period models before pruning: LL={llRef:.3f}, k={kRef}')

# ---- one model for all structures: period-specific fixed columns (k*T + s); structures enter as linear maps ----
Ay = np.column_stack([X[:, k] * (t == s) for k in range(K) for s in range(T)]).astype(float)
Kf = K * T
m = RPBL(Ay, A3[:, rnd], y, HM=HMm, HV=HVm, Z=A3, R=R)
L = m.nD + J + nHV
other = np.r_[np.arange(Kf + J, L)]                     # HM, SD, HV (means of the random terms are 0: carried by Ay)
nO = len(other)

def gmap(G):
    gidx = np.zeros(Kf, int); pos = 0; starts = []
    for k in range(K):
        starts.append(pos)
        for g in G[k]:
            for s in g: gidx[k * T + s] = pos
            pos += 1
    return gidx, pos, starts

def make_f(G):
    gidx, ng, starts = gmap(G)
    def f(z):
        th = np.zeros(L); th[:Kf] = z[gidx]; th[other] = z[ng:]
        ll, g = m.loglik_grad(th)
        return ll, np.concatenate([np.bincount(gidx, weights=g[:Kf], minlength=ng), g[other]])
    return f, ng, starts

def bfgs(f, z0):
    res = minimize(lambda v: tuple(-x for x in f(v)), z0, jac=True, method='BFGS', options=dict(maxiter=5000, gtol=1e-6))
    return res.x, -res.fun

def info(f, z, g0):
    """information matrix = -(d grad / dz), forward differences of the analytic gradient, symmetrised."""
    P = len(z); H = np.zeros((P, P))
    for i in range(P):
        h = 1e-5 * max(1.0, abs(z[i])); zz = z.copy(); zz[i] += h
        H[:, i] = -(f(zz)[1] - g0) / h
    return 0.5 * (H + H.T)

def psolve(I, g):
    w, V = np.linalg.eigh(I); keep = w > 1e-8 * max(w.max(), 1e-12)
    return V[:, keep] @ ((V[:, keep].T @ g) / w[keep])

def newton_fit(f, z, I, maxit=30):
    """Newton steps with a fixed information matrix and step halving; returns z, LL, converged."""
    llc, g = f(z)
    for it in range(maxit):
        step = psolve(I, g)
        if g @ step < 2e-8: return z, llc, True
        a = 1.0
        while a > 1e-4:
            zn = z + a * step; lln, gn = f(zn)
            if lln >= llc - 1e-10: break
            a /= 2
        if lln < llc - 1e-10: return z, llc, False
        z, llc, g = zn, lln, gn
    return z, llc, False

# ---- start: period-specific means from the yearly fixed-parameter logits; HM/SD/HV from the Stage-1 PC model ----
G = [[[s] for s in range(T)] for _ in range(K)]
by, _, _ = newton(sp.csr_matrix(Ay), y)
th3 = np.asarray(F['th']); n3 = len(th3)
z = np.concatenate([by, th3[n3 - nHV - J - nHM: n3 - nHV - J], np.abs(th3[n3 - nHV - J: n3 - nHV]), th3[n3 - nHV:]])
rounds = []
while True:
    f, ng, starts = make_f(G)
    z, llU = bfgs(f, z)
    llU, gU = f(z); I = info(f, z, gU)
    z, llU, _ = newton_fit(f, z, I)                       # polish
    kU = ng + nO
    log(f' round {len(rounds) + 1}: current structure k={kU}, LL={llU:.3f}  ({round(time.time() - t0)} s)')
    new, merged, tests = [], 0, []
    for k in range(K):
        Gk = G[k]; best, bj = None, None
        for j in range(len(Gk) - 1):
            a = starts[k] + j
            Gc = [list(g) for g in G]; Gc[k] = Gk[:j] + [Gk[j] + Gk[j + 1]] + Gk[j + 2:]
            fc, _, _ = make_f(Gc)
            I2 = I.copy(); I2[a, :] += I2[a + 1, :]; I2[:, a] += I2[:, a + 1]
            I2 = np.delete(np.delete(I2, a + 1, 0), a + 1, 1)
            zc = z.copy(); zc[a] = 0.5 * (z[a] + z[a + 1]); zc = np.delete(zc, a + 1)
            zc, llC, ok = newton_fit(fc, zc, I2)
            if not ok:                                   # not converged: BFGS from three starts, keep the best
                fits = [bfgs(fc, zc), bfgs(fc, np.delete(z, a + 1)), bfgs(fc, np.delete(z, a))]
                zc, llC = max(fits, key=lambda r_: r_[1])
            lr = 2 * (llU - llC)
            if best is None or lr < best: best, bj = lr, j
        if best is not None:
            tests.append(dict(var=names[k], pair=[Gk[bj], Gk[bj + 1]], LR=best, merged=bool(best < CRIT)))
        if best is not None and best < CRIT:
            Gk = Gk[:bj] + [Gk[bj] + Gk[bj + 1]] + Gk[bj + 2:]; merged += 1
        new.append(Gk)
    rounds.append(dict(k=kU, LL=llU, merges=merged, tests=tests, seconds=round(time.time() - t0)))
    log(f'   {merged} merges  ({round(time.time() - t0)} s)')
    if merged == 0: break
    # start of the next round: merged groups take the mean of their two coefficients
    zn, pos = [], 0
    for k in range(K):
        old = {tuple(g): z[starts[k] + i] for i, g in enumerate(G[k])}
        for g in new[k]:
            if tuple(g) in old: zn.append(old[tuple(g)])
            else:
                parts = [v for gg, v in old.items() if set(gg) <= set(g)]; zn.append(np.mean(parts))
    z = np.concatenate([zn, z[ng:]]); G = new

LR, df = 2 * (llRef - llU), kRef - kU; conf = float(chi2.cdf(max(LR, 0), df))
out = dict(subset=SUB, rule=f'greedy adjacent merging by LR tests at {CONF:.0%} inside the random parameters model ({R} draws)',
           groups=G, k=kU, LL=llU, LR_ref=LR, df_ref=df, conf_ref=conf, ref_LL=llRef, ref_k=kRef, draws=R,
           stage1_groups=groups1, random=[cn3[j] for j in rnd], HM=[[cn3[r], cn3[c]] for r, c in hm],
           HV=[[cn3[r], cn3[c]] for r, c in hv], rounds=rounds, seconds=round(time.time() - t0))
json.dump(out, open(os.path.join(OUT, 'pc_rp_structure.json'), 'w', encoding='utf-8'), indent=1, default=float)
n_changed = sum(G[k] != groups1[k] for k in range(K))
log(f'DONE: k={kU}, LL={llU:.3f}; against the yearly models before pruning LR={LR:.1f}, df={df}, conf={conf:.3f}; '
    f'{n_changed} of {K} variables differ from the Stage-1 PC structure  ({round(time.time() - t0)} s)')
