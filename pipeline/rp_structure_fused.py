"""Adaptive fused structure re-selected INSIDE the random parameters model (Section 2.2, random parameters stage).

Model during selection: every variable enters the utility with the fused basis (level + steps between adjacent periods,
steps penalised with the Stage-1 adaptive weights); the random parameters and heterogeneity terms of the random parameters
model estimated on the Stage-1 fused structure are kept as they are (random terms add sigma * nu * x over their Stage-1
period span; their means are carried by the fused basis). The penalised simulated log-likelihood is maximised at
100 Halton draws over a lambda path (lambda_max * 0.75^i, then 6 bisection steps); each new structure is refitted
without the penalty at 2,000 draws. Rule: sparsest lambda whose refitted structure is not rejected at 90% against the
separate period random parameters models before pruning (LR < chi2_0.90(df)). The next five distinct, less sparse
structures on the path are stored as step-down candidates.

usage: python pipeline/rp_structure_fused.py <subset> [--backend torch|numpy]
needs  results/stage1/struct_<subset>_fused.json, results/rp/<subset>/fused_s1_fit.json, results/rp/<subset>/uncon_p*_search.json
writes results/rp/<subset>/fused_rp_structure.json and fused_rp_structure.log
"""
import os, sys, json, time, argparse
import _paths  # noqa: F401
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.sparse import csr_matrix
from settings import SUBSETS, RESULTS, DRAWS_SEARCH, DRAWS_FINAL, CONF, RP_FUSED, subset_file
from temporal import newton, build, spec_fused, spec_year, implied_paths, group_design

ap = argparse.ArgumentParser(); ap.add_argument('subset'); ap.add_argument('--backend', default='torch', choices=['torch', 'numpy'])
ap.add_argument('--out'); a = ap.parse_args()
if a.backend == 'torch':
    from rpbl_torch import RPBLTorch as RPBL, DEVICE
else:
    from rpbl import RPBL
    DEVICE = 'cpu (numpy)'
SUB = a.subset; T = SUBSETS[SUB]['T']
OUT = os.path.abspath(a.out or os.path.join(RESULTS, 'rp', SUB))
RS = int(os.environ.get('AFTI_DRAWS_SEARCH', DRAWS_SEARCH)); RF = int(os.environ.get('AFTI_DRAWS_FINAL', DRAWS_FINAL))
SHRINK, NBISECT, NEXTRA = RP_FUSED['shrink'], RP_FUSED['nbisect'], RP_FUSED['n_stepdown']
def passes_rule(LR, df): return bool(LR < chi2.ppf(CONF, df))
LOG = open(os.path.join(OUT, 'fused_rp_structure.log'), 'a', encoding='utf-8')
def log(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

Zd = np.load(subset_file(SUB), allow_pickle=True)
X, y, t, names = Zd['X'], Zd['y'], Zd['t'], [str(n) for n in Zd['names']]; K, N = X.shape[1], len(y)
t0 = time.time()

# ---- Stage-1 adaptive weights; the Stage-1 fused structure maps the random / heterogeneity columns ----
Af_s, meta = build(X, t, spec_fused(K, T)); Ay, my = build(X, t, spec_year(K, T))
pen_f = np.array([m[2] for m in meta])
by, _, _ = newton(Ay, y); Py = implied_paths(my, by, K, T); w_f = np.ones(Af_s.shape[1])
for j, (k, lab, p_, f) in enumerate(meta):
    if p_: s = int(lab[4:]); w_f[j] = 1 / max(abs(Py[k, s] - Py[k, s - 1]), 1e-4)
groups1 = json.load(open(os.path.join(RESULTS, 'stage1', f'struct_{SUB}_fused.json'), encoding='utf-8'))['groups']
A3, cn3, var3 = group_design(X, t, names, groups1, T)
F = json.load(open(os.path.join(OUT, 'fused_s1_fit.json'), encoding='utf-8')); assert F['cn'] == cn3, 'starting structure mismatch'
rnd = list(F['spec']['rnd']); hm = [tuple(x) for x in F['spec']['hm']]; hv = [tuple(x) for x in F['spec']['hv']]
J = len(rnd)
log(f'== fused structure in the RP model {SUB} [refit {RF} draws, penalised fits {RS} draws] [{a.backend}, {DEVICE}]: N={N}, K={K}, T={T}; '
    f'random={[cn3[j] for j in rnd]}, HM={[(cn3[r], cn3[c]) for r, c in hm]}, HV={[(cn3[r], cn3[c]) for r, c in hv]}')

# ---- reference: separate period models before pruning ----
U = [json.load(open(os.path.join(OUT, f'uncon_p{p}_search.json'), encoding='utf-8')) for p in range(T)]
llU, kU = sum(u['LL'] for u in U), sum(u['k'] for u in U)
log(f'separate period models before pruning: LL={llU:.3f}, k={kU}')

Af = Af_s.toarray()
Xr = A3[:, rnd]
HMm = [(rnd.index(r), c) for r, c in hm]; HVm = [(rnd.index(r), c) for r, c in hv]


class Wrapped:
    """Random parameters model whose random terms have no separate mean (means come from the fused basis)."""
    def __init__(self, cols, R):
        self.cols = np.asarray(cols); nf = len(cols)
        self.m = RPBL(Af[:, cols], Xr, y, HM=HMm, HV=HVm, Z=A3, R=R)
        L = self.m.nD + J + len(HVm)
        self.free = np.array([i for i in range(L) if not (nf <= i < nf + J)])
        self.L, self.nf = L, nf
    def full(self, z):
        th = np.zeros(self.L); th[self.free] = z; return th
    def llg(self, z):
        ll, g = self.m.loglik_grad(self.full(z)); return ll, g[self.free]
    def nfree(self): return len(self.free)


def penalized(lam, z0):
    """min -LL/N + lam * sum w |step| at RS draws (L-BFGS-B, split positive/negative step coefficients)."""
    W = Wrapped(np.arange(Af.shape[1]), RS)
    pidx = np.where(pen_f)[0]
    uidx = np.array([i for i in range(W.nfree()) if i not in set(pidx)])
    wl = lam * w_f[pidx]
    def unpack(v):
        z = np.zeros(W.nfree()); z[uidx] = v[:len(uidx)]; z[pidx] = v[len(uidx):len(uidx) + len(pidx)] - v[len(uidx) + len(pidx):]; return z
    def fg(v):
        z = unpack(v); ll, g = W.llg(z)
        f = -ll / N + np.sum(wl * (v[len(uidx):len(uidx) + len(pidx)] + v[len(uidx) + len(pidx):]))
        gz = -g / N
        return f, np.concatenate([gz[uidx], gz[pidx] + wl, -gz[pidx] + wl])
    v0 = np.concatenate([z0[uidx], np.maximum(z0[pidx], 0), np.maximum(-z0[pidx], 0)])
    bounds = [(None, None)] * len(uidx) + [(0, None)] * (2 * len(pidx))
    res = minimize(fg, v0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=3000, maxcor=30, ftol=1e-12, gtol=1e-7))
    z = unpack(res.x); z[pidx[np.abs(z[pidx]) < 1e-7]] = 0.0
    return z, W


def structure(z):
    """period groups per variable from the active steps."""
    groups = []
    for k in range(K):
        brk = [int(meta[j][1][4:]) for j in range(len(meta)) if meta[j][0] == k and meta[j][2] and z[j] != 0]
        cuts = [0] + sorted(brk) + [T]
        groups.append([list(range(cuts[i], cuts[i + 1])) for i in range(len(cuts) - 1)])
    return groups


def refit(z):
    """unpenalised refit of the selected structure at RF draws -> LL, k."""
    cols = [j for j in range(Af.shape[1]) if not pen_f[j] or z[j] != 0]
    W = Wrapped(cols, RF)
    zf = np.concatenate([z[cols], z[Af.shape[1]:]])
    res = minimize(lambda v: tuple(-x for x in W.llg(v)), zf, jac=True, method='BFGS', options=dict(maxiter=3000, gtol=1e-6))
    return -res.fun, W.nfree()


# ---- start: no steps, with the random / heterogeneity terms of the Stage-1 model, at RS draws ----
W0 = Wrapped(np.arange(Af.shape[1]), RS)
zstart = np.zeros(W0.nfree())
lev = [j for j in range(Af.shape[1]) if not pen_f[j]]
bl, _, _ = newton(csr_matrix(Af[:, lev]), y); zstart[lev] = bl
nf = Af.shape[1]; zstart[nf + len(HMm):nf + len(HMm) + J] = 0.5          # SD starts
zfix, _ = penalized(1e6, zstart)                                         # very large lambda: all steps zero
ll0, g0 = W0.llg(zfix)
lam_max = float(np.max(np.abs(-g0[np.where(pen_f)[0]] / N) / w_f[pen_f]))
log(f'start (no steps): LL={ll0:.3f}; lambda_max={lam_max:.4e}  ({round(time.time() - t0)} s)')

path, seen = [], {}
def visit(lam, zw):
    z, _ = penalized(lam, zw)
    g = structure(z); key = json.dumps(g)
    if key not in seen:
        ll, k = refit(z); LR, df = 2 * (llU - ll), kU - k
        ok = passes_rule(LR, df); conf = float(chi2.cdf(max(LR, 0), df))
        seen[key] = dict(lam=lam, groups=g, LL=ll, k=k, LR=LR, df=df, conf=conf, passes=ok)
        log(f'  lambda={lam:.4e}: k={k}, LL={ll:.3f}, LR={LR:.1f}, df={df}, conf={conf:.3f}, passes: {ok}  ({round(time.time() - t0)} s)')
    rec = dict(seen[key]); rec['lam'] = lam; path.append(rec)
    return z, rec

hi = lo = None; zcur = zfix; zhi = zfix
for i in range(1, 80):
    lam = lam_max * SHRINK ** i
    zcur, rec = visit(lam, zcur)
    if rec['passes']: lo, zlo = lam, zcur; break
    hi, zhi = lam, zcur
if lo is None: sys.exit('no lambda on the path gives a structure that is not rejected')
best = rec
if hi is not None:
    for it in range(NBISECT):
        mid = np.sqrt(hi * lo); zm, rm = visit(mid, zhi)
        if rm['passes']: lo, zlo, best = mid, zm, rm
        else: hi, zhi = mid, zm
cands, keys = [best], {json.dumps(best['groups'])}
lam, zc = lo, zlo
while len(cands) < 1 + NEXTRA:
    lam *= SHRINK
    if lam < lam_max * SHRINK ** 80: break
    zc, r = visit(lam, zc)
    if json.dumps(r['groups']) not in keys: keys.add(json.dumps(r['groups'])); cands.append(r)
out = dict(subset=SUB, rule=f'sparsest lambda not rejected at {CONF:.0%} against the separate period models before pruning '
           f'(penalised fit {RS} draws, refit {RF} draws)', ref_LL=llU, ref_k=kU, lambda_max=lam_max, chosen=best,
           candidates=cands, path=path, seconds=round(time.time() - t0))
json.dump(out, open(os.path.join(OUT, 'fused_rp_structure.json'), 'w', encoding='utf-8'), indent=1, default=float)
log(f'CHOSEN lambda={best["lam"]:.4e}: k={best["k"]}, LR={best["LR"]:.1f}, df={best["df"]}, conf={best["conf"]:.3f}; '
    f'{len(cands) - 1} step-down candidates  ({round(time.time() - t0)} s)')
