"""Stage 1: temporal structures in the fixed-parameter binary logit (Section 2.2 and Table 2 of the paper).

  period_logits : fully unconstrained structure, a separate parameter for every variable in every period
  pc_structure  : partially constrained structure, greedy merging of adjacent period groups by likelihood ratio tests
  fused_structure: adaptive fused lasso, sparsest lambda whose structure is not rejected at 90% against the period logits
  spline_model  : spline-indicator structure of Shabab et al. (2024), backward elimination at 90%
"""
import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2, norm
from scipy.special import expit
from settings import CONF, FUSED_SHRINK, FUSED_NBISECT
from temporal import (build, spec_year, spec_fused, f_spline, newton, lasso_logit, implied_paths, groups_from_paths)


def period_logits(X, y, t, T):
    K = X.shape[1]
    A, meta = build(X, t, spec_year(K, T)); b, ll, cov = newton(A, y)
    return dict(k=A.shape[1], LL=ll, b=b, cov=cov, meta=meta)


def lr_conf(ll, k, llU, kU):
    LR, df = 2 * (llU - ll), kU - k
    return LR, df, float(chi2.cdf(max(LR, 0), df)) if df > 0 else 0.0


def pc_structure(X, y, t, T, conf=CONF):
    """Partially constrained structure by likelihood ratio tests: starting from period-specific parameters, in each round
    every variable's adjacent pair of period groups whose merging reduces the log-likelihood least is merged if
    LR = 2[LL(current) - LL(merged)] < chi2_conf(1); the model is re-estimated and merging continues until no pair of any
    variable can be merged. Returns (groups, LL of the final structure)."""
    K = X.shape[1]; crit = chi2.ppf(conf, 1)
    groups = [[[s] for s in range(T)] for _ in range(K)]
    cache = {}
    def col(k, g):                                    # sparse column x_k * 1[t in g], built once
        key = (k, tuple(g))
        if key not in cache: cache[key] = sp.csc_matrix((X[:, k] * np.isin(t, g))[:, None])
        return cache[key]
    def design(G): return sp.hstack([col(k, g) for k in range(K) for g in G[k]], format='csr')
    while True:
        b, llU, _ = newton(design(groups), y); starts, pos, merged, new = [], 0, 0, []
        for k in range(K): starts.append(pos); pos += len(groups[k])
        for k in range(K):
            G = groups[k]; best, bj = None, None
            for j in range(len(G) - 1):
                Gc = [list(g) for g in groups]; Gc[k] = G[:j] + [G[j] + G[j + 1]] + G[j + 2:]
                b0 = np.delete(b.copy(), starts[k] + j + 1); b0[starts[k] + j] = 0.5 * (b[starts[k] + j] + b[starts[k] + j + 1])
                _, llC, _ = newton(design(Gc), y, b0=b0); lr = 2 * (llU - llC)
                if best is None or lr < best: best, bj = lr, j
            if best is not None and best < crit: G = G[:bj] + [G[bj] + G[bj + 1]] + G[bj + 2:]; merged += 1
            new.append(G)
        if merged == 0: return groups, llU
        groups = new


class AdaptiveFused:
    """Adaptive fused lasso on the level + step basis; weights w = 1 / |step of the period logits|."""
    def __init__(self, X, y, t, T):
        self.X, self.y, self.t, self.T, self.K = X, y, t, T, X.shape[1]
        self.A, self.meta = build(X, t, spec_fused(self.K, T)); self.Ay, self.my = build(X, t, spec_year(self.K, T))
        self.pen = np.array([m[2] for m in self.meta]); self.iu = np.where(~self.pen)[0]
        by, _, _ = newton(self.Ay, y); Py = implied_paths(self.my, by, self.K, T); self.w = np.ones(self.A.shape[1])
        for j, (k, lab, p_, f) in enumerate(self.meta):
            if p_: s = int(lab[4:]); self.w[j] = 1 / max(abs(Py[k, s] - Py[k, s - 1]), 1e-4)

    def fit(self, lam, x0=None):
        """Penalised fit at lam, then unpenalised refit of the selected structure. Returns bp, refit b, k, LL."""
        A, y = self.A, self.y
        if x0 is None:
            bu, _, _ = newton(A[:, self.iu], y); x0 = np.zeros(A.shape[1]); x0[self.iu] = bu
        bp = lasso_logit(A, y, self.pen, lam, self.w, x0=x0); act = np.where((~self.pen) | (bp != 0))[0]
        br, ll, _ = newton(A[:, act], y, b0=bp[act]); bb = np.zeros(A.shape[1]); bb[act] = br
        return bp, bb, len(act), ll

    def groups(self, bb):
        return groups_from_paths(implied_paths(self.meta, bb, self.K, self.T), self.T)


def fused_structure(X, y, t, T, llU, kU, conf=CONF):
    """Sparsest lambda whose structure is not rejected at `conf` against the period logits (LR < chi2_conf(df)):
    lambda = lambda_max * 0.75^i until the first structure not rejected, then bisection between the last rejected and the
    first accepted lambda. Returns a dict with lambda, groups, k, LL, LR, df, conf and the path."""
    m = AdaptiveFused(X, y, t, T); N = len(y)
    bu, _, _ = newton(m.A[:, m.iu], y); b0 = np.zeros(m.A.shape[1]); b0[m.iu] = bu
    g = m.A.T @ (expit(m.A @ b0) - y) / N; lam_max = float(np.max(np.abs(g[m.pen]) / m.w[m.pen]))
    path = []
    def rec(lam, k, ll):
        LR, df, cf = lr_conf(ll, k, llU, kU); path.append(dict(lam=lam, k=k, LL=ll, LR=LR, df=df, conf=cf)); return cf
    hi, lo, x0, acc, hix = None, None, b0, None, None
    for i in range(1, 40):
        lam = lam_max * FUSED_SHRINK ** i
        bp, bb, k, ll = m.fit(lam, x0); x0 = bp
        if rec(lam, k, ll) < conf: lo, acc = lam, (lam, bb, k, ll); break
        hi, hix = lam, bp
    if acc is None: raise RuntimeError('no lambda on the path gives a structure that is not rejected')
    if hi is not None:
        x0 = hix
        for it in range(FUSED_NBISECT):
            mid = np.sqrt(hi * lo); bp, bb, k, ll = m.fit(mid, x0)
            if rec(mid, k, ll) < conf: lo, acc = mid, (mid, bb, k, ll)
            else: hi, x0 = mid, bp
    lam, bb, k, ll = acc; LR, df, cf = lr_conf(ll, k, llU, kU)
    return dict(lam=lam, lam_max=lam_max, groups=m.groups(bb), k=k, LL=ll, LR=LR, df=df, conf=cf, path=path)


def spline_model(X, y, t, T, conf=CONF):
    """Spline-indicator model: every variable starts with all terms f_j, j = 1..T; in each round the least significant
    term j > 1 of every variable is removed if |t| < z_(1-(1-conf)/2); repeated until no term is removed."""
    K = X.shape[1]; zc = norm.ppf(1 - (1 - conf) / 2)
    keep = [list(range(1, T + 1)) for _ in range(K)]
    while True:
        spec = [(k, f'sp{j}', f_spline(j), j > 1) for k in range(K) for j in keep[k]]
        A, meta = build(X, t, spec); b, ll, cov = newton(A, y)
        ts = b / np.sqrt(np.maximum(np.diag(cov), 1e-16)); pos, dropped = 0, 0
        for k in range(K):
            js = keep[k]; idx = list(range(pos, pos + len(js))); pos += len(js)
            cand = [(abs(ts[i]), j) for i, j in zip(idx, js) if j > 1]
            if cand:
                mn, j = min(cand)
                if mn < zc: keep[k] = [x for x in js if x != j]; dropped += 1
        if dropped == 0: break
    P = implied_paths(meta, b, K, T)
    return dict(keep=keep, k=A.shape[1], LL=ll, b=b, cov=cov, meta=meta, paths=P)


def structure_fit(X, y, t, groups):
    """Fixed-parameter logit on a temporal structure. Returns k, LL."""
    K = X.shape[1]
    A = sp.csr_matrix(np.column_stack([X[:, k] * np.isin(t, g) for k in range(K) for g in groups[k]]))
    b, ll, _ = newton(A, y)
    return A.shape[1], ll
