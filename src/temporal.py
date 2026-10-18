"""Temporal bases, the fixed-parameter binary logit (Newton-Raphson), the L1-penalised logit and the period-group design.

Notation: T periods, t = 0..T-1. Each variable k (incl. the constant) enters through basis columns X_k * f(t):
  separate periods : f = 1[t = s], s = 0..T-1
  adaptive fused   : level f = 1 plus steps f = 1[t >= s], s = 1..T-1 (penalised), so a step is the change between
                     adjacent periods
  spline indicator : f_j(t) = max(t + 2 - j, 0), j = 1..T (Shabab et al., 2024); f_1 always kept
A temporal structure is a list, per variable, of groups of adjacent periods sharing one parameter.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit
from settings import PERIOD_LABELS


def f_year(s):   return lambda t: (t == s).astype(float)
def f_level():   return lambda t: np.ones_like(t, dtype=float)
def f_step(s):   return lambda t: (t >= s).astype(float)
def f_spline(j): return lambda t: np.maximum(t + 2 - j, 0).astype(float)


def build(X, t, spec):
    """spec: list of (k, label, f, penalised). Returns a sparse design and the column metadata."""
    Xs = sp.csc_matrix(X)
    cols, meta = [], []
    for k, lab, f, pen in spec:
        cols.append(Xs[:, k].multiply(f(t)[:, None]))
        meta.append((k, lab, pen, f))
    A = sp.hstack(cols, format='csr'); A.eliminate_zeros()
    return A, meta


def spec_year(K, T):   return [(k, f'y{s}', f_year(s), False) for k in range(K) for s in range(T)]
def spec_fused(K, T):  return [(k, 'level', f_level(), False) for k in range(K)] + \
                              [(k, f'step{s}', f_step(s), True) for k in range(K) for s in range(1, T)]


def newton(A, y, b0=None, tol=1e-9, maxit=50):
    """Unpenalised binary logit by Newton-Raphson with step halving. Returns b, LL, covariance."""
    n, p = A.shape
    b = np.zeros(p) if b0 is None else b0.copy()
    At = A.T.tocsr()
    for it in range(maxit):
        eta = A @ b
        mu = expit(eta)
        g = At @ (y - mu)
        W = mu * (1 - mu)
        H = (At @ sp.diags(W) @ A).toarray()
        H[np.diag_indices_from(H)] += 1e-10
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        ll_old = loglik(A, y, b)
        a = 1.0
        while a > 1e-4:
            bn = b + a * step
            if loglik(A, y, bn) >= ll_old - 1e-10: break
            a /= 2
        b = bn
        if np.max(np.abs(a * step)) < tol * 10 or abs(g @ step) < tol: break
    eta = A @ b; mu = expit(eta)
    H = (At @ sp.diags(mu * (1 - mu)) @ A).toarray()
    cov = np.linalg.pinv(H)
    return b, loglik(A, y, b), cov


def loglik(A, y, b):
    eta = A @ b
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def lasso_logit(A, y, pen, lam, w=None, x0=None, maxiter=5000):
    """min -LL/n + lam * sum_j w_j |b_j| over the penalised columns (L-BFGS-B, split positive/negative parts)."""
    n, p = A.shape
    pen = np.asarray(pen, bool)
    w = np.ones(p) if w is None else w
    iu, ip = np.where(~pen)[0], np.where(pen)[0]
    nu, npn = len(iu), len(ip)
    At = A.T.tocsr()
    wl = lam * w[ip]

    def unpack(z):
        b = np.zeros(p)
        b[iu] = z[:nu]
        b[ip] = z[nu:nu + npn] - z[nu + npn:]
        return b

    def fg(z):
        b = unpack(z)
        eta = A @ b
        f = -(np.sum(y * eta - np.logaddexp(0, eta))) / n
        r = (expit(eta) - y) / n
        gb = At @ r
        f += np.sum(wl * (z[nu:nu + npn] + z[nu + npn:]))
        g = np.concatenate([gb[iu], gb[ip] + wl, -gb[ip] + wl])
        return f, g

    if x0 is None:
        z0 = np.zeros(nu + 2 * npn)
    else:
        z0 = np.concatenate([x0[iu], np.maximum(x0[ip], 0), np.maximum(-x0[ip], 0)])
    bounds = [(None, None)] * nu + [(0, None)] * (2 * npn)
    res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': maxiter, 'maxcor': 30, 'ftol': 1e-13, 'gtol': 1e-9})
    b = unpack(res.x)
    small = np.zeros(p, bool); small[ip] = np.abs(b[ip]) < 1e-7
    b[small] = 0
    return b


def implied_paths(meta, b, K, T):
    """K x T matrix of the period parameters beta_k(t) implied by basis coefficients b."""
    tt = np.arange(T)
    P = np.zeros((K, T))
    for (k, lab, pen, f), bj in zip(meta, b):
        P[k] += bj * f(tt)
    return P


def groups_from_paths(P, T, tol=1e-9):
    """Temporal structure (groups of adjacent periods with equal parameters) from a K x T path matrix."""
    groups = []
    for k in range(P.shape[0]):
        G = [[0]]
        for s in range(1, T):
            if abs(P[k, s] - P[k, s - 1]) > tol: G.append([s])
            else: G[-1].append(s)
        groups.append(G)
    return groups


def group_design(X, t, names, groups, T):
    """Dense design with one column per variable and period group; column names carry the period span in brackets."""
    lab = PERIOD_LABELS[T]; cols, cn, var = [], [], []
    for k, G in enumerate(groups):
        for g in G:
            cols.append(X[:, k] * np.isin(t, g))
            span = lab[g[0]] if len(g) == 1 else f'{lab[g[0]].split("-")[0]}-{lab[g[-1]].split("-")[-1]}' if T == 4 else f'{lab[g[0]]}-{lab[g[-1]]}'
            cn.append(names[k] if len(G) == 1 else f'{names[k]} [{span}]'); var.append(k)
    return np.column_stack(cols), cn, np.array(var)
