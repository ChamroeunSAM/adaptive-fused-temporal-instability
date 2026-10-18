"""Random parameters binary logit with heterogeneity in means and variances (simulated maximum likelihood).

Utility of the severe outcome for crash i and draw r:
    U_ir = Xf_i b + sum_j Xr_ij * beta_ijr
    beta_ijr = mu_j + delta_j' Z_ij + sigma_j * exp(omega_j' W_ij) * nu_ijr,   nu ~ N(0,1) (Halton)
P(y_i) = (1/R) sum_r Lambda(s_i U_ir), s_i = 2 y_i - 1.

Xr_ij is a period-group column (x_ij * 1[period in group]), so heterogeneity terms Z, W (raw indicators) only act on
crashes of that parameter's own period group: the period rule holds by construction.
Analytic gradient; standard errors from the Hessian obtained by central differences of the analytic gradient.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

R_DEFAULT = 1000


_CACHE = {}
PRIMES = [p for p in range(2, 1300) if all(p % d for d in range(2, int(p ** 0.5) + 1))][:200]   # first 200 primes
assert PRIMES[:25] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def radical_inverse(p, n0, n):
    """Van der Corput sequence in base p for indices n0 .. n0+n-1 (standard, unscrambled Halton coordinate)."""
    i = np.arange(n0, n0 + n, dtype=np.int64); out = np.zeros(n); f = 1.0 / p
    while i.any():
        out += f * (i % p); i //= p; f /= p
    return out


def halton_normal(N, R, d, skip=10):
    """Standard Halton draws: dimension j uses the j-th prime, first `skip` points discarded, observation i uses
    points i*R .. i*R+R-1. Returns float32 array (d, N, R) of standard normal draws (cached per dimension)."""
    out = np.zeros((d, N, R), np.float32)
    for j in range(d):
        key = (N, R, skip, j)
        if key not in _CACHE:
            _CACHE[key] = norm.ppf(radical_inverse(PRIMES[j], skip, N * R)).astype(np.float32).reshape(N, R)
        out[j] = _CACHE[key]
    return out


class RPBL:
    def __init__(self, Xf, Xr, y, HM=(), HV=(), Z=None, R=R_DEFAULT, draws=None, names_f=None, names_r=None,
                 names_z=None, chunk=2000):
        """Xf (N,Kf) fixed columns; Xr (N,J) random columns; HM/HV: lists of (j, col index in Z);
        Z (N,Q) raw heterogeneity candidate indicators."""
        self.Xf, self.Xr, self.y = np.asarray(Xf, float), np.asarray(Xr, float), np.asarray(y, float)
        self.N, self.Kf = self.Xf.shape; self.J = self.Xr.shape[1]; self.R = R
        self.s = 2 * self.y - 1
        self.HM, self.HV = list(HM), list(HV); self.Z = Z
        self.Hm = np.column_stack([self.Xr[:, j] * Z[:, q] for j, q in self.HM]) if self.HM else np.zeros((self.N, 0))
        self.D = np.hstack([self.Xf, self.Xr, self.Hm])           # columns entering the utility linearly
        self.nD = self.D.shape[1]; self.nHV = len(self.HV)
        self.nu = halton_normal(self.N, R, self.J) if draws is None else draws
        self.chunk = chunk
        self.ridx = np.where(np.any(self.Xr != 0, axis=1))[0]          # rows needing simulation
        self.fidx = np.setdiff1d(np.arange(self.N), self.ridx)
        self.nu_r = np.ascontiguousarray(self.nu[:, self.ridx, :])
        nz = names_z or [f'z{q}' for q in range(0 if Z is None else Z.shape[1])]
        self.names = ([*(names_f or [f'f{k}' for k in range(self.Kf)])] +
                      [f'mean:{n}' for n in (names_r or [f'r{j}' for j in range(self.J)])] +
                      [f'HM:{(names_r or [f"r{j}" for j in range(self.J)])[j]}|{nz[q]}' for j, q in self.HM] +
                      [f'SD:{n}' for n in (names_r or [f'r{j}' for j in range(self.J)])] +
                      [f'HV:{(names_r or [f"r{j}" for j in range(self.J)])[j]}|{nz[q]}' for j, q in self.HV])

    def unpack(self, th):
        a = self.nD; lin = th[:a]; sig = th[a:a + self.J]; om = th[a + self.J:]
        return lin, sig, om

    def _scale(self, sig, om, rows):
        """exp(omega' W_ij) for the rows in the chunk -> (J, n); sigma_ij = sigma_j * this."""
        E = np.ones((self.J, len(rows)))
        for h, (j, q) in enumerate(self.HV):
            E[j] *= np.exp(om[h] * self.Z[rows, q])
        return E

    def loglik_grad(self, th, per_obs=False):
        lin, sig, om = self.unpack(th)
        g = np.zeros_like(th); G = np.zeros((self.N, len(th))) if per_obs else None
        lli = np.zeros(self.N)
        # rows without any random column: ordinary logit
        f = self.fidx
        if len(f):
            v = self.D[f] @ lin; sv = self.s[f] * v
            lli[f] = -np.logaddexp(0, -sv); wsum = self.s[f] * expit(-sv)
            gl = self.D[f] * wsum[:, None]; g[:self.nD] += gl.sum(0)
            if per_obs: G[f, :self.nD] = gl
        nr = len(self.ridx)
        for c0 in range(0, nr, self.chunk):
            loc = slice(c0, min(c0 + self.chunk, nr)); rows = self.ridx[loc]
            nu = self.nu_r[:, loc, :]
            v = self.D[rows] @ lin                                   # (n,)
            E = self._scale(sig, om, rows); S = sig[:, None] * E    # (J, n)
            XS = self.Xr[rows].T * S                                # (J, n)
            U = v[:, None] + np.einsum('jn,jnr->nr', XS, nu, optimize=True)
            s = self.s[rows][:, None]
            L = expit(s * U)                                        # (n, R)
            P = L.mean(1); lli[rows] = np.log(np.maximum(P, 1e-300))
            w = (L * (1 - L) * s) / (self.R * np.maximum(P, 1e-300)[:, None])   # d logP / dU_ir
            wsum = w.sum(1)
            wnu = np.einsum('nr,jnr->jn', w, nu, optimize=True)     # (J, n)
            gl = self.D[rows] * wsum[:, None]
            gs = (self.Xr[rows].T * E * wnu).T                       # d/d sigma_j
            gom = np.zeros((len(rows), self.nHV))
            for h, (j, q) in enumerate(self.HV):
                gom[:, h] = XS[j] * self.Z[rows, q] * wnu[j]
            gi = np.hstack([gl, gs, gom]); g += gi.sum(0)
            if per_obs: G[rows] = gi
        ll = lli.sum()
        return (ll, g, G, lli) if per_obs else (ll, g)

    def fit(self, th0, maxiter=2000, gtol=1e-6):
        f = lambda th: tuple(-x for x in self.loglik_grad(th))
        res = minimize(f, th0, jac=True, method='BFGS', options=dict(maxiter=maxiter, gtol=gtol))
        self.th, self.LL, self.res = res.x, -res.fun, res
        return self

    def hessian_se(self, eps=1e-5):
        th = self.th; k = len(th); H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k); e[i] = eps * max(1, abs(th[i]))
            H[:, i] = (self.loglik_grad(th + e)[1] - self.loglik_grad(th - e)[1]) / (2 * e[i])
        H = (H + H.T) / 2
        self.cov = np.linalg.pinv(-H); self.se = np.sqrt(np.maximum(np.diag(self.cov), 0))
        return self.se

    def start(self, b_fixed_logit, sd0=0.1):
        """b_fixed_logit: coefficients of [Xf, Xr] from a fixed-parameter logit."""
        return np.concatenate([b_fixed_logit, np.zeros(len(self.HM)), np.full(self.J, sd0), np.zeros(self.nHV)])

    def table(self):
        import pandas as pd
        t = self.th / np.where(self.se > 0, self.se, np.nan)
        return pd.DataFrame(dict(parameter=self.names, estimate=self.th, se=self.se, t=t))

    def predict_proba(self, Xf, Xr, Z=None, draws=None):
        """Unconditional probability of the severe outcome, simulated over the full random-parameter distribution."""
        lin, sig, om = self.unpack(self.th)
        N = Xf.shape[0]
        Hm = np.column_stack([Xr[:, j] * Z[:, q] for j, q in self.HM]) if self.HM else np.zeros((N, 0))
        D = np.hstack([Xf, Xr, Hm]); nu = halton_normal(N, self.R, self.J) if draws is None else draws
        out = np.zeros(N)
        for c0 in range(0, N, self.chunk):
            rows = np.arange(c0, min(c0 + self.chunk, N))
            S = np.repeat(sig[:, None], len(rows), axis=1)
            for h, (j, q) in enumerate(self.HV): S[j] *= np.exp(om[h] * Z[rows, q])
            U = (D[rows] @ lin)[:, None] + np.einsum('jn,jnr->nr', Xr[rows].T * S, nu[:, rows, :], optimize=True)
            out[rows] = expit(U).mean(1)
        return out
