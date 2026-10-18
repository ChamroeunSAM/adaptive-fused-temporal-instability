"""GPU (PyTorch) backend for the random parameters binary logit with heterogeneity in means and variances.

Same model, same Halton draws, same analytic gradient and same optimiser (scipy BFGS) as rpbl.py; only the
log-likelihood/gradient and the out-of-sample simulation run on the GPU, in float64. Results therefore match the CPU
code up to floating-point rounding (summation order differs on the GPU).
"""
import os
import numpy as np
import torch
from rpbl import RPBL

DEVICE = torch.device(os.environ.get('RPBL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
DT = torch.float64
CHUNK = int(os.environ.get('RPBL_GPU_CHUNK', '20000'))   # rows simulated per GPU batch
HOST_DRAWS = os.environ.get('RPBL_DRAWS_HOST', '0') == '1'   # keep the Halton draws in (pinned) CPU memory, copy each batch to the GPU


_GCACHE = {}


def gpu_draws(N, R, j, skip=10):
    """Standard-normal Halton draws for dimension j, built exactly as rpbl.halton_normal (float32) but kept on the GPU."""
    key = (N, R, skip, j)
    if key not in _GCACHE:
        from rpbl import radical_inverse, PRIMES
        from scipy.stats import norm
        a = norm.ppf(radical_inverse(PRIMES[j], skip, N * R)).astype(np.float32).reshape(N, R)
        t_ = torch.from_numpy(a)
        _GCACHE[key] = t_.pin_memory() if HOST_DRAWS and DEVICE.type == 'cuda' else t_.to(DEVICE) if not HOST_DRAWS else t_
    return _GCACHE[key]


def clear_cache():
    _GCACHE.clear()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()


def _t(a, dtype=DT):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


class RPBLTorch(RPBL):
    def __init__(self, *args, **kw):
        kw.setdefault('chunk', CHUNK)
        user_draws = kw.get('draws')
        Xr_ = np.asarray(args[1]); N_, J_ = Xr_.shape
        if user_draws is None: kw['draws'] = np.zeros((J_, N_, 1), np.float32)   # host draws not needed: GPU cache
        super().__init__(*args, **kw)
        self.tD, self.ts = _t(self.D), _t(self.s)
        self.tXr = _t(self.Xr)
        self.tf = torch.as_tensor(self.fidx, device=DEVICE, dtype=torch.long)
        self.tr = torch.as_tensor(self.ridx, device=DEVICE, dtype=torch.long)
        if user_draws is None:                                           # per-dimension draws, gathered by row on the fly
            self.gnu = [gpu_draws(self.N, self.R, j) for j in range(self.J)]
        else:
            self.gnu = [_t(user_draws[j], torch.float32) for j in range(self.J)]
        self.tW = [_t(self.Z[:, q]) for _, q in self.HV]                  # variance-heterogeneity columns
        del self.nu_r, self.nu

    def loglik_grad(self, th, per_obs=False):
        if per_obs: raise NotImplementedError('per-observation scores are only available in the CPU backend')
        th_t = _t(th); a = self.nD
        lin, sig, om = th_t[:a], th_t[a:a + self.J], th_t[a + self.J:]
        g = torch.zeros(len(th), dtype=DT, device=DEVICE); ll = torch.zeros((), dtype=DT, device=DEVICE)
        if len(self.fidx):
            Df, sf = self.tD[self.tf], self.ts[self.tf]
            sv = sf * (Df @ lin)
            ll = ll - torch.logaddexp(torch.zeros_like(sv), -sv).sum()
            g[:a] += Df.T @ (sf * torch.sigmoid(-sv))
        nr = len(self.ridx); R = self.R
        for c0 in range(0, nr, self.chunk):
            c1 = min(c0 + self.chunk, nr); rows = self.tr[c0:c1]
            D, s = self.tD[rows], self.ts[rows][:, None]
            Xr = self.tXr[rows].T                                         # (J, n)
            E = torch.ones((self.J, c1 - c0), dtype=DT, device=DEVICE)
            for h, (j, q) in enumerate(self.HV):
                E[j] = E[j] * torch.exp(om[h] * self.tW[h][rows])
            XS = Xr * sig[:, None] * E                                    # (J, n)
            U = (D @ lin)[:, None].expand(-1, R).clone()
            NU = ([self.gnu[j][torch.as_tensor(self.ridx[c0:c1])].to(DEVICE, non_blocking=True) for j in range(self.J)] if HOST_DRAWS
                  else [self.gnu[j][rows] for j in range(self.J)])               # (n, R) float32 each
            for j in range(self.J):
                U += XS[j][:, None] * NU[j].to(DT)
            L = torch.sigmoid(s * U)
            P = torch.clamp(L.mean(1), min=1e-300)
            ll = ll + torch.log(P).sum()
            w = L * (1 - L) * s / (R * P[:, None])
            g[:a] += D.T @ w.sum(1)
            wnu = torch.stack([(w * NU[j].to(DT)).sum(1) for j in range(self.J)]) if self.J else None
            if self.J:
                g[a:a + self.J] += (Xr * E * wnu).sum(1)
            for h, (j, q) in enumerate(self.HV):
                g[a + self.J + h] += (XS[j] * self.tW[h][rows] * wnu[j]).sum()
        return float(ll.item()), g.cpu().numpy()

    def predict_proba(self, Xf, Xr, Z=None, draws=None):
        from rpbl import halton_normal
        lin, sig, om = (_t(x) for x in self.unpack(self.th))
        N = Xf.shape[0]
        Hm = np.column_stack([Xr[:, j] * Z[:, q] for j, q in self.HM]) if self.HM else np.zeros((N, 0))
        D = _t(np.hstack([Xf, Xr, Hm])); tXr = _t(Xr)
        nu = halton_normal(N, self.R, self.J) if draws is None else draws
        out = np.zeros(N)
        for c0 in range(0, N, self.chunk):
            c1 = min(c0 + self.chunk, N)
            S = sig[:, None].expand(-1, c1 - c0).clone()
            for h, (j, q) in enumerate(self.HV): S[j] = S[j] * torch.exp(om[h] * _t(Z[c0:c1, q]))
            U = (D[c0:c1] @ lin)[:, None].expand(-1, self.R).clone()
            for j in range(self.J):
                U += (tXr[c0:c1, j] * S[j])[:, None] * _t(nu[j, c0:c1], torch.float32).to(DT)
            out[c0:c1] = torch.sigmoid(U).mean(1).cpu().numpy()
        return out
