"""GPU vs CPU: does the PyTorch backend compute the same simulated log-likelihood and gradient as the NumPy backend?

Evaluates both backends on S1 (partially constrained Stage-1 structure, four random parameters, two heterogeneity-in-mean
and one heterogeneity-in-variance terms) at three parameter vectors, with 2,000 Halton draws. Pass = |LL difference| < 1e-6
and max |gradient difference| < 1e-4 at every point. Run it before the random parameters models (after steps 1 and 2).

usage: python validation/parity_check.py
"""
import time
import numpy as np
from check_data import check_data_4
from rpbl import RPBL
from rpbl_torch import RPBLTorch, DEVICE
import torch

A, y, cn = check_data_4()
rnd = [cn.index(n) for n in ['songkran', 'new_year [2018-19]', 'Northeastern [2016-21]', 'concrete_surface [2018-23]']]
z1, z2, z3 = cn.index('Eastern [2022-23]'), cn.index('Southern'), cn.index('speeding [2020-23]')
HM, HV = [(0, z2), (3, z1)], [(2, z3)]
moved = {z1, z2, z3}; fx = [j for j in range(A.shape[1]) if j not in rnd and j not in moved]
args = (A[:, fx], A[:, rnd], y); kw = dict(HM=HM, HV=HV, Z=A, R=2000)
print(f'torch {torch.__version__}, device {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''))
cpu, gpu = RPBL(*args, **kw), RPBLTorch(*args, **kw)
rng = np.random.default_rng(1); ok = True
for i in range(3):
    th = np.concatenate([rng.normal(0, 0.3, cpu.nD), rng.uniform(0.2, 1.5, len(rnd)), rng.normal(0, 0.3, len(HV))])
    t0 = time.perf_counter(); llc, gc = cpu.loglik_grad(th); tc = time.perf_counter() - t0
    gpu.loglik_grad(th); t0 = time.perf_counter(); llg, gg = gpu.loglik_grad(th)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    tg = time.perf_counter() - t0
    dll, dg = abs(llc - llg), float(np.abs(gc - gg).max()); passed = dll < 1e-6 and dg < 1e-4; ok &= passed
    print(f'point {i + 1}: LL cpu {llc:.8f}, gpu {llg:.8f}, |dLL| {dll:.2e}, max |dgrad| {dg:.2e}, '
          f'time cpu {tc:.2f} s, gpu {tg:.3f} s -> {"ok" if passed else "FAILED"}')
print('PARITY CHECK', 'PASSED' if ok else 'FAILED')
