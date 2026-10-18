"""Step 2: Stage-1 temporal structures in the fixed-parameter binary logit (CPU, a few minutes per subset).

For each subset: separate-period logits (fully unconstrained), partially constrained structure (LR tests, 90%),
adaptive fused structure (90% rule) and the spline-indicator model (90%).

usage: python pipeline/s2_stage1_structures.py [S1 S2 S3 S4]
writes results/stage1/<subset>.json (Table 2), results/stage1/struct_<subset>_pc.json and struct_<subset>_fused.json
       (the structures used by the random parameters models)
"""
import os, sys, json, time
import _paths  # noqa: F401
import numpy as np
from settings import RESULTS, SUBSETS, subset_file
from stage1 import period_logits, pc_structure, fused_structure, spline_model, structure_fit, lr_conf

OUT = RESULTS / 'stage1'; OUT.mkdir(parents=True, exist_ok=True)
subs = sys.argv[1:] or list(SUBSETS)


def stats(k, ll, N, llU=None, kU=None):
    d = dict(k=int(k), LL=float(ll), AIC=-2 * ll + 2 * k, BIC=-2 * ll + np.log(N) * k)
    if llU is not None:
        LR, df, cf = lr_conf(ll, k, llU, kU); d.update(LR=LR, df=int(df), conf=cf)
    return d


for sub in subs:
    t0 = time.time(); T = SUBSETS[sub]['T']
    Z = np.load(subset_file(sub), allow_pickle=True); X, y, t, names = Z['X'], Z['y'], Z['t'], [str(n) for n in Z['names']]
    K, N = X.shape[1], len(y)
    U = period_logits(X, y, t, T); llU, kU = U['LL'], U['k']
    pc_groups, _ = pc_structure(X, y, t, T); k_pc, ll_pc = structure_fit(X, y, t, pc_groups)
    print(f'{sub}: period logits k={kU}, LL={llU:.3f}; partially constrained k={k_pc}, LL={ll_pc:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    F = fused_structure(X, y, t, T, llU, kU)
    print(f'{sub}: adaptive fused lambda={F["lam"]:.4e}, k={F["k"]}, LL={F["LL"]:.3f}, conf={F["conf"]:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    S = spline_model(X, y, t, T)
    print(f'{sub}: spline k={S["k"]}, LL={S["LL"]:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    n_vary = lambda G: int(sum(len(g) > 1 for g in G))
    res = dict(subset=sub, N=N, K=K, T=T, names=names,
               uncon=dict(stats(kU, llU, N), varying=K),
               pc=dict(stats(k_pc, ll_pc, N, llU, kU), varying=n_vary(pc_groups), groups=pc_groups),
               fused=dict(stats(F['k'], F['LL'], N, llU, kU), varying=n_vary(F['groups']), lam=F['lam'], lam_max=F['lam_max'],
                          groups=F['groups'], path=F['path']),
               spline=dict(stats(S['k'], S['LL'], N, llU, kU), varying=int(K - (np.abs(np.diff(S['paths'], axis=1)).max(1) < 1e-8).sum()),
                           proportional=int(sum(len(kp) == 1 for kp in S['keep'])), keep=S['keep']),
               seconds=round(time.time() - t0))
    json.dump(res, open(OUT / f'{sub}.json', 'w', encoding='utf-8'), indent=1, default=float)
    for app in ['pc', 'fused']:                       # written atomically: step 3 may be waiting for these files
        tmp = OUT / f'struct_{sub}_{app}.json.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(dict(subset=sub, approach=app, stage='fixed-parameter logit', groups=res[app]['groups'], k=res[app]['k'],
                           LL=res[app]['LL'], conf=res[app]['conf'], **({'lam': F['lam']} if app == 'fused' else {})),
                      f, indent=1, default=float)
        os.replace(tmp, OUT / f'struct_{sub}_{app}.json')
print('Stage 1 done:', ', '.join(subs))
