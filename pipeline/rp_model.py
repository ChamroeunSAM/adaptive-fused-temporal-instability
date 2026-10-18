"""Random parameters binary logit with heterogeneity in means and variances on a given temporal structure
(Section 2.3 of the paper). Called by s3_stage2_models.py; one model per call.

mode search (specification search at 100 Halton draws, then confirmation at 2,000):
  1. random parameters: each non-constant period-group parameter tested alone (normal, SD |t| > 1.645); the significant
     ones are ranked by their log-likelihood gain and the top 16 enter a joint model; the least significant SD is dropped
     until all |t| > 1.645 and no more than the cap remain. Each joint model is estimated from three starts (previous
     joint estimates, all SDs = 0.1, all SDs = 1.0) and the best is kept.
  2. heterogeneity in means: candidates = non-significant fixed parameters (|t| < 1.645, constants excluded). A candidate
     column enters the mean of ONE random parameter whose periods overlap (term = x_random * candidate column, so only
     crashes of the shared periods count) and leaves the main utility. Kept if |t| > 1.645 and LL gain > 1.353
     (LR > 2.706, 1 df). Best LL gain added first; candidates re-evaluated after each addition.
  3. heterogeneity in variances: same rule, the candidate enters exp(omega * z) of the SD.
  confirmation: the specification is re-estimated at 2,000 draws; heterogeneity terms and SDs with |t| < 1.645 are
  removed (the variable returns to the main utility as a fixed parameter) and the model is re-estimated once.
mode prune (at 2,000 draws, starting from a search result): non-significant parameters are removed until every
  remaining parameter is significant at 90%: each round removes every term with |t| < 1.0, or else the single least
  significant term; a non-significant fixed parameter is dropped, a non-significant SD makes the parameter fixed (its
  heterogeneity variables return to the main utility), a non-significant heterogeneity term returns its variable to the
  main utility. Constants are always kept; a random parameter is kept while its SD is significant.

usage: python pipeline/rp_model.py --subset S1 --tag pc_s1_search --mode search --structure results/stage1/struct_S1_pc.json
       python pipeline/rp_model.py --subset S1 --tag pc_s1 --mode prune --start pc_s1_search --structure ...
       python pipeline/rp_model.py --subset S2 --period 3 --tag uncon_p3_search --mode search      (separate period model)
writes <out>/<tag>.json (summary), <tag>_fit.json (specification and estimates), <tag>_estimates.csv, <tag>.log
"""
import os, sys, json, time, argparse
import _paths  # noqa: F401
import numpy as np
import scipy.sparse as sp
from settings import (ROOT, SUBSETS, PERIOD_LABELS, RESULTS, DRAWS_SEARCH, DRAWS_FINAL, Z_CRIT, LL_GAIN, CAPS_MAIN, CAPS_PERIOD,
                      subset_file)
from temporal import newton, group_design

ap = argparse.ArgumentParser()
ap.add_argument('--subset', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--mode', choices=['search', 'prune'], required=True)
ap.add_argument('--structure', help='JSON file with "groups" (temporal structure); omit for a separate period model')
ap.add_argument('--period', type=int, help='period index for a separate period model')
ap.add_argument('--start', help='prune mode: tag of the search result to start from')
ap.add_argument('--out', help='output folder (default results/rp/<subset>)')
ap.add_argument('--backend', default='torch', choices=['torch', 'numpy'])
a = ap.parse_args()

if a.backend == 'torch':
    from rpbl_torch import RPBLTorch as RPBL, DEVICE
else:
    from rpbl import RPBL
    DEVICE = 'cpu (numpy)'
RS = int(os.environ.get('AFTI_DRAWS_SEARCH', DRAWS_SEARCH)); RF = int(os.environ.get('AFTI_DRAWS_FINAL', DRAWS_FINAL))
ZC, LLGAIN = Z_CRIT, LL_GAIN
CAPS = CAPS_PERIOD if a.period is not None else CAPS_MAIN
MAXRP, RP_ENTRY, MAXHM, MAXHV = CAPS['max_rp'], CAPS['rp_entry'], CAPS['max_hm'], CAPS['max_hv']
SUB, tag, PERIOD = a.subset, a.tag, a.period
OUT = os.path.abspath(a.out or os.path.join(RESULTS, 'rp', SUB)); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, f'{tag}.log'), 'a', encoding='utf-8')
def log(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

Zd = np.load(subset_file(SUB), allow_pickle=True)
X, y, t, names = Zd['X'], Zd['y'], Zd['t'], [str(n) for n in Zd['names']]; T = SUBSETS[SUB]['T']
K = X.shape[1]
t0 = time.time()

if PERIOD is not None: groups = [[[PERIOD]] for _ in range(K)]
else: groups = json.load(open(a.structure, encoding='utf-8'))['groups']
rows = np.arange(len(y)) if PERIOD is None else np.where(t == PERIOD)[0]
A, cn, var = group_design(X, t, names, groups, T)
if PERIOD is not None: cn = [f'{c} [{PERIOD_LABELS[T][PERIOD]}]' for c in cn]
A, Y, TP = A[rows], y[rows], t[rows]; N, P = A.shape
b_fix, ll_fix, _ = newton(sp.csr_matrix(A), Y)
log(f'== {SUB} {tag} [mode {a.mode}, backend {a.backend}, device {DEVICE}, draws {RS}/{RF}, caps RP/HM/HV={MAXRP}/{MAXHM}/{MAXHV}]: '
    f'N={N}, parameters={P}, fixed-parameter LL={ll_fix:.3f}')
is_const = np.array([var[j] == 0 for j in range(P)])
per_of = [set(np.unique(TP[A[:, j] != 0])) for j in range(P)]


# ---------------- model fitting ----------------
def build_model(spec, R):
    rnd, hm, hv = spec['rnd'], spec['hm'], spec['hv']
    moved = {c for _, c in hm} | {c for _, c in hv}
    fx = [j for j in range(P) if j not in rnd and j not in moved and j not in spec.get('drop', [])]
    m = RPBL(A[:, fx], A[:, rnd], Y, HM=[(rnd.index(r), c) for r, c in hm], HV=[(rnd.index(r), c) for r, c in hv],
             Z=A, R=R, names_f=[cn[j] for j in fx], names_r=[cn[j] for j in rnd], names_z=cn)
    return m, fx


def fit(spec, R, warm=None, starts=(0.1,), se=True, cold=None):
    """warm: dict of starting values by name (missing SDs start at 0.1). starts: SD starting values for cold starts
    from the fixed-parameter estimates. cold=None -> cold starts only when no warm start is given."""
    m, fx = build_model(spec, R)
    base = {**{cn[j]: b_fix[j] for j in range(P)}, **{f'mean:{cn[j]}': b_fix[j] for j in range(P)}}
    cand0 = []
    if warm is not None:
        cand0.append(np.array([warm.get(n, warm.get('mean:' + n, base.get(n, 0.1 if n.startswith('SD:') else 0.0))) for n in m.names]))
    if cold or (cold is None and warm is None):
        for sd0 in starts:
            cand0.append(np.array([sd0 if n.startswith('SD:') else base.get(n, 0.0) for n in m.names]))
    best = None
    for th0 in cand0:
        m.fit(th0)
        if best is None or m.LL > best[0]: best = (m.LL, m.th.copy())
    m.th, m.LL = best[1], best[0]
    if se: m.hessian_se()
    return m


def est(m): return dict(zip(m.names, m.th))
def tval(m, name): i = m.names.index(name); return abs(m.th[i] / m.se[i]) if m.se[i] > 0 else 0.0
def fit_path(tg): return os.path.join(OUT, f'{tg}_fit.json')


def save(fin, spec, extra):
    k = len(fin.th)
    res = dict(subset=SUB, tag=tag, mode=a.mode, structure=(os.path.relpath(a.structure, ROOT).replace(os.sep, '/') if a.structure else None), period=PERIOD, draws_search=RS, draws_final=RF,
               caps=dict(random_parameters=MAXRP, het_means=MAXHM, het_variances=MAXHV), N=int(N), P_structure=int(P),
               LL_fixed=float(ll_fix), LL=float(fin.LL), k=int(k), AIC=float(-2 * fin.LL + 2 * k),
               BIC=float(-2 * fin.LL + np.log(N) * k), random=[cn[j] for j in spec['rnd']],
               HM=[[cn[r], cn[c]] for r, c in spec['hm']], HV=[[cn[r], cn[c]] for r, c in spec['hv']],
               dropped=[cn[j] for j in spec.get('drop', [])], **extra, seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(OUT, f'{tag}.json'), 'w', encoding='utf-8'), indent=1, default=float)
    fin.table().to_csv(os.path.join(OUT, f'{tag}_estimates.csv'), index=False)
    sp_ = dict(rnd=[int(j) for j in spec['rnd']], hm=[[int(r), int(c)] for r, c in spec['hm']],
               hv=[[int(r), int(c)] for r, c in spec['hv']], drop=[int(j) for j in spec.get('drop', [])])
    json.dump(dict(spec=sp_, th=[float(v) for v in fin.th], se=[float(v) for v in fin.se], names=list(fin.names), cn=cn,
                   groups=groups, period=PERIOD), open(fit_path(tag), 'w', encoding='utf-8'), indent=1)
    log(f'DONE {tag}: LL={fin.LL:.3f} k={k}  ({round(time.time() - t0)} s)')


# ======================================================================================== prune
if a.mode == 'prune':
    F = json.load(open(fit_path(a.start), encoding='utf-8'))
    assert F['cn'] == cn, 'structure mismatch with the search result'
    spec = dict(rnd=list(F['spec']['rnd']), hm=[tuple(x) for x in F['spec']['hm']], hv=[tuple(x) for x in F['spec']['hv']], drop=[])
    start_spec = json.loads(json.dumps(spec))
    fin = fit(spec, RF, warm=dict(zip(F['names'], F['th'])))
    log(f'start from {a.start}: k={len(fin.th)}, LL={fin.LL:.3f}')
    history = []
    while True:
        moved = {c for _, c in spec['hm']} | {c for _, c in spec['hv']}
        items = [('fixed', j, tval(fin, cn[j])) for j in range(P)
                 if not is_const[j] and j not in spec['rnd'] and j not in moved and j not in spec['drop']]
        items += [('sd', r, tval(fin, f'SD:{cn[r]}')) for r in spec['rnd']]
        items += [('hm', rc, tval(fin, f'HM:{cn[rc[0]]}|{cn[rc[1]]}')) for rc in spec['hm']]
        items += [('hv', rc, tval(fin, f'HV:{cn[rc[0]]}|{cn[rc[1]]}')) for rc in spec['hv']]
        insig = [it for it in items if it[2] < ZC]
        if not insig: break
        batch = [it for it in insig if it[2] < 1.0] or [min(insig, key=lambda it: it[2])]
        for kind, obj, tv in batch:
            if kind == 'fixed': spec['drop'].append(obj)
            elif kind == 'sd':
                spec['rnd'] = [r for r in spec['rnd'] if r != obj]
                spec['hm'] = [rc for rc in spec['hm'] if rc[0] != obj]; spec['hv'] = [rc for rc in spec['hv'] if rc[0] != obj]
            elif kind == 'hm': spec['hm'] = [rc for rc in spec['hm'] if rc != obj]
            elif kind == 'hv': spec['hv'] = [rc for rc in spec['hv'] if rc != obj]
        desc = [f'{kind}:{cn[obj] if kind in ("fixed", "sd") else cn[obj[0]] + "|" + cn[obj[1]]} (t={tv:.2f})' for kind, obj, tv in batch]
        fin = fit(spec, RF, warm=est(fin))
        history.append(dict(removed=desc, k=len(fin.th), LL=fin.LL))
        log(f'  removed {len(batch)}: {desc} -> k={len(fin.th)}, LL={fin.LL:.3f}  ({round(time.time() - t0)} s)')
    log(f'FINAL (all significant at 90%): random={[cn[j] for j in spec["rnd"]]}, HM={[(cn[r], cn[c]) for r, c in spec["hm"]]}, '
        f'HV={[(cn[r], cn[c]) for r, c in spec["hv"]]}, dropped={len(spec["drop"])}, LL={fin.LL:.3f}, k={len(fin.th)}')
    save(fin, spec, dict(start=a.start, start_spec=start_spec, prune_history=history))
    sys.exit(0)

# ======================================================================================== search
# ---------------- 1. random parameters (RS draws) ----------------
spec = dict(rnd=[], hm=[], hv=[])
cand = [j for j in range(P) if not is_const[j]]
screen = {}
for j in cand:
    m = fit(dict(rnd=[j], hm=[], hv=[]), RS, starts=(0.1, 1.0))
    screen[cn[j]] = dict(LL=m.LL, LL_gain=m.LL - ll_fix, sd_t=tval(m, f'SD:{cn[j]}'))
sig = sorted([j for j in cand if screen[cn[j]]['sd_t'] > ZC], key=lambda j: -screen[cn[j]]['LL_gain'])
rnd = sig[:RP_ENTRY]
log(f'RP screen ({round(time.time() - t0)} s): {len(sig)} of {len(cand)} significant; top {len(rnd)} by LL gain enter the joint model: '
    f'{[cn[j] for j in rnd]}')
cur, prev = None, None
while rnd:
    cur = fit(dict(rnd=rnd, hm=[], hv=[]), RS, warm=prev, starts=(0.1, 1.0), cold=True)
    ts = [tval(cur, f'SD:{cn[j]}') for j in rnd]
    log(f'  joint RP: {len(rnd)} random, LL={cur.LL:.3f}, min SD t={min(ts):.2f}')
    if min(ts) >= ZC and len(rnd) <= MAXRP: break
    prev = est(cur)
    rnd = [j for i, j in enumerate(rnd) if i != int(np.argmin(ts))]
spec['rnd'] = rnd
if not rnd: cur = fit(spec, RS)
log(f'RP model: {[cn[j] for j in rnd]}  LL={cur.LL:.3f}')


# ---------------- 2./3. heterogeneity (RS draws) ----------------
def candidates(m, spec):
    moved = {c for _, c in spec['hm']} | {c for _, c in spec['hv']}
    out = []
    for j in range(P):
        if is_const[j] or j in spec['rnd'] or j in moved: continue
        if tval(m, cn[j]) < ZC: out.append(j)
    return out


def het_search(kind, cap):
    global cur
    while len(spec[kind]) < cap and spec['rnd']:
        cs = candidates(cur, spec); best = None; tried = 0
        for c in cs:
            for r in spec['rnd']:
                if var[c] == var[r] or not (per_of[c] & per_of[r]): continue
                trial = {**spec, kind: spec[kind] + [(r, c)]}
                m = fit(trial, RS, warm=est(cur), se=False); tried += 1
                if m.LL - cur.LL <= LLGAIN or (best is not None and m.LL <= best[1].LL): continue
                m.hessian_se()
                nm = f'{"HM" if kind == "hm" else "HV"}:{cn[r]}|{cn[c]}'
                if tval(m, nm) > ZC: best = ((r, c), m)
        if best is None:
            log(f'{kind}: no qualifying term ({tried} pairings tried, {round(time.time() - t0)} s)'); break
        spec[kind].append(best[0]); gain = best[1].LL - cur.LL; cur = best[1]
        log(f'{kind}: added {cn[best[0][1]]} -> {cn[best[0][0]]}  (LL gain {gain:.2f}, LL={cur.LL:.3f}, {tried} tried, {round(time.time() - t0)} s)')

het_search('hm', MAXHM)
het_search('hv', MAXHV)
search_spec = json.loads(json.dumps(spec))

# ---------------- confirmation at RF draws ----------------
fin = fit(spec, RF, warm=est(cur))
drop_hm = [(r, c) for r, c in spec['hm'] if tval(fin, f'HM:{cn[r]}|{cn[c]}') < ZC]
drop_hv = [(r, c) for r, c in spec['hv'] if tval(fin, f'HV:{cn[r]}|{cn[c]}') < ZC]
drop_rp = [r for r in spec['rnd'] if tval(fin, f'SD:{cn[r]}') < ZC]
if drop_hm or drop_hv or drop_rp:
    spec = dict(rnd=[r for r in spec['rnd'] if r not in drop_rp],
                hm=[(r, c) for r, c in spec['hm'] if (r, c) not in drop_hm and r not in drop_rp],
                hv=[(r, c) for r, c in spec['hv'] if (r, c) not in drop_hv and r not in drop_rp])
    log(f'{RF}-draw confirmation removed: SD {[cn[r] for r in drop_rp]}, HM {[cn[c] for _, c in drop_hm]}, HV {[cn[c] for _, c in drop_hv]}')
    fin = fit(spec, RF, warm=est(fin))
log(f'FINAL ({RF} draws): random={[cn[j] for j in spec["rnd"]]}, HM={[(cn[r], cn[c]) for r, c in spec["hm"]]}, '
    f'HV={[(cn[r], cn[c]) for r, c in spec["hv"]]}, LL={fin.LL:.3f}, k={len(fin.th)}')
save(fin, spec, dict(search_spec=search_spec, rp_screen=screen))
