"""Step 3: all random parameters binary logit models with heterogeneity in means and variances (GPU; hours).

For each subset (restartable: every finished piece is skipped):
  A. fully unconstrained: a separate model for every period (search, then pruning); the models before pruning are the
     reference for the structure selection inside the random parameters model, the pruned models the reference for
     the final likelihood ratio tests
  B. partially constrained, Stage-1 structure: search + pruning on results/stage1/struct_<subset>_pc.json   -> pc_s1
  C. adaptive fused, Stage-1 structure: search + pruning on results/stage1/struct_<subset>_fused.json       -> fused_s1
  D. adaptive fused, structure re-selected in the random parameters model (rp_structure_fused.py); candidate structures
     are rebuilt (search + pruning) in order of sparsity until one is not rejected at 90% against the pruned period
     models                                                                                                   -> fused_rp_c<i>
  E. partially constrained, structure re-selected in the random parameters model (rp_structure_pc.py); the structure is
     rebuilt (search + pruning)                                                                               -> pc_rp
  F. stage choice (Section 2.2): for each approach, the final model with the lower BIC among those not rejected at
     the 90% level against the pruned period models -> results/rp/<subset>/decision.json, results/final/<subset>.json

usage: python pipeline/s3_stage2_models.py [S1 S2 S3 S4] [--backend torch|numpy]
"""
import os, sys, json, time, platform, subprocess, argparse
import _paths  # noqa: F401
import numpy as np
from scipy.stats import chi2
from settings import RESULTS, SUBSETS, CONF, DRAWS_SEARCH, DRAWS_FINAL

ap = argparse.ArgumentParser(); ap.add_argument('subsets', nargs='*'); ap.add_argument('--backend', default='torch', choices=['torch', 'numpy'])
a = ap.parse_args(); subs = a.subsets or ['S1', 'S2', 'S4', 'S3']
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = RESULTS / 'final'; FINAL.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')
LOG = open(RESULTS / 'stage2_run.log', 'a', encoding='utf-8')
t0 = time.time()
def say(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def environment():
    info = dict(python=platform.python_version(), platform=platform.platform(), backend=a.backend,
                draws_search=int(os.environ.get('AFTI_DRAWS_SEARCH', DRAWS_SEARCH)), draws_final=int(os.environ.get('AFTI_DRAWS_FINAL', DRAWS_FINAL)))
    import numpy, scipy, pandas
    info.update(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
    if a.backend == 'torch':
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    return info


def run_script(script, *args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args, '--backend', a.backend], env=ENV, cwd=HERE)
    if r.returncode != 0: sys.exit(f'FAILED: {script} {" ".join(args)}')


def model(sub, tag, mode, structure=None, period=None, start=None):
    out = RESULTS / 'rp' / sub
    if (out / f'{tag}.json').exists(): say(f'  skip (done): {sub} {tag}'); return
    say(f'  {sub} {tag} ({mode}) started {time.strftime("%H:%M:%S")}  (elapsed {(time.time() - t0) / 3600:.2f} h)')
    args = ['--subset', sub, '--tag', tag, '--mode', mode]
    if structure: args += ['--structure', str(structure)]
    if period is not None: args += ['--period', str(period)]
    if start: args += ['--start', start]
    run_script('rp_model.py', *args)


def res(sub, tag): return json.load(open(RESULTS / 'rp' / sub / f'{tag}.json', encoding='utf-8'))


def test(r, llU, kU):
    LR, df = 2 * (llU - r['LL']), kU - r['k']
    return dict(k=r['k'], LL=r['LL'], AIC=r['AIC'], BIC=r['BIC'], N=r['N'], LR=LR, df=df,
                conf=float(chi2.cdf(max(LR, 0), df)), passes=bool(LR < chi2.ppf(CONF, df)))


json.dump(environment(), open(RESULTS / 'environment.json', 'w', encoding='utf-8'), indent=1)
for sub in subs:
    T = SUBSETS[sub]['T']; D = RESULTS / 'rp' / sub; D.mkdir(parents=True, exist_ok=True)
    S1 = RESULTS / 'stage1'
    say(f'=== {sub} started {time.strftime("%Y-%m-%d %H:%M:%S")}')
    # ---- A. separate period models
    for p in range(T):
        model(sub, f'uncon_p{p}_search', 'search', period=p)
        model(sub, f'uncon_p{p}', 'prune', period=p, start=f'uncon_p{p}_search')
    Ub = [res(sub, f'uncon_p{p}') for p in range(T)]
    llU, kU = sum(u['LL'] for u in Ub), sum(u['k'] for u in Ub)
    say(f'  pruned period models: LL={llU:.3f}, k={kU}')
    # ---- B., C. Stage-1 structures
    if not all((S1 / f'struct_{sub}_{app}.json').exists() for app in ('pc', 'fused')):   # step 2 may still be running
        say(f'  waiting for the Stage-1 structures of {sub} (pipeline/s2_stage1_structures.py)')
        tw = time.time()
        while not all((S1 / f'struct_{sub}_{app}.json').exists() for app in ('pc', 'fused')):
            if time.time() - tw > 8 * 3600: sys.exit(f'Stage-1 structures of {sub} still missing after 8 hours: run pipeline/s2_stage1_structures.py {sub}')
            time.sleep(60)
    for app in ['pc', 'fused']:
        st = S1 / f'struct_{sub}_{app}.json'
        model(sub, f'{app}_s1_search', 'search', structure=st)
        model(sub, f'{app}_s1', 'prune', structure=st, start=f'{app}_s1_search')
    # ---- D. adaptive fused, structure re-selected in the random parameters model
    if not (D / 'fused_rp_structure.json').exists():
        say(f'  {sub} fused structure in the RP model started {time.strftime("%H:%M:%S")}')
        run_script('rp_structure_fused.py', sub)
    Sf = json.load(open(D / 'fused_rp_structure.json', encoding='utf-8')); tried, fused_rp = [], None
    for i, cand in enumerate(Sf['candidates']):
        cf = D / f'fused_rp_c{i}_struct.json'
        if not cf.exists(): json.dump(cand, open(cf, 'w', encoding='utf-8'), indent=1, default=float)
        model(sub, f'fused_rp_c{i}_search', 'search', structure=cf)
        model(sub, f'fused_rp_c{i}', 'prune', structure=cf, start=f'fused_rp_c{i}_search')
        r_ = test(res(sub, f'fused_rp_c{i}'), llU, kU)
        tried.append(dict(tag=f'fused_rp_c{i}', candidate=i, lam=cand['lam'], k_selection=cand['k'], conf_selection=cand['conf'], **r_))
        say(f'  fused candidate {i}: k={r_["k"]}, LL={r_["LL"]:.3f}, LR={r_["LR"]:.1f}, df={r_["df"]}, conf={r_["conf"]:.3f} -> '
            f'{"not rejected" if r_["passes"] else "rejected, next candidate"}')
        if r_['passes']: fused_rp = tried[-1]; break
    # ---- E. partially constrained, structure re-selected in the random parameters model
    if not (D / 'pc_rp_structure.json').exists():
        say(f'  {sub} PC structure in the RP model started {time.strftime("%H:%M:%S")}')
        run_script('rp_structure_pc.py', sub)
    Sp = json.load(open(D / 'pc_rp_structure.json', encoding='utf-8'))
    model(sub, 'pc_rp_search', 'search', structure=D / 'pc_rp_structure.json')
    model(sub, 'pc_rp', 'prune', structure=D / 'pc_rp_structure.json', start='pc_rp_search')
    # ---- F. stage choice
    M = {tag: test(res(sub, tag), llU, kU) for tag in ['pc_s1', 'fused_s1', 'pc_rp']}
    M.update({d['tag']: {k: d[k] for k in M['pc_s1']} for d in tried})
    choice, flagged = {}, {}
    for app, opts in [('pc', ['pc_s1', 'pc_rp']), ('fused', ['fused_s1'] + ([fused_rp['tag']] if fused_rp else []))]:
        ok = [o for o in opts if M[o]['passes']]
        if ok: choice[app] = min(ok, key=lambda o: M[o]['BIC']); flagged[app] = False
        else:                                       # no final model passes: lowest confidence of rejection, flagged
            pool = opts + ([d['tag'] for d in tried] if app == 'fused' else [])
            choice[app] = min(pool, key=lambda o: M[o]['conf']); flagged[app] = True
    dec = dict(subset=sub, reference=dict(LL=llU, k=kU, N=sum(u['N'] for u in Ub)), models=M,
               fused_candidates_tried=tried, fused_rp_accepted=fused_rp['tag'] if fused_rp else None,
               fused_rp_selection=dict(lam=Sf['chosen']['lam'], k=Sf['chosen']['k'], conf=Sf['chosen']['conf']),
               pc_rp_selection=dict(k=Sp['k'], conf=Sp['conf_ref'], rounds=len(Sp['rounds'])),
               choice=choice, flagged=flagged)
    json.dump(dec, open(D / 'decision.json', 'w', encoding='utf-8'), indent=1, default=float)
    json.dump(dict(subset=sub, choice=choice, flagged=flagged, period_models=[f'uncon_p{p}' for p in range(T)]),
              open(FINAL / f'{sub}.json', 'w', encoding='utf-8'), indent=1)
    say(f'=== {sub} DONE: adaptive fused -> {choice["fused"]}, partially constrained -> {choice["pc"]}'
        f'{" (FLAGGED: rejected at 90%)" if any(flagged.values()) else ""}; elapsed {(time.time() - t0) / 3600:.2f} h')
say('ALL DONE')
