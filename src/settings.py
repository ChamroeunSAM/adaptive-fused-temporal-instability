"""Paths, subsets and estimation settings used throughout the pipeline.

Every setting that defines a model in the paper is here, so the protocol can be read in one place.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = Path(os.environ.get('AFTI_RAW_CSV', ROOT / 'data' / 'raw' / 'crashes_thailand_2016_2023.csv'))
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = Path(os.environ.get('AFTI_RESULTS', ROOT / 'results'))

YEARS = list(range(2016, 2024))

# Crash subsets. S1 is analysed in two-year periods (smaller sample), S2-S4 in yearly periods.
SUBSETS = {
    'S1': dict(name='single_motorcycle', T=4, title='single-motorcycle crashes'),
    'S2': dict(name='motorcycle_with_other_vehicle', T=8, title='crashes involving a motorcycle and another vehicle type'),
    'S3': dict(name='single_vehicle_no_motorcycle', T=8, title='single-vehicle crashes without a motorcycle'),
    'S4': dict(name='multi_vehicle_no_motorcycle', T=8, title='multi-vehicle crashes without a motorcycle'),
}
PERIOD_LABELS = {4: ['2016-17', '2018-19', '2020-21', '2022-23'], 8: [str(y) for y in YEARS]}

# ---- Stage 1 (fixed-parameter binary logit) ----
CONF = 0.90                  # level of every test in the paper (PC merges, fused rule, spline elimination, pruning)
FUSED_SHRINK = 0.75          # lambda path: lambda_max * 0.75^i until the first structure not rejected ...
FUSED_NBISECT = 8            # ... then bisection between the last rejected and the first accepted lambda

# ---- Stage 2 (random parameters binary logit with heterogeneity in means and variances) ----
DRAWS_SEARCH = 100           # Halton draws during the specification search
DRAWS_FINAL = 2000           # Halton draws for confirmation, pruning, final estimates and marginal effects
Z_CRIT = 1.645               # |t| threshold (90%)
LL_GAIN = 1.353              # minimum LL gain for a heterogeneity term (LR > 2.706, 1 df, 90%)
CAPS_MAIN = dict(max_rp=8, rp_entry=16, max_hm=3, max_hv=1)       # restricted temporal models
CAPS_PERIOD = dict(max_rp=2, rp_entry=16, max_hm=2, max_hv=1)     # separate period (fully unconstrained) models

# ---- random parameters stage of the structure selection ----
RP_FUSED = dict(shrink=0.75, nbisect=6, n_stepdown=5)    # penalised path (DRAWS_SEARCH), refits at DRAWS_FINAL


def subset_file(sub):
    return PROCESSED / f'{sub}_{SUBSETS[sub]["name"]}.npz'
