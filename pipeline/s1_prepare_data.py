"""Step 1: build the four crash subsets from the raw HAIMS extract.

usage: python pipeline/s1_prepare_data.py
reads  data/raw/crashes_thailand_2016_2023.csv (or the file named by AFTI_RAW_CSV)
writes data/processed/<subset>.npz (X with the constant first, y, t = period index, names) and subsets.json (counts and
       the indicators excluded in each subset, with the reason)
"""
import json
import _paths  # noqa: F401
import numpy as np
from settings import PROCESSED, SUBSETS, subset_file
from data_prep import build_all

PROCESSED.mkdir(parents=True, exist_ok=True)
subsets, info = build_all()
log = dict(info)
for sub, (X, y, t, names, dropped) in subsets.items():
    np.savez(subset_file(sub), X=X, y=y, t=t, names=np.array(names))
    log[sub] = dict(name=SUBSETS[sub]['name'], periods=SUBSETS[sub]['T'], n=len(y), serious_fatal_share=float(y.mean()),
                    n_by_period={int(p): int((t == p).sum()) for p in range(SUBSETS[sub]['T'])},
                    candidate_variables=len(names) - 1, excluded=dropped)
    print(f'{sub}: {len(y):,} crashes, {100 * y.mean():.1f}% serious/fatal, {len(names) - 1} candidate indicators')
print(f'{info["n_crashes"]:,} crashes in the extract, {info["n_in_no_subset"]:,} in no subset')
json.dump(log, open(PROCESSED / 'subsets.json', 'w', encoding='utf-8'), indent=1)
