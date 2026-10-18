"""Data for the verification checks (Appendix E), built from the processed S1 subset.

check_data_1to3()  -> DataFrame with crashid, sev, period (1..4) and the columns of check_specs.json (checks 1-3)
check_data_4()     -> design of the S1 partially constrained structure (results/stage1/struct_S1_pc.json), column names
usage: python validation/check_data.py   (writes validation/output/S1_check_data.csv for Rchoice)
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
import numpy as np
import pandas as pd
from settings import RESULTS, subset_file
from temporal import group_design

SPEC = json.load(open(HERE / 'check_specs.json', encoding='utf-8'))
OUTDIR = HERE / 'output'; OUTDIR.mkdir(exist_ok=True)


def load_s1():
    Z = np.load(subset_file('S1'), allow_pickle=True)
    return Z['X'], Z['y'], Z['t'], [str(n) for n in Z['names']]


def check_data_1to3():
    X, y, t, names = load_s1(); year0 = 2016 + 2 * t                 # S1: t = two-year period, first year of the period
    df = pd.DataFrame(dict(crashid=np.arange(1, len(y) + 1), sev=y.astype(int), period=(t + 1).astype(int)))
    for short, var, a, b in SPEC['columns']:
        df[short] = (X[:, names.index(var)] * ((year0 >= a) & (year0 <= b))).astype(int)
    return df


def check_data_4():
    X, y, t, names = load_s1()
    groups = json.load(open(RESULTS / 'stage1' / 'struct_S1_pc.json', encoding='utf-8'))['groups']
    A, cn, var = group_design(X, t, names, groups, 4)
    return A, y, cn


if __name__ == '__main__':
    df = check_data_1to3(); df.to_csv(OUTDIR / 'S1_check_data.csv', index=False)
    print('written', OUTDIR / 'S1_check_data.csv', df.shape)
