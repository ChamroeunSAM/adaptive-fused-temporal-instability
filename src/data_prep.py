"""Crash subsets and candidate indicators (Section 3 of the paper).

Vehicle units per crash = sum of the single-vehicle indicators (incl. bike, three_wheels, van, large_bas) + 2 x the
'two or more' indicators.
  S1 single motorcycle            : motor_1 = 1 and no other vehicle unit
  S2 motorcycle with other vehicle: any motorcycle and at least one non-motorcycle vehicle
  S3 single vehicle, no motorcycle: exactly one vehicle unit, not a motorcycle
  S4 multi-vehicle, no motorcycle : two or more vehicle units, no motorcycle
Crashes with no recorded vehicle and crashes involving only two or more motorcycles belong to no subset.

Candidate indicators in each subset: all indicators except (1) the reference categories (Central region, daytime,
no median; and the most frequent vehicle type when the vehicle indicators always sum to one), (2) indicators constant
in the subset, (3) indicators with fewer than 10 serious/fatal or 10 PDO/minor crashes with x = 1 in some period, so that
every separate-period model is identified, and (4) indicators exactly collinear with the constant and earlier columns.
"""
import numpy as np
import pandas as pd
from settings import RAW_CSV, YEARS, SUBSETS

ONE = ['motor_1', 'car_1', 'pickup_1', 'truck_6_1', 'truck_10_1', 'truck_>10_1', 'bike', 'three_wheels', 'van', 'large_bas']
MANY = ['motor_>1', 'car_>1', 'pickup_>1', 'truck_6_>1', 'truck_10_>1', 'truck_>10_>1']
VEH = ONE + MANY
GROUP_REF = {'region': 'Central', 'time_of_day': 'Daytime', 'median': 'no_median'}


def read_raw(path=RAW_CSV):
    d = pd.read_csv(path, encoding='utf-8-sig', dtype={'date': str})
    s = d['date']; ser = s.str.fullmatch(r'\d{5}')       # Excel serial dates mixed with d/m/yyyy text; only the year is used
    yr = pd.Series(0, index=d.index)
    yr[ser] = (pd.Timestamp('1899-12-30') + pd.to_timedelta(s[ser].astype(int), unit='D')).dt.year
    yr[~ser] = s[~ser].str.split('/').str[2].astype(int)
    d['year'] = yr.astype(int)
    assert (d['Injury'].isin([3, 4]).astype(int) == d['Injury_binary']).all()   # 3, 4 = serious/fatal; 1, 2 = PDO/minor
    return d


def subset_masks(d):
    nveh = d[ONE].sum(axis=1) + 2 * d[MANY].sum(axis=1)
    moto = (d['motor_1'] + d['motor_>1']) > 0
    other = (nveh - d['motor_1'] - 2 * d['motor_>1']) > 0
    return {'S1': (d['motor_1'] == 1) & (nveh == 1), 'S2': moto & other,
            'S3': (~moto) & (nveh == 1), 'S4': (~moto) & (nveh >= 2)}


def build_subset(d, mask, T):
    """Returns X (constant first), y, t (period index 0..T-1), names, dropped (column -> reason)."""
    x = d[mask].reset_index(drop=True)
    per = (x['year'] - 2016) // (8 // T)                  # T = 8: years; T = 4: two-year periods
    cols = [c for c in d.columns if c not in ['date', 'Injury', 'Injury_binary', 'year']]
    dropped = {}
    for g, ref in GROUP_REF.items():
        cols.remove(ref); dropped[ref] = f'reference ({g})'
    vc = [c for c in VEH if c in cols and x[c].sum() > 0]
    if len(vc) > 0 and (x[vc].sum(axis=1) == 1).all():
        ref = x[vc].sum().idxmax(); cols.remove(ref); dropped[ref] = 'reference (vehicle type within subset)'
    for c in list(cols):
        if x[c].nunique() < 2:
            cols.remove(c); dropped[c] = 'constant in subset'
    for c in list(cols):
        g = x[x[c] == 1].groupby(per[x[c] == 1])['Injury_binary'].agg(['sum', 'count']).reindex(range(T), fill_value=0)
        if not ((g['sum'] >= 10).all() and ((g['count'] - g['sum']) >= 10).all()):
            cols.remove(c); dropped[c] = 'fewer than 10 serious/fatal or 10 PDO/minor crashes in some period'
    M = np.column_stack([np.ones(len(x))] + [x[c].values for c in cols]).astype(float)
    keep = [0]
    for j in range(1, M.shape[1]):
        if np.linalg.matrix_rank(M[:, keep + [j]]) > len(keep): keep.append(j)
        else: dropped[cols[j - 1]] = 'collinear'
    cols = [cols[j - 1] for j in keep[1:]]
    X = np.column_stack([np.ones(len(x))] + [x[c].values for c in cols]).astype(float)
    return X, x['Injury_binary'].values.astype(float), per.values.astype(int), ['constant'] + cols, dropped


def build_all(path=RAW_CSV):
    d = read_raw(path); masks = subset_masks(d); out = {}
    for sub, m in masks.items():
        out[sub] = build_subset(d, m, SUBSETS[sub]['T'])
    n_none = int((~pd.concat(masks, axis=1).any(axis=1)).sum())
    return out, dict(n_crashes=len(d), n_in_no_subset=n_none)
