"""Table E1: verification of the Python estimator against R glm, Rchoice and Biogeme (from validation/output/).
usage: python validation/summarize.py      writes validation/output/Table_E1_verification.csv"""
import json
import pandas as pd
from check_data import OUTDIR

L = lambda f: json.load(open(OUTDIR / f)) if (OUTDIR / f).exists() else {}
p = {c: L(f'python_check{c}.json') for c in (1, 2, 3, 4)}
r1, r2, b3, b4 = L('rchoice_check1.json'), L('rchoice_check2.json'), L('biogeme_check3.json'), L('biogeme_check4.json')
rows = [dict(check=f'Fixed-parameter logit ({p[1].get("k")} parameters)', reference='R glm', draws='–', LL_python=p[1].get('LL'),
             diff_at_python=r1.get('difference'), LL_reference=r1.get('LL_glm'), max_param_diff=None, max_se_diff=None),
        dict(check=f'Random parameters ({p[2].get("k")} parameters)', reference=f'Rchoice {r2.get("rchoice_version", "")}', draws=500,
             LL_python=p[2].get('LL'), diff_at_python=r2.get('difference'), LL_reference=r2.get('LL_rchoice_estimated'),
             max_param_diff=r2.get('max_abs_coef_diff'), max_se_diff=r2.get('max_abs_se_diff')),
        dict(check=f'Random parameters with heterogeneity in means and variances ({p[3].get("k")} parameters)', reference='Biogeme',
             draws=500, LL_python=p[3].get('LL'), diff_at_python=b3.get('difference'), LL_reference=b3.get('LL_biogeme_estimated'),
             max_param_diff=b3.get('max_abs_coef_diff'), max_se_diff=b3.get('max_abs_se_diff'), max_rel_se_diff=b3.get('max_rel_se_diff')),
        dict(check='Random parameters, S1 partially constrained structure', reference='Biogeme', draws=1000, LL_python=p[4].get('LL'),
             diff_at_python=b4.get('difference'), LL_reference=b4.get('LL_biogeme_estimated'), max_param_diff=b4.get('max_abs_coef_diff'),
             max_se_diff=b4.get('max_abs_se_diff'), max_rel_se_diff=b4.get('max_rel_se_diff'))]
T = pd.DataFrame(rows); T.to_csv(OUTDIR / 'Table_E1_verification.csv', index=False)
pd.set_option('display.width', 250); print(T.to_string(index=False))
