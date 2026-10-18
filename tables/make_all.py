"""Builds every table and figure of the paper from results/ (CPU). Writes PDF + CSV to results/tables/.

usage: python tables/make_all.py [--skip-coef]
--skip-coef leaves out Tables 4-7 and A1-A4, whose marginal effects take the longest (about 20 minutes for all eight).
"""
import sys, subprocess, os
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = ['table1_descriptive.py', 'table2_stage1.py', 'table3_comparison.py', 'appendixB_spline.py', 'appendixC_unconstrained.py',
        'appendixD_candidates.py', 'figure1_method.py', 'figure2_paths.py']
if '--skip-coef' not in sys.argv: jobs.insert(3, 'coef_tables.py')
for j in jobs:
    print(f'--- {j}', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, j)], cwd=HERE)
    if r.returncode != 0: sys.exit(f'FAILED: {j}')
print('all tables and figures written to results/tables/')
