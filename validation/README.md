# Validation

## GPU and CPU backends

`parity_check.py` evaluates the simulated log-likelihood and gradient of the same model with the NumPy and PyTorch backends. It needs steps 1 and 2 to have been run for S1.

## Verification against R glm, Rchoice and Biogeme (Appendix E, Table E1)

All checks use the single-motorcycle subset (S1, 7,325 crashes). The specifications are in `check_specs.json`.

| Check | Model | Reference package | Halton draws |
|---|---|---|---|
| 1 | fixed-parameter logit, 43 parameters | R glm | – |
| 2 | random parameters logit, 45 parameters: New Year and Northeastern random | Rchoice | 500 |
| 3 | check 2 plus heterogeneity in the mean of New Year (Eastern) and in the variances of both random parameters (speeding 2018–2023), 48 parameters | Biogeme 3.3.5 | 500 |
| 4 | random parameters logit on the S1 partially constrained structure, eight random parameters | Biogeme 3.3.5 | 1,000 |

Rchoice and Biogeme are given the same standard Halton draws as the Python estimator, so the log-likelihoods agree exactly at the same parameter values. Each check has two parts:

- **Evaluation:** the reference package evaluates the log-likelihood and its gradient at the Python estimates.
- **Estimation:** the reference package estimates the model itself, starting from the fixed-parameter logit with SDs of 0.1.

Run from the repository root, after steps 1 and 2:

```
python validation/check_data.py
python validation/estimate_python.py
Rscript validation/check_rchoice.R
python validation/check_biogeme.py --estimate
python validation/summarize.py
```

- `check_data.py` writes `output/S1_check_data.csv`, the data for checks 1–3.
- `estimate_python.py` produces the Python estimates; checks 2–4 take a few minutes on a CPU.
- `check_rchoice.R` runs checks 1 and 2.
- `check_biogeme.py --estimate` runs checks 3 and 4 and takes about 15 minutes.
- `summarize.py` writes `output/Table_E1_verification.csv`.

The Python log-likelihoods in the paper are −4,602.868, −4,590.248, −4,565.577 and −4,521.658.
