# Temporal instability in crash injury severity: an adaptive fused random parameters approach

Code for the paper *Temporal instability in crash injury severity: an adaptive fused random parameters approach*
(Chamroeun Se). It reproduces every model, table and figure in the paper, including the appendices, from the raw crash
records.

The adaptive fused approach writes each parameter of a binary logit model as a level plus steps between adjacent periods. An adaptive fused lasso penalty shrinks small steps to exactly zero, which selects the temporal structure of all parameters jointly. The penalty is set to the sparsest structure that is not rejected at the 90% level against separate period models.

The approach is applied in two places:

- in the fixed-parameter binary logit;
- inside a random parameters binary logit with heterogeneity in means and variances.

It is compared with three alternatives: fully unconstrained (separate period) models, partially constrained models (likelihood ratio tests) and spline-indicator models.

## Repository layout

```
src/            estimation code
  settings.py     paths, subsets and every estimation setting of the paper (draws, caps, test levels)
  data_prep.py    crash subsets and candidate indicators (Section 3)
  temporal.py     temporal bases, fixed-parameter logit, L1-penalised logit, period-group design
  stage1.py       Stage 1: separate-period logits, partially constrained (LR tests), adaptive fused (90% rule), spline
  rpbl.py         random parameters binary logit with heterogeneity in means and variances (simulated ML, CPU)
  rpbl_torch.py   the same likelihood and gradient on a GPU (PyTorch, float64)
  labels.py       variable descriptions used in the tables
pipeline/       steps 1-3 (run in this order)
  s1_prepare_data.py        raw records -> four subsets
  s2_stage1_structures.py   Stage-1 structures (CPU)
  s3_stage2_models.py       all random parameters models and the stage choice (GPU), calling
    rp_model.py               specification search and pruning of one model
    rp_structure_fused.py     adaptive fused structure re-selected inside the random parameters model
    rp_structure_pc.py        partially constrained structure re-selected inside the random parameters model
tables/         tables and figures of the paper (make_all.py builds them all)
validation/     GPU/CPU parity check and the verification against R glm, Rchoice and Biogeme (Appendix E)
data/           raw and processed data (not included; see data/README.md)
results/        created by the pipeline
```

## Tables and figures

| Paper | Script | Main inputs |
|---|---|---|
| Table 1 | `tables/table1_descriptive.py` | `data/processed/` |
| Table 2 | `tables/table2_stage1.py` | `results/stage1/<subset>.json` |
| Table 3 | `tables/table3_comparison.py` | `results/rp/<subset>/decision.json` and model results |
| Tables 4–7 (adaptive fused) | `tables/coef_tables.py` | chosen models, `results/rp/<subset>/<tag>_fit.json` |
| Tables A1–A4 (partially constrained) | `tables/coef_tables.py` | chosen models |
| Tables B1–B4 (spline indicator) | `tables/appendixB_spline.py` | `results/stage1/` |
| Tables C1–C4 (fully unconstrained) | `tables/appendixC_unconstrained.py` | `results/rp/<subset>/uncon_p*` |
| Table D1 (all structures estimated) | `tables/appendixD_candidates.py` | `decision.json`, structure files |
| Table E1 (verification) | `validation/summarize.py` | `validation/output/` |
| Figure 1 | `tables/figure1_method.py` | none (schematic) |
| Figure 2 | `tables/figure2_paths.py` | chosen models and period models |

Tables are written to `results/tables/` as PDF and CSV. The marginal effects of Tables 4–7 and A1–A4 are also written as JSON.

## Results included in this repository

`results/` holds the estimates reported in the paper, produced by one run of steps 2–4 on an NVIDIA RTX 5060 Ti (October 2026). Software versions are in `results/environment.json`, and the run log is `results/stage2_run.log`. No crash records are included.

| Folder | Content |
|---|---|
| `results/stage1/` | Stage-1 models of each subset (Table 2) and the temporal structures used in step 3 |
| `results/rp/<subset>/` | every random parameters model estimated: `<tag>.json` (summary), `<tag>_estimates.csv` (estimates, standard errors, t-statistics), `<tag>_fit.json` (specification and estimates), `<tag>.log`; the structure selections inside the random parameters model (`*_structure.json`); and `decision.json` (likelihood ratio tests and the stage choice) |
| `results/final/` | the model reported for each approach and subset (Table 3) |
| `results/tables/` | the tables and figures of the paper as PDF and CSV, and the marginal effects as JSON |

Model tags, for each subset:

| Tag | Model |
|---|---|
| `uncon_p<period>_search`, `uncon_p<period>` | separate period model before and after pruning |
| `pc_s1`, `fused_s1` | partially constrained / adaptive fused structure selected in the fixed-parameter logit |
| `pc_rp`, `fused_rp_c<i>` | structure re-selected inside the random parameters model (`c<i>`: candidates in order of sparsity) |

Tables 2, 3, C1–C4 and D1 and Figure 2 can be rebuilt from `results/` alone. Tables 1, 4–7, A1–A4 and B1–B4 also need the crash data: they recompute descriptive statistics, marginal effects or the spline models.

## Requirements

- Python 3.10 or newer with `pip install -r requirements.txt`.
- **GPU (recommended for step 3).** Use a CUDA build of PyTorch from pytorch.org; RTX 50-series cards need CUDA 12.8 or newer.
- **CPU only.** Step 3 also runs on a CPU (`--backend numpy`), but much more slowly.
- **Validation only (Appendix E).** R with the packages Rchoice, maxLik, sandwich and jsonlite, and Biogeme 3.3.5.

## Data

The crash records come from the Highways Accident Information Management System (HAIMS) of Thailand's Department of Highways. They are not included in this repository; `data/README.md` explains how to request them and which columns the code expects.

Place the file at `data/raw/crashes_thailand_2016_2023.csv`, or set the environment variable `AFTI_RAW_CSV` to its path.

## Running

Run every step from the repository root.

| Step | Command | Time |
|---|---|---|
| 1 | `python pipeline/s1_prepare_data.py` | under a minute |
| 2 | `python pipeline/s2_stage1_structures.py` | CPU, 1–3 hours; S3 takes longest |
| check | `python validation/parity_check.py` | must print `PARITY CHECK PASSED` |
| 3 | `python pipeline/s3_stage2_models.py` | GPU, several hours (see below) |
| 4 | `python tables/make_all.py` | CPU, about 30 minutes, mostly the marginal effects |

**Step 1** builds the four crash subsets.

**Step 2** estimates the Stage-1 structures. Its outputs are:

- the separate-period logits;
- the partially constrained structure from likelihood ratio tests;
- the adaptive fused structure from the 90% rule;
- the spline-indicator model.

**Step 2 can run in parallel**, one process per subset (`python pipeline/s2_stage1_structures.py S3`, and so on). Step 3 can start as soon as S1 is done, because it waits for the Stage-1 structures of each subset.

**Step 3** estimates every random parameters model. It is restartable: run the same command again after an interruption and finished models are skipped. Progress is written to `results/stage2_run.log` and to one log per model in `results/rp/<subset>/`. On a single RTX-class GPU, step 3 takes several hours. In it, the partially constrained re-selection inside the random parameters model takes the longest.

**Step 4** builds the tables and figures.

Optional validation for Appendix E:

```
python validation/check_data.py
python validation/estimate_python.py
Rscript validation/check_rchoice.R
python validation/check_biogeme.py --estimate
python validation/summarize.py
```

## Protocol

All settings are in `src/settings.py`.

- **Halton draws.** The specification search uses 100 standard Halton draws (primes 2, 3, 5, …, first 10 points discarded, one block of draws per crash). Confirmation, pruning, final estimates and marginal effects use 2,000 draws.
- **Test level.** Every test in the paper is at the 90% level: partially constrained merges, the adaptive fused rule, spline elimination, significance of random parameters and heterogeneity terms, and pruning.
- **Caps.** Restricted temporal models have at most 8 random parameters (16 enter the joint model), 3 heterogeneity-in-mean terms and 1 heterogeneity-in-variance term. Separate period models have at most 2 random parameters, 2 heterogeneity-in-mean terms and 1 heterogeneity-in-variance term.
- **Stage choice.** For each approach, the stage reported in Table 3 is the one whose final model has the lower BIC among the models not rejected at the 90% level against the pruned separate period models.

## Licence

MIT (see `LICENSE`). Please cite the paper when using the code.
