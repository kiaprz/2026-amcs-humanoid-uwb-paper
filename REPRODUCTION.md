# Reproduction guide

All commands are run from the repository root. `reproduce.py` sets the paths of the bundled model and of the geometry adapter itself; the individual programs in `poc/` use the same defaults (`G1_XML` may point to another copy of the model).

## 1. Environment

The stored results in `poc/results/` were produced by the authors with Python 3.12.10, MuJoCo 3.2.6, NumPy 1.26.4, SciPy 1.14.1 and Matplotlib 3.9.2 on an Apple M1 MacBook Pro (macOS, arm64); every stored `result.json` records these versions. `requirements.txt` pins MuJoCo 3.2.6 and gives the original versions of the other packages as lower bounds, so that newer Python versions can be used.

The robot model in `g1_model/` is `unitree_g1` from MuJoCo Menagerie at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. `poc/model_manifest.json` lists its 39 files with SHA-256 hashes; `reproduce.py` checks them before every task, and

```sh
python poc/menagerie_shim/fetch_model.py poc/model_manifest.json g1_model
```

re-downloads missing files and verifies all hashes.

## 2. Two kinds of evidence

* **Stored results** are the authors' original outputs. They are not overwritten by `reproduce.py`.
* **Replays** recompute the experiments on the current machine and write difference reports: `poc/results/replay_report.json` (static searches), `poc/results/article_estimator/` (estimator and calibration) and `poc/results/reviewer/replay_sensitivity/` (Section 6.3). The reports shipped here were produced on 15–16 November 2026 with Python 3.13, NumPy 2.5, SciPy 1.18 and MuJoCo 3.2.6 on Windows (x64).

The four checkers `poc/check_article_*.py` compare every number printed in the article with both. Rounding follows the article: in the sphere example, lower (upper) bounds are rounded downwards (upwards); other values are rounded to the nearest last digit.

Across platforms the floating-point results are not bit-for-bit identical. In the shipped replays, all decisions, numbers of subdivisions and posture evaluations, selected postures and evaluated posture sets are identical to the stored ones. Quantities printed in the article differ by less than 6e-10 in the static searches; over all stored values the largest difference is 3.9e-8 (a link weight), caused by clearances that differ by about 3e-9 m and are multiplied by the envelope slopes. In the search with margin 0.36, boxes with numerically tied upper bounds are subdivided in a different order, with the same set of 3559 evaluated postures and the same final cover. Search times are hardware-dependent and are only recorded, not compared.

## 3. Map from the article to programs and files

| Article | Programs | Stored results | Checker |
|---|---|---|---|
| Sec. 3 "Illustrative instance", Example 1, Fig. 1 | `poc/verified_sphere.py`, `poc/paper_example.py`, `figsrc/make_figs.py` | `poc/results/reviewer/verified_sphere.json`, `poc/results/paper_validation.json`, `figsrc/plot_*.dat` | `check_article_sphere.py` |
| Sec. 6, Table 1, Examples 2 and 3 | same | same | `check_article_sphere.py` |
| Fig. 2 | `figsrc/fig_decision_n10_EN.tex` (schematic, no data) | — | — |
| Sec. 6.1 baseline G1 configuration, Fig. 3(a),(b) | `poc/run.py` with `poc/config.json`, `poc/replay_g1_static.py`, `poc/plot_postures_gray.py` | `poc/results/result.json`, `inconclusive/`, `excluded/`, `summary.csv` | `check_article_g1_static.py` |
| Sec. 6.2 full-rank configuration, Fig. 3(c),(d), Fig. 4 | `poc/run.py` with `poc/config_hard.json`, `poc/config_bnb.json`, `poc/replay_g1_static.py`, `figsrc/make_fig5.py` | `poc/results/hard/`, `poc/results/bnb/` | `check_article_g1_static.py` |
| Sec. 6.2 "Estimator mismatch"; Sec. 2 calibration protocol | `poc/estimator_experiments.py`, `poc/reviewer_experiments.py`, `poc/replay_article_estimator.py` | `poc/results/reviewer/estimator/result.json`, `calibration.json`, `calibration_ranges.csv`, `monte_carlo.json` | `check_article_estimator.py` |
| Sec. 6.3 diffraction | `poc/diffraction_model.py`, `poc/reviewer_experiments.py` | `poc/results/reviewer/physical_sensitivity.json`, `diffraction_assumed_2mm.json` | `check_article_sensitivity.py` |
| Sec. 6.3 standing robot | `poc/dynamic_standing.py`, `poc/menagerie_shim/humanoid_uwb_sim/standing.py`, `poc/static_sensitivity.py` | `poc/results/reviewer/dynamics_static/`, `dynamics_static_dt1ms/`, `static_sensitivity/` | `check_article_sensitivity.py` |
| Sec. 6.3 local bounds, stress cases | `poc/local_bounds.py`, `poc/local_experiments.py` | `poc/results/reviewer/local/` | `check_article_sensitivity.py` |
| Independent recomputation of stored decisions | `poc/audit_decision.py`, `poc/check_result.py`, `poc/audit_reviewer.py`, `poc/comparison_models.py` | `poc/results/bnb/decision_audit.json`, `poc/results/reviewer/verification.json` | — |

`poc/replay_sensitivity.py` re-runs the whole chain of Section 6.3 (standing simulations, then the searches that use their budgets). The programs `dynamic_standing.py`, `static_sensitivity.py`, `local_experiments.py` and `estimator_experiments.py` accept `--output`; without it, and likewise `run.py` and `reviewer_experiments.py`, they overwrite the stored files of the authors' runs.

Fig. 3 is produced by

```sh
python poc/plot_postures_gray.py poc/results/result.json poc/results/hard/result.json fig_g1postures.pdf
```

(Ghostscript, if available, converts the fonts to outlines; otherwise the Matplotlib PDF is kept), and the sources of Figs. 1, 2 and 4 by `python figsrc/make_figs.py` and `python figsrc/make_fig5.py` (pdfLaTeX with pgfplots is needed to compile them).

## 4. Results reported here for the article

### 4.1 Exact arithmetic in the sphere example

`poc/verified_sphere.py` evaluates the sphere example with Python `fractions.Fraction` intervals. Square roots and the sine and cosine values are enclosed by rational bounds with explicit remainder terms, π by Machin's formula, and every comparison used in a decision is made between exact rational endpoints. The program encloses the clearances, weights, index bounds and ceiling of Example 1 and Table 1 for budgets A and B, the widths for M = 1, …, 256 equal intervals, the right-hand side of (12), the one-box exclusion of Example 3 and the number of equal intervals (159) needed by the scalar Lipschitz bound. The gain lower bounds exceed 0.5986 (budget A) and 0.6394 (budget B); the one-box exclusion margins exceed 0.007641 (A) and 0.04305 (B). `poc/paper_example.py` repeats the computation in float64 and produces the data of Fig. 1.

### 4.2 Baseline G1 configuration: three margins (Section 6.1)

| Run | Margin | Outcome | LB | UB | Evaluations | Subdivisions | Stopped by |
|---|---|---|---:|---:|---:|---:|---|
| `poc/results/` | epsilon_g=0.02 | guaranteed improvement | 0.6030 | 0.7014 | 28 | 0 | decision inequality |
| `poc/results/inconclusive/` | epsilon_g=0.4 | inconclusive | 0.6030 | 0.7014 | 44 | 8 | subdivision limit |
| `poc/results/excluded/` | epsilon_g=0.8 | excluded improvement | — | 0.7014 | 1 | 0 | calibration ceiling |

The guaranteed gain lower bound of the first run is 0.3358. Full-rank configuration (Section 6.2): margin 0.02 gives guaranteed improvement with 3 evaluations and no subdivision (gain lower bound 0.05506); margin 0.36 gives excluded improvement after 1778 subdivisions and 3559 evaluations (LB 0.2315, UB 0.4437).

### 4.3 Estimator experiment (Section 6.2)

`poc/estimator_experiments.py`: 20 000 range vectors per posture, seed 124 with the same standard normal draws at q0 and qb, lambda_ref = (0.03 m)^-2, weights equal to the mean of the two clip envelopes at the grid clearance, weighted Gauss–Newton from p + (0.02, −0.02, 0.02) m with backtracking and a 100-iteration limit. Mean squared errors with Monte Carlo standard errors, in 1e-3 m^2:

| Case | rho (m) | q0 | qb | Lower MSE |
|---|---|---:|---:|---|
| unbiased | (0, 0, 0, 0) | 9.572 ± 0.085 | 4.686 ± 0.031 | qb |
| mild bias | (0.03, 0, 0.08, 0) | 9.622 ± 0.074 | 5.296 ± 0.035 | qb |
| strong bias (article) | (0.03, 0, 0.50, 0) | 14.888 ± 0.101 | 29.832 ± 0.116 | q0 |

In the unbiased case the ordering agrees with the information model (normalized index 0.1289 at q0 and 0.3219 at qb; Cramér–Rao traces 9.192 and 4.679 in 1e-3 m^2). Only the strong-bias case is printed in the article.

### 4.4 Synthetic calibration-coverage experiment and Monte Carlo checks

`poc/reviewer_experiments.py --mode physical` (seed 20261011): 68 calibration cells (17 clearances × 4 links), 120 Gaussian residuals per cell, delta_2 = 0.05, Bonferroni-corrected chi-square intervals as in Section 2 of the article. All 68 true weights lie inside the intervals of the stored data set. In 2000 repeated calibrations all cells were covered simultaneously 1905 times (0.9525; the Bonferroni bound is 0.95 and the exact probability for independent cells is 0.9512).

Finite Monte Carlo check of the index bounds (`--mode mc`): 80 postures × 8 admissible realizations, no value outside the computed interval. These experiments test the implementation on synthetic data; they do not validate radio envelopes on hardware.

### 4.5 Standing simulations and local bounds (Section 6.3)

Five simulations with the free-base model (floor contact, position actuators of the model with gain 500 N m/rad, no balance feedback), differing only in the initial base velocity. During the held postures the tag moves by at most 22.67 mm (q0) and 16.24 mm (qb); the rigid-motion bound of the anchors in the torso frame reaches 137.1 and 95.85 mm; joint tracking errors stay below 8.1 mrad. With these values as uniform budgets, the full-rank comparisons are inconclusive after 32 subdivisions; with the local profiles of Lemma 2 the diffraction comparison is decided without subdivision. In the 32 stress cases no clearance bound was violated, and cell omission gave the same bounds as full evaluation at all 32 postures. The joint errors and the tag displacement of the stress cases lie exactly on the boundary of the uncertainty set; the rigid-motion term uses 90–96% of its bound.

## 5. Tests

```sh
python -m pytest -q poc                         # 20 unit tests
python -m pytest -q poc/check_article_sphere.py poc/check_article_g1_static.py poc/check_article_estimator.py poc/check_article_sensitivity.py
```

If the environment variable `AMCS_ARTICLE_TEX` points to the LaTeX source of the article, `check_article_g1_static.py` additionally requires every quoted fragment to occur verbatim in it.
