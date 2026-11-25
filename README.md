# Calibration-aware posture certificates for humanoid UWB localization — software

This repository contains the source code, configuration files and stored results for the article

> J. Kluska, M. Goral, T. Mączka, *Calibration-aware posture certificates for humanoid ultra-wideband localization*, submitted to the International Journal of Applied Mathematics and Computer Science (AMCS).

It reproduces every numerical result of the article: the exact-arithmetic sphere example (Table 1, Examples 1–3, Fig. 1), the static Unitree G1 searches (Sections 6.1–6.2, Figs. 3–4), the estimator-mismatch and calibration-coverage experiments, and the radio and torso-motion sensitivity study (Section 6.3), including the finite standing-sensitivity simulations.

## Quick start

Python 3.12 or newer is required.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # Windows: .venv\Scripts\python
.venv/bin/python reproduce.py
```

`python reproduce.py` takes about two minutes. It verifies the bundled robot model, runs the unit tests and compares every number printed in the article with the stored results, printing one `PASS`/`FAIL` line per claim (253 claims).

To recompute the experiments on your machine:

```sh
python reproduce.py --task sphere      # exact sphere example and Fig. 1 data (seconds)
python reproduce.py --task searches    # the five static G1 searches (about 8 min)
python reproduce.py --task estimator   # estimator and calibration experiments (about 4 min)
python reproduce.py --task standing    # standing, diffraction and local-profile studies (about 10 min)
python reproduce.py --task audit       # recomputation audit of the stored results
python reproduce.py --task all         # everything, then the check
```

[REPRODUCTION.md](REPRODUCTION.md) maps each part of the article to the programs and result files, and reports the additional results that the article attributes to the repository documentation.

## Contents

| Path | Content |
|---|---|
| `reproduce.py` | single entry point (tasks above) |
| `poc/` | programs: bounds and branch and bound (`bounds.py`), G1 geometry (`g1_geometry.py`), local profiles (`local_bounds.py`), diffraction model (`diffraction_model.py`), experiments, replays, checkers and tests |
| `poc/config*.json` | configurations of the baseline and full-rank G1 searches |
| `poc/results/` | stored results of the authors' runs and the replay reports |
| `poc/menagerie_shim/` | geometry adapter replacing the authors' simulation engine, model download/verification script |
| `poc/model_manifest.json` | SHA-256 manifest of the pinned robot model |
| `g1_model/` | Unitree G1 model from MuJoCo Menagerie, commit `71f066a` (39 files) |
| `figsrc/` | generators and TikZ sources of Figs. 1, 2 and 4 |

## Scope and limitations

The G1 results are a model study. The radio envelopes and the numerical error budgets are assumed, not measured, and all G1 computations use floating-point arithmetic; exact rational interval arithmetic is used for the sphere example only. The authors' simulation engine `humanoid-uwb-sim` is not distributed; the adapter in `poc/menagerie_shim/` provides the kinematics and the standing model with plain MuJoCo, and repeats the stored searches. Search times reported in the article were measured on an Apple M1 MacBook Pro and are recorded in the stored result files; they are hardware-dependent.

## Licences

The code and results of this repository are released under the MIT licence ([LICENSE](LICENSE)). The robot model in `g1_model/` is redistributed unchanged from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under its own BSD-3-Clause licence (`g1_model/MODEL_LICENSE`, `g1_model/REPOSITORY_LICENSE`).

## Acknowledgment

This work was supported by institutional funding for the maintenance and development of research potential (UPB) of the Rzeszów University of Technology.
