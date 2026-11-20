# V17C.14 theory package

This directory contains the generalized model, analytical derivations, reproduced source data, numerical checks, eight figure sets, and Russian/English LaTeX theory drafts for V17C.14.

## Start here

1. `REPORT_RU.md` — result and scope summary.
2. `THEORY_STATUS_RU.md` — completed and unresolved claims.
3. `FIGURE_AUDIT_RU.md` — per-figure scientific question, parameters, protocol and limitations.
4. `NUMERICAL_VALIDATION_RU.md` — checks, errors and source-run provenance.
5. `SESSION_HANDOFF.md` — reproducibility and follow-up state.

## Reproduce deterministic outputs

The scripts were run with Python 3.12.10, NumPy 2.5.3, Matplotlib 3.11.2 and SymPy 1.14.0 (`requirements.txt`). From repository root, create `.venv` in this directory and install those pinned packages:

```powershell
py -3.12 -m venv research/v17c/v17c_14/.venv
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' -m pip install -r research/v17c/v17c_14/requirements.txt
```

Then run:

```powershell
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' -m unittest discover -s research/v17c/v17c_14/tests -v
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/symbolic/derive_generalized_identities.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/validate_averaged_harmonic.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/compute_leading_branches.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/make_figures.py
```

The checks and plots are deterministic. V17C.14 did not integrate new trajectories. Figure inputs from V17C.13 are pinned by commit in `results/v17c13_reference_data.json`.

The Russian and English LaTeX manuscripts are source drafts. No TeX engine was installed in the available environment, so they were not compiled to PDF; figure PDF assets themselves were generated and inspected.

## Figure assets

`figures/` has 8 figures in PDF, SVG and PNG at 600 dpi. Corresponding tabular source data is under `results/figure_data/`. The `.py` scripts that generate plots and calculations are in `numerical/`.

## Scope warning

Reduced averaged stability, frozen-phase eigenvalues and fixed-radius H4 extrema are not full RF Floquet/Lyapunov stability, a random-start capture probability, or proof of an experimentally realizable trap. See the audit docs before reusing the diagrams in a paper.
