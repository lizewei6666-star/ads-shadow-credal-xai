# Shadow-calibrated multi-explainer analysis of ADS crashes

This repository accompanies a study of class-specific predictive relevance in
automated driving system (ADS) crashes. The analysis retains several admissible
predictive models and four non-interchangeable explanation sources: permutation
SHAP, permutation feature importance (PFI), leave-one-covariate-out refitting
(LOCO), and FIRM. Each source is calibrated against a feature-matched
replacement-shadow reference before the retained model-explainer specifications
are summarized by relevance bounds, strict partial orders, and maximal sets.

A separate Fast Causal Inference (FCI) workflow describes local structural
proximity around the injury outcome in partial ancestral graphs (PAGs). The
predictive and structural results are complementary analytical views; the PAG
workflow does not alter or validate the predictive ordering.

## Repository contents

| Path | Purpose |
| --- | --- |
| `analysis/run_predictive_credal.py` | Nested model admission, four explainers, shadow calibration, and credal ordering |
| `analysis/run_pag_structural.py` | FCI multiverse, label-null analysis, adjacency, proximity, and bridge summaries |
| `analysis/merge_distributed_checkpoints.py` | Verified merge of checkpoint shards from a second computer |
| `figures/` | Scripts for the methodological framework and manuscript result figures |
| `data/` | Data-access instructions and the expected local workbook location |
| `results/` | Aggregate inputs required to reproduce the manuscript figures |
| `provenance/` | Formal-run metadata, dataset identity, and release checksums |
| `docs/` | Statistical contract and reproducibility notes |
| `tests/` | Lightweight release-contract tests |

## Environment

The reported run used Python 3.10.20. Create an isolated environment and install
the pinned dependencies:

```powershell
conda create -n ads_shadow_credal python=3.10 -y
conda activate ads_shadow_credal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_environment.py
```

An equivalent Conda specification is provided in `environment.yml`.

## Reproduce the figures

The aggregate tables needed by the figure scripts are included. Figure
reproduction therefore does not require rerunning the computationally intensive
shadow or PAG analyses.

```powershell
python figures/make_figure1.py
python figures/make_figure2.py
python figures/make_figure3.py
python figures/make_figures4_5.py
python figures/make_figure6.py
python figures/make_figure7.py
```

Outputs are written to `figures/generated/`. Editable SVG and PDF files are
produced where supported by the original figure program.

## Full analysis

Start a new predictive run from the repository root:

```powershell
$env:ADS_XAI_EXCEL_PATH = (Resolve-Path "data/ads_crashes.xlsx").Path
$env:ADS_XAI_BASE_OUTPUT_DIR = (Join-Path $PWD "outputs")
$env:ADS_XAI_RESUME_RUN_DIR = ""
$env:ADS_XAI_SHADOW_PARALLEL_WORKERS = "4"
$env:ADS_XAI_SHADOW_MAX_PENDING_TASKS = "8"
python analysis/run_predictive_credal.py
```

The program creates a timestamped run directory under `outputs/`. To resume an
interrupted run, set `ADS_XAI_RESUME_RUN_DIR` to that directory and rerun the
same command. Checkpoints are accepted only when the dataset, software, split,
model, and method fingerprints agree.

After the predictive run is complete, run the PAG extension:

```powershell
python analysis/run_pag_structural.py `
  --excel data/ads_crashes.xlsx `
  --source-run outputs/<completed-predictive-run> `
  --output-dir outputs/pag_multiscale_extension_v1 `
  --workers 6
```

The full workflow is computationally intensive and its duration depends on CPU,
memory, storage, and model-library builds. See
[`docs/DISTRIBUTED_EXECUTION.md`](docs/DISTRIBUTED_EXECUTION.md) for the verified
two-computer workflow.

## Reproducibility scope

The formal study used 1,469 records, 14 predictors, three injury classes, five
outer folds, five inner folds, 49 replacement-shadow replicates, 45 observed
PAG specifications, and 99 label-null replicates per structural specification.
The complete contract is recorded in
[`docs/METHOD_CONTRACT.md`](docs/METHOD_CONTRACT.md).

The public-facing scripts differ from the frozen formal scripts only in module
documentation and portable path defaults. Statistical settings and computational
functions are unchanged. Formal source hashes and sanitized run metadata are
preserved under `provenance/`.

## Data source

The analytical workbook is not distributed with this repository. It was derived
from publicly downloadable crash reports released under the U.S. National
Highway Traffic Safety Administration Standing General Order on crash reporting.
See the [NHTSA data page](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting)
for source files, reporting scope, and documentation. Instructions for supplying
the expected local workbook are in [`data/README.md`](data/README.md).

## Citation and release status

Citation metadata are available in `CITATION.cff`. No open-source license is
assigned in this release; use and redistribution remain subject to the rights
holders' permission.
