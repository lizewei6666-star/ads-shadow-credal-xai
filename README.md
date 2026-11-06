# Matched-shadow feature prioritization

Research code for class-specific feature priorities in automated driving system crash analysis. Matched shadows qualify relevance within each explanation source. SHAP, PFI, LOCO, and FIRM then contribute directional comparisons across admissible learners. Twelve bootstrap resamples determine recurrence-based trust tiers.

This repository contains code only. Raw records, synthetic datasets, fitted models, and precomputed results are not included.

## Installation

The fitting environment uses Python 3.10.20 and the package versions in `requirements.txt`.

```powershell
conda env create -f environment.yml
conda activate ads-shadow-xai-v2
python -m unittest discover -s tests -v
```

## Simulation

The generator creates nine datasets locally: three scenarios and three random seeds, with 1,500 records and ten predictors per dataset. Preflight performs model admission and one fitting unit. The full run resumes completed work.

```powershell
python launch.py run --dataset simulation --out outputs/simulation --stage preflight --jobs 2
python launch.py run --dataset simulation --out outputs/simulation --jobs 4
```

## ADS Analysis

The original analysis workbook is required. Its schema and checksum are specified in `configs/ads_schema.json`. Keep the workbook outside version control and supply its local path:

```powershell
python launch.py run --dataset ads --data "data/private/ads_crashes.xlsx" --out outputs/ads --jobs 4
```

Outputs are written to the selected local directory. After an interruption, rerun the same command to reuse verified checkpoints. Each worker uses one numerical thread. Progress is printed in the terminal and recorded in `progress.jsonl`.

Existing local profiles can be summarized without refitting:

```powershell
python launch.py summarize --dataset simulation --input outputs/simulation --out outputs/simulation/tables
```

## Organization

- `shadow_xai/numerics.py`: learner fitting and four explanation statistics.
- `shadow_xai/controls.py`: matched shadows and resampling roles.
- `shadow_xai/population.py`: simulation mechanisms and reference targets.
- `shadow_xai/evidence.py`: source integration, recurrence tiers, and table export.
- `shadow_xai/runtime.py`: execution and checkpoint management.
- `configs/`: analysis settings and input schema.
- `tests/`: decision and execution checks using synthetic test inputs.
- `provenance/`: distribution file checksums.

Run analyses through `launch.py`. [Method details](docs/method.md) describe the analysis settings and directional decision rules. Plain fits, single-source decisions, and fixed-penalty fits are retained as comparators for the current study.

Checkpoints are tied to the code and settings that produced them. Use a new output directory when changing code versions. Previously completed profiles can still be passed to `summarize` without refitting.

## Citation and Permissions

Software citation metadata is in `CITATION.cff`. The authors have not assigned an open-source license. Contact the repository owner regarding reuse permissions. Third-party dependencies retain their own licenses.
