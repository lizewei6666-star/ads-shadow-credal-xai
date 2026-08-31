# Reproducibility notes

## Two reproduction levels

**Aggregate reproduction** regenerates the manuscript tables and figures from
the compact CSV files under `results/`. This is the recommended first check and
finishes without fitting models.

**Full reproduction** starts from `data/ads_crashes.xlsx`, reruns nested model
admission, all observed explanation statistics, 49 replacement-shadow
replicates, credal ordering, and the PAG structural extension. The workbook is
not distributed with this repository; see [`data/README.md`](../data/README.md)
for access and placement instructions. This path is computationally intensive
and writes interruption-safe checkpoints.

## Formal-run identity

The completed predictive run recorded the following source SHA-256:

```text
43483d00b1e36a0a5da520d33efd7ad0cc66c8b4bfde56b60b286f8732dc453f
```

The completed PAG source SHA-256 was:

```text
4dbad15793dfcf8c48037a80c4cfb3ceb4ce0ce80bd5ed6172b3fa9a718310b3
```

The analytical workbook SHA-256 was:

```text
5d70b42898355269607df61a7ca8b2ac3255193d5cbd846fef9398a7c577c205
```

The repository copies use portable path defaults and shorter module
documentation. These publication edits change the script file hashes but do
not change model definitions, seeds, resampling, explainers, calibration,
aggregation, or structural settings. The formal metadata are preserved in
`provenance/formal_run_meta_sanitized.json`; only local machine paths were
replaced.

## Verification

Run the deterministic release checks from the repository root:

```powershell
python scripts/verify_release.py
python -m pytest -q
```

The checks validate Python syntax, required result tables, method constants,
absence of personal absolute paths, and the release checksum manifest. When a
local analytical workbook is present, its identity and schema are also checked.
Use `python scripts/verify_release.py --require-data` to require that check.
