# Changelog

## Unreleased

- Moved the current analysis into `shadow_xai`, removing historical experiment directories and unused analysis entry points.
- Added tests for shadow sampling, resampling roles, tuning splits, and population targets.
- Kept the current study's comparator analyses and command-line interface.

## 2.0.0

- Added matched-shadow relevance qualification, joint directional comparisons, and bootstrap recurrence tiers.
- Added simulation generators, population-reference calculations, and paired comparator workflows.
- Unified simulation and ADS execution under `launch.py`, with checkpoint verification and bounded parallelism.
- Set the main reporting threshold to eight of twelve supporting resamples.
- Removed the earlier percentile-ranking and PAG workflows from the supported interface.
- Distributed code, configuration, and tests only; datasets and computed results remain outside the repository.
