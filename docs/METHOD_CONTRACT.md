# Statistical method contract

This document records the settings that define the reported estimands. Changes
to these settings create a different analysis rather than an execution variant.

## Analytical sample

- 1,469 ordered crash records after exclusion of unknown vehicle age.
- 14 raw predictors grouped into temporal, roadway, crash-interaction, and
  vehicle-history domains.
- Three outcomes: No injury, Minor injury, and Moderate-plus injury.
- Preprocessing is fitted within the corresponding training boundary.

## Predictive analysis

- One stratified five-fold outer pass.
- Five-fold inner cross-validation within each outer training partition.
- Candidate models: logistic regression, random forest, extra trees, histogram
  gradient boosting, XGBoost, LightGBM, and CatBoost.
- Model admission: one-standard-error rule based on inner-CV balanced accuracy.
- Explanation sources retained separately: permutation SHAP, class-balanced
  signed Brier-loss PFI, refitted class-balanced signed Brier-loss LOCO, and
  class-balanced FIRM.
- Independent-masker background: a seeded simple random sample of at most 120
  raw training records, with common raw row positions across models in a
  partition.

## Shadow calibration

- A focal raw feature is replaced from its empirical marginal distribution
  before preprocessing and model refitting.
- Null statistics are computed by five-fold inner cross-fitting.
- 49 independent feature-replacement replicates are used for each retained
  model, explanation source, class, and feature.
- Observed and shadow statistics use the same explanation definition, feature
  grouping, class probability, weighting rule, and numerical budget.
- Calibration produces a finite empirical relevance percentile. It does not
  average the raw magnitudes of the four explanation sources.

## Credal ordering

- Only models with complete finite results for every feature, source, and
  shadow replicate enter the common support for a class and outer partition.
- A specification consists of one explanation source and one retained model in
  each outer partition.
- Outer partitions are weighted by holdout sample size.
- Relevance bounds are the lower and upper expectations over the complete
  retained specification set.
- Feature `j` strictly precedes feature `k` only when the worst-case matched
  difference remains above the numerical tolerance of `1e-12`.
- The maximal set contains features that are not strictly dominated by another
  feature. Incomparability is retained rather than resolved by a tie breaker.

## PAG structural interpretation

- Five outer training partitions and nine prespecified FCI settings produce 45
  observed PAG specifications.
- G-square conditional-independence tests and the FCI algorithm are used.
- The structural summaries are direct injury-node adjacency, within-two-step
  skeleton proximity, and recurrent exact-two-step bridge configurations.
- Each structural specification has 99 outcome-label permutation replicates.
- PAG summaries are outcome-level structural evidence. They do not modify the
  class-specific predictive bounds, partial orders, or maximal sets.

## Randomization and execution

Seeds, stable record identifiers, background row positions, permutation plans,
and method fingerprints are deterministic. Parallelism and checkpoint sharding
change scheduling only. Checkpoint reuse is rejected when any data, software,
split, model, or method fingerprint differs.

