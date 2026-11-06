# Analysis Settings

The candidate learners are logistic regression, random forest, Extra Trees, histogram gradient boosting, XGBoost, LightGBM, and CatBoost. Admission follows the class-balanced one-standard-error rule. Explanation profiles are pooled by the coordinatewise median within each source and partition. ADS profiles are then aggregated across partitions using external-block size weights, separately for calibration and external evaluation.

Two shadows per original predictor are sampled from the fitting sample's empirical marginal distribution. A second orientation exchanges their positions. Their identities are realigned before averaging. Bootstrap copies of the same record share the same shadow values.

A source supports `A > B` when A exceeds both its matched shadows and B's native importance score on both evaluation blocks. Only A must qualify. Joint4 requires at least two supporting sources and strictly more support than opposition. Comparisons use a tolerance of `1e-12`.

Edges within nontrivial strongly connected components of the original joint graph are removed. The resulting candidate directions are assessed in twelve bootstrap resamples that preserve the original fold and outcome strata. Learner admission remains fixed; logistic tuning is repeated within fitting records. Candidate relations are grouped as high support (8-12), medium support (4-7), or low support (0-3). The main analysis uses high support. Noncandidate pairs are not assigned to the low tier.

The simulation crosses additive, rare-interaction, and null mechanisms with seeds 2026101003, 2026101004, and 2026101005. Each dataset contains 1,500 records and ten independent predictors, partitioned into 900 fitting, 300 calibration, and 300 external-evaluation records. X1-X3 determine outcomes in the signal scenarios; the null scenario contains no informative predictor. The generator and exact population reference are defined in `shadow_xai/sampling.py` and `shadow_xai/population.py`.

Simulation explanation budgets are 12 SHAP background records, one nominal permutation cycle, and three PFI permutations. ADS budgets are 120 background records, five cycles, and 30 PFI permutations. Comparators within each experiment share these budgets.

Simulation precision is the number of correct declarations divided by all declarations. Recovery divides correct declarations by the reference-relation count. A reference direction must hold across all four population source targets. Precision is undefined for an empty declaration set; recovery is undefined for the null reference. Ratios are calculated after summing counts. Recurrence tiers are not calibrated probabilities of correctness.
