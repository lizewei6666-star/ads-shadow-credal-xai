import unittest

import numpy as np
import pandas as pd

from shadow_xai import controls, logistic, numerics, population, sampling


class NumericalProtocolTests(unittest.TestCase):
    def test_seven_learner_parameters(self):
        expected = {
            "lr": ("max_iter", 3000),
            "rf": ("n_estimators", 600),
            "et": ("n_estimators", 800),
            "hgb": ("max_iter", 350),
            "xgb": ("n_estimators", 500),
            "lgbm": ("n_estimators", 500),
            "catboost": ("iterations", 500),
        }
        for model, (parameter, value) in expected.items():
            with self.subTest(model=model):
                self.assertEqual(
                    numerics.build_estimator(model, 42, 1).get_params()[parameter], value
                )

    def test_sampling_is_reproducible(self):
        a, ya, sa = sampling.sample("rare_interaction", master=2026101004)
        b, yb, sb = sampling.sample("rare_interaction", master=2026101004)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(ya, yb)
        self.assertEqual(sa, sb)
        self.assertEqual(len(a), 1500)
        self.assertEqual(len(sa.feature_cols), 10)

    def test_bootstrap_keeps_roles_and_classes_separate(self):
        raw, y, schema = sampling.sample("additive", master=2026101003)
        folds = controls.partitions(y)
        original = controls.split_roles(folds, y, 0)
        for repeat in range(1, 13):
            roles = controls.split_roles(folds, y, 0, repeat)
            for reference, current in zip(original, roles):
                self.assertTrue(set(current) <= set(reference))
                np.testing.assert_array_equal(np.bincount(y[current]), np.bincount(y[reference]))
            self.assertFalse(set(roles[0]) & set(roles[1]))
            self.assertFalse(set(roles[0]) & set(roles[2]))

    def test_shadows_use_fitting_donors(self):
        raw, y, schema = sampling.sample("additive", master=2026101003)
        fit, _, _ = controls.split_roles(controls.partitions(y), y, 0, 1)
        augmented, expanded, donors = controls.augment(raw, schema, fit, 0, 1)
        self.assertEqual(len(expanded.feature_cols), 30)
        for name, record in donors.items():
            self.assertTrue(set(record["donor_ids"]) <= set(fit))
            np.testing.assert_array_equal(
                augmented[name], raw.iloc[record["donor_ids"]][record["parent"]]
            )
        swapped = controls.oriented(augmented, schema.feature_cols, 1)
        aligned = controls.aligned_names(schema.feature_cols, expanded, 1)
        np.testing.assert_array_equal(
            swapped[aligned].to_numpy(), augmented[expanded.feature_cols].to_numpy()
        )

    def test_tuning_never_splits_duplicate_records(self):
        _, y, _ = sampling.sample("additive", master=2026101003)
        fit, _, _ = controls.split_roles(controls.partitions(y), y, 0, 1)
        self.assertLess(len(np.unique(fit)), len(fit))
        for train, valid in logistic.tuning_splits(fit, y, "additive", 1):
            self.assertFalse(set(train) & set(valid))
            np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.sort(fit))

    def test_population_targets(self):
        for scenario in population.SCENARIOS:
            _, mass, probabilities, prior = population.population(scenario)
            np.testing.assert_allclose(mass @ probabilities, prior, rtol=0, atol=1e-11)
            scores = population.population_scores(scenario, prior)
            self.assertEqual(scores.shape, (4, 3, 10))
            np.testing.assert_array_equal(scores[..., 3:], 0)
            if scenario == "null":
                self.assertFalse(population.truth(scores).any())


if __name__ == "__main__":
    unittest.main()
