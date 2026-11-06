import unittest

import numpy as np
import pandas as pd

from shadow_xai.evidence import graph_for, grade, ratios


class DirectionTests(unittest.TestCase):
    def center(self):
        values = np.zeros((13, 2, 4, 3, 6))
        values[..., 0] = 3
        values[..., 1] = 1
        return values

    def test_both_roles_required(self):
        values = self.center()
        values[:, 1, :, :, 0] = .5
        self.assertFalse(graph_for(values, 2, "shadow", "Joint4")[0].any())

    def test_both_shadows_required(self):
        values = self.center()
        values[..., 3] = 4
        self.assertFalse(graph_for(values, 2, "shadow", "Joint4")[0].any())

    def test_only_leader_needs_qualification(self):
        values = self.center()
        values[..., 4:] = 2
        graphs, counts, _ = graph_for(values, 2, "shadow", "Joint4")
        self.assertTrue(graphs[8, :, 0, 1].all())
        self.assertTrue((counts[:, 0, 1] == 12).all())

    def test_support_must_exceed_opposition(self):
        values = self.center()
        values[:, :, 2, :, :2] = [1, 3]
        values[:, :, 3, :, :2] = [0, 0]
        self.assertTrue(graph_for(values, 2, "shadow", "Joint4")[0][8, :, 0, 1].all())
        values[:, :, 3, :, :2] = [1, 3]
        self.assertFalse(graph_for(values, 2, "shadow", "Joint4")[0].any())

    def test_cycle_removed(self):
        values = np.zeros((13, 2, 4, 3, 9))
        for source, ranking in enumerate(([3, 2, 1], [1, 3, 2], [2, 1, 3])):
            values[:, :, source, :, :3] = ranking
        self.assertFalse(graph_for(values, 3, "shadow", "Joint4")[0].any())

    def test_original_candidate_and_support_threshold(self):
        values = self.center()
        values[9:, ..., 0] = .5
        graphs, counts, _ = graph_for(values, 2, "shadow", "Joint4")
        self.assertTrue(graphs[8, :, 0, 1].all())
        self.assertFalse(graphs[9, :, 0, 1].any())
        self.assertTrue((counts[:, 0, 1] == 8).all())
        values[0, ..., 0] = .5
        self.assertFalse(graph_for(values, 2, "shadow", "Joint4")[0][:, :, 0, 1].any())

    def test_invalid_input(self):
        values = self.center()
        values[0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            graph_for(values, 2, "shadow", "Joint4")

    def test_reference_ties_are_not_correct_directions(self):
        scores = np.array([[2., 1., 1.]]*4)
        graph = np.array([[False, True, False], [False, False, True], [True, False, False]])
        result = grade(graph, scores)
        self.assertEqual(result, dict(correct=1, wrong=2, declared=3, true_total=2))

    def test_empty_denominators_remain_undefined(self):
        frame = pd.DataFrame([dict(correct=0, declared=0, true_total=0)])
        result = ratios(frame)
        self.assertTrue(np.isnan(result.precision.iloc[0]))
        self.assertTrue(np.isnan(result.recovery.iloc[0]))


if __name__ == "__main__":
    unittest.main()
