import unittest

from scripts.check_release import check_names


class ReleasePolicyTests(unittest.TestCase):
    def test_data_files_are_rejected(self):
        for name in ("crashes.xlsx", "summary.csv", "profiles.npz", "model.pkl", "outputs/roles.json", "results/table.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_names([name])

    def test_source_and_configuration_are_allowed(self):
        check_names(["launch.py", "configs/ads_schema.json", "README.md", "provenance/numerical_sources.json"])


if __name__ == "__main__":
    unittest.main()
