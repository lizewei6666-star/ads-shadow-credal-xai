from pathlib import Path
import tempfile
import unittest

from shadow_xai.runtime import run_lock, write, read


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            path = Path(folder) / "value.json"
            write(path, {"value": 3})
            self.assertEqual(read(path), {"value": 3})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            with run_lock(folder):
                with self.assertRaises(RuntimeError):
                    with run_lock(folder):
                        pass
            with run_lock(folder):
                pass


if __name__ == "__main__":
    unittest.main()
