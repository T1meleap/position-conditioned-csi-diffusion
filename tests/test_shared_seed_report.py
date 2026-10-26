import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shared_seed_report import SCORES, aggregate_runs


class TrainingSeedAggregationTests(unittest.TestCase):
    def setUp(self):
        self.results = []
        for seed, offset in zip((2026, 2027, 2028), (1, 2, 3)):
            scores = [{"sampling_seed": s, **{m: float(v + offset) for m in SCORES}}
                      for s, v in zip((12345, 23456, 34567), (-100, 0, 100))]
            self.results.append({"training_seed": seed, "config": {"seed": seed, "mode": "independent", "timesteps": 200,
                                 "test_sampling_seeds": [12345, 23456, 34567]},
                                 "parameters": 10, "train_samples": 16, "validation_samples": 8, "test_samples": 16,
                                 "per_seed": scores, "aggregate": {m + "_mean": float(offset) for m in SCORES}})

    def test_averages_sampling_then_training_and_uses_sample_sd(self):
        result = aggregate_runs(self.results)
        self.assertAlmostEqual(result[SCORES[0]]["mean"], 2)
        self.assertAlmostEqual(result[SCORES[0]]["training_seed_std"], 1)
        self.assertEqual(list(result[SCORES[0]]["per_training_seed"]), ["2026", "2027", "2028"])

    def test_rejects_duplicate_or_missing_training_seed(self):
        with self.assertRaises(ValueError):
            aggregate_runs(self.results[:2])
        duplicate = copy.deepcopy(self.results)
        duplicate[2] = copy.deepcopy(duplicate[1])
        with self.assertRaises(ValueError):
            aggregate_runs(duplicate)

    def test_rejects_changed_configuration_and_incorrect_means(self):
        other = copy.deepcopy(self.results)
        other[2]["config"]["timesteps"] = 100
        with self.assertRaises(ValueError):
            aggregate_runs(other)
        other = copy.deepcopy(self.results)
        other[0]["aggregate"][SCORES[0] + "_mean"] = 0
        with self.assertRaises(ValueError):
            aggregate_runs(other)


if __name__ == "__main__":
    unittest.main()
