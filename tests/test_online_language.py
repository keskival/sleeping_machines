"""The target may affect future parameters, never the current prediction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("online_language_ablation",
    Path(__file__).resolve().parents[1] / "experiments/online_language_ablation.py")
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)


class OnlineLanguageTests(unittest.TestCase):
    def test_prediction_is_independent_of_current_target(self):
        weights = np.array([.5, .5])
        experts = np.log([[.8, .2], [.3, .7]])
        left = online.predict_then_update(weights, experts, 0, .01, True)
        right = online.predict_then_update(weights, experts, 1, .01, True)
        np.testing.assert_array_equal(left[0], right[0])
        self.assertFalse(np.array_equal(left[2], right[2]))
        np.testing.assert_array_equal(weights, [.5, .5])

    def test_zero_update_arm_preserves_checkpoint(self):
        weights = np.array([.5, .5])
        experts = np.log([[.8, .2], [.3, .7]])
        probability, loss, next_weights = online.predict_then_update(weights, experts, 0, .01, False)
        np.testing.assert_array_equal(weights, next_weights)
        self.assertAlmostEqual(loss, -np.log2(probability[0]))


if __name__ == "__main__":
    unittest.main()
