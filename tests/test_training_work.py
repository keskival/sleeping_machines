"""Training estimates must retain padding, optimizer steps and fused attention."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("estimate_training_work", ROOT / "experiments/estimate_training_work.py")
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)


class TrainingWorkTests(unittest.TestCase):
    def test_padding_and_short_final_batch_are_charged(self):
        shape = work.transformer_contractions([3, 3, 3, 3, 3], 4, 2, 6, 7)
        self.assertEqual(shape["optimizer_steps"], 4)
        self.assertEqual(shape["padded_tokens"], 30)
        self.assertEqual(shape["padded_token_pairs"], 90)

    def test_fused_attention_trace_reproduces_its_own_batch(self):
        records = json.loads((ROOT / "experiments/results/e172/complete_work_v2_20260930.json").read_text())
        row = next(r for r in records["rows"] if r["task"] == "dvs")
        sample = row["transformer"]
        shape = work.transformer_contractions(row["query_events"], 4, 1, 6, 11, True)
        # A bare contraction-operator sum misses the fused attention kernel.
        self.assertGreater(shape["flops"], 4 * work.contractions(sample["stages"]["forward_and_loss"]))
        stages = work.scaled_stages(sample, shape["flops"], 1, shape["flops"])
        self.assertEqual(stages, {k: v["arithmetic_flops"] for k, v in sample["stages"].items()})

    def test_adam_is_charged_per_step_instead_of_per_query(self):
        records = json.loads((ROOT / "experiments/results/e172/complete_work_v2_20260930.json").read_text())
        sample = records["rows"][0]["common"]
        matrix = work.contractions(sample["stages"]["forward_and_loss"])
        stages = work.scaled_stages(sample, matrix * 100, 25)
        self.assertEqual(stages["optimizer"], sample["stages"]["optimizer"]["arithmetic_flops"] * 25)
        self.assertEqual(stages["backward"], sample["stages"]["backward"]["arithmetic_flops"] * 100)


if __name__ == "__main__":
    unittest.main()
