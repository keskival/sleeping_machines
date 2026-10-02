"""Completed 90M references must become visible independently of each other."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'readable_report', Path(__file__).resolve().parents[1] / 'report/readable_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReferenceTextTests(unittest.TestCase):
    def render(self, lstm=None, transformer=None):
        return report.language_90m_reference_text(
            {'lstm90': lstm, 'tf90': transformer})

    def test_pending_controls_do_not_claim_completed_scores(self):
        text = self.render()
        self.assertIn('LSTM and four-layer Transformer reference results are pending', text)
        self.assertIn('single-seed', text)
        self.assertIn('not matched', text)

    def test_lstm_completion_is_visible_before_transformer_finishes(self):
        text = self.render(lstm=1.6)
        self.assertIn('1.600 for the LSTM', text)
        self.assertIn('four-layer Transformer reference results are pending', text)
        self.assertNotIn('LSTM reference results are pending', text)

    def test_transformer_completion_is_visible_independently(self):
        text = self.render(transformer=1.7)
        self.assertIn('1.700 for the four-layer Transformer', text)
        self.assertIn('LSTM reference results are pending', text)

    def test_both_completed_scores_replace_pending_notice(self):
        text = self.render(1.6, 1.7)
        self.assertIn('1.600 for the LSTM', text)
        self.assertIn('1.700 for the four-layer Transformer', text)
        self.assertNotIn('pending', text)


if __name__ == '__main__':
    unittest.main()
