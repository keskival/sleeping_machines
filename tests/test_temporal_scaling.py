"""Limits of the declared full-bank substitution and sampling hypothesis."""
import math
import runpy
from pathlib import Path

API=runpy.run_path(str(Path(__file__).resolve().parents[1]/'report/language_scaling.py'))


def test_scoring_retained_prevents_unbounded_arithmetic_saving():
    row=API['full_bank_comparison'](context=10**10)
    assert math.isclose(row['transformer_inference']/row['ours_inference'],2,rel_tol=1e-5)
    assert math.isclose(row['transformer_training']/row['ours_training'],1.2,rel_tol=1e-5)


def test_value_only_and_total_access_have_different_limits():
    n=8192;row=API['full_bank_comparison'](context=n)
    assert row['transformer_value_bytes']/row['ours_value_bytes']==n
    assert 1.99<row['transformer_key_value_bytes']/row['ours_key_value_bytes']<2


def test_sampling_does_not_remove_missing_history_bias():
    error=API['winner_average_error_bound']
    assert error(4,1)==2
    assert error(4,16)==.5
    assert error(4,10**12,.25,2)>=1
