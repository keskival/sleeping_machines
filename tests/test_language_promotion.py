"""Development-only promotion should favor affordable capacity without hiding loss."""
import pytest

from scripts.run_language_scaling_ladder import choose_width


def test_smaller_model_wins_when_quality_is_within_declared_margin():
    assert choose_width({128:2.20,256:2.18},.03)==128


def test_larger_model_is_selected_for_material_quality_gain():
    assert choose_width({128:2.24,256:2.18},.03)==256


@pytest.mark.parametrize('scores,tolerance', [({},.03),({128:float('nan')},.03),({128:2.1},-.01)])
def test_invalid_promotion_evidence_is_rejected(scores,tolerance):
    with pytest.raises(ValueError):choose_width(scores,tolerance)
