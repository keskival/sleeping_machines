"""Withdrawn target-dependent language evidence cannot reenter publications."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
QUARANTINE=ROOT/'experiments/archive/invalid_protocol/target_leakage_20261002'
spec=importlib.util.spec_from_file_location('quarantine_report',ROOT/'report/readable_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_quarantined_files_have_left_active_results_with_exact_audit_bytes():
    manifest=json.loads((QUARANTINE/'manifest.json').read_text())
    assert manifest['status']=='invalid_protocol' and not manifest['eligible_for_benchmark']
    assert len(manifest['files'])==10
    for row in manifest['files']:
        assert not (ROOT/row['original_path']).exists()
        assert hashlib.sha256((ROOT/row['quarantined_path']).read_bytes()).hexdigest()==row['sha256']


@pytest.mark.parametrize('path',[
    'e79/race_mixer_D90000000_K7_e77none.json',
    'e63/mix_D10000000_K6.json',
    'aws_20260929/aws_e79wk_D1M_20260929/race_mixer_D1000000_K5_e77none_wk.json',
])
def test_stale_result_references_fail_with_the_causal_replacement(path):
    with pytest.raises(ValueError,match='Quarantined.*causal E173'):
        report.read(path)
