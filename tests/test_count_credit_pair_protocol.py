"""Reject misleading comparisons before their scores enter the credit ledger."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('credit_analysis',ROOT/'experiments/count_credit_pair_analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


@pytest.fixture
def comparison(tmp_path,monkeypatch):
    names={
        'full':'curie_count_carrying_D2048_K4_gate_s6_20261002T032000Z',
        'minimal':'curie_count_carrying_D2048_K4_gate_minimal_p2_d1_s6_20261002T050000Z'}
    # Copies are schema fixtures only: no artificial result is written to active paths.
    source={k:json.loads((ROOT/'experiments/results/count_carrying_language'/(v+'.json')).read_text()) for k,v in names.items()}
    plan={'comparison_results':{}}
    for core in ('full','minimal'):
        for credit in (16,64):
            arm=f'{core}{credit}';row=copy.deepcopy(source[core]);row['args'].update(tag=arm,chunk=credit)
            row['source_sha256']={};path=tmp_path/(arm+'.json');path.write_text(json.dumps(row));plan['comparison_results'][arm]=path.name
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    return plan,tmp_path


def alter(directory,arm,action):
    path=directory/(arm+'.json');row=json.loads(path.read_text());action(row);path.write_text(json.dumps(row))


def test_ledger_uses_same_target_denominators(comparison):
    plan,_=comparison;report=analysis.analyze(plan)
    for row in report['common_unit_ledger']:
        assert row['fitting_targets']==8188 and row['optimizer_updates']==128
        assert row['cpu_fit_mflops_per_target']==pytest.approx(row['cpu_whole_fit_gflops']*1000/8188)
        assert row['projected_fit_mflops_per_target']==pytest.approx(row['projected_whole_fit_gflops']*1000/8188)


@pytest.mark.parametrize('tamper,reason',[
    (lambda r:r.update(development_data_sha256='different population'),'Data mismatch'),
    (lambda r:r['args'].update(seed=7),'protocol difference'),
    (lambda r:r['work'].update(optimizer_steps=127),'budget mismatch'),
])
def test_reject_unmatched_population_seed_or_updates(comparison,tamper,reason):
    plan,directory=comparison;alter(directory,'minimal64',tamper)
    with pytest.raises(ValueError,match=reason):analysis.analyze(plan)


def test_refuse_pending_quality(comparison):
    plan,directory=comparison;alter(directory,'full64',lambda r:r.update(status='running'))
    with pytest.raises(ValueError,match='Completed matching result'):analysis.analyze(plan)
