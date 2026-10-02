"""Recovery must reject live reservations and incomparable completed evidence."""
import hashlib
import json

import pytest

from scripts import run_local_frozen_recovery as recovery


def plan_at(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda *a, **k: 'main\n')
    queue = tmp_path / 'queue.txt'
    queue.write_text('job experiments/driver.py --tag job\n')
    return dict(host=recovery.os.uname().nodename, source_sha256={},
        jobs=[dict(tag='job', queue='queue.txt', queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest())],
        absent_predecessors=[dict(pid=100, start_ticks='42')])


def test_live_original_reservation_blocks_recovery(tmp_path, monkeypatch):
    plan = plan_at(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'identity', lambda pid: dict(state='T', start_ticks='42'))
    with pytest.raises(ValueError, match='still alive'):
        recovery.verify(plan)


def test_reused_pid_does_not_get_signalled_or_block_recovery(tmp_path, monkeypatch):
    plan = plan_at(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'identity', lambda pid: dict(state='S', start_ticks='43'))
    recovery.verify(plan)


def test_multiple_jobs_in_one_queue_are_rejected_even_with_matching_hash(tmp_path, monkeypatch):
    plan = plan_at(tmp_path, monkeypatch)
    queue = tmp_path / 'queue.txt'
    queue.write_text(queue.read_text() + 'second experiments/driver.py\n')
    plan['jobs'][0]['queue_sha256'] = hashlib.sha256(queue.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Exactly one'):
        recovery.verify(plan)


def comparison_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    jobs = []
    for strength in (0, 1):
        tag = 'credit' + str(strength)
        row = dict(status='completed', source_sha256={},
            args=dict(tag=tag, state_credit=strength, seed=6), data_sha256=dict(fit='same', dev='same'),
            final=dict(dev=dict(accuracy=.5 + .1 * strength, nll=1. - .2 * strength)),
            work=dict(total_training_unit_special_flops=100 + 50 * strength))
        (tmp_path / (tag + '.json')).write_text(json.dumps(row))
        jobs.append(dict(tag=tag, stage='pilot', result=tag + '.json'))
    return dict(jobs=jobs)


def test_matching_completed_pair_can_pass_gate(tmp_path, monkeypatch):
    result = recovery.compare(comparison_plan(tmp_path, monkeypatch))
    assert result['followup_gate_passed']
    assert result['whole_fitting_work_ratio'] == 1.5


@pytest.mark.parametrize('field', ['seed', 'dev'])
def test_changed_seed_or_population_cannot_form_pair(tmp_path, monkeypatch, field):
    plan = comparison_plan(tmp_path, monkeypatch)
    path = tmp_path / 'credit1.json'
    row = json.loads(path.read_text())
    if field == 'seed':
        row['args']['seed'] = 7
    else:
        row['data_sha256']['dev'] = 'different'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Matched data|Only added credit'):
        recovery.compare(plan)
