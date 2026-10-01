"""Admission tests use empty queues; no training is launched."""
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'experiments/queue/run_safe.sh'


def test_slot_requires_inherited_host_reservation(tmp_path):
    queue = tmp_path / 'missing_reservation.txt'
    queue.write_text('# no training\n')
    result = subprocess.run(['bash', str(RUNNER), str(queue)], cwd=ROOT,
        env=dict(os.environ, AWS_GYM_SLOT='1', AWS_GYM_HOST_LOCK_FD='999'), capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'inherited exclusive host reservation' in result.stderr


def test_reserved_host_allows_bounded_slots_but_blocks_default_runner(tmp_path):
    queue = tmp_path / 'empty.txt'; queue.write_text('# no training\n')
    with open('/tmp/experiments-runner.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.skip('An actual guarded experiment owns the host')
        for slot in (1, 2, 3):
            result = subprocess.run(['bash', str(RUNNER), str(queue)], cwd=ROOT,
                env=dict(os.environ, AWS_GYM_SLOT=str(slot), AWS_GYM_HOST_LOCK_FD=str(lock.fileno())),
                pass_fds=(lock.fileno(),), capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, result.stderr + result.stdout
        environment = dict(os.environ)
        for name in ('AWS_GYM_SLOT', 'AWS_GYM_HOST_LOCK_FD', 'WAIT'):
            environment.pop(name, None)
        blocked = subprocess.run(['bash', str(RUNNER), str(queue)], cwd=ROOT,
            env=environment, capture_output=True, text=True, timeout=5)
        assert blocked.returncode == 1
        assert 'another runner holds' in blocked.stdout


def test_fourth_slot_is_rejected(tmp_path):
    queue = tmp_path / 'invalid_slot.txt'; queue.write_text('# no training\n')
    result = subprocess.run(['bash', str(RUNNER), str(queue)], cwd=ROOT,
        env=dict(os.environ, AWS_GYM_SLOT='4', AWS_GYM_HOST_LOCK_FD='3'), capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'requires 1..3' in result.stderr
