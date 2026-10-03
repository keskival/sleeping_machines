"""Publish read-only optimizer summaries once replay's first milestone is archived."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'experiments/analysis')]
import torch
import publish_aws_language_progress as P
from aws_language_checkpoint_plasticity import analyze

MANIFEST='experiments/queue/aws_language_winner_matrix_20261003T014100Z/manifest.json'
TAG='aws_language_winner_matrix_20261003T014100Z_private_replay_10m_s7'
OUT='experiments/results/diagnostics/aws_replay_250k_checkpoint_plasticity_20261003T040000Z.json'


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    checkpoint=ROOT/'experiments/results/aws_language_progress'/f'{TAG}_milestone001.pt'
    metadata=checkpoint.with_suffix('.json')
    if (ROOT/OUT).exists():
        raise RuntimeError('Result already exists; do not overwrite')
    while True:
        publisher=json.loads((Path(MANIFEST).parent/'progress_publisher.status.json').read_text())
        # Require completed Git publication, not just files in the publication window.
        if any(r['tag']==TAG for r in publisher['published']):break
        if publisher['status']!='monitoring':raise RuntimeError('Checkpoint publisher stopped before replay milestone')
        lifecycle=json.loads((Path(MANIFEST).parent/'worker_recovery.status.json').read_text())
        if lifecycle['status'] in ('needs_review','completed'):
            raise RuntimeError('Matrix stopped before replay milestone')
        time.sleep(30)
    rows=[analyze(checkpoint)]
    for family in ('private','depth'):
        rows.append(analyze(ROOT/'experiments/results/aws_language_progress'/
            f'aws_depth8_language_20261002T234100Z_{family}_pilot_s7_milestone001.pt'))
    result=dict(status='completed',scope='Read-only saved weights and Adam moments at first archived milestone. No model execution, new FIT/DEV reads or quality evaluation. Bias-only sigmoid is not actual gate activity; stored moment direction is not a new functional update.',
        equal_target_exposure=len({r['targets'] for r in rows})==1,
        equal_optimizer_updates=len({r['updates'] for r in rows})==1,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['scripts/watch_aws_replay_checkpoint_plasticity.py','experiments/analysis/aws_language_checkpoint_plasticity.py']},
        checkpoints=rows)
    (ROOT/OUT).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pid=P.coordinator(MANIFEST);os.kill(pid,signal.SIGSTOP)
        try:
            for _ in range(30):
                if not P.git_children(pid):break
                time.sleep(1)
            else:raise RuntimeError('Coordinator Git still active')
            P.git('add','--',OUT)
            P.git('commit','--only','-m','Analyze replay and teacher moments at published 250k checkpoints','--',OUT)
            for _ in range(6):
                P.git('pull','--rebase')
                try:P.git('push','origin','main');break
                except Exception:time.sleep(5)
            else:raise RuntimeError('Push retries exhausted')
        finally:os.kill(pid,signal.SIGCONT)
    print(json.dumps(dict(status='published',result=OUT,
        equal_target_exposure=result['equal_target_exposure'])),flush=True)


if __name__=='__main__':main()
