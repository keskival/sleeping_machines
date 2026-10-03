import subprocess
import sys
import json
from pathlib import Path


def test_language_driver_smoke_end_to_end(tmp_path):
    tag = 'pytest_language_batched_smoke_tmp'
    out = Path('experiments/results/language_batched') / f'{tag}.json'
    out.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, 'experiments/language_batched_benchmark.py', '--tag', tag, '--fit', '20000',
                        '--test', '3000', '--dev', '3000', '--segment', '32', '--lanes', '16', '--passes', '.5',
                        '--depth', '2', '--payload', '8'], check=True, capture_output=True, timeout=600)
        r = json.loads(out.read_text())
        assert r['status'] == 'completed' and 0 < r['test_bpc'] < 6 and r['work']['fit_unit_special_flops_per_char_estimate'] > 0
    finally:
        out.unlink(missing_ok=True)
        (out.parent / 'checkpoints' / f'{tag}_final.pt').unlink(missing_ok=True)


def test_language_driver_cosine_and_eval_segment(tmp_path):
    tag = 'pytest_language_batched_cosine_tmp'
    out = Path('experiments/results/language_batched') / f'{tag}.json'
    out.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, 'experiments/language_batched_benchmark.py', '--tag', tag, '--fit', '20000',
                        '--test', '3000', '--dev', '3000', '--segment', '32', '--lanes', '16', '--passes', '.5',
                        '--depth', '2', '--payload', '8', '--cosine', '--eval-segment', '64'],
                       check=True, capture_output=True, timeout=600)
        r = json.loads(out.read_text())
        assert r['protocol']['schedule'].startswith('cosine') and r['eval_segment'] == 64
        assert 0 < r['test_bpc_eval_segment'] < 6
    finally:
        out.unlink(missing_ok=True)
        (out.parent / 'checkpoints' / f'{tag}_final.pt').unlink(missing_ok=True)


def test_language_driver_compiled_smoke(tmp_path):
    tag = 'pytest_language_batched_compiled_tmp'
    out = Path('experiments/results/language_batched') / f'{tag}.json'
    out.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, 'experiments/language_batched_benchmark.py', '--tag', tag, '--fit', '20000',
                        '--test', '3000', '--dev', '3000', '--segment', '32', '--lanes', '16', '--passes', '.5',
                        '--depth', '2', '--payload', '8', '--compiled', '--route-credit', 'linear_rw'],
                       check=True, capture_output=True, timeout=900)
        r = json.loads(out.read_text())
        assert r['protocol']['kernels'].startswith('compiled') and 0 < r['test_bpc'] < 6
        assert 'write slot' in r['protocol']['route_credit']
        assert r['work']['fit_unit_special_flops_per_char_estimate'] > 0
    finally:
        out.unlink(missing_ok=True)
        (out.parent / 'checkpoints' / f'{tag}_final.pt').unlink(missing_ok=True)


def test_language_driver_resume_is_exact(tmp_path):
    import torch
    base = Path('experiments/results/language_batched')
    tags = ['pytest_language_resume_a_tmp', 'pytest_language_resume_b_tmp']
    paths = ([base / f'{t}.json' for t in tags] + [base / 'checkpoints' / f'{t}.pt' for t in tags]
             + [base / 'checkpoints' / f'{t}_final.pt' for t in tags])
    for q in paths:
        q.unlink(missing_ok=True)
    common = ['--fit', '20000', '--test', '3000', '--dev', '3000', '--segment', '32', '--lanes', '16', '--passes', '.5',
              '--depth', '2', '--payload', '8', '--cosine', '--checkpoint-every', '5', '--eval-every', '4']
    try:
        subprocess.run([sys.executable, 'experiments/language_batched_benchmark.py', '--tag', tags[0], *common],
                       check=True, capture_output=True, timeout=600)
        state = torch.load(base / 'checkpoints' / f'{tags[0]}.pt', weights_only=False)
        assert state['window'] == 15
        state['args']['tag'] = tags[1]
        torch.save(state, base / 'checkpoints' / f'{tags[1]}.pt')
        subprocess.run([sys.executable, 'experiments/language_batched_benchmark.py', '--tag', tags[1], *common, '--resume'],
                       check=True, capture_output=True, timeout=600)
        a, b = (json.loads((base / f'{t}.json').read_text()) for t in tags)
        assert a['test_bpc'] == b['test_bpc'] and a['dev_bpc'] == b['dev_bpc'] and a['curve'] == b['curve']
        assert b['resumed_from_window'] == [15] and a['work'] == b['work']
        wa, wb = (torch.load(base / 'checkpoints' / f'{t}_final.pt') for t in tags)
        assert all(torch.equal(wa[k], wb[k]) for k in wa)
    finally:
        for q in paths:
            q.unlink(missing_ok=True)


def test_chunked_uint8_loader_equals_text_slice():
    import numpy as np
    sys.path.insert(0, 'experiments')
    from e120_shared_tasks import text_slice
    from language_batched_benchmark import load_text
    x = load_text(89_999_000, 25_001, chunk=7_000)
    assert x.dtype == np.uint8 and np.array_equal(x.astype(np.int64), text_slice(89_999_000, 25_001))
