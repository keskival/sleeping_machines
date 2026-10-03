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
