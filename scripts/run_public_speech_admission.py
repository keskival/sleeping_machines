"""Owner-host serial admission conductor; every numerical stage uses run_safe.

Rejects the unshared workspace container. Run in tmux after existing owners'
campaigns, or in their declared scheduler order. No benchmark runs here directly.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from public_speech_admission import preflight
from public_speech_packets import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if Path('/.dockerenv').exists():
        raise ValueError('No workspace-container training admission; physical owner reservation required')
    if not os.environ.get('TMUX'):
        raise ValueError('Preserve this campaign in the owning physical host tmux session')
    path = ROOT / args.manifest
    if file_sha(path) != args.manifest_sha256:
        raise ValueError('Changed frozen manifest')
    manifest = json.loads(path.read_text())
    status_path = path.parent / 'owner_status.json'
    if status_path.exists():
        raise ValueError('Preserve prior conductor state; use the individual exact recovery queue')
    state = dict(status='preflight', host=socket.gethostname(),
        started_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=args.manifest_sha256,
        completed=[], numerical_work_in_conductor=False, official_test_read=False)
    def persist():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(dict(status=state['status'], completed=state['completed'])), flush=True)
    persist()
    try:
        for stage in ('contracts', 'smoke', 'pilot'):
            manifest, cfg = preflight(path, stage, args.manifest_sha256)
            output = ROOT / cfg['output']
            if output.exists():
                result = json.loads(output.read_text())
            else:
                info = {}
                for line in Path('/proc/meminfo').read_text().splitlines():
                    if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:')):
                        key, value, *_ = line.split()
                        info[key.rstrip(':')] = int(value)
                state['host_memory_kib'] = info
                state['last_stage'] = stage
                if info['MemAvailable'] < cfg['min_available_mb'] * 1024:
                    raise ValueError('Host available-memory floor prevents admission')
                if shutil.which('nvidia-smi'):
                    occupancy = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
                    state['gpu_occupancy'] = occupancy.stdout.strip()
                else:
                    state['gpu_occupancy'] = 'No visible nvidia-smi; CPU-only stages'
                state['status'] = 'waiting_for_physical_run_safe_lock_' + stage
                persist()
                env = dict(os.environ, WAIT='1', WAIT_TIMEOUT_S='86400',
                    MEM_CAP_KB=str(cfg['vms_cap_kb']), MEM_CAP_RSS_KB=str(cfg['rss_cap_kb']),
                    MIN_AVAIL_MB=str(cfg['min_available_mb']), JOB_TIMEOUT_S=str(cfg['timeout_s']))
                subprocess.run(['bash', 'experiments/queue/run_safe.sh', cfg['queue']], cwd=ROOT,
                               env=env, check=True)
                result = json.loads(output.read_text())
            if (result['status'] != 'completed' or result['config'] != cfg or
                    result['source_sha256'] != manifest['source_sha256'] or
                    result['manifest_sha256'] != args.manifest_sha256):
                raise ValueError('Exact completed stage required')
            state['completed'].append(stage)
            state['status'] = 'completed_' + stage
            persist()
        state['small_fit_learning_passed'] = result['small_fit_learning_passed']
        state['status'] = 'admission_complete_no_benchmark_claim' if result['small_fit_learning_passed'] else 'learning_gate_failed_no_promotion'
        persist()
    except Exception as exc:
        state.update(status='stopped_before_promotion', reason=str(exc))
        persist()
        raise


if __name__ == '__main__':
    main()
