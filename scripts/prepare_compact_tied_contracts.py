"""Freeze deferred compact-map contracts, no numerical execution or scheduling."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from compact_tied_inference_contracts import SOURCES, sha


def prepare(tag, parent_name=None):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in tag):
        raise ValueError('Unique alphanumeric/underscore tag required')
    parent, parent_hash, weights_hash = None, None, None
    if parent_name:
        parent_path = ROOT / parent_name
        parent = json.loads(parent_path.read_text())
        if (parent.get('status') != 'completed' or not parent['args'].get('tie_pools')
                or parent['args'].get('skip_init_from', 0)):
            raise ValueError('Actual completed nongrown tied-map parent required')
        parent_hash = sha(parent_path)
        weights_hash = sha(ROOT / parent['final_weights'])
        for name, digest in parent['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Changed completed-fit source: ' + name)
    queue = ROOT / 'experiments/queue' / (tag + '.txt')
    manifest_path = queue.with_suffix('.sources.json')
    if queue.exists() or manifest_path.exists():
        raise ValueError('Unique unused queue and manifest required')
    arguments = dict(manifest=str(manifest_path.relative_to(ROOT)),
                     out=f'experiments/results/diagnostics/{tag}.json', result=parent_name)
    command = tag + ' experiments/compact_tied_inference_contracts.py ' + ' '.join(
        '--' + k + ' ' + v for k, v in arguments.items() if v is not None)
    with queue.open('x') as stream:
        stream.write('# PREPARED UNRUN. Actual physical-host reservation required; never infer it from Docker-local locks.\n'
                     '# run_safe only: threads1 VMS3000000KiB RSS1250000KiB available8192MiB timeout420s.\n'
                     '# Preserve existing curie/AWS training chains. No quality or speed claim from synthetic cases.\n'
                     + command + '\n')
    result = dict(status='prepared_unrun', source_sha256={name: sha(ROOT / name) for name in SOURCES},
                  arguments=arguments, parent_sha256=parent_hash, weights_sha256=weights_hash,
                  queue=str(queue.relative_to(ROOT)), queue_sha256=sha(queue),
                  preparation_source_sha256=sha(Path(__file__)),
                  guards=dict(threads=1, address_space_kib=3000000, rss_kib=1250000,
                              min_available_mib=8192, timeout_s=420),
                  physical_reservation='Required after owner chains or on a genuinely separate idle host; no waiter or coordinator launched',
                  scope='Native synthetic/tied FIT parity pending; no optimizer, training or heldout quality assignment',
                  planned_synthetic_execution=dict(pair_cases=32, worker_lifecycles=8,
                      forward_calls=144, evaluated_lane_positions=5616, active_lane_positions=3888,
                      backward_calls=0, optimizer_updates=0))
    if parent is not None:
        result['planned_actual_execution'] = dict(pair_cases=8, worker_lifecycles=2,
            forward_calls=36, evaluated_lane_positions=3456, active_lane_positions=2628)
    with manifest_path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='prepared_unrun', queue=result['queue'], actual_tied_parent=bool(parent))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--parent')
    args = parser.parse_args()
    prepare(args.tag, args.parent)
