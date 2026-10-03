"""Freeze a deferred sparse-quality ladder from a completed native fit, no tensor runtime."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from sparse_rescore_protocol import (CONTRACT_SOURCES, RESCORE_SOURCES, parent_quality,
                                     sha, source_hashes, window_plan)


def dump(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(parent_name, tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in tag):
        raise ValueError('Unique alphanumeric/underscore tag required')
    parent_path = ROOT / parent_name
    parent = json.loads(parent_path.read_text())
    a = parent['args']
    if a.get('skip_init_from', 0) or a.get('tie_pools', False):
        raise ValueError('Current admission covers nongrown, untied models')
    if a['dev'] < 8192 or a['test'] < 8192:
        raise ValueError('Full split must contain a nontrivial prefix')
    for segment in (a['segment'], a['eval_segment']):
        parent_quality(parent, 'dev', segment)
        parent_quality(parent, 'test', segment)
    if a['segment'] != 128 or a['eval_segment'] != 256:
        raise ValueError('Current ladder is source-exact T128/T256')
    for name, digest in parent['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed completed-fit producer: ' + name)
    weights = ROOT / parent['final_weights']
    weights_hash = sha(weights)  # actual weights must exist; never synthesize them
    parent_hash = sha(parent_path)
    contract_sources, rescore_sources = source_hashes(CONTRACT_SOURCES), source_hashes(RESCORE_SOURCES)
    directory = ROOT / 'experiments/queue' / tag
    directory.mkdir(exist_ok=False)
    guard = dict(threads=1, address_space_kib=3000000, rss_kib=1250000, min_available_mib=8192)
    warning = '# PREPARED UNRUN; physical-host reservation required, never use a container-local free lock as admission.\n'
    jobs = []
    name = tag + '_contracts'
    queue = directory / (name + '.txt')
    manifest_path = directory / (name + '.sources.json')
    arguments = dict(result=parent_name, out=f'experiments/results/diagnostics/{name}.json',
                     manifest=str(manifest_path.relative_to(ROOT)))
    queue.write_text(warning + '# One thread; VMS3000000KiB RSS1250000KiB MIN_AVAIL8192MiB timeout420s; run_safe only.\n' +
                     name + ' experiments/prepacked_sparse_contracts.py ' +
                     ' '.join('--' + k + ' ' + v for k, v in arguments.items()) + '\n')
    contract_manifest = dict(status='prepared_unrun', source_sha256=contract_sources,
        parent_result_sha256=parent_hash, weights_sha256=weights_hash,
        queue=str(queue.relative_to(ROOT)), queue_sha256=sha(queue), arguments=arguments,
        guards=dict(guard, timeout_s=420), physical_reservation='Required outside Docker on the actual host',
        scope='Completed 90M quality checkpoint, not an admission pilot; native parity remains unrun')
    dump(manifest_path, contract_manifest)
    required = dict(manifest=arguments['manifest'], manifest_sha256=sha(manifest_path), output=arguments['out'])
    jobs.append(dict(stage='contracts', queue=str(queue.relative_to(ROOT)), manifest=arguments['manifest'],
                     output=arguments['out'], guards=contract_manifest['guards']))
    prefixes = {}
    for split, length, segment in [('dev', 8192, 128), ('dev', 8192, 256)] + [
            (split, a[split], segment) for split in ('dev', 'test') for segment in (128, 256)]:
        name = f'{tag}_{split}_{length}_T{segment}'
        queue = directory / (name + '.txt')
        manifest_path = directory / (name + '.sources.json')
        output = f'experiments/results/diagnostics/{name}.json'
        lanes = max(1, a['lanes'] * a['segment'] // segment)
        plan = window_plan(length, segment, lanes)
        timeout_s = 600 if length == 8192 else 21600
        queue.write_text(warning + '# Paired quality only; full jobs require passed DEV prefix. run_safe only.\n' +
                         f'# One thread; VMS3000000KiB RSS1250000KiB MIN_AVAIL8192MiB timeout{timeout_s}s.\n' +
                         name + ' experiments/trained_sparse_rescore.py --manifest ' + str(manifest_path.relative_to(ROOT)) + '\n')
        item = dict(status='prepared_unrun', parent=parent_name, parent_sha256=parent_hash,
            weights_sha256=weights_hash, source_sha256=rescore_sources,
            queue=str(queue.relative_to(ROOT)), queue_sha256=sha(queue), output=output,
            split=split, length=length, segment=segment,
            execution_plan={k: v for k, v in plan.items() if k != 'batches'},
            required_contracts=required, guards=dict(guard, timeout_s=timeout_s),
            physical_reservation='Required outside Docker; no auto-admission or running-chain displacement',
            timeout_scope='Conservative ceiling, not a throughput forecast. Reassess physical resources and measured prefix workload before full admission; new settings require new tags.')
        if length != 8192:
            item['required_prefix'] = prefixes[segment]
        dump(manifest_path, item)
        if length == 8192:
            prefixes[segment] = dict(output=output, manifest=str(manifest_path.relative_to(ROOT)),
                                     manifest_sha256=sha(manifest_path))
        jobs.append(dict(stage='prefix' if length == 8192 else 'full', split=split, segment=segment,
                         queue=item['queue'], manifest=str(manifest_path.relative_to(ROOT)),
                         output=output, guards=item['guards'], execution_plan=item['execution_plan']))
    dump(directory / 'manifest.json', dict(status='prepared_unrun', parent=parent_name,
        parent_sha256=parent_hash, weights_sha256=weights_hash, jobs=jobs,
        producer_sha256=sha(Path(__file__)),
        scope='Contracts -> two DEV prefixes -> full DEV/test T128/T256. Each is one guarded job; no scheduler started. Preserve original pilot queue and active owner chains.'))
    print(json.dumps(dict(status='prepared_unrun', jobs=len(jobs), directory=str(directory.relative_to(ROOT)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    prepare(args.parent, args.tag)
