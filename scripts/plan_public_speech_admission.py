"""Prepare three frozen one-job queues; no numerical imports or execution."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from public_speech_packets import file_sha


def local_dependencies(starts):
    pending, seen = [ROOT / name for name in starts], set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            candidates = []
            if isinstance(node, ast.Import):
                modules = [x.name for x in node.names]
                for module in modules:
                    for base in (ROOT, ROOT / 'experiments'):
                        part = base / module.replace('.', '/')
                        candidates.extend((part.with_suffix('.py'), part / '__init__.py'))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    bases = [base]
                else:
                    bases = [ROOT, ROOT / 'experiments']
                for base in bases:
                    part = base / (node.module or '').replace('.', '/')
                    if node.module:
                        candidates.extend((part.with_suffix('.py'), part / '__init__.py'))
                    for alias in node.names:
                        child = part / alias.name
                        candidates.extend((child.with_suffix('.py'), child / '__init__.py'))
            for candidate in candidates:
                if candidate.is_file() and candidate.is_relative_to(ROOT):
                    pending.append(candidate)
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    if not args.stamp.isalnum():
        raise ValueError('Plain unique timestamp required')
    prefix = 'public_speech_admission_' + args.stamp
    directory = ROOT / 'experiments/queue' / prefix
    directory.mkdir()
    sources = local_dependencies(['experiments/public_speech_admission.py',
        'experiments/public_speech_packets.py', 'scripts/plan_public_speech_admission.py',
        'scripts/run_public_speech_admission.py',
        'tests/test_public_speech_packets_stdlib.py'])
    sources.update({name: file_sha(ROOT / name) for name in [
        'experiments/queue/run_safe.sh',
        'experiments/theory/149_public_benchmarks_and_information_preserving_admission.md']})
    common = dict(payload=16, depth=2, heads=2, pool=2, seed=6, packet_us=16000,
        lanes=4, lr=.003, clip=1., weight_decay=.01, backend='compiled', credit='linear',
        rss_cap_kb=2_000_000, vms_cap_kb=6_000_000, min_available_mb=8192)
    stages = {}
    for name, fit, dev, epochs, timeout, requires in [
            ('contracts', 0, 0, 2, 600, []), ('smoke', 8, 8, 1, 600, ['contracts']),
            ('pilot', 64, 64, 4, 1800, ['contracts', 'smoke'])]:
        tag = prefix + '_' + name
        stages[name] = dict(common, tag=tag, fit=fit, dev=dev, epochs=epochs,
            timeout_s=timeout, requires=requires,
            output='experiments/results/public_speech_admission/' + tag + '.json',
            queue=str((directory / (tag + '.txt')).relative_to(ROOT)))
    manifest = dict(status='prepared_unrun', architecture='Existing native temporal sparse race core; all-channel speech packets',
        source_sha256=sources, stages=stages,
        data=dict(path='data/shd/shd_train.h5', sha256=file_sha(ROOT / 'data/shd/shd_train.h5')),
        admission=dict(reject_unshared_container=True, threads=1, min_available_mb=8192,
                       mode='Owner physical run_safe reservation; no auto-start from the workspace container'),
        priorities=['Preserve active curie/AWS language and DVS owner queues',
                    'Next new event admission: contracts -> smoke -> gated pilot'],
        official_test_read=False, scores=None, hypothesis_only=True)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    digest = file_sha(path)
    for name, cfg in stages.items():
        queue = ROOT / cfg['queue']
        command = (cfg['tag'] + ' experiments/public_speech_admission.py --manifest ' +
                   str(path.relative_to(ROOT)) + ' --manifest-sha256 ' + digest + ' --stage ' + name)
        queue.write_text('# UNRUN: owning physical host admission only; no container waiter.\n' +
            '# One job, one CPU thread, watchdog and 8GiB availability floor required.\n' +
            '# Guards: MEM_CAP_KB=6000000 MEM_CAP_RSS_KB=2000000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=' +
            str(cfg['timeout_s']) + '\n' + command + '\n')
    print(json.dumps(dict(status='prepared_unrun', manifest=str(path.relative_to(ROOT)),
                          manifest_sha256=digest, frozen_sources=len(sources), jobs=len(stages))))


if __name__ == '__main__':
    main()
