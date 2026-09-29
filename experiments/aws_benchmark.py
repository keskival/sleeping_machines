"""Execute an unchanged benchmark with an isolated OUT directory and provenance.

Invoke only through queue/run_safe.sh. This wrapper changes the top-level OUT
assignment, not model code, arguments, data splits, or random seeds.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--stdout-only', action='store_true')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    script = (root / args.script).resolve()
    if not script.is_relative_to(root / 'experiments'):
        parser.error('script must be inside experiments/')
    if not args.run_tag.startswith('aws_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.run_tag):
        parser.error('run tag must be an aws_ filename component')
    out = root / 'experiments/results/aws_20260929' / args.run_tag
    out.mkdir(parents=True, exist_ok=False)
    source = script.read_bytes()
    tree = ast.parse(source, filename=str(script))
    redirected = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in node.targets):
            node.value = ast.Constant(str(out))
            redirected = True
    if not redirected and not args.stdout_only:
        raise ValueError(f'{script} has no top-level OUT assignment; review manually')
    ast.fix_missing_locations(tree)
    argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    metadata = dict(run_tag=args.run_tag, script=str(script.relative_to(root)), arguments=argv,
                    output_redirected=redirected,
                    source_sha256=hashlib.sha256(source).hexdigest(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    limits={k: os.environ.get(k) for k in ('MEM_CAP_KB', 'MEM_CAP_RSS_KB', 'MIN_AVAIL_MB', 'JOB_TIMEOUT_S')},
                    start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='running')
    provenance = out / 'provenance.json'
    provenance.write_text(json.dumps(metadata, indent=2) + '\n')
    start = time.monotonic()
    sys.argv = [str(script), *argv]
    sys.path.insert(0, str(script.parent))
    try:
        exec(compile(tree, str(script), 'exec'), {'__name__': '__main__', '__file__': str(script)})
        def check_metrics(value, path=''):
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, float) and (key.endswith('_bpc') or key.endswith('_acc')):
                        if not math.isfinite(item):
                            raise ValueError(f'Non-finite benchmark metric: {path}/{key}')
                    check_metrics(item, path + '/' + key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_metrics(item, path + '/' + str(i))
        for result in out.glob('*.json'):
            if result != provenance:
                check_metrics(json.loads(result.read_text()), result.name)
        metadata['status'] = 'completed'
    except BaseException:
        metadata['status'] = 'failed'
        metadata['error'] = traceback.format_exc()
        raise
    finally:
        metadata.update(wall_s=round(time.monotonic() - start, 3),
                        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        provenance.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
