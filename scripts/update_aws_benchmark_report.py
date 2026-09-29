#!/usr/bin/env python3
"""Refresh the Markdown research report and PDF from completed AWS result files."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results/aws_20260929'
START = '<!-- AWS_BENCHMARKS_START -->'
END = '<!-- AWS_BENCHMARKS_END -->'
KEYS = {'acc', 'test_acc', 'held_out_acc', 'prequential_acc', 'test_bpc', 'best_valid_bpc'}


def metrics_from(value):
    found = []
    def walk(item, path=''):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in KEYS and isinstance(child, (int, float)) and math.isfinite(child):
                    found.append((path + key, float(child)))
                walk(child, path + key + '/')
        elif isinstance(item, list):
            if path.endswith('rows/'):
                for i, child in enumerate(item):
                    walk(child, path + f'{i}/')
                return
            # Curves often repeat the same metric at every checkpoint; retain
            # the final point for each configuration/seed.
            latest = {}
            for i, child in enumerate(item):
                before = len(found)
                walk(child, path + f'{i}/')
                for k, val in found[before:]:
                    latest[k.rsplit('/', 1)[-1]] = (k, val)
                del found[before:]
            found.extend(latest.values())
    walk(value)
    unique = {}
    for key, val in found:
        unique[key] = val
    return list(unique.items())


def completed_runs():
    rows = []
    if not RESULTS.exists():
        return rows
    for provenance_path in RESULTS.glob('*/provenance.json'):
        try:
            provenance = json.loads(provenance_path.read_text())
            if provenance.get('status') != 'completed':
                continue
            outputs = []
            for result_path in provenance_path.parent.glob('*.json'):
                if result_path == provenance_path:
                    continue
                result = json.loads(result_path.read_text())
                outputs.append(result)
            metric_pairs = []
            for result in outputs:
                metric_pairs.extend(metrics_from(result))
            rows.append((provenance.get('start_utc', ''), provenance, metric_pairs))
        except (OSError, ValueError, TypeError):
            continue
    rows.sort(key=lambda row: (row[0], row[1].get('run_tag', '')))
    return rows


def metric_text(pairs):
    if not pairs:
        return 'Completed; inspect the saved result for measurements.'
    return '; '.join(f'{key.rsplit("/", 1)[-1]}={value:.5g}' for key, value in pairs[:10])


def update_report(rows):
    path = ROOT / 'REPORT.md'
    content = path.read_text()
    recalls = []
    for result_path in sorted((ROOT / 'experiments/results/e68').glob('recall_R*_s0.json')):
        try:
            result = json.loads(result_path.read_text())
            recalls.append((result['args']['R'], result['curve'][-1]['test_acc']))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    lines = [START, '## AWS benchmark updates', '',
             'Runs below passed the runner and finite-metric checks. These early outcomes are diagnostics; single seeds do not establish a comparative advantage.', '',
             '| Run | Benchmark | Result | Wall time | Peak RSS |', '|---|---|---|---:|---:|']
    for _when, provenance, pairs in rows[-30:]:
        metrics = metric_text(pairs).replace('|', '\\|')
        lines.append(f"| `{provenance.get('run_tag','')}` | `{provenance.get('script','')}` | {metrics} | {provenance.get('wall_s','')} s | {provenance.get('peak_rss_kb','')} KB |")
    if recalls:
        lines += ['', 'E68 seed-0 synthetic recall, 8,000 updates (512,000 sequences): ' + ', '.join(
            f'R={r}: {acc:.1%}' for r, acc in recalls) + '.']
    lines += [END]
    block = '\n'.join(lines)
    if START in content and END in content:
        content = re.sub(re.escape(START) + r'.*?' + re.escape(END), block, content, count=1, flags=re.S)
    else:
        intro = re.search(r'(?m)^\*\d{1,2} [A-Za-z]+ \d{4}.*?\n\n', content, flags=re.S)
        if intro is None:
            raise RuntimeError('Could not locate report introduction for the benchmark table')
        idx = intro.end()
        content = content[:idx] + block + '\n\n' + content[idx:]
    path.write_text(content)
    return block


def update_findings(rows):
    path = ROOT / 'experiments/FINDINGS.md'
    content = path.read_text()
    existing = set(re.findall(r'\*\*AWS run `([^`]+)`\.', content))
    additions = []
    for _when, provenance, pairs in rows:
        tag = provenance.get('run_tag', '')
        if tag and tag not in existing:
            additions.append(f"**AWS run `{tag}`.** {provenance.get('script','')} "
                             f"with arguments `{json.dumps(provenance.get('arguments', []), separators=(',', ':'))}`. "
                             f"Validated metrics: {metric_text(pairs)}. Wall time {provenance.get('wall_s','')} s; "
                             f"peak process RSS {provenance.get('peak_rss_kb','')} KB. Single-run result; interpret "
                             "under the experiment's preregistered comparisons and limits.")
    if additions:
        heading = '## 2026-09-29\n\n'
        if heading not in content:
            content = content.replace('# Findings log\n', '# Findings log\n', 1)
            pos = content.find('\n## ')
            if pos < 0:
                content += '\n' + heading
                pos = len(content)
            else:
                content = content[:pos] + '\n' + heading + content[pos:]
                pos = content.find(heading) + len(heading)
        else:
            pos = content.find(heading) + len(heading)
        content = content[:pos] + '\n\n'.join(additions) + '\n\n' + content[pos:]
        path.write_text(content)


def main():
    rows = completed_runs()
    update_report(rows)
    update_findings(rows)
    print(f'Updated report sources with {len(rows)} completed AWS runs.', flush=True)


if __name__ == '__main__':
    main()
