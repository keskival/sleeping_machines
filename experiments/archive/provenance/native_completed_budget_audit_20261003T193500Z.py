"""Completed native budget tradeoffs; stdlib saved-result arithmetic only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]


def audit():
    appendix = runpy.run_path(str(ROOT / 'report/native_language_batched_appendix.py'))
    parents = {}
    def read(name):
        path = ROOT / 'experiments/results' / name
        raw = path.read_bytes()
        parents[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    data = appendix['load'](read)
    def selected(rows, label):
        return next(row for row in rows if row['label'] == label)
    one = selected(data['native'], 'p64/d4 + route credit')
    four = selected(data['native'], 'p64/d4 + route credit, 4 passes')
    tf = selected(data['controls'], 'Transformer-256x4, 4 passes')
    large = selected(data['native90'], 'p32/d4/pool4 + route credit')
    # Native90/control loading in the appendix uses direct reads; bind them too.
    for row in data['native90']:
        path = ROOT / row['path']
        parents[row['path']] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT / 'experiments/results/aws_20260929').glob('*/provenance.json')):
        meta = json.loads(path.read_text())
        if meta.get('status') == 'completed' and meta.get('script') == 'experiments/e64_lm_baselines.py':
            for result in sorted(path.parent.glob('*.json')):
                row = json.loads(result.read_text())
                if 'test_bpc' in row and 'params' in row and 'steps' in row:
                    parents[str(result.relative_to(ROOT))] = hashlib.sha256(result.read_bytes()).hexdigest()
            parents[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for kind, source in (('native10M', data['native']), ('control10M', data['controls']),
                         ('native90M', data['native90']), ('control90M', data['controls90'])):
        for row in source:
            if kind == 'native10M' and row not in (one, four):
                continue
            rows.append(dict(kind=kind, model=row['label'], parameters=row['parameters'],
                updates=row['updates'], test_bpc_T256=row.get('test256') or row['test'],
                whole_fit_TF_estimate=row['whole'] / 1e12,
                fit_MF_per_presentation_estimate=row['fit'] / 1e6,
                inference_quality_verified_on_prepacked_backend=False if kind.startswith('native') else None))
    # Never infer equal-quality/energy/modern-frontier supremacy from these ratios.
    deltas = dict(
        p64_one_to_four_pass_bpc_improvement=one['test256'] - four['test256'],
        p64_one_to_four_pass_perplexity_reduction=1 - 2 ** (four['test256'] - one['test256']),
        p64_four_to_one_whole_fit_work_ratio=four['whole'] / one['whole'],
        four_pass_native_vs_transformer_bpc_gap=four['test256'] - tf['test'],
        four_pass_native_vs_transformer_perplexity_ratio=2 ** (four['test256'] - tf['test']),
        four_pass_transformer_to_native_whole_fit_work_ratio=tf['whole'] / four['whole'],
        four_pass_transformer_to_native_parameter_ratio=tf['parameters'] / four['parameters'],
        native90m_to_four_pass_native10m_whole_fit_ratio=large['whole'] / four['whole'],
        native90m_minus_four_pass_native10m_test_bpc=large['test256'] - four['test256'])
    assert all(math.isfinite(v) for v in deltas.values())
    return dict(status='completed_metadata_analysis', parent_sha256=parents, rows=rows, deltas=deltas,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Saved single-seed quality and differing operation-estimate conventions; no numerical runtime. Test is reporting-only. Ratios are raw unequal-quality work gaps, not isoquality or energy advantage.',
        decisions=[
            'Preserve integrated temporal races/private state/key-value separation/alternative-value credit.',
            'Training budget is a material D4 lever with clip1 unchanged; this does not identify clipping effects in deeper models.',
            'At approximately107TF the two native outcomes confound width, pool, data variety, presentations, update count and cosine schedule. Do not infer a scaling law or data preference.',
            'Close actual trained sparse quality parity before advertising backend quality/resource comparisons.',
            'Retain current AWS pool2/depth8/width64 and curie tied-pool/seeds/horizon owner chains. Stronger comparisons remain on AWS.',
            'Further model selection uses development evidence and a reserved confirmation protocol, not these test gaps.',
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = ROOT / args.out
    result = audit()
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], rows=len(result['rows']), deltas=result['deltas'])))
