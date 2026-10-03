"""Current completed native language evidence and separately labelled AWS progress."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = 'experiments/results/diagnostics/aws_language_matched_progress_1m_20261003T133500Z.json'
ANALYSIS = 'experiments/analysis/aws_language_matched_progress_1m.py'
ARMS = [('private_replay', 'Private full replay'), ('private_teacher', 'Private teacher'),
        ('shared_teacher', 'Depth-shared teacher')]
SELECT = [
    ('p32/d4', 'p32/D4: timing credit'),
    ('p32/d4 + route credit', 'p32/D4: value credit'),
    ('p32/d4/pool4 + route credit', 'p32/D4/U4: value credit'),
    ('p32/d8, skip2', 'p32/D8: timing credit'),
    ('p32/d8, skip2 + route credit', 'p32/D8: value credit'),
    ('p64/d4 + route credit', 'p64/D4: value credit'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(native):
    by_label = {r['label']: r for r in native['native']}
    selected = [dict(row=by_label[name], name=label, ours=True) for name, label in SELECT]
    selected += [dict(row=r, name=r['label'], ours=False) for r in native['controls']]
    best = min((r for r in native['native'] if 'route credit' in r['label'] and r['test256'] is not None),
               key=lambda r: r['test256'])
    if not any(entry['row']['label'] == best['label'] for entry in selected):
        selected.insert(len(SELECT), dict(row=best, name=best['label'], ours=True))
    controls = {r['label']: r for r in native['controls']}
    saved = json.loads((ROOT/PROGRESS).read_text())
    if saved['status'] != 'completed' or sha(ROOT/ANALYSIS) != saved['analysis_source_sha256']:
        raise ValueError('Completed unchanged AWS checkpoint analysis required')
    latest = []
    for arm, label in ARMS:
        row = next(r for r in saved['rows'] if r['arm'] == arm and r['milestone'] == 4)
        prior = next(r for r in saved['rows'] if r['arm'] == arm and r['milestone'] == 3)
        metadata = json.loads((ROOT/row['checkpoint']).with_suffix('.json').read_text())
        earlier = json.loads((ROOT/prior['checkpoint']).with_suffix('.json').read_text())
        assert row['checkpoint_sha256'] == metadata['checkpoint_sha256'] == sha(ROOT/row['checkpoint'])
        assert prior['checkpoint_sha256'] == earlier['checkpoint_sha256'] == sha(ROOT/prior['checkpoint'])
        assert row['targets'] == metadata['trained_targets'] == 1003520
        assert row['optimizer_updates'] == metadata['optimizer_updates'] == 3920
        measured = (metadata['cursor']['total_loss']-earlier['cursor']['total_loss'])/(row['targets']-prior['targets'])/math.log(2)
        assert abs(measured-row['online_interval_bpc']) < 1e-10
        latest.append(dict(label=label, **row))
    return dict(selected=selected, best=best, controls=controls, progress=latest, native=by_label)


def figure(data, path):
    import matplotlib.pyplot as plt
    rows = data['selected']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharey=True, gridspec_kw={'width_ratios': [1.2, 1.]})
    for index, entry in enumerate(rows):
        row = entry['row']
        quality = row['test256'] if entry['ours'] else row['test']
        color = '#2a78d6' if entry['ours'] else '#8a8984'
        if entry['ours'] and 'route credit' not in row['label']:
            color = '#a9c8ef'
        axes[0].scatter(quality, index, s=32, color=color, zorder=3)
        axes[0].annotate(f'{quality:.3f}', (quality, index), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
        work = row['whole']/1e12
        axes[1].scatter(work, index, s=32, color=color, zorder=3)
        axes[1].annotate(f'{work:.2f}', (work, index), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
    axes[0].set_yticks(range(len(rows)), [entry['name'] for entry in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(2.10, 2.56)
    axes[0].set_xlabel('Matched T256 test bpc (lower is better)')
    axes[0].set_title('Completed language quality')
    axes[1].set_xscale('log')
    axes[1].set_xlim(5, 160)
    axes[1].set_xticks([10, 30, 100], ['10', '30', '100'])
    axes[1].set_xlabel('Whole fitting TFLOPs estimate (log scale)')
    axes[1].set_title('Full fitting work, same unit')
    for axis in axes:
        axis.grid(axis='y', visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
    vector = Path(path).with_suffix('.svg')
    fig.savefig(vector, bbox_inches='tight', facecolor='white')
    # Matplotlib emits trailing spaces in path attributes; keep generated
    # artifacts clean while retaining the separating newline/XML whitespace.
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    plt.close(fig)


def pages(data):
    best, native, controls = data['best'], data['native'], data['controls']
    p2, p4 = native['p32/d4 + route credit'], native['p32/d4/pool4 + route credit']
    p8, no8 = native['p32/d8, skip2 + route credit'], native['p32/d8, skip2']
    tf, lstm = controls['Transformer-256x2'], controls['LSTM-256']
    gap = best['test256']-lstm['test']
    relative_quality = 'behind' if gap >= 0 else 'ahead of'
    replay, private, shared = data['progress']
    progress_rows = [[r['label'], f"{r['online_interval_bpc']:.6f}", f"{r['cumulative_online_bpc']:.6f}"] for r in data['progress']]
    return [[
        ('h1', 'Current language evidence — 3 October 2026'),
        ('p', f"<b>The integrated native learner now reaches {best['test256']:.3f} bpc at the controls’ T256 test window.</b> "
              f"The saved one-pass LSTM scores {lstm['test']:.3f} and Transformer {tf['test']:.3f}. "
              'These are completed single-seed comparisons; replication and large-data advantage remain open.'),
        ('figure', ('current_native_language_status', 174)),
        ('small', 'Blue: native temporal races, sparse addressed persistent writes and learned messages; light blue: timing-only route credit. '
                  'Gray: saved dense controls. Same text8 test[95M:96M], T256 evaluation, nominal one-pass 10M fitting budget, 1,220 updates. '
                  'Native training samples random segments; order differs from the controls. Work is traced/extrapolated for native and '
                  'shape-estimated for controls. Both panels use the same denominator for every model. Full resource table is in the native appendix.'),
        ('h2', 'The gains are about learned routing and useful capacity'),
        ('bullets', [
            f"Value-informed categorical credit improves p32/D4 by {native['p32/d4']['test256']-p2['test256']:.3f} bpc "
            'with about 0.3% extra counted fitting work. The successful rule keeps hard forward choices and messages unchanged.',
            f"At eight selected writes per position, doubling p32 slots improves {p2['test256']:.3f}→{p4['test256']:.3f} bpc. "
            f"Fitting work rises {p4['whole']/p2['whole']:.2f}×; unchanged selected activity is not unchanged total cost.",
            f"The credited depth-8 model reaches {p8['test256']:.3f} versus {no8['test256']:.3f} without that credit. "
            'The gain survives a deeper stack; width, initialization and capacity still need controlled comparisons.'
        ]),
        ('p', f"The best native model is {abs(gap):.3f} bpc {relative_quality} the LSTM, "
              f"using {best['whole']/1e12:.2f} versus {lstm['whole']/1e12:.2f} estimated fitting TFLOPs. "
              'This is substantial progress, not comparable-quality superiority in total resources.')
    ], [
        ('h1', 'Learning diagnosis and the next decisive checks'),
        ('h2', 'Keep the successful value credit; withdraw failed write credit'),
        ('p', 'The earlier fast law taught the winner’s content and first-time clocks without an explicit alternative-value choice term. '
              'Adding that term helped both depth-4 and depth-8 language models. Stored-memory write credit diverged. Written-only '
              'credit trained stably but worse at pool2 and also diverged at pool4, so it is withdrawn. Removing lazy transport from '
              'the coefficient was insufficient; memory norms, cotangents, timestamp/seen effects and feedback remain to be measured.'),
        ('h2', 'Inference arithmetic is promising; the practical boundary is wider'),
        ('p', 'Winner-only inference computes selected proposals and refreshes their cached stored-memory key reads. The saved shape traces '
              'give 0.163→0.164 MFLOPs per input position when p32 capacity doubles, and 0.605 for p64/D4. All keys are scored. '
              'Every call still stacks all unit matrices; copying, extra cache state and wall time are outside these arithmetic counts. '
              'Small float64 output contracts passed; actual trained float32 winner/state/cache parity and full rescoring remain pending. '
              'The reported test scores use the compiled training evaluator, not a completed sparse-backend rescore.'),
        ('h2', 'AWS depth-8 replay: supported online progress, a separate protocol'),
        ('table', (['Ongoing fitting arm', 'Latest interval online bpc', 'Cumulative online bpc'], progress_rows, [70, 52, 52])),
        ('small', f"Saved matched checkpoints: 1,003,520 targets / 3,920 Adam updates; latest interval[753,664:1,003,520]. "
                  f"Full replay’s interval lead is {private['online_interval_bpc']-replay['online_interval_bpc']:.6f}/"
                  f"{shared['online_interval_bpc']-replay['online_interval_bpc']:.6f} bpc. Identical fitting-data hash/exposure; "
                  'single seed, changing parameters and no asserted RNG pairing. Full replay costs much more learning work. '
                  'These are training predictions, not completed heldout scores or useful-depth/iso-FLOP proof.'),
        ('h2', 'Prioritize discriminating evidence'),
        ('bullets', [
            'Complete current multi-pass/width and queued tied-pool/seed comparisons. AWS90M pool4 has started after '
            '15 contracts and its throughput pilot; completed90M quality is pending.',
            'Prepared, unrun trained-FIT factorial checks separate message effects, private commit effects and their interaction at fixed first time/future noise.',
            'Calibrate optional write credit against unexplained value utility; check shared scales, feedback and actual updates before another fit.',
            'Datacenter serving: a prepared worker reuses one packed matrix stack. Standard-library lifecycle checks pass; '
            'trained parity, measured runtime and quality rescore remain pending. Snapshot/setup/residency costs are charged.'
        ]),
        ('small', 'Theory143–146; DATACENTER_VALUE_MILESTONES.md. No proof of a mathematical barrier or general supremacy; neither follows from this evidence. '
                  'Counts remain strong references in their established region. Current gains retain time as computation, hard-route credit, '
                  'deep persistent state, separate keys/values and capacity beyond selected activity.')
    ]]
