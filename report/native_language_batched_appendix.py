"""Completed segment-batched native language fits (THEORY §§409-413) beside the matched one-pass E64 controls.

Only completed result files enter; every row states its parameters, updates, protocol and work in the same units.
"""
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments/results'
NATIVE = [
    ('language_batched/curie_language_batched_10M_p16d8_s6_20261003T023000Z.json', 'p16/d8, v1 (610 updates)'),
    ('language_batched/curie_language_batched_10M_p16d8_skip2_l64_lr004_cmp_s6_20261003T054000Z.json', 'p16/d8, skip2'),
    ('language_batched/curie_language_batched_10M_p32d4_l64_lr004_cmp_s6_20261003T054000Z.json', 'p32/d4'),
    ('language_batched/curie_language_batched_10M_p32d8_skip2_l64_lr004_cmp_s6_20261003T054000Z.json', 'p32/d8, skip2'),
    ('language_batched/curie_language_batched_10M_p32d8_pool4_skip2_l64_lr004_cmp_s6_20261003T054000Z.json',
     'p32/d8/pool4, skip2'),
    ('language_batched/curie_language_batched_10M_p32d4_pool1_l64_lr004_cmp_s6_20261003T070000Z.json',
     'p32/d4/pool1 (control: no selection)'),
    ('language_batched/curie_language_batched_10M_p32d4_pool2_linear_l64_lr004_cmp_s6_20261003T070000Z.json',
     'p32/d4 + route credit'),
    ('language_batched/curie_language_batched_10M_p32d4_pool4_linear_l64_lr004_cmp_s6_20261003T101000Z.json',
     'p32/d4/pool4 + route credit'),
    ('language_batched/curie_language_batched_10M_p32d4_pool2_linear_rwn_l64_lr004_cmp_s6_20261003T101000Z.json',
     'p32/d4 + read and write credit'),
    ('language_batched/curie_language_batched_10M_p64d4_pool2_linear_l64_lr004_cmp_s6_20261003T101000Z.json',
     'p64/d4 + route credit'),
    ('language_batched/curie_language_batched_10M_p32d8_skip2_pool2_linear_l64_lr004_cmp_s6_20261003T101000Z.json',
     'p32/d8, skip2 + route credit'),
    ('language_batched/curie_language_batched_10M_p64d4_pool4_linear_l64_lr004_cmp_s6_20261003T120000Z.json',
     'p64/d4/pool4 + route credit'),
    ('language_batched/curie_language_batched_10M_p96d4_pool2_linear_l64_lr004_cmp_s6_20261003T120000Z.json',
     'p96/d4 + route credit'),
    ('language_batched/curie_language_batched_10M_p64d4_pool2_linear_4pass_l64_lr004_cmp_s6_20261003T150000Z.json',
     'p64/d4 + route credit, 4 passes'),
]
INFERENCE = 'language_batched/curie_language_batched_inference_work_20261003T064000Z.json'
INFERENCE_MORE = ['language_batched/curie_language_batched_inference_work_20261003T120000Z.json',
                  'language_batched/curie_language_batched_inference_work_20261003T143000Z.json']
SPARSE = ['language_batched/curie_language_sparse_inference_work_20261003T120000Z.json',      # §414 winner-only, exact
          'language_batched/curie_language_sparse_inference_work_20261003T143000Z.json']
CONTROLS = [('e64/lstm_D10000000_s256_p1.json', 'LSTM-256'), ('e64/tf_D10000000_s256_p1.json', 'Transformer-256x2'),
            ('e64/tf_D10000000_s256_L4_p4_dr0.1_v.json', 'Transformer-256x4, 4 passes'),
            ('e64/lstm_D10000000_s512_p6_dr0.1_v.json', 'LSTM-512, 6 passes')]


def load(read):
    inference = {(r['payload'], r['depth'], r['pool']): r['unit_special_flops_per_evaluated_position']
                 for path in [INFERENCE, *INFERENCE_MORE] for r in read(path)['rows']}
    sparse = {(r['payload'], r['depth'], r['pool']): r['unit_special_flops_per_evaluated_position']
              for path in SPARSE for r in read(path)['rows']}
    native = []
    for path, label in NATIVE:
        if not (RES / path).exists():
            continue
        r = read(path); a = r['args']; w = r['work']
        updates = int(a['passes'] * (a['fit'] - 1) // (a['segment'] * a['lanes']))
        native.append(dict(label=label, parameters=r['parameters'], updates=updates, dev=r.get('dev_bpc'),
                           test=r['test_bpc'], test256=r.get('test_bpc_eval_segment'),
                           whole=w['whole_fit_unit_special_flops_estimate'], fit=w['fit_unit_special_flops_per_char_estimate'],
                           infer=inference.get((a['payload'], a['depth'], a['pool'])), dev_window=a['dev'],
                           sparse=sparse.get((a['payload'], a['depth'], a['pool'])),
                           available_slots=a['depth']*a['heads']*a['pool'],
                           state_value_scalars=a['depth']*a['heads']*a['pool']*a['payload'],
                           selected_writes=a['depth']*a['heads'],
                           scored_keys=a['depth']*a['heads']*a['pool'],
                           computed_values=a['depth']*a['heads']*a['pool']))
    estimate = runpy.run_path(str(ROOT / 'experiments/lm_training_flops.py'))['estimate_training_flops']
    controls = []
    for path, label in CONTROLS:
        r = json.loads((RES / path).read_text()); w = estimate(r['args'], r['params'], r['steps'])
        controls.append(dict(label=label, parameters=r['params'], updates=r['steps'], test=r['test_bpc'],
                             whole=w['total_training_flops'], fit=w['total_training_flops'] / w['training_token_positions'],
                             infer=w['forward_flops'] / w['training_token_positions']))
    native90 = []
    for path in sorted((RES / 'language_batched').glob('aws_language_batched_90M_r*_s6_*.json')):
        r = json.loads(path.read_text()); a = r['args']
        if r.get('status') != 'completed' or a.get('max_windows'):
            continue
        w = r['work']
        native90.append(dict(label=f"p{a['payload']}/d{a['depth']}/pool{a['pool']}" + (' + route credit' if a.get('route_credit', 'none') != 'none' else ''),
                             parameters=r['parameters'], updates=r['windows'], passes=a['passes'], test=r['test_bpc'],
                             test256=r.get('test_bpc_eval_segment'), whole=w['whole_fit_unit_special_flops_estimate'],
                             fit=w['fit_unit_special_flops_per_char_estimate'], path=str(path.relative_to(ROOT))))
    controls90 = []
    for model, label in (('lstm', 'LSTM-512'), ('tf', 'Transformer-256x4')):
        for prov in sorted((RES / 'aws_20260929').glob('*/provenance.json')):
            meta = json.loads(prov.read_text()); args = meta.get('arguments', [])
            if (meta.get('status') != 'completed' or meta.get('script') != 'experiments/e64_lm_baselines.py'
                    or '--model' not in args or args[args.index('--model') + 1] != model
                    or '--D' not in args or int(args[args.index('--D') + 1]) != 90_000_000):
                continue
            for rp in sorted(prov.parent.glob('*.json')):
                if rp.name == 'provenance.json':
                    continue
                row = json.loads(rp.read_text())
                if isinstance(row.get('test_bpc'), float) and 'params' in row and 'steps' in row:
                    w = estimate(row['args'], row['params'], row['steps'])
                    controls90.append(dict(label=f"{label}, {row['args']['passes']:g} passes", parameters=row['params'],
                                           updates=row['steps'], test=row['test_bpc'], whole=w['total_training_flops'],
                                           fit=w['total_training_flops'] / w['training_token_positions']))
                    break
            if controls90 and controls90[-1]['label'].startswith(label):
                break
    return dict(native=native, controls=controls, native90=native90, controls90=controls90)


def pages(data):
    columns = ['Model (10M)', 'Params', 'Steps', 'Test bpc T128/T256', 'Whole fit TF est.',
               'Fit MF/char', 'Infer MF/pos. emulator', 'Infer MF/pos. winner-only']
    widths = [40, 20, 15, 25, 18, 16, 21, 21]
    rows = []
    for r in data['native']:
        t = f"{r['test']:.3f} / {r['test256']:.3f}" if r['test256'] is not None else f"{r['test']:.3f} / —"
        rows.append([f"Ours {r['label']}", f"{r['parameters']:,}", f"{r['updates']:,}", t, f"{r['whole'] / 1e12:.2f}",
                     f"{r['fit'] / 1e6:.2f}", f"{r['infer'] / 1e6:.2f}" if r['infer'] else '—',
                     f"{r['sparse'] / 1e6:.2f}" if r.get('sparse') else '—'])
    for r in data['controls']:
        rows.append([f"E64 {r['label']}", f"{r['parameters']:,}", f"{r['updates']:,}", f"— / {r['test']:.3f}",
                     f"{r['whole'] / 1e12:.1f}", f"{r['fit'] / 1e6:.2f}", f"{r['infer'] / 1e6:.2f}", f"{r['infer'] / 1e6:.2f}"])
    activity_columns = ['Native model', 'Slots', 'Memory<br/>scalars', 'Writes', 'Keys',
                        'Emulator<br/>values', 'Winner<br/>values']
    activity_rows = [[r['label'], str(r['available_slots']), str(r['state_value_scalars']),
                      str(r['selected_writes']), str(r['scored_keys']), str(r['computed_values']),
                      str(r['selected_writes'])]
                     for r in data['native']]
    return [[('h1', 'Appendix. Native language at 10M: the integrated core, segment-batched'),
             ('table', (columns, rows, widths)),
             ('figure', ('native_language_frontier', 172)),
             *([('table', (['Model (90M)', 'Params', 'Updates', 'Test bpc T128/T256', 'Whole fit TF est.', 'Fit MF/char'],
                           [[f"Ours {r['label']} (AWS, one pass)", f"{r['parameters']:,}", f"{r['updates']:,}",
                             f"{r['test']:.3f} / {r['test256']:.3f}", f"{r['whole'] / 1e12:.1f}", f"{r['fit'] / 1e6:.2f}"]
                            for r in data.get('native90', [])] +
                           [[f"E64 {r['label']} (AWS)", f"{r['parameters']:,}", f"{r['updates']:,}", f"— / {r['test']:.3f}",
                             f"{r['whole'] / 1e12:.0f}", f"{r['fit'] / 1e6:.2f}"] for r in data.get('controls90', [])],
                           [52, 20, 20, 30, 26, 22])),
                ('small', '90M rows: text8[0:90M], same test interval and E64 windows. The native rows are one pass of the '
                          'segment-batched protocol on AWS (compiled, 64 x 128 windows, lr .004 cosine); the references '
                          'are multi-pass with larger models and are listed for scale, not as matched comparisons.')]
               if data.get('native90') else []),
             ('table', (activity_columns, activity_rows, [49, 20, 25, 20, 20, 21, 21])),
             ('small', 'Native mechanism counts per input position. Memory scalars are '
                       'available unit-value storage per lane; timestamps, readiness bits and source context are '
                       'additional. One value is delivered per selected head/layer write. Every candidate key and '
                       'proposal value is computed before selection in the emulator; the winner-only evaluator '
                       'computes only selected proposals and also caches one key-read vector per slot. Every key '
                       'is still scored. These are shape counts, not traffic or energy measurements.'),
             ('p', 'Integrated native core only: temporal races (factorized law), sparse addressed writes into persistent '
                   'rotating memories, transport between layers; no dense carrier and no count statistics. One pass over '
                   'text8[0:10M] in 64 lanes x 128 characters (state reset per segment, exact credit within it), lr .004 '
                   'with cosine annealing, compiled layer steps (contract-tested against the batched path). The v1 row '
                   'used 128 lanes, about 610 updates and a constant lr; it is kept as measured. Test text8[95M:96M] '
                   'on E64 windows, scored at the training length and at the controls\' 256 with the same weights. '
                   'Single seed per row; exploratory, not a benchmark claim.'),
             ('p', 'The E64 rows are the matched one-pass controls (1,220 steps of 32 x 256, cosine). Work: ours traced '
                   'unit/special operations (fitting extrapolated from traced windows). Inference is traced twice: the '
                   'batched emulator, which computes every proposal, and the exact winner-only evaluator (THEORY §414), '
                   'whose small random float64 fixtures match emulator logits within 1e-10. Direct winner/state/cache '
                   'contracts on the actual trained float32 weights are prepared and pending (note 143); the saved '
                   'test scores use the compiled training evaluator. For fixed weights, cached stored-memory key reads '
                   'need refreshing only on a slot\'s write: arithmetic scales as U.P plus winner maps per race. '
                   'The present implementation also stacks every unit\'s matrices at each call, an O(U.P²) '
                   'parameter-copy/allocation cost outside these FLOP counts. Cache storage, traffic and wall time '
                   'must be measured before claiming total-resource scaling. Every key is scored and counted. '
                   'Control columns repeat their single shape estimate. '
                   'The conventions differ, so work comparisons are estimates.'),
             ('p', 'Reading: update calibration took p16/d8 from 2.899 to 2.719. Width beat depth (p32/d4 2.507), and depth '
                   'then helped at width 64 (p32/d8 2.456). Without route credit the fast path trains the race address only '
                   'through first-time clock credit (THEORY §413), and more units then cost quality: pool 4 is worse than pool 2 '
                   '(2.498 vs 2.456), and the no-selection pool-1 control beats pool 2 at depth 4 (2.439 vs 2.507). With the '
                   'linearized local-expectation route credit (forward values unchanged, about 0.3% more counted fitting work) the same p32/d4 '
                   'pool-2 model scores 2.370: .137 better than without it, .069 better than the control, and .057 better '
                   'than the one-pass Transformer; at the matched T256 window it scores 2.371 versus 2.427, a .0554 bpc '
                   'advantage, with about 1/15 of its parameters and estimated fitting work. It remains '
                   '.199 behind the one-pass LSTM. With credit, pool 4 at the same 8 selected writes per character scores 2.343 '
                   '(T256 2.345): more stored units now improve quality instead of costing it. Width is the strongest lever: '
                   'p64/d4 with credit scores 2.184 (T256 2.183), .012 behind the one-pass LSTM, with exact winner-only '
                   'inference of 0.60 MFLOPs per position against the LSTM estimate of 0.68 and more estimated fitting work '
                   '(2.68 vs 2.03 MFLOPs per character). A write-address credit on stored coordinates diverged; the corrected '
                   'variant trained stably at pool 2 without improving on value credit (2.384 vs 2.370) and diverged at pool 4, '
                   'so write-address credit is withdrawn. Single seeds; '
                   'pending arms are not filled.')]]


def figure(data, out):
    """Quality against whole-fit and inference work for every completed native row and the saved E64 references."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, xlabel in ((axes[0], 'whole', 'Whole-fit work, TFLOPs (estimate, log scale)'),
                            (axes[1], 'infer', 'Inference work per position, MFLOPs (log scale)')):
        for r in data['native']:
            x = (r['whole'] / 1e12) if key == 'whole' else ((r.get('sparse') or r['infer'] or 0) / 1e6)
            if not x:
                continue
            y = r['test256'] if r['test256'] is not None else r['test']
            credit = 'route credit' in r['label']
            ax.scatter(x, y, color='#1f5fa8' if credit else '#9db8d9', s=34, zorder=3)
            if credit or 'v1' in r['label']:
                ax.annotate(r['label'].replace(' + route credit', '+c'), (x, y), fontsize=6.5, xytext=(4, 2),
                            textcoords='offset points', color='#1f5fa8')
        for r in data['controls']:
            x = (r['whole'] / 1e12) if key == 'whole' else r['infer'] / 1e6
            ax.scatter(x, r['test'], color='#d2691e', marker='s', s=34, zorder=3)
            ax.annotate(r['label'], (x, r['test']), fontsize=6.5, xytext=(4, -9), textcoords='offset points',
                        color='#d2691e')
        ax.set_xscale('log'); ax.set_xlabel(xlabel); ax.set_ylabel('Test bpc (T256 where scored) — lower is better')
        ax.grid(alpha=.25)
    axes[0].set_title('Native core (blue; dark = route credit) vs E64 controls (orange)', fontsize=9)
    axes[1].set_title('Native inference: exact winner-only evaluator where traced', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)
