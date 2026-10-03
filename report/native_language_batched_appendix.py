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
]
INFERENCE = 'language_batched/curie_language_batched_inference_work_20261003T064000Z.json'
CONTROLS = [('e64/lstm_D10000000_s256_p1.json', 'LSTM-256'), ('e64/tf_D10000000_s256_p1.json', 'Transformer-256x2')]


def load(read):
    inference = {(r['payload'], r['depth'], r['pool']): r['unit_special_flops_per_evaluated_position']
                 for r in read(INFERENCE)['rows']}
    native = []
    for path, label in NATIVE:
        if not (RES / path).exists():
            continue
        r = read(path); a = r['args']; w = r['work']
        updates = int(a['passes'] * (a['fit'] - 1) // (a['segment'] * a['lanes']))
        native.append(dict(label=label, parameters=r['parameters'], updates=updates, dev=r.get('dev_bpc'),
                           test=r['test_bpc'], test256=r.get('test_bpc_eval_segment'),
                           whole=w['whole_fit_unit_special_flops_estimate'], fit=w['fit_unit_special_flops_per_char_estimate'],
                           infer=inference.get((a['payload'], a['depth'], a['pool'])), dev_window=a['dev']))
    estimate = runpy.run_path(str(ROOT / 'experiments/lm_training_flops.py'))['estimate_training_flops']
    controls = []
    for path, label in CONTROLS:
        r = json.loads((RES / path).read_text()); w = estimate(r['args'], r['params'], r['steps'])
        controls.append(dict(label=label, parameters=r['params'], updates=r['steps'], test=r['test_bpc'],
                             whole=w['total_training_flops'], fit=w['total_training_flops'] / w['training_token_positions'],
                             infer=w['forward_flops'] / w['training_token_positions']))
    return dict(native=native, controls=controls)


def pages(data):
    columns = ['Model (10M, one pass)', 'Params', 'Updates', 'Test bpc T128/T256', 'Whole fit TF est.',
               'Fit MF/char', 'Infer MF/position']
    widths = [44, 20, 18, 30, 22, 20, 22]
    rows = []
    for r in data['native']:
        t = f"{r['test']:.3f} / {r['test256']:.3f}" if r['test256'] is not None else f"{r['test']:.3f} / —"
        rows.append([f"Ours {r['label']}", f"{r['parameters']:,}", f"{r['updates']:,}", t, f"{r['whole'] / 1e12:.2f}",
                     f"{r['fit'] / 1e6:.2f}", f"{r['infer'] / 1e6:.2f}" if r['infer'] else '—'])
    for r in data['controls']:
        rows.append([f"E64 {r['label']}", f"{r['parameters']:,}", f"{r['updates']:,}", f"— / {r['test']:.3f}",
                     f"{r['whole'] / 1e12:.1f}", f"{r['fit'] / 1e6:.2f}", f"{r['infer'] / 1e6:.2f}"])
    return [[('h1', 'Appendix. Native language at 10M: the integrated core, segment-batched'),
             ('table', (columns, rows, widths)),
             ('p', 'Integrated native core only: temporal races (factorized law), sparse addressed writes into persistent '
                   'rotating memories, transport between layers; no dense carrier and no count statistics. One pass over '
                   'text8[0:10M] in 64 lanes x 128 characters (state reset per segment, exact credit within it), lr .004 '
                   'with cosine annealing, compiled layer steps (contract-tested against the batched path). The v1 row '
                   'used 128 lanes, about 610 updates and a constant lr; it is kept as measured. Test text8[95M:96M] '
                   'on E64 windows, scored at the training length and at the controls\' 256 with the same weights. '
                   'Single seed per row; exploratory, not a benchmark claim.'),
             ('p', 'The E64 rows are the matched one-pass controls (1,220 steps of 32 x 256, cosine). Work: ours traced '
                   'unit/special operations (fitting extrapolated from traced windows; inference is the batched emulator, '
                   'which computes every proposal); controls are shape estimates. The conventions differ, so work '
                   'comparisons are estimates.'),
             ('p', 'Reading: update calibration took p16/d8 from 2.899 to 2.719. Width beat depth (p32/d4 2.507), and depth '
                   'then helped at width 64 (p32/d8 2.456). The best native row is .03 bpc behind the one-pass Transformer '
                   'with about 1/8 of its parameters and estimated fitting work, and .29 bpc behind the one-pass LSTM. '
                   'THEORY §413: in this fast path the race address receives only first-time clock credit, so pools fragment '
                   'memory. The counterfactual route credit of the architecture is absent there. Diagnostics with '
                   'linearized read and write-address credit and with wider units are queued; no pending cell is filled.')]]
