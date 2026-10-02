"""Audit synthetic target alphabets and causal local controls before neural fits.

Each local order is reported separately; selecting an order using each target
would leak that target and is never a predictor here. No neural fitting occurs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
from long_range_core_benchmark import make_stream
from sleeping_machines.count_carrying_language import eval_stream_counts


def audit(task, distance, seed):
    fit, fit_mask = make_stream(task, 8192, distance, seed)
    dev, mask = make_stream(task, 4096, distance, seed+100)
    positions = np.flatnonzero(mask[1:]); labels = dev[positions+1]
    counts = eval_stream_counts(fit, dev, 8)
    controls = []
    for k in range(8):
        c = counts[k, positions]; p = (c+1)/(c.sum(-1, keepdims=True)+27)
        controls.append(dict(order=k+1, target_bpc=float(-np.log2(p[np.arange(len(labels)), labels]).mean()),
                             accuracy=float((p.argmax(-1)==labels).mean())))
    fit_targets = fit[fit_mask]; prior = np.bincount(fit_targets, minlength=27)+1
    prior = prior/prior.sum()
    freq = np.bincount(labels, minlength=27); prob = freq[freq>0]/freq.sum()
    return dict(task=task, distance=distance, seed=seed, fit_tokens=8192, dev_tokens=4096,
        target_fraction=float(mask[1:].mean()), fit_targets=int(fit_mask[1:].sum()), dev_targets=len(labels),
        observed_target_symbol_counts=freq.tolist(), empirical_target_entropy_bits=float(-(prob*np.log2(prob)).sum()),
        targets_outside_assumed_24_symbols=int((labels>=24).sum()),
        target_only_fitted_unigram_bpc=float(-np.log2(prior[labels]).mean()),
        uniform_27_bpc=math.log2(27), claimed_uniform_24_bpc=math.log2(24),
        causal_prequential_add_one_controls=controls,
        fit_sha256=hashlib.sha256(fit.astype('uint8').tobytes()).hexdigest(),
        dev_sha256=hashlib.sha256(dev.astype('uint8').tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    started=time.perf_counter()
    rows=[audit(task,d,seed) for task,d in [('lag',48),('induction',128)] for seed in (2,3,4)]
    names=['experiments/long_range_protocol_audit.py','experiments/long_range_core_benchmark.py',
           'sleeping_machines/count_carrying_language.py','tests/test_long_range_core.py']
    result=dict(status='completed',rows=rows,neural_optimizer_steps=0,
        prospective_credit_comparison=dict(fit_tokens=8192,passes=4,
            current_driver_steps={str(c):4*math.ceil(8191/c) for c in (16,64)},
            required_matched_optimizer_interval=64,
            current_driver_changes_optimizer_budget_with_credit=True),
        protocol_findings=[
            'Lag targets copy arbitrary prior tokens, including earlier cues; iid filler does not imply iid uniform-24 targets.',
            'Induction queries are sampled from keys observed in the window; this selection and earlier query/target insertions invalidate the asserted all-orders independence proof.',
            'Existing per-target minimum over order losses uses the target to choose its predictor; it is an optimistic audit bound, not a deployable count control.',
            'Whole-stream training includes predictable cues, random filler and copied targets; target fraction is recorded, not a measured gradient share.',
            'Driver resets addressed neural memory at development start, whereas saved count experiments preload fit counts; preserve cold-state comparisons but do not treat history access as matched.',
            'Credit length must vary independently of the optimizer interval, normalization, clipping, learning rate and target budget.'
        ],
        scope='Three independent synthetic stream pairs per task; no neural fits, no semantic claim or all-orders impossibility proof. All eight causal local orders reported individually; additive smoothing over27 symbols. Target-only unigram is a fitted diagnostic with target annotations, not the integrated model.',
        official_test_read=False,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
