"""Chronological stream RNG and target-weighted replay-accumulator evidence."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/native_language_horizon_evidence.py'))
FILE='diagnostics/local_causal_language_replay_accumulator_20261003T012100Z.json'


def load(read):
    data={'prior':BASE['load'](read),'accumulator':read(FILE)};r=data['accumulator']
    if r['status']!='completed' or r['contracts_passed']!=10:raise ValueError('Completed stateful accumulation contracts required')
    for name,digest in r['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed stateful replay source '+name)
    return data


def pages(data):
    pages=BASE['pages'](data['prior']);r=data['accumulator'];rows=[]
    for family in r['families']:
        rows.append([family['family'],str(family['parameters']),str(family['targets']),str(family['credit_chunks']),
            str(family['shadow_lanes']),str(family['shadow_events']),f"{family['actual_partial_learning_rate']:.4f}",
            str(family['final_live_state_bytes'])])
    pages.append([('h1','Appendix B. Stateful language replay accumulator: chronological RNG retained'),
        ('table',(['L8/p4 family','Params','Targets','Chunks','Shadow lanes','Shadow events','Partial LR','Live bytes'],rows,[30,22,20,18,29,30,21,22])),
        ('p','New sibling RNG-state kernel/helper and ReplayAccumulator preserve '
         'the original native stream randomness cadence: enter a chunk from its '
         'actual Torch RNG state, replay all alternatives from that SAMEstate, '
         'and advance the real stream only to the factual end RNG. No extra '
         'seed draw, repeated chunk seed or input/label/index randomness channel. '
         'Private addressed state crosses chunks/optimizer updates while its '
         'credit graph detaches each microchunk. Physical races/message evolution '
         'and original hard score map remain; numerical shadows are fully charged.'),
        ('p','Ten double L8/H2/p4/pool2 private/shared contracts pass. Every '
         'factual prediction and ALLstate match original chronological-teacher '
         'forward before a delayed optimizer update; end RNG is exact. Independent '
         'sequential actual-write all-target returns match every accumulated '
         'parameter gradient over two-plus-one target microchunks. Save partly '
         'filled gradients/private state/Adam/cursor/RNG and all replay counters: '
         'recovered predictions/state/gradients and next actual update are bitwise exact.'),
        ('p','Target labels change pending gradients but leave both chunks\' '
         'pre-update factual predictions/state/RNG identical. Target-weighted '
         'normalization,clip1,warmup total3/4,partial lr.0015 and Adam match '
         'independently summed sequential credit. Complete actual shadow/backward '
         'plus normalize/clip/warmup/optimizer operation coverage.96shadow lanes/ '
         '160shadow events across the2+1chunks are paid; this is not the288 '
         'events of one unbroken three-target credit chunk. Different boundaries '
         'are explicitly different training objectives.'),
        ('small','Theory113;14.510s/345160KiB. Correctness-test optimizer steps, '
         'no trained data/DEV/test/quality claim. Successfully contracted note108 '
         'seeded sources are preserved; new stateful siblings are frozen too. '
         'Existing AWS10M original-teacher drivers and checkpoints unchanged. '
         'A new fitted driver must use this credit in BOTH traced and untraced '
         'windows, save replay counters and preserve accounting/chronological '
         'controls; merely swapping an accumulator into an old traced loop '
         'would omit replay in traced windows. No fitted comparator launched here.')])
    return pages
