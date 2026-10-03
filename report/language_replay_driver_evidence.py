"""Restartable full-credit language benchmark driver numerical admission."""
import hashlib
import copy
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/stateful_language_replay_evidence.py'))
FILE='diagnostics/local_language_replay_driver_contracts_20261003T012900Z.json'
ARCHIVE_SOURCE='experiments/dvs_batched_reg_benchmark.py'
ARCHIVE_SHA='ee3d8ada5648c9a152407d7b78850d37838537dfcb11387f06b462779298285c'
ARCHIVE_PATH=f'experiments/archive/frozen_sources/{ARCHIVE_SHA}/dvs_batched_reg_benchmark.py'


def historical_read(read,path):
    r=copy.deepcopy(read(path));hashes=r.get('source_sha256',{})
    if hashes.get(ARCHIVE_SOURCE)==ARCHIVE_SHA:
        if hashlib.sha256((ROOT/ARCHIVE_PATH).read_bytes()).hexdigest()!=ARCHIVE_SHA:
            raise ValueError('Changed exact historical DVS source archive')
        # Only the publication copy resolves its original binding to archived bytes.
        # The stored result and its original source-key/digest remain immutable.
        del hashes[ARCHIVE_SOURCE];hashes[ARCHIVE_PATH]=ARCHIVE_SHA
        r['publication_source_archive']={ARCHIVE_SOURCE:ARCHIVE_PATH}
    return r


def load(read):
    data={'prior':BASE['load'](lambda p:historical_read(read,p)),'driver':read(FILE)};r=data['driver']
    if r['status']!='completed' or r['contracts_passed']!=12:raise ValueError('Completed replay driver contracts required')
    for name,digest in r['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed contracted driver source '+name)
    return data


def load_score_evidence(read):
    module=runpy.run_path(str(ROOT/'report/score_bound_evidence.py'))
    return module['load'](lambda p:historical_read(read,p))


def pages(data):
    pages=BASE['pages'](data['prior']);r=data['driver'];rows=[]
    for family in r['families']:
        work=family['fitting_work'];rows.append([family['family'],str(family['parameters']),str(family['targets']),
            str(family['optimizer_updates']),f"{work['whole_fit_unit_special_flops_estimate']/1e9:.6f}",
            f"{work['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
            f"{work['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    pages.append([('h1','Appendix B. Replay fitting driver: interrupted learning recovers exactly'),
        ('table',(['L8/p4 family','Params','Targets','Updates','Whole fit GF','Fit MF/target','Infer MF/target'],rows,[29,22,18,20,32,35,36])),
        ('p','New sibling benchmark driver ALWAYS calls the stateful full-write '
         'ReplayAccumulator. Operation tracing wraps that same call, preserving '
         'all-target shadow returns in both audited and ordinary windows. '
         'Twelve end-to-end private/shared depth8 contracts pass. Synthetic '
         'eight-target fits use two-target credit chunks, four-target optimizer '
         'windows, target-weighted normalization, clipping and warmup. '
         'This closes the traced-loop omission hazard identified in Theory114; '
         'the existing AWS original-teacher controls remain unchanged.'),
        ('p','Interrupt after the first microchunk: two targets of pending '
         'gradients, zero updates, live private memory and an unfinished work '
         'trace. Interrupt separately after the first completed update. '
         'Both resumptions reproduce EVERY learned weight, Adam field, '
         'remaining gradient, state, RNG, counter and cursor bitwise, together '
         'with all output curves, DEV selection and fitting work. '
         'Tracing and untraced learning also agree bitwise; untraced runs '
         'explicitly carry no whole-fitting work estimate.'),
        ('p','Changed source hashes, settings or input data refuse recovery; '
         'completed outputs refuse overwrite. All factual/shadow forward and '
         'backward, normalization, clipping and optimizer operations have '
         'complete numerical accounting coverage. The eight-target row pays '
         '256 shadow lanes and512 shadow events across four detached chunks. '
         'Inference is the original cold native selected-value prefix. '
         'Whole-fit and per-target columns use the SAME actual eight-target '
         'denominator for both families; synthetic correctness fits provide '
         'no text8 quality comparison or benchmark advantage.'),
        ('small',f"Theory114;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'All numerical arms ran serially inside one guarded job with an8GiB '
         'host-memory floor. Original races, score clamp, key/value separation, '
         'deep persistent state and chronological noise are retained. '
         'No larger credit horizon, new timing operator or Transformer fit '
         'is admitted by these contracts; a real-data replay fit still requires '
         'its own frozen protocol and measured throughput budget. '
         'Older AWS DVS results retain their exact original source hashes: '
         'the shared regularization driver later evolved, so publication '
         'validates its archived original bytes from Git rather than requiring '
         'the current driver. Stored results and frozen report modules are unchanged.')])
    return pages
