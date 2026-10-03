"""Verified historical driver and publication view; saved evidence stays unchanged."""
import copy
import hashlib
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='experiments/dvs_batched_le_benchmark.py'
DIGEST='4b163a25008ea8261acbee465770bad84dff3dbc1aa3c0ff74f39571fd83e8f8'
ARCHIVE=f'experiments/archive/frozen_sources/{DIGEST}/dvs_batched_le_benchmark.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolved_sources(sources):
    output=dict(sources)
    if output.get(NAME)==DIGEST and sha(ROOT/NAME)!=DIGEST:
        if sha(ROOT/ARCHIVE)!=DIGEST:raise ValueError('Changed archived historical driver')
        if ARCHIVE in output and output[ARCHIVE]!=DIGEST:raise ValueError('Conflicting historical archive hash')
        output.pop(NAME);output[ARCHIVE]=DIGEST
    return output
def historical_view(row):
    result=copy.deepcopy(row)
    if isinstance(result.get('source_sha256'),dict):result['source_sha256']=resolved_sources(result['source_sha256'])
    return result
def load_driver():
    if sha(ROOT/ARCHIVE)!=DIGEST:raise ValueError('Historical driver hash mismatch')
    spec=importlib.util.spec_from_file_location('_source_bound_batched_driver',ROOT/ARCHIVE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=ROOT
    return module
