"""Verified historical driver and publication view; saved evidence stays unchanged."""
import copy
import hashlib
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='experiments/dvs_batched_le_benchmark.py'
DIGEST='4b163a25008ea8261acbee465770bad84dff3dbc1aa3c0ff74f39571fd83e8f8'
ARCHIVE=f'experiments/archive/frozen_sources/{DIGEST}/dvs_batched_le_benchmark.py'
BATCH_NAME='sleeping_machines/batched_episodes.py'
BATCH_DIGEST='83265f63633a666e6bc788c80b4ebbde3ede468b15d38c2f60438427c75a3bc4'
BATCH_ARCHIVE=f'experiments/archive/frozen_sources/{BATCH_DIGEST}/batched_episodes.py'
# Verified unchanged at both archived producer anchors (07eaaae^/e096ca6^).
BATCH_DEPENDENCY_NAME='sleeping_machines/parallel_stream_language.py'
BATCH_DEPENDENCY_DIGEST='7237331f34b4b7802fe98896511a301480a17ebb647297dd47df94246d6ae5ce'
BINDINGS=((NAME,DIGEST,ARCHIVE),(BATCH_NAME,BATCH_DIGEST,BATCH_ARCHIVE))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolved_sources(sources):
    output=dict(sources)
    for name,digest,archive in BINDINGS:
        if output.get(name)==digest and sha(ROOT/name)!=digest:
            if sha(ROOT/archive)!=digest:raise ValueError('Changed archived historical source '+name)
            if archive in output and output[archive]!=digest:raise ValueError('Conflicting historical archive hash '+name)
            output.pop(name);output[archive]=digest
    return output
def historical_view(row):
    result=copy.deepcopy(row)
    if isinstance(result.get('source_sha256'),dict):result['source_sha256']=resolved_sources(result['source_sha256'])
    return result
def load_batched():
    if sha(ROOT/BATCH_ARCHIVE)!=BATCH_DIGEST:raise ValueError('Historical batched episode hash mismatch')
    if sha(ROOT/BATCH_DEPENDENCY_NAME)!=BATCH_DEPENDENCY_DIGEST:raise ValueError('Historical temporal transport dependency changed')
    batch_spec=importlib.util.spec_from_file_location('sleeping_machines._source_bound_batched_episodes',ROOT/BATCH_ARCHIVE);batch=importlib.util.module_from_spec(batch_spec);batch_spec.loader.exec_module(batch)
    return batch
def load_driver():
    if sha(ROOT/ARCHIVE)!=DIGEST:raise ValueError('Historical driver hash mismatch')
    batch=load_batched()
    spec=importlib.util.spec_from_file_location('_source_bound_batched_driver',ROOT/ARCHIVE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=ROOT
    module.batched_logits=batch.batched_logits
    return module
