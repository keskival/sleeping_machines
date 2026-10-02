"""Accounted causal generator-informed timing reference, never a fitted control."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.paired_timing_tasks import paired_timing_episodes
from experiments.native_event_tasks import data_hash
from sleeping_machines.decay_timing_reference import DecayTimingReference
from sleeping_machines.operation_audit import OperationAudit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh unique tag required')
    torch.set_num_threads(1)
    rows=paired_timing_episodes(4,1024,3201);start=time.perf_counter();correct=0;queries=0;events=0
    audit=OperationAudit()
    with audit:
        for row in rows:
            model=DecayTimingReference()
            for e in row:
                margin=model.consume(e.source,e.time,e.mark);events+=1
                if margin is not None:queries+=1;correct+=int((float(margin)>0)==e.target)
    work=audit.result()
    if not work['formula_coverage_complete']:raise ValueError('Complete operator coverage required')
    names=('experiments/decay_timing_reference_benchmark.py','sleeping_machines/decay_timing_reference.py',
           'experiments/paired_timing_tasks.py','experiments/native_event_tasks.py','sleeping_machines/operation_audit.py')
    result=dict(status='completed',args=vars(a),accuracy=correct/queries,correct=correct,queries=queries,
        events=events,data_sha256=data_hash(rows),work=work,
        inference_arithmetic_flops_per_query=work['arithmetic_flops']/queries,
        inference_special_functions_per_query=work['special_function_evaluations']/queries,
        fitting_targets=0,fitting_flops=0,learned_parameters=0,occupied_sources=4,
        persistent_tensor_bytes=4*3*8,wall_s=time.perf_counter()-start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='Diagnostic using known generator time constants and coefficient. Same1024-query seed3201 confirmation population; no fitting or calibrated NLL. Two causal traces per observed address, no race selection/counterfactual learning. Not an information-matched learned control or supremacy claim; validates timestamp sufficiency and task-specific sufficient state. Arithmetic excludes Python control/indexing and is not energy.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(accuracy=result['accuracy'],per_query_flops=result['inference_arithmetic_flops_per_query'])))


if __name__=='__main__':main()
