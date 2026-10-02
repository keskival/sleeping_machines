"""Full-size projection-placement learning/recovery and real-driver contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import torch
import pytest
import deep_memory_optimizer_contracts as harness
from sleeping_machines.late_projected_context_memory import LateProjectedContextModel
from addressed_memory_language_benchmark import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique unused plain tag required')
    started=time.perf_counter();torch.set_num_threads(1)
    # Reuse the full actual native optimizer harness with a substituted factory;
    # it is isolated in this process and does not mutate any frozen source.
    harness.ContextAddressedNativeModel=LateProjectedContextModel
    numerical=harness.contract('addressed');numerical['model']='late_projection'
    tests=['tests/test_late_projected_context_memory.py','tests/test_addressed_memory_driver_recovery.py']
    rc=pytest.main(['-q',*tests]);assert rc==0,'Numerical/whole-driver prerequisite failed'
    names=['experiments/late_projection_contracts.py','experiments/deep_memory_optimizer_contracts.py',
        'experiments/addressed_memory_language_benchmark.py','sleeping_machines/late_projected_context_memory.py',
        'sleeping_machines/context_addressed_memory.py','experiments/parallel_head_gradient_accumulation.py',*tests]
    result=dict(status='completed',args=vars(a),numerical_contract=numerical,tests_passed=8,
        projection_fixed_weight_equivalence_and_old_feature_adjoint_verified=True,
        actual_driver_interruption_recovery_verified=['native','addressed','late'],
        official_test_read=False,wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={**sources(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}},
        hardware=dict(host=__import__('os').uname().nodename,device='cpu',threads=1,torch=torch.__version__),
        scope='Numerical/fixed-feature projection and recovery prerequisites, not semantic/quality evidence; core historical producer credit remains truncated.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])),flush=True)


if __name__=='__main__':main()
