"""Bounded joint-delivery derivative, protected-state and real recovery checks."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from joint_outcome_benchmark import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh plain tag required')
    started=time.perf_counter()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_joint_outcome_race_query.py'],cwd=ROOT,check=True)
    result=dict(status='completed',args=vars(a),source_sha256=sources(),
        contract_sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['experiments/joint_outcome_contracts.py','tests/test_joint_outcome_race_query.py']},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Exact terminal categorical score credit, generic causal state, zero parent nesting, local forward identity and actual interrupted Adam recovery. Earlier core surrogate and write alternatives not promoted into exact gradients.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
