"""Run bounded real-driver and full-shape numerical contracts under run_safe."""
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
from balanced_joint_benchmark import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh plain tag required')
    started=time.perf_counter()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_balanced_joint_learning.py'],cwd=ROOT,check=True)
    result=dict(status='completed',source_sha256=sources(),
        tests='Seven balanced-marginal/count, full p8/L4 zero nesting, actual driver recovery native/tapped, target normalization and trained causality contracts',
        contract_sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['experiments/balanced_joint_contracts.py','tests/test_balanced_joint_learning.py']},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Numerical/real optimizer recovery admission only; no fitted advantage claim')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
