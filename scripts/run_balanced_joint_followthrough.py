"""Wait for the reserved integrated cycle, then reuse the serial guarded worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    status=(ROOT/a.parent).with_suffix('.status.json')
    while True:
        if status.exists():
            value=json.loads(status.read_text())
            if value['status']=='completed':break
            if value['status']=='needs_review':raise RuntimeError('Reserved parent stopped: '+str(value.get('error')))
        time.sleep(10)
    subprocess.run([sys.executable,'scripts/run_addressed_memory_pair.py','--plan',a.plan],cwd=ROOT,check=True)


if __name__=='__main__':main()
