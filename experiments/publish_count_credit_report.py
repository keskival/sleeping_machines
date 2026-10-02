"""Guarded publication after the full/minimal credit pair and its matched analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import pymupdf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_aws_matrix_recovery import validate_result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused publication tag required')
    path=ROOT/a.plan;plan=json.loads(path.read_text());state=json.loads(path.with_suffix('.status.json').read_text())
    if state['status']!='completed':raise ValueError('Completed pair required')
    job=next(j for j in plan['jobs'] if j['stage']=='analysis');analysis=validate_result(ROOT,job)
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures']
    if subprocess.check_output(['git','status','--porcelain','--',*artifacts],cwd=ROOT):raise ValueError('Preserve concurrent report edits')
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='count-credit-report-') as backup:
        for name in artifacts[:2]:shutil.copy2(ROOT/name,Path(backup)/Path(name).name)
        try:
            subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,check=True,timeout=90)
            pdf=pymupdf.open(ROOT/artifacts[1]);pages=len(pdf)
            for index,page in enumerate(pdf):
                if len(page.get_text())<400:raise ValueError('Orphan report page '+str(index+1))
                for x0,y0,x1,y1,*_ in page.get_text('blocks'):
                    if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:raise ValueError('Out-of-bounds text')
            pdf.close();subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
        except BaseException:
            for name in artifacts[:2]:shutil.copy2(Path(backup)/Path(name).name,ROOT/name)
            raise
    names=['experiments/publish_count_credit_report.py','report/readable_report.py','report/make_pdf.py']
    result=dict(status='completed',args=vars(a),kind='completed_credit_pair_publication',pdf_pages=pages,
        analysis_result=job['result'],analysis_sha256=hashlib.sha256((ROOT/job['result']).read_bytes()).hexdigest(),
        followup_gate_passed=analysis['followup_gate_passed'],wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='Completed exploratory evidence only, with common units and raw/clockless costs separated; no pending score or automatic scale-up.')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
