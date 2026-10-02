"""Guarded completed value-credit/frozen-fusion report publication."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    path=ROOT/a.plan;plan=json.loads(path.read_text());assert json.loads(path.with_suffix('.status.json').read_text())['status']=='completed'
    analysis_path=next(j['result'] for j in plan['jobs'] if j['stage']=='analysis')
    audit_path='experiments/results/diagnostics/local_value_credit_frozen_20261002T090800Z.json'
    analysis=json.loads((ROOT/analysis_path).read_text());audit=json.loads((ROOT/audit_path).read_text())
    assert analysis['status']==audit['status']=='completed'
    assert audit['analysis_sha256']==hashlib.sha256((ROOT/analysis_path).read_bytes()).hexdigest()
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures/value_credit_learning_20261002T090800Z.png']
    if subprocess.check_output(['git','status','--porcelain','--',*artifacts],cwd=ROOT):raise ValueError('Preserve concurrent report artifacts')
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='value-credit-report-') as tmp:
        backups={}
        for n,name in enumerate(artifacts):
            if (ROOT/name).exists():
                backups[name]=Path(tmp)/str(n);shutil.copy2(ROOT/name,backups[name])
        try:
            subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,check=True,timeout=90)
            pdf=pymupdf.open(ROOT/artifacts[1]);pages=len(pdf)
            for index,page in enumerate(pdf):
                if len(page.get_text())<400:raise ValueError('Orphan report page '+str(index+1))
                for x0,y0,x1,y1,*_ in page.get_text('blocks'):
                    if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:raise ValueError('Out-of-bounds text')
            pdf.close();subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
        except BaseException:
            for name in artifacts:
                if name in backups:shutil.copy2(backups[name],ROOT/name)
                elif (ROOT/name).exists():(ROOT/name).unlink()
            raise
    names=['experiments/publish_value_credit_report.py','report/readable_report.py','report/make_pdf.py']
    result=dict(status='completed',args=vars(a),pdf_pages=pages,
        analysis_result=analysis_path,analysis_sha256=hashlib.sha256((ROOT/analysis_path).read_bytes()).hexdigest(),
        frozen_audit=audit_path,frozen_audit_sha256=hashlib.sha256((ROOT/audit_path).read_bytes()).hexdigest(),
        followup_gate_passed=analysis['followup_gate_passed'],wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='Completed exploratory paired quality/resource and frozen algebra/conditional-information evidence; no pending prediction, energy or supremacy claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
