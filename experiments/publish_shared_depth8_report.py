"""Guarded publication of completed bounded-score training/utility evidence."""
import argparse
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
sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf']
    if subprocess.check_output(['git','status','--porcelain','--',*artifacts],cwd=ROOT):
        raise ValueError('Preserve concurrent report edits')
    sources=['report/readable_report.py','report/make_pdf.py','report/cross_host_bridge_training_evidence.py',
             'experiments/publish_shared_depth8_report.py']
    hashes={name:N.sha(ROOT/name) for name in sources};begin=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='bounded-training-report-') as tmp:
        backups={}
        for idx,name in enumerate(artifacts):
            backups[name]=Path(tmp)/str(idx);shutil.copy2(ROOT/name,backups[name])
        try:
            subprocess.run([sys.executable,'report/make_pdf.py','report/cross_host_bridge_training_evidence.py'],
                cwd=ROOT,check=True,timeout=120)
            content=(ROOT/'REPORT.md').read_text()
            for title in ['AWS depth8 replay: seed7 positive pilot', 'AWS depth8 replay: seed8 confirmation fails',
                          'Deep causal language: admission only', 'Bounded score bridge: integrated training contracts',
                          'Restored native sensitivity has small, mixed route utility',
                          'Native conditional choice-clock','Full coarse matrix','Query-only native admission']:
                if title not in content:raise ValueError('Missing completed/history page '+title)
            with pymupdf.open(ROOT/artifacts[1]) as pdf:
                pages=len(pdf)
                for idx,page in enumerate(pdf):
                    if len(page.get_text())<400:raise ValueError('Orphan page '+str(idx+1))
                    for x0,y0,x1,y1,*_ in page.get_text('blocks'):
                        if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:
                            raise ValueError('Bounds page '+str(idx+1))
            subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
            if hashes!={name:N.sha(ROOT/name) for name in sources}:raise ValueError('Concurrent source edit')
        except BaseException:
            failed=ROOT/'.git/report-validation';failed.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/artifacts[1],failed/(a.tag+'.pdf'))
            for name,path in backups.items():shutil.copy2(path,ROOT/name)
            raise
    result=dict(status='completed',args=vars(a),source_sha256=hashes,pdf_pages=pages,
        report_sha256=N.sha(ROOT/artifacts[0]),pdf_sha256=N.sha(ROOT/artifacts[1]),
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Completed integrated bridge contracts, matched restricted smokes and native conditional utility; prior evidence retained.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',pages=pages,wall_s=result['wall_s'])))


if __name__=='__main__':main()
