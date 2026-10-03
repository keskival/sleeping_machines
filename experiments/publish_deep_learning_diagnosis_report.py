"""Guarded publication of deep-learning diagnosis and causal-prefix resource tradeoff."""
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
    sources=['report/readable_report.py','report/make_pdf.py','report/deep_learning_diagnosis_evidence.py',
             'experiments/publish_deep_learning_diagnosis_report.py']
    hashes={name:N.sha(ROOT/name) for name in sources};begin=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='bounded-training-report-') as tmp:
        backups={}
        for idx,name in enumerate(artifacts):
            backups[name]=Path(tmp)/str(idx);shutil.copy2(ROOT/name,backups[name])
        try:
            subprocess.run([sys.executable,'report/make_pdf.py','report/deep_learning_diagnosis_evidence.py'],
                cwd=ROOT,check=True,timeout=120)
            content=(ROOT/'REPORT.md').read_text()
            for title in ['Added message branches: float32 silence and Adam epsilon',
                          'Cached causal prefixes: equivalent credit with fewer operations',
                          'Cached-prefix prototype: CPU wall regression prevents promotion',
                          'Deep fitting diagnosis: looser clipping fails its prediction',
                          'Progressive depth growth: preserve the actual parent computation',
                          'Actual Adam updates: clipping scale and moment history differ',
                          'Production precision: equivalent gradients and every shadow route agree',
                          'Production causal depth8 replay driver: numerical admission complete',
                          'Full corrected language replay: avoid the redundant winning shadow',
                          'AWS deep language smokes: same learning with half the replay work',
                          'Factual-winner reuse: measured work and numerical limits',
                          'Replay fitting driver: interrupted learning recovers exactly',
                          'Stateful language replay accumulator: chronological RNG retained',
                          'Native text8 horizon return: four credit reversals',
                          'Horizon confirmation: downstream effects vary by context',
                          'Corrected causal language replay: sixteen contracts pass',
                          'Production-size language replay: paid resource admission',
                          'Persistent representation and missing delayed credit', 'AWS depth8 replay: seed7 positive pilot', 'AWS depth8 replay: seed8 confirmation fails',
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
        scope='Completed cached-prefix proofs/work/negative wall, deep cap controls, actual-moment forks and source-bound growth repair; all earlier evidence retained; quality advantage pending.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',pages=pages,wall_s=result['wall_s'])))


if __name__=='__main__':main()
