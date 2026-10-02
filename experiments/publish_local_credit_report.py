"""Guarded publication of a completed credit pair and its learning diagnostics."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_local_frozen_recovery import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True); parser.add_argument('--parent-plan',required=True)
    args=parser.parse_args(); out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json')
    if Path(args.tag).name!=args.tag or out.exists(): raise ValueError('Unused publication tag required')
    plan=json.loads((ROOT/args.parent_plan).read_text())
    state=json.loads((ROOT/args.parent_plan).with_suffix('.status.json').read_text())
    if state['status']!='completed': raise ValueError('Complete comparison required')
    analysis=compare(plan); started=time.perf_counter()
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures']
    if subprocess.check_output(['git','status','--porcelain','--',*artifacts],cwd=ROOT):
        raise ValueError('Preserve concurrent report edits before publication')
    # Preserve both valid rendered documents if a rebuild/validation fails.
    with tempfile.TemporaryDirectory(prefix='local-credit-report-') as backup:
        for name in artifacts[:2]: shutil.copy2(ROOT/name,Path(backup)/Path(name).name)
        try:
            subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,timeout=90,check=True)
            pdf=pymupdf.open(ROOT/'report/sleeping_machines_status.pdf')
            for index,page in enumerate(pdf):
                if len(page.get_text())<400: raise ValueError('Orphan report page '+str(index+1))
                for x0,y0,x1,y1,*_ in page.get_text('blocks'):
                    if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:
                        raise ValueError('Out-of-bounds report page '+str(index+1))
            pages=len(pdf); pdf.close()
            subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
        except BaseException:
            for name in artifacts[:2]: shutil.copy2(Path(backup)/Path(name).name,ROOT/name)
            raise
    rows=[]
    for job in plan['jobs']:
        if job['stage']!='pilot': continue
        r=json.loads((ROOT/job['result']).read_text()); w=r['work']
        rows.append(dict(result=job['result'],accuracy=r['final']['dev']['accuracy'],nll=r['final']['dev']['nll'],
            fitting_targets=w['fitting_query_targets'],whole_fit_gflops=w['total_training_unit_special_flops']/1e9,
            fit_mflops_per_query=w['total_training_unit_special_flops']/w['fitting_query_targets']/1e6,
            inference_mflops_per_query=(w['inference_arithmetic_flops_per_query']+w['inference_special_functions_per_query'])/1e6,
            available_receivers=w['available_receivers'],selected_updates_per_event=w['selected_updates_per_event'],
            scored_keys_per_event=w['key_scores_per_event']))
    sources=[Path(__file__),ROOT/'report/readable_report.py',ROOT/'report/make_pdf.py']
    result=dict(status='completed',args=vars(args),kind='completed_evidence_publication',pdf_pages=pages,
        comparison=analysis,common_unit_ledger=rows,wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Completed single-seed exploratory fits; no pending score, comparable-quality supremacy or physical-energy claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
