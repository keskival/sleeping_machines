"""Publish a completed local language result into reviewable main commits.

Called serially after the training guard exits. No remote push is performed.
Refuse to overwrite report edits or mix another writer's staged changes.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def main():
    if len(sys.argv)!=2:
        raise ValueError('One completed language result path required')
    path=(ROOT/sys.argv[1]).resolve()
    if not path.is_relative_to(ROOT/'experiments/results/parallel_language'):
        raise ValueError('Result must be in the declared language directory')
    result=json.loads(path.read_text())
    if result['status']!='completed':raise ValueError('Do not publish live progress')
    if run('git','branch','--show-current')!='main':raise ValueError('Commit on main only')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        raise ValueError('Another writer has staged changes; preserve them and review manually')
    report_paths=['REPORT.md','report/readable_report.py','report/make_pdf.py',
                  'report/sleeping_machines_status.pdf','report/figures']
    if subprocess.run(['git','diff','--quiet','--',*report_paths],cwd=ROOT).returncode:
        raise ValueError('Report edits are present; preserve them and review manually')
    for source,digest in result['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Completed result/source mismatch: '+source)
    subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,check=True)
    pdf=pymupdf.open(ROOT/'report/sleeping_machines_status.pdf')
    for i,page in enumerate(pdf):
        if len(page.get_text())<400:raise ValueError(f'Orphan report page {i+1}')
        for x0,y0,x1,y1,*_ in page.get_text('blocks'):
            if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:
                raise ValueError(f'Report text outside page {i+1}')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    figures=run('git','ls-files','report/figures').splitlines()
    subprocess.run(['git','add','--',str(path.relative_to(ROOT)),'REPORT.md',
                    'report/sleeping_machines_status.pdf',*figures],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',f"Report completed language stage: {result['args']['tag']}"],cwd=ROOT,check=True)
    print('Committed completed evidence on main; authenticated host can push.',flush=True)


if __name__=='__main__':main()
