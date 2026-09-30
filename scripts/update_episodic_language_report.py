"""Publish completed matched episodic development evidence on main."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pymupdf

ROOT=Path(__file__).resolve().parents[1]


def main():
    if not sys.argv[1:]:raise ValueError('Completed result paths required')
    for name in sys.argv[1:]:
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT/'experiments/results/episodic_language'):
            raise ValueError('Unexpected result directory')
        r=json.loads(path.read_text())
        if r['status']!='completed' or r['protocol']['official_test_read']:
            raise ValueError('Completed development result required')
        for source,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
                raise ValueError('Source changed: '+source)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':
        raise ValueError('Work on main')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        raise ValueError('Preserve another writer staged changes')
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures']
    if subprocess.run(['git','diff','--quiet','--',*artifacts,'report/readable_report.py','report/make_pdf.py'],cwd=ROOT).returncode:
        raise ValueError('Preserve existing report edits; review and publish manually')
    subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,check=True)
    pdf=pymupdf.open(ROOT/'report/sleeping_machines_status.pdf')
    for i,page in enumerate(pdf):
        if len(page.get_text())<400:raise ValueError(f'Orphan page {i+1}')
        for x0,y0,x1,y1,*_ in page.get_text('blocks'):
            if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:
                raise ValueError(f'Text outside page {i+1}')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','add',*sys.argv[1:],*artifacts],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Report matched integrated episodic race memory experiments'],cwd=ROOT,check=True)


if __name__=='__main__':main()
