"""Publish completed guarded credit fits or frozen diagnostics on main."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pymupdf

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=sys.argv[1:]
    if not paths:raise ValueError('Completed records required')
    for name in paths:
        path=(ROOT/name).resolve()
        if not any(path.is_relative_to(ROOT/'experiments/results'/d) for d in ('episodic_language','diagnostics')):
            raise ValueError('Unexpected result directory')
        row=json.loads(path.read_text())
        if row['status']!='completed':raise ValueError('Completed record required')
        if row.get('protocol',{}).get('official_test_read'):raise ValueError('Development only')
        for f,sha in row['source_sha256'].items():
            if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=sha:raise ValueError('Source changed: '+f)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='main':raise ValueError('Main required')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:raise ValueError('Preserve staged changes')
    artifacts=['REPORT.md','report/sleeping_machines_status.pdf','report/figures']
    if subprocess.run(['git','diff','--quiet','--',*artifacts,'report/readable_report.py','report/make_pdf.py'],cwd=ROOT).returncode:
        raise ValueError('Preserve report edits')
    subprocess.run([sys.executable,'report/make_pdf.py'],cwd=ROOT,check=True)
    pdf=pymupdf.open(ROOT/'report/sleeping_machines_status.pdf')
    for i,page in enumerate(pdf):
        if len(page.get_text())<400:raise ValueError(f'Orphan page {i+1}')
        for x0,y0,x1,y1,*_ in page.get_text('blocks'):
            if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height:raise ValueError(f'Text outside page {i+1}')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','add',*paths,*artifacts],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Publish integrated language credit stage and frozen diagnosis'],cwd=ROOT,check=True)


if __name__=='__main__':main()
