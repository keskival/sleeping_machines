"""Bounded, source-guarded rendering of the investment memo and short pitch.

Saved text/evidence only. No numerical model runtime or training imports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['investment/INVESTMENT_CASE.md', 'investment/PITCH.md',
          'investment/evidence_20261003.json', 'scripts/build_investment_case.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))


def verify_evidence():
    data = json.loads((ROOT/'investment/evidence_20261003.json').read_text())
    for name, digest in data['completed_evidence_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed investor evidence: '+name)


def render(stage):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    # Locate bundled fonts without importing matplotlib or its numerical stack.
    font_dir = Path(importlib.util.find_spec('matplotlib').origin).parent/'mpl-data/fonts/ttf'
    for name, file in [('Inv', 'DejaVuSans.ttf'), ('InvB', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir/file)))
    pdfmetrics.registerFontFamily('Inv', normal='Inv', bold='InvB', italic='Inv', boldItalic='InvB')
    styles = dict(body=ParagraphStyle('Body', fontName='Inv', fontSize=9.2, leading=13.2, spaceAfter=7,
                                     allowWidows=0, allowOrphans=0),
                  title=ParagraphStyle('Title', fontName='InvB', fontSize=23, leading=29, spaceAfter=12),
                  heading=ParagraphStyle('Heading', fontName='InvB', fontSize=13.5, leading=18,
                                         spaceBefore=12, spaceAfter=7, keepWithNext=True),
                  cell=ParagraphStyle('Cell', fontName='Inv', fontSize=7.1, leading=10))

    def inline(value):
        value = html.escape(value)
        value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#2463a0">\1</link>', value)
        return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Inv', 7)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(18*mm, 12*mm, 'Sleeping Machines | investment thesis | 3 October 2026')
        canvas.drawRightString(A4[0]-18*mm, 12*mm, str(doc.page))
        canvas.restoreState()

    for source, output in [('INVESTMENT_CASE.md', 'sleeping_machines_investment_case.pdf'),
                           ('PITCH.md', 'sleeping_machines_investor_pitch.pdf')]:
        lines = (ROOT/'investment'/source).read_text().splitlines()
        story, paragraph, index = [], [], 0

        def flush():
            if paragraph:
                story.append(Paragraph(inline(' '.join(paragraph)), styles['body']))
                paragraph.clear()

        while index < len(lines):
            line = lines[index].strip()
            if line.startswith('|'):
                flush(); rows = []
                while index < len(lines) and lines[index].strip().startswith('|'):
                    text = lines[index].strip()
                    if not re.fullmatch(r'[\s|:-]+', text):
                        rows.append([Paragraph(inline(cell.strip()), styles['cell'])
                                     for cell in text.strip('|').split('|')])
                    index += 1
                count = len(rows[0]); width = A4[0]-36*mm
                if count == 5: proportions = [.33, .13, .17, .18, .19]
                elif 'Mechanism' in rows[0][0].getPlainText(): proportions = [.23, .36, .41]
                else: proportions = [.22, .55, .23]
                table = Table(rows, colWidths=[width*x for x in proportions], repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf3fc')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafbfd')]),
                    ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#2a78d6')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
                story.append(KeepTogether([table, Spacer(1, 9)]))
                continue
            if not line:
                flush()
            elif line.startswith('# '):
                flush(); story.append(Paragraph(inline(line[2:]), styles['title']))
            elif line.startswith('## '):
                flush(); story.append(Paragraph(inline(line[3:]), styles['heading']))
            elif line.startswith('- ') or re.match(r'^\d+\. ', line):
                flush(); paragraph.append(line)
            else:
                paragraph.append(line)
            index += 1
        flush()
        doc = SimpleDocTemplate(str(stage/output), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=18*mm, bottomMargin=22*mm,
                                title='Sleeping Machines: '+('investment case' if source.startswith('INV') else 'investor pitch'),
                                author='Sleeping Machines')
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (1_000_000*1024, 1_000_000*1024))
    os.nice(19)


def publish(tag):
    if not tag or Path(tag).name != tag:
        raise ValueError('Fresh plain tag required')
    stage = ROOT/'.git/investment-preview'/tag
    if stage.exists(): raise ValueError('Preserve previous preview; use a fresh tag')
    record_path = ROOT/'investment'/('publication_'+tag+'.json')
    if record_path.exists(): raise ValueError('Preserve previous publication record')
    if available() < 8192*1024: raise ValueError('Less than 8 GiB available')
    verify_evidence()
    hashes = {name: sha(ROOT/name) for name in INPUTS}
    outputs = ['sleeping_machines_investment_case.pdf', 'sleeping_machines_investor_pitch.pdf']
    previous = {name: sha(ROOT/'investment'/name) if (ROOT/'investment'/name).exists() else None for name in outputs}
    stage.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start, peak = time.monotonic(), 0
    with (stage/'render.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--render-stage', str(stage)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
        try:
            while child.poll() is None:
                status = Path(f'/proc/{child.pid}/status')
                try:
                    rss = next((int(line.split()[1]) for line in status.read_text().splitlines()
                                if line.startswith('VmRSS:')), 0)
                except FileNotFoundError:
                    rss = 0
                peak = max(peak, rss)
                if rss > 300000 or available() < 8192*1024 or time.monotonic()-start > 120:
                    raise RuntimeError('Investment render resource guard tripped')
                time.sleep(.1)
            if child.returncode: raise RuntimeError('Render failed; inspect '+str(stage/'render.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    import pymupdf
    counts = {}
    for name in outputs:
        with pymupdf.open(stage/name) as doc:
            counts[name] = len(doc)
            for index, page in enumerate(doc):
                if len(page.get_text().strip()) < 120: raise ValueError('Sparse investor page')
                for x0, y0, x1, y1, *_ in page.get_text('blocks'):
                    if min(x0, y0) < 0 or x1 > page.rect.width or y1 > page.rect.height:
                        raise ValueError('Out-of-bounds investor text')
                page.get_pixmap().save(str(stage/(name.replace('.pdf', f'_page{index+1}.png'))))
    verify_evidence()
    if hashes != {name: sha(ROOT/name) for name in INPUTS}: raise ValueError('Concurrent memo/source change')
    for name, digest in previous.items():
        path = ROOT/'investment'/name
        if digest != (sha(path) if path.exists() else None): raise ValueError('Concurrent publication')
    record = dict(status='completed', tag=tag, published_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashes, previous_sha256=previous,
                  outputs_sha256={name: sha(stage/name) for name in outputs}, pages=counts,
                  render_wall_s=time.monotonic()-start, peak_render_rss_kib=peak,
                  guards=dict(threads=1, rss_kib=300000, address_space_kib=1000000,
                              min_available_mib=8192, timeout_s=120),
                  scope='Investment narrative and saved evidence; no model runtime or numerical import')
    for name in outputs:
        path = ROOT/'investment'/name
        if path.exists():
            archive = ROOT/'investment/archive'/(tag+'_previous_'+name)
            if archive.exists(): raise ValueError('Preserve previous investor artifact')
            archive.parent.mkdir(exist_ok=True)
            shutil.copy2(path, archive)
        temporary = path.with_suffix('.publishing')
        shutil.copy2(stage/name, temporary); temporary.replace(path)
    with record_path.open('x') as file:
        file.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(pages=counts, peak_rss_kib=peak, preview=str(stage))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tag'); group.add_argument('--render-stage')
    args = parser.parse_args()
    render(Path(args.render_stage)) if args.render_stage else publish(args.tag)
