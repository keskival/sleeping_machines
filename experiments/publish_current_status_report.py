"""Render/validate an isolated status PDF, then publish the reviewed manifest.

Only saved evidence/plotting: no model runtime, fitting or Torch imports.
One render thread, bounded RSS/time and an 8 GiB available-memory floor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = list((ROOT/'report').glob('*.py'))
    paths += [ROOT/'experiments/legacy_batched_driver_binding.py', ROOT/'experiments/lm_training_flops.py',
              Path(__file__).resolve(), ROOT/'experiments/analysis/aws_language_matched_progress_1m.py']
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def evidence_hashes():
    native = runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))
    current = runpy.run_path(str(ROOT/'report/current_language_status.py'))
    paths = [ROOT/'experiments/results'/name for name, _ in native['NATIVE']+native['CONTROLS']]
    paths += [ROOT/'experiments/results'/native[key] for key in ('INFERENCE', 'INFERENCE_MORE', 'SPARSE')]
    paths.append(ROOT/current['PROGRESS'])
    progress = json.loads(paths[-1].read_text())
    for row in progress['rows']:
        if row['milestone'] in (3, 4):
            checkpoint = ROOT/row['checkpoint']
            paths.extend([checkpoint, checkpoint.with_suffix('.json')])
    return {str(path.relative_to(ROOT)): sha(path) for path in paths if path.exists()}


def available_kib():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))


def rss_kib(pid):
    try:
        return int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:')))
    except FileNotFoundError:
        return 0


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2_000_000*1024, 2_000_000*1024))
    os.nice(19)


def prepare(tag):
    if Path(tag).name != tag or not tag:
        raise ValueError('Fresh plain tag required')
    stage = ROOT/'.git/report-preview'/tag
    if stage.exists():
        raise ValueError('Preserve previous preview; use a fresh tag')
    if available_kib() < 8192*1024:
        raise ValueError('Report render refused: less than 8 GiB available')
    initial_sources = sources()
    initial_evidence = evidence_hashes()
    previous = {name: sha(ROOT/name) for name in ('REPORT.md', 'report/sleeping_machines_status.pdf')}
    stage.mkdir(parents=True)
    tree = stage/'workspace'
    tree.mkdir()
    for path in ROOT.iterdir():
        if path.name == 'report':
            shutil.copytree(path, tree/path.name, ignore=shutil.ignore_patterns('__pycache__', '*.building.pdf'))
        elif path.name == 'REPORT.md':
            shutil.copy2(path, tree/path.name)
        else:
            (tree/path.name).symlink_to(path, target_is_directory=path.is_dir())
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    code = "import runpy,sys;sys.path[:0]=['report','.'];ns=runpy.run_path('report/make_pdf.py');ns['build']();assert 'torch' not in sys.modules"
    start, peak = time.monotonic(), 0
    with (stage/'render.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-c', code], cwd=tree, env=env, stdout=log,
                                stderr=subprocess.STDOUT, preexec_fn=limits, start_new_session=True)
        try:
            while proc.poll() is None:
                peak = max(peak, rss_kib(proc.pid))
                if peak > 750_000 or available_kib() < 8192*1024 or time.monotonic()-start > 420:
                    raise RuntimeError('Render stopped by RSS / 8 GiB floor / 420s guard; canonical files unchanged')
                time.sleep(.5)
            if proc.returncode:
                raise RuntimeError(f'Render failed (exit {proc.returncode}); inspect {stage}/render.log')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    import pymupdf
    pdf_path = tree/'report/sleeping_machines_status.pdf'
    headings = ['Current language evidence', 'Learning diagnosis and the next decisive checks',
                'Native language at 10M', 'Deep fitting diagnosis: looser clipping fails its prediction',
                'Existing route coverage: actual fitting cost']
    with pymupdf.open(pdf_path) as doc:
        text = '\n'.join(page.get_text() for page in doc)
        for heading in headings:
            if heading not in text.replace('\n', ' '):
                raise ValueError('Missing report section: '+heading)
        for index, page in enumerate(doc):
            if len(page.get_text().strip()) < 300:
                raise ValueError('Sparse/orphan page '+str(index+1))
            for x0, y0, x1, y1, *_ in page.get_text('blocks'):
                if min(x0, y0) < 0 or x1 > page.rect.width or y1 > page.rect.height:
                    raise ValueError('Out-of-bounds text on page '+str(index+1))
        pages = len(doc)
        for index in range(min(4, pages)):
            doc[index].get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(str(stage/f'page{index+1}.png'))
        native_pages = []
        for index, page in enumerate(doc):
            if 'Native language at 10M' in page.get_text().replace('\n', ' '):
                native_pages = list(range(index+1, min(index+4, pages)+1))
        for number in native_pages:
            doc[number-1].get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(str(stage/f'page{number}.png'))
    if initial_sources != sources() or initial_evidence != evidence_hashes() or previous != {name: sha(ROOT/name) for name in previous}:
        raise ValueError('Concurrent source/canonical edit; preserve preview without publishing')
    result = dict(status='validated_preview', tag=tag, created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=initial_sources, evidence_sha256=initial_evidence,
                  previous_sha256=previous, pdf_pages=pages, native_pages=native_pages,
                  staged_report_sha256=sha(tree/'REPORT.md'), staged_pdf_sha256=sha(pdf_path),
                  render_wall_s=time.monotonic()-start, peak_render_rss_kib=peak,
                  guards=dict(threads=1, rss_kib=750000, address_space_kib=2000000, min_available_mib=8192, timeout_s=420),
                  scope='Saved-evidence publication only; no model forward/backward/training or Torch import')
    manifest = stage/'manifest.json'
    manifest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(preview=str(stage), manifest=str(manifest), pages=pages, peak_rss_kib=peak)))


def publish(manifest_path):
    manifest = Path(manifest_path).resolve()
    if not manifest.is_relative_to(ROOT/'.git/report-preview'):
        raise ValueError('Expected a local validated-preview manifest')
    result = json.loads(manifest.read_text())
    stage, tag = manifest.parent, result['tag']
    if result['status'] != 'validated_preview' or sources() != result['source_sha256'] or evidence_hashes() != result['evidence_sha256']:
        raise ValueError('Stale preview/source hashes; rebuild')
    for name, digest in result['previous_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Concurrent canonical publication; rebuild without overwriting')
    tree = stage/'workspace'
    if sha(tree/'REPORT.md') != result['staged_report_sha256'] or sha(tree/'report/sleeping_machines_status.pdf') != result['staged_pdf_sha256']:
        raise ValueError('Changed reviewed artifact')
    record = ROOT/'experiments/results/publication'/(tag+'.json')
    if record.exists():
        raise ValueError('Publication tag already exists')
    archive = ROOT/'report/archive'/(tag+'_previous.pdf')
    if archive.exists():
        raise ValueError('Preserve previous archive')
    archive.parent.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'report/sleeping_machines_status.pdf', archive)
    for path in (tree/'report/figures').iterdir():
        if path.is_file() and path.suffix in ('.png', '.svg'):
            target = ROOT/'report/figures'/path.name
            if not target.exists() or sha(target) != sha(path):
                shutil.copy2(path, target)
    for name in ('REPORT.md', 'report/sleeping_machines_status.pdf'):
        destination = ROOT/name
        temporary = destination.with_suffix(destination.suffix+'.publishing')
        shutil.copy2(tree/name, temporary)
        temporary.replace(destination)
    result.update(status='completed', published_utc=datetime.now(timezone.utc).isoformat(),
                  previous_pdf_archive=str(archive.relative_to(ROOT)), previous_pdf_archive_sha256=sha(archive))
    record.parent.mkdir(exist_ok=True)
    with record.open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(pdf='report/sleeping_machines_status.pdf', pages=result['pdf_pages'], record=str(record))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tag')
    group.add_argument('--publish')
    args = parser.parse_args()
    prepare(args.tag) if args.tag else publish(args.publish)


if __name__ == '__main__':
    main()
