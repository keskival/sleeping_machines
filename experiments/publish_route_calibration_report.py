"""Publish completed cross-host evidence, preserving prior report on failure."""
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


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--analysis', action='append', required=True)
    a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unused plain tag required')
    analyses = {}
    for name in a.analysis:
        data = json.loads((ROOT / name).read_text())
        if data['status'] != 'completed':
            raise ValueError('Completed evidence required')
        for row in data.get('common_unit_ledger', []):
            parent = row.get('result', row.get('parent_result'))
            if parent and digest(parent) != row.get('result_sha256', row.get('parent_result_sha256')):
                raise ValueError('Changed parent evidence')
        analyses[name] = digest(name)
    artifacts = ['REPORT.md', 'report/sleeping_machines_status.pdf']
    artifacts += ['report/figures/' + json.loads((ROOT / name).read_text())['args']['tag'] + '_learning.png'
                  for name in a.analysis if 'common_unit_ledger' in json.loads((ROOT / name).read_text())]
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *artifacts], cwd=ROOT):
        raise ValueError('Preserve concurrent report artifacts')
    sources = ['experiments/publish_route_calibration_report.py', 'report/readable_report.py', 'report/make_pdf.py', 'report/current_dvs_appendix.py', 'report/reception_evidence.py', 'report/route_calibration_evidence.py']
    source_before = {name: digest(name) for name in sources}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='joint-learning-report-') as tmp:
        backups = {}
        for n, name in enumerate(artifacts):
            if (ROOT / name).exists():
                backups[name] = Path(tmp) / str(n)
                shutil.copy2(ROOT / name, backups[name])
        try:
            subprocess.run([sys.executable, 'report/make_pdf.py', 'report/current_dvs_appendix.py', 'report/reception_evidence.py', 'report/route_calibration_evidence.py'], cwd=ROOT, check=True, timeout=120)
            markdown=(ROOT/'REPORT.md').read_text()
            required=['AWS coarse temporal screens','Full coarse matrix','Strong coarse controls','Integrated quadratic head','Producer-held selection', 'Receiving losing arrivals', 'Native reception intervention', 'AWS frozen polynomial decoder', 'Race-scaled reception', 'Memory-conditioned keys', 'Clock-preserving route calibration', 'fresh score gains', 'Integrated exact replay', 'compact context head']
            if any(title not in markdown for title in required):
                raise ValueError('Cross-host report section missing')
            with pymupdf.open(ROOT / artifacts[1]) as pdf:
                pages = len(pdf)
                for index, page in enumerate(pdf):
                    if len(page.get_text()) < 400:
                        raise ValueError('Orphan report page ' + str(index + 1))
                    for x0, y0, x1, y1, *_ in page.get_text('blocks'):
                        if min(x0, y0) < 0 or x1 > page.rect.width or y1 > page.rect.height:
                            raise ValueError('Out-of-bounds text on page ' + str(index + 1))
            subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
            if {name: digest(name) for name in sources} != source_before:
                raise ValueError('Report source changed during rendering')
        except BaseException:
            failed = ROOT / '.git' / 'report-validation'
            failed.mkdir(parents=True, exist_ok=True)
            if (ROOT / artifacts[1]).exists():
                shutil.copy2(ROOT / artifacts[1], failed / (a.tag + '.pdf'))
            for name in artifacts:
                if name in backups:
                    shutil.copy2(backups[name], ROOT / name)
                elif (ROOT / name).exists():
                    (ROOT / name).unlink()
            raise
    result = dict(status='completed', args=vars(a), analysis_sha256=analyses, pdf_pages=pages,
                  wall_s=time.perf_counter() - started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={name: digest(name) for name in sources},
                  scope='Completed cross-host DVS evidence, historical failures and common-unit boundaries preserved; exploratory scope retained.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
