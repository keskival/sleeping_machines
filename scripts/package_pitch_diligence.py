"""Package the private deck, source notes and frozen evidence; no distribution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))


def package(tag):
    if not tag or Path(tag).name != tag: raise ValueError('Fresh plain tag required')
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (1_000_000 * 1024, 1_000_000 * 1024))
    if available() < 8192 * 1024: raise ValueError('Less than 8 GiB available')
    target = ROOT / 'investment' / ('sleeping_machines_private_diligence_' + tag + '.zip')
    record = target.with_suffix('.json')
    if target.exists() or record.exists(): raise ValueError('Preserve existing private pack')
    data = json.loads((ROOT / 'investment/pitch_deck_evidence_20261003.json').read_text())
    paths = set(data['completed_evidence_sha256'])
    paths.update(['investment/README.md', 'investment/PITCH_DECK.json', 'investment/PITCH_DECK_NOTES.md',
                  'investment/pitch_deck_evidence_20261003.json', 'investment/pitch_deck_benchmarks.csv',
                  'investment/pitch_deck_financial_sensitivity.csv', 'investment/sleeping_machines_pitch_deck.pdf',
                  'investment/sleeping_machines_pitch_deck_main.pdf', 'investment/INVESTOR_READING_REVIEW.md',
                  'report/sleeping_machines_status.pdf', 'scripts/build_pitch_deck.py',
                  'scripts/package_pitch_diligence.py'])
    deck_sha = sha(ROOT / 'investment/sleeping_machines_pitch_deck.pdf')
    publications = []
    for path in (ROOT / 'investment').glob('publication_pitch_deck_*.json'):
        value = json.loads(path.read_text())
        if value.get('output_sha256') == deck_sha: publications.append(path)
    if len(publications) != 1: raise ValueError('One matching completed deck publication required')
    paths.add(str(publications[0].relative_to(ROOT)))
    for path in paths:
        p = Path(path)
        if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe archive member')
    hashes = {path: sha(ROOT / path) for path in sorted(paths)}
    for path, digest in data['completed_evidence_sha256'].items():
        if hashes[path] != digest: raise ValueError('Changed frozen parent: ' + path)
    total = sum((ROOT / path).stat().st_size for path in paths)
    if total > 32 * 1024 * 1024: raise ValueError('Private pack exceeds bounded 32 MiB input size')
    manifest = dict(classification='PRIVATE REVIEW — founder-controlled disclosure; no external distribution',
                    packaged_utc=datetime.now(timezone.utc).isoformat(),
                    file_sha256=hashes, total_input_bytes=total,
                    scope='Curated deck and evidence records. No weights, credentials or private contact data. '
                          'Older evidence snapshot includes historical pricing assumptions; new full deck governs the current proposal.')
    temporary = target.with_suffix('.packing')
    if temporary.exists(): raise ValueError('Preserve previous partial pack; use a fresh tag')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr('PRIVATE_PACK_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        for path in sorted(paths):
            if available() < 8192 * 1024: raise ValueError('Available-memory guard tripped')
            archive.write(ROOT / path, path)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != paths | {'PRIVATE_PACK_MANIFEST.json'}: raise ValueError('Archive membership mismatch')
        for path, digest in hashes.items():
            if hashlib.sha256(archive.read(path)).hexdigest() != digest: raise ValueError('Archive hash mismatch')
    if hashes != {path: sha(ROOT / path) for path in sorted(paths)}: raise ValueError('Concurrent pack input change')
    temporary.replace(target)
    record.write_text(json.dumps(dict(manifest, archive_sha256=sha(target), archive_bytes=target.stat().st_size), indent=2) + '\n')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(archive=str(target), members=len(paths)+1, bytes=target.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag', required=True)
    package(parser.parse_args().tag)
