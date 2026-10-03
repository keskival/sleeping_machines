"""Fetch/recreate exact official archives from committed download provenance."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    destination=ROOT/'data/public_benchmarks/raw';destination.mkdir(parents=True,exist_ok=True)
    for item in json.loads((ROOT/'experiments/public_benchmarks/downloads.json').read_text()):
        archive=destination/(item['dataset']+'.zip')
        if not archive.exists():
            with urllib.request.urlopen(item['url'],timeout=60) as stream:blob=stream.read()
            assert hashlib.sha256(blob).hexdigest()==item['sha256'];archive.write_bytes(blob)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['sha256']
        with zipfile.ZipFile(archive) as stream:
            for name in stream.namelist():
                if name.lower().endswith('.ts'):
                    assert name==Path(name).name
                    target=destination/name;blob=stream.read(name)
                    if target.exists():assert target.read_bytes()==blob
                    else:target.write_bytes(blob)
        print(item['dataset'],'verified')
