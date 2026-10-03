"""Download the official NeuroBench MG bytes; never regenerate or score series."""
import ast,hashlib,json,struct,tarfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
URL='https://huggingface.co/datasets/NeuroBench/mackey_glass/resolve/main/data.tar.gz'
if __name__=='__main__':
    destination=ROOT/'data/neurobench/mackey_glass/data';destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'data.tar.gz'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=60) as stream:blob=stream.read(2_000_000)
        assert len(blob)<2_000_000,'Unexpected archive growth; inspect before extraction'
        archive.write_bytes(blob)
    files=[]
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            if not member.isfile() or not member.name.endswith('.npy'):continue
            assert member.size<2_000_000
            source=stream.extractfile(member);blob=source.read();name=Path(member.name).name;target=destination/name
            if target.exists():assert target.read_bytes()==blob
            else:target.write_bytes(blob)
            assert blob[:6]==b'\x93NUMPY'
            version=blob[6:8];width=2 if version[0]==1 else 4
            size=int.from_bytes(blob[8:8+width],'little');header=ast.literal_eval(blob[8+width:8+width+size].decode().strip())
            files.append(dict(path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(blob).hexdigest(),header=header,size_bytes=len(blob)))
    assert any(Path(r['path']).name=='mg_17.npy' for r in files)
    out=ROOT/'experiments/public_benchmarks/neurobench_mg_data_manifest.json';assert not out.exists()
    out.write_text(json.dumps(dict(status='data_prepared',url=URL,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=files,
        official_loader='experiments/vendor/neurobench_2_3_0/neurobench/datasets/mackey_glass.py',
        official_loader_sha256=hashlib.sha256((ROOT/'experiments/vendor/neurobench_2_3_0/neurobench/datasets/mackey_glass.py').read_bytes()).hexdigest(),
        scope='Official downloaded bytes and NPY shape headers only; no numeric array parsing, model execution, development selection or TEST score. tau17 remains unscored on AWS.'),indent=2)+'\n')
    print('Prepared official MG series:',len(files))
