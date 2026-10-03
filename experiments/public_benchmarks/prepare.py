"""Record archived public data, fixed splits and exact contracts without fitting."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.public_benchmarks.data import EXPECTED,load_train,split,sha,contracts

if __name__=='__main__':
    records=[]
    for name in EXPECTED:
        rows,mapping,meta=load_train(name);fit,dev=split(rows)
        records.append(dict(dataset=name,train_rows=len(rows),expected_official_test_rows=EXPECTED[name][1],
            dimensions=EXPECTED[name][2],classes=len(mapping),labels=mapping,
            min_length=min(len(r['values']) for r in rows),max_length=max(len(r['values']) for r in rows),
            fit_indices=fit,dev_indices=dev,split_seed=20261004,
            metadata=meta,train_sha256=sha(ROOT/'data/public_benchmarks/raw'/f'{name}_TRAIN.ts'),
            test_sha256=sha(ROOT/'data/public_benchmarks/raw'/f'{name}_TEST.ts'),
            test_access='Bytes hashed only; no test parsing, model scoring or configuration selection.'))
    out=ROOT/'experiments/public_benchmarks/data_manifest.json'
    assert not out.exists();out.write_text(json.dumps(dict(status='prepared',contract=contracts(),datasets=records),indent=2)+'\n')
    print([(r['dataset'],r['train_rows'],len(r['fit_indices']),len(r['dev_indices'])) for r in records])
