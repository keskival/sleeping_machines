"""Pin completed E118 depth/credit comparisons and verify their protocols."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).parent/"results"/"e118"
    output = root/"matched_summary_20260929.json"
    if output.exists():
        raise FileExistsError(output)
    pairs = (("credit_d8_n128", "race_d8_cf0_n128_e8_s6", "race_d8_cf1_n128_e8_s6", {"tag","credit"}),
             ("depth_cf0_n128", "race_d1_cf0_n128_e8_s6", "race_d8_cf0_n128_e8_s6", {"tag","depth"}),
             ("depth_cf1_n128", "race_d1_cf1_n128_e8_s6", "race_d8_cf1_n128_e8_s6", {"tag","depth"}),
             ("depth_cf1_n512", "race_d1_cf1_n512_e4_s6", "race_d8_cf1_n512_e4_s6", {"tag","depth"}))
    result = {"scope":"Single-seed held-out-speaker development comparisons, not official SHD test",
              "pairs":{}}
    for name,left,right,allowed in pairs:
        paths = [root/(tag+'.json') for tag in (left,right)]
        a,b = [json.loads(p.read_text()) for p in paths]
        assert a['status'] == b['status'] == 'completed'
        for key in ('fit_ids','dev_ids','fit_labels','dev_labels'):
            assert a[key] == b[key], (name,key)
        assert {k:v for k,v in a['args'].items() if k not in allowed} == {
            k:v for k,v in b['args'].items() if k not in allowed},name
        if name.startswith('credit_'):
            assert a['initial'] == b['initial'] and a['readout_calibration'] == b['readout_calibration']
        rows = []
        for source,record in zip(paths,(a,b)):
            final = record['curve'][-1]
            rows.append({"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                         "args":record['args'],"parameters":record['parameters'],
                         "fit_correct":final['fit']['correct'],"fit_n":final['fit']['n'],
                         "dev_correct":final['dev']['correct'],"dev_n":final['dev']['n'],
                         "fit_nll":final['fit']['nll'],"dev_nll":final['dev']['nll'],
                         "wall_s":record['wall_s'],"max_rss_kb":record['max_rss_kb']})
        p,q = a['curve'][-1]['dev']['predictions'],b['curve'][-1]['dev']['predictions']
        y = a['dev_labels']
        result['pairs'][name] = {"arms":rows,
            "right_only_correct":sum(j==t and i!=t for i,j,t in zip(p,q,y)),
            "left_only_correct":sum(i==t and j!=t for i,j,t in zip(p,q,y))}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
