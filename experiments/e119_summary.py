"""Read-only aggregation of completed E119 evidence, with input hashes."""
import hashlib
import json
from pathlib import Path
import platform


def main():
    files = {
        'earlier':Path('experiments/results/e118/race_d8_cf1_n512_e4_s6.json'),
        'training':Path('experiments/results/e119/race_d8_linear_n1024_e8_s6.json'),
        'scan':Path('experiments/results/e119/scan_audit_s6.json'),
        'packets':Path('experiments/results/e119/packet_pareto_d8_n1024_s6.json')}
    out = Path('experiments/results/e119/summary_s6.json')
    if out.exists(): raise FileExistsError(out)
    r = {k:json.loads(p.read_text()) for k,p in files.items()}
    assert all(v['status']=='completed' for v in r.values())
    old, new, audit, packet = [r[k] for k in files]
    assert old['dev_ids'] == new['dev_ids'] == packet['dev_ids']
    assert old['dev_labels'] == new['dev_labels'] == packet['dev_labels']
    assert old['fit_ids'] == new['fit_ids'][:len(old['fit_ids'])]
    assert old['parameters'] == new['parameters'] == 53296
    assert len(new['curve']) == new['args']['epochs'] == 8
    final = new['curve'][-1]
    assert final['dev']['predictions'] == packet['rows']['1']['predictions']
    assert audit['dev']['doubling']['predictions'] == audit['dev']['linear']['predictions'] == old['curve'][-1]['dev']['predictions']
    labels = new['dev_labels']
    pp, qq = old['curve'][-1]['dev']['predictions'], final['dev']['predictions']
    result = {'status':'completed','input_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},
              'data_checks':'same development IDs/labels; earlier fitting subset is prefix; final checkpoint packet audit agrees',
              'earlier_dev':{k:old['curve'][-1]['dev'][k] for k in ('correct','n','accuracy','nll')},
              'final_dev':{k:final['dev'][k] for k in ('correct','n','accuracy','nll')},
              'final_fit':{k:final['fit'][k] for k in ('correct','n','accuracy','nll')},
              'paired_corrected':sum(p!=y and q==y for p,q,y in zip(pp,qq,labels)),
              'paired_lost':sum(p==y and q!=y for p,q,y in zip(pp,qq,labels)),
              'comparison_scope':'same architecture and seed; more fitting data/updates and changed schedule/fit-only calibration values; not a single-factor causal comparison',
              'scan_combine_reduction':audit['dev']['doubling']['scan_compositions']/audit['dev']['linear']['scan_compositions'],
              'training_forward_backward_speedup':audit['median_timings']['doubling']['training_step_s']/audit['median_timings']['linear']['training_step_s'],
              'inference_speedup':audit['median_timings']['doubling']['inference_s']/audit['median_timings']['linear']['inference_s'],
              'training_wall_s':new['wall_s'],'training_max_rss_kb':new['max_rss_kb'],
              'hardware':{'platform':platform.platform(),'cpu_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')},
              'energy_joules':None,'official_test_access':False}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
