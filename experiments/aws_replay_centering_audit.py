"""Guarded diagnostic: centered returns against each raw double reference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
import causal_language_precision_audit as A
from aws_replay_centered_returns import CenteredProgram


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--sources', required=True)
    args = p.parse_args()
    sources = json.loads((ROOT / args.sources).read_text())
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    out = ROOT / 'experiments/results/diagnostics' / (args.tag + '.json')
    assert not out.exists()
    torch.set_num_threads(1)
    start = time.perf_counter()
    observed = torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2])
    rows = []
    contracts = []
    for family in ('private', 'depth'):
        model = A.P.make(family)
        state = A.P.initial(model)
        double = copy.deepcopy(model).double()
        ds = A.promote(state)
        torch.manual_seed(116329)
        seed = torch.get_rng_state().clone()
        for kind, raw in [('original', A.Old), ('reuse', A.New)]:
            arms = {}
            for precision, base, entering in [('float32', model, state), ('float64', double, ds)]:
                for centered, helper in [(False, raw), (True, CenteredProgram(kind == 'reuse'))]:
                    arms[precision, centered] = A.evaluate(helper, copy.deepcopy(base),
                        observed[:16], observed[1:], copy.deepcopy(entering), seed)
                    A.C.close(torch.get_rng_state(), seed, True)
                left, right = arms[precision, False], arms[precision, True]
                A.C.close(left['logits'], right['logits'], True)
                A.C.state_close(left['state'], right['state'], True)
                A.C.close(left['activity']['factual_end_rng'], right['activity']['factual_end_rng'], True)
                assert left['activity']['shadow_lanes'] == right['activity']['shadow_lanes']
                assert left['activity']['shadow_events'] == right['activity']['shadow_events']
                assert not any(r['mismatches'] for r in A.route_mismatches(left['traces'], right['traces']))
            A.C.grads_close(arms['float64', False]['gradients'], arms['float64', True]['gradients'])
            contracts.append(f'{family}/{kind}: double all-gradient agreement; bitwise factual state/logits/RNG and all route histories retained')
            reference = arms['float64', False]['gradients']
            rows.append(dict(family=family, estimator=kind,
                raw_float32_against_raw_double=A.errors(arms['float32', False]['gradients'], reference),
                centered_float32_against_raw_double=A.errors(arms['float32', True]['gradients'], reference),
                centered_double_against_raw_double=A.errors(arms['float64', True]['gradients'], reference),
                precision_route_mismatches=A.route_mismatches(arms['float32', True]['traces'], arms['float64', True]['traces'])))
    result = dict(status='completed', contracts=contracts, rows=rows, source_sha256=sources,
        wall_s=time.perf_counter()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Diagnostic only; original float32 coordinate thresholds retained. No quality/optimizer admission; diagnostic FLOPs, traffic and energy unknown, not zero.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed', contracts_passed=len(contracts), wall_s=result['wall_s'])))


if __name__ == '__main__':
    main()
