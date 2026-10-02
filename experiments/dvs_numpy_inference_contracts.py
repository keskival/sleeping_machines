"""Validate the numeric inference port against frozen real native checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as N
from sleeping_machines.numpy_addressed_inference import NumpyAddressedInference, pack


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def check(model, rows, tight):
    port = NumpyAddressedInference(pack(model)); maximum = 0.; arrival_error = 0.
    tolerance = 1e-10 if tight else 5e-4
    for row in rows:
        choices = []; original_race = model.race
        def traced(scores, values=None):
            value, delay, winner = original_race(scores, values)
            choices.append(int(winner)); return value, delay, winner
        model.race = traced
        try:
            reference, reference_state = N.predict(model, row, 314159, False)
        finally:
            model.race = original_race
        result, state = port.predict(row['events'])
        if choices != state['winners']: raise ValueError('Different hard route choices')
        error = float(np.max(np.abs(result-reference.numpy())))
        maximum = max(maximum, error)
        np.testing.assert_allclose(result, reference.numpy(), rtol=tolerance, atol=tolerance)
        for address, value in reference_state.memories.items():
            depth, head, source, unit = address
            assert source == 0 and state['seen'][depth, head, unit]
            np.testing.assert_allclose(state['memories'][depth, head, unit], value.numpy(), rtol=tolerance, atol=tolerance)
            gap = abs(float(reference_state.arrivals[address])-state['arrivals'][depth, head, unit])
            arrival_error = max(arrival_error, gap)
        context, times = reference_state.contexts[0]
        np.testing.assert_allclose(state['context'].reshape(-1), context.numpy(), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(state['context_times'], times.numpy(), rtol=tolerance, atol=tolerance)
        assert state['events'] == reference_state.events
        assert state['noise_cursor'] == state['events']*model.depth*model.heads
    return dict(targets=len(rows),maximum_logit_absolute_error=maximum,
        maximum_memory_arrival_absolute_error=arrival_error,all_hard_route_choices_equal=True,
        context_and_selected_memory_contracts_passed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--native', action='append', required=True)
    a = p.parse_args(); start = time.perf_counter(); torch.set_num_threads(1)
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    rows = []; sources = {}
    for name in a.native:
        path = ROOT/name; parent = json.loads(path.read_text())
        if parent['status'] != 'completed': raise ValueError('Completed checkpoint required')
        for source, digest in parent['source_sha256'].items():
            if sha(ROOT/source) != digest: raise ValueError('Changed native source')
        sources.update(parent['source_sha256']); args = argparse.Namespace(**parent['args'])
        fit, dev, info = N.load(args)
        if info != parent['data']: raise ValueError('Changed data')
        cp = path.with_suffix('.progress.pt'); ck = torch.load(cp, weights_only=False)
        if ck['source_sha256'] != parent['source_sha256'] or ck['data'] != parent['data']:
            raise ValueError('Changed checkpoint')
        model = N.make_model(args); model.load_state_dict(ck['best_state']); model.eval()
        ordinary = check(model, dev, False)
        original = NumpyAddressedInference(pack(model)); first, _ = original.predict(dev[0]['events'])
        again, _ = original.predict(dev[0]['events']); np.testing.assert_array_equal(first, again)
        changed = dict(dev[0], target=-99, identity='different', index=-77)
        alternate, _ = original.predict(changed['events']); np.testing.assert_array_equal(first, alternate)
        state = original.new_state(); before = []
        for timestamp, content in dev[0]['events']: before.append(original.consume(timestamp,content,state)[0])
        later = [(t, c.copy()) for t,c in dev[0]['events']]; later[-1][1][0] += 7
        state = original.new_state(); after = []
        for timestamp, content in later: after.append(original.consume(timestamp,content,state)[0])
        np.testing.assert_array_equal(np.array(before[:-1]),np.array(after[:-1]))
        # High precision tests the mathematical port independently of payload rounding.
        precise = check(model.double(), fit[:4]+dev[:4], True)
        rows.append(dict(native=name,native_result_sha256=sha(path),native_checkpoint_sha256=sha(cp),
            float32_full_development=ordinary,float64_prefixes=precise,
            repeat_metadata_and_future_content_contracts_passed=True))
    sources.update({n:sha(ROOT/n) for n in ['sleeping_machines/numpy_addressed_inference.py','experiments/dvs_numpy_inference_contracts.py']})
    result=dict(status='completed',args=vars(a),rows=rows,contracts_passed=4*len(rows),
        source_sha256=sources,wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Inference-only one-source port: frozen full-development routes/state/logits plus double precision, repeat/reset, metadata and future-content contracts. Float32 finite tolerance is numerical reproduction on these clips, not universal bitwise equivalence near route ties. Static transformed parameters and target-independent reference draw cache are paid setup/storage; training and counterfactual credit remain unchanged. No fitting, test or advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':main()
