"""Read-only E83 spike-margin audit conditioned on actual delivered messages."""
import argparse, json, math, os
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S
from e71_event_cde import events
from e83_deep_shd import DeepSHD, batch_to_events, stratified_limit

torch.set_num_threads(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--examples', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--output', required=True)
    acli = ap.parse_args()
    ckpt = torch.load(acli.checkpoint, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ckpt['args'])
    torch.manual_seed(int(getattr(a, 'seed', acli.seed)))
    eval_rng = np.random.default_rng(int(getattr(a, 'seed', acli.seed)) + 300_007)
    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances('train', a.bands, 'val_spk') if len(t) > 1]
    examples = stratified_limit(data, acli.examples, eval_rng)
    widths_sd = [float(v) for v in str(a.w_sd).split(',')]
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, widths_sd, a.seed, event_readout=True,
                  readout_fusion=a.readout_fusion,
                  input_count_payload=a.input_count_payload == 'additive',
                  early_event_skip=bool(getattr(a, 'early_event_skip', False)),
                  route_topk=int(getattr(a, 'route_topk', 0)))
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    # Bands are disjoint intervals in signed margin m=V-theta.
    bins = [(-0.25, 0.0), (-0.5, -0.25), (-1.0, -0.5), (-2.0, -1.0), (-float('inf'), -2.0)]
    names = ['[-.25,0)', '[-.5,-.25)', '[-1,-.5)', '[-2,-1)', '(-inf,-2)']
    quantile_rng = np.random.default_rng(acli.seed + 19)
    stats = [{'valid_cells': 0, 'nonrefractory_cells': 0, 'cells_after_input': 0,
              'candidate_counts': [0] * len(bins), 'examples_with_candidate': [0] * len(bins),
              'min_margin': None, 'quantile_sample': []} for _ in range(a.depth)]
    example_band_support = [np.zeros((len(examples), len(bins)), dtype=bool)
                            for _ in range(a.depth)]
    with torch.no_grad():
        for start in range(0, len(examples), acli.batch_size):
            items = examples[start:start + acli.batch_size]
            eb, ei, et, input_counts, labels, tmax, seq_end = batch_to_events(
                items, a.bands, 0, np.random.default_rng(0), 0.0)
            B = len(items)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            _, info = net(eb, ei, et, B, grid, return_taps=True, collect_routes=True,
                          return_spike_diagnostics=True, input_counts=input_counts)
            for k, diag in enumerate(info['spike_diagnostics']):
                margins = diag['spike_margin_trace']
                fired = diag['spike_fire_mask']
                refractory = diag['spike_refractory_trace']
                G, batch_n, M = margins.shape
                earliest = torch.full((B, M), float('inf'))
                route = info['route_candidates'][k]
                active = route['active'].cpu().numpy().astype(bool)
                if active.any():
                    rec = route['receiver'].cpu().numpy()[active]
                    eid = route['event_index'].cpu().numpy()[active]
                    src_b = route['source_batch'].cpu().numpy()[active]
                    src_t = route['source_time'].cpu().numpy()[active]
                    scores = route['score'].cpu().numpy()[active]
                    src_units = info['route_inputs'][k][1].cpu().numpy()
                    layer = net.layers[k]
                    rec_t = torch.as_tensor(rec, dtype=torch.long)
                    if layer.cdelay:
                        delay_score = torch.as_tensor(scores, dtype=layer.log_rate.dtype)
                    else:
                        source_units = torch.as_tensor(src_units[eid], dtype=torch.long)
                        delay_score = F.softplus(layer.c[source_units, rec_t]).detach()
                    delays = (torch.exp(layer.log_td.detach()[rec_t])
                              * delay_score.clamp(min=0)).clamp(max=layer.dmax).cpu().numpy()
                    arrival_grid = np.ceil(src_t + delays).astype(np.int64)
                    for b, j, arrival in zip(src_b, rec, arrival_grid):
                        if 0 <= arrival < G and arrival < float(earliest[int(b), int(j)]):
                            earliest[int(b), int(j)] = float(arrival)
                time = torch.arange(G, dtype=seq_end.dtype)[:, None, None]
                max_causal_time = (seq_end + (k + 1) * float(a.dmax))[None, :, None]
                valid = (time > 0) & (time <= max_causal_time)
                nonref = valid & (refractory < 1e-3) & ~fired
                arrived = torch.as_tensor(earliest)[None, :, :] <= time
                conditioned = nonref & arrived
                st = stats[k]
                st['valid_cells'] += int(valid.expand_as(margins).sum())
                st['nonrefractory_cells'] += int(nonref.sum())
                st['cells_after_input'] += int(conditioned.sum())
                vals = margins[conditioned].cpu().numpy()
                if len(vals):
                    st['min_margin'] = float(np.min(vals)) if st['min_margin'] is None else min(st['min_margin'], float(np.min(vals)))
                    take = min(1024, len(vals))
                    sample_idx = quantile_rng.choice(len(vals), size=take, replace=False)
                    st['quantile_sample'].extend(vals[sample_idx].tolist())
                for bi, (lo, hi) in enumerate(bins):
                    mask = conditioned & (margins >= lo) & (margins < hi)
                    st['candidate_counts'][bi] += int(mask.sum())
                    by_item = mask.any(dim=(0, 2)).cpu().numpy()
                    st['examples_with_candidate'][bi] += int(by_item.sum())
                    example_band_support[k][start:start + len(items), bi] |= by_item
    result = {
        'checkpoint': os.path.abspath(acli.checkpoint),
        'split': 'train/val_spk', 'examples': len(examples), 'batch_size': acli.batch_size,
        'conditioning': 'nonfiring, nonrefractory at a receiver with at least one actual selected upstream message already arrived; candidate time limited to seq_end + (layer+1)*dmax',
        'margin_definition': 'signed pre-reset voltage margin m=Vd-theta from actual frozen forward pass',
        'bins': names, 'per_layer': []}
    for k, st in enumerate(stats):
        st['examples_with_candidate_within_0p25'] = int(example_band_support[k][:, 0].sum())
        st['examples_with_candidate_within_0p5'] = int(example_band_support[k][:, :2].any(axis=1).sum())
        qs = np.quantile(np.asarray(st['quantile_sample']), [0, .1, .5, .9, .99, 1]).tolist() if st['quantile_sample'] else []
        result['per_layer'].append({'layer': k + 1, **st, 'candidate_margin_quantiles_min_p10_p50_p90_p99_max': qs, 'quantile_sample': None})
    with open(acli.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'output': acli.output, 'examples': len(examples), 'layers': [
        {'layer': x['layer'], 'valid_cells': x['valid_cells'], 'cells_after_input': x['cells_after_input'],
         'candidates_by_band': x['candidate_counts'], 'examples_by_band': x['examples_with_candidate'],
         'margin_quantiles': x['candidate_margin_quantiles_min_p10_p50_p90_p99_max']}
        for x in result['per_layer']]}), flush=True)

if __name__ == '__main__':
    main()
