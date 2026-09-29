"""Summarize the matched marked-input screen and check its routing contract."""
import hashlib
import json
from pathlib import Path

import torch
from e74_time_vector_net import TVLayer
from e83_deep_shd import DeepSHD

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'e83'
PREFIX = 'deep_d8_n4_M16-16_depth4_aux0.2_objevent_prefix_rfdeepest'
SUFFIX = '_recongrid_cfnorm0_b0.5_sg0.25_w1_dl5_lr0.001_gc1_spk_s6.json'
FILES = {
    'off': PREFIX + '_rngsplit_cntmark_control' + SUFFIX,
    'additive': PREFIX + '_cntadd_rngsplit_cntmark_additive' + SUFFIX,
    'address_neutral': PREFIX + '_cntaddr_rngsplit_cntmark_address_neutral' + SUFFIX,
}


def contract():
    torch.set_num_threads(1)
    torch.manual_seed(813)
    layer = TVLayer(2, 1, 2, 2, 1, 2, spiking=False)
    with torch.no_grad():
        layer.q.copy_(torch.tensor([[1., 0.]])); layer.c.zero_()
        layer.Bre.fill_(1); layer.Bim.zero_(); layer.wre.fill_(1); layer.wim.zero_()
        layer.freq.zero_(); layer.log_td.zero_()
    eb = torch.tensor([0, 0]); ei = torch.tensor([0, 1]); et = torch.tensor([1., 2.])
    key = torch.tensor([[1., 0.], [-1., 0.]])
    changed = torch.tensor([[-10., 2.], [10., 3.]], requires_grad=True)
    base, _, rb = layer(eb, ei, et, key, 1, 12, return_routes=True)
    explicit, _, _ = layer(eb, ei, et, key, 1, 12, route_ev=key, return_routes=True)
    val, _, rv = layer(eb, ei, et, changed, 1, 12, route_ev=key, return_routes=True)
    _, _, coupled = layer(eb, ei, et, changed, 1, 12, return_routes=True)
    assert torch.equal(base, explicit), 'default routing changed'
    assert torch.equal(rb['active'], rv['active']), 'mark altered neutral route mask'
    assert torch.equal(rb['score'], rv['score']), 'mark altered neutral delays'
    assert not torch.equal(rb['active'], coupled['active']), 'witness did not cross a routing boundary'
    assert not torch.equal(base, val), 'value path lost the mark'
    val.sum().backward()
    assert float(changed.grad.norm()) > 0, 'value path lost credit'

    net = DeepSHD(2, 2, 1, 2, 2, 1, 4, .5, .5, 2, [.05, .05],
                  event_readout=True, input_count_payload=True,
                  input_count_route_neutral=True, spike_reconstruction='grid')
    with torch.no_grad(): net.count_proj.weight.fill_(.25)
    _, info = net(eb, ei, et, 1, 12, return_taps=True, collect_routes=True,
                  input_counts=torch.tensor([1., 2.]))
    _, units, routing_input = info['route_inputs'][0]
    rc = info['route_candidates'][0]; pe = rc['event_index']; pj = rc['receiver']
    score = (net.layers[0].q[pj] * routing_input[pe]).sum(-1) + net.layers[0].c[units[pe], pj]
    assert torch.equal(score.detach(), rc['score']), 'boundary credit used the wrong input vector'
    return {'default_matches_explicit_key': True, 'neutral_mask_and_delay_invariant': True,
            'coupled_mask_changes': True, 'payload_changes_and_has_gradient': True,
            'counterfactual_score_reconstruction_matches': True}


def main():
    records = {}
    baseline_args = None
    baseline_labels = None
    for name, filename in FILES.items():
        path = RESULTS / filename; data = json.loads(path.read_text())
        args = {k: v for k, v in data['args'].items() if k not in ('run_tag', 'input_count_payload')}
        if baseline_args is None: baseline_args = args
        assert args == baseline_args, f'configuration mismatch in {name}'
        outputs = data['eval_output_payloads'][-1]
        labels = [row['true_class'] for row in outputs]
        if baseline_labels is None: baseline_labels = labels
        assert labels == baseline_labels, 'evaluation label order mismatch'
        final = data['curve'][-1]
        correct = sum(row['true_class'] == row['predicted_class'] for row in outputs)
        records[name] = {'path': str(path.relative_to(ROOT.parent)),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'curve': data['curve'], 'anytime_correct': correct,
                         'terminal_correct': round(final['event_terminal_accuracy'] * len(outputs)),
                         'evaluation_examples': len(outputs)}
    out = {'scope': 'one seed; 512 fitting / 256 held-out-speaker development utterances; two epochs',
           'contract': contract(), 'arms': records}
    target = RESULTS / 'countmark_matched_20260929.json'
    target.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'contract': out['contract'], 'summary': str(target)}))


if __name__ == '__main__':
    main()
