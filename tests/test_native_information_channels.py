"""Read-only contracts for distinguishing key, value and context memory."""
from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from native_event_contracts import predict
from native_event_tasks import episodes
from native_information_channels import CHANNELS, predict_channels
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def test_information_probe_preserves_original_and_matches_full_clear():
    torch.manual_seed(164)
    model = AddressedEventHeads(payload=8, depth=2).eval()
    row = episodes('order',4,4,17)[0]
    noise = torch.get_rng_state()
    with torch.no_grad():
        expected, _, _, _ = predict(model,row)
        torch.set_rng_state(noise)
        actual, _, _ = predict_channels(model,row,'full')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.set_rng_state(noise)
        expected, _, _, _ = predict(model,row,clear=True)
        torch.set_rng_state(noise)
        actual, _, _ = predict_channels(model,row,'no_history')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_no_history_removes_past_signs_and_labels_do_not_activate_channels():
    torch.manual_seed(165)
    model = AddressedEventHeads(payload=8, depth=2).eval()
    row = episodes('order',4,4,18)[0]
    different = [replace(e,mark=(-e.mark[0],e.mark[1])) if not e.mark[1] else e for e in row]
    labels = [replace(e,target=(e.target+1)%4) if e.target is not None else e for e in row]
    noise = torch.get_rng_state()
    a, _, _ = predict_channels(model,row,'no_history')
    torch.set_rng_state(noise)
    b, _, _ = predict_channels(model,different,'no_history')
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for condition in CHANNELS:
        torch.set_rng_state(noise)
        a, _, _ = predict_channels(model,row,condition)
        torch.set_rng_state(noise)
        b, _, _ = predict_channels(model,labels,condition)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_information_hooks_and_parameters_restore_even_on_error():
    torch.manual_seed(166)
    model = AddressedEventHeads(payload=8, depth=2).eval()
    before = {name:p.clone() for name,p in model.state_dict().items()}
    row = episodes('order',4,4,19)[0]
    broken = [replace(e,target=None) if e.mark[1] else e for e in row]
    try:
        predict_channels(model,broken,'keys_only')
    except ValueError as error:
        assert 'external target' in str(error)
    else:
        raise AssertionError('Missing scored target should fail')
    for unit in model.modules():
        if hasattr(unit,'key_read'):
            assert 'propose' not in unit.__dict__ and not unit.key_read._forward_hooks
    for name,p in model.state_dict().items():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
