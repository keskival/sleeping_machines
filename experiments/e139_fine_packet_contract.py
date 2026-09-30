"""Bounded representation/credit audit, invoked only through run_safe.sh."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from e117_serial_event_shd import batch, load_items
from e139_fine_packet_model import (FinePacketModel, SLOTS, augment_marked,
    load_marked, marked_batch, marked_packets, warm_optimizer)
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--checkpoint", default="experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    a = p.parse_args()
    out = Path("experiments/results/e139")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError("Unique simple output tag required")
    torch.set_num_threads(1)
    torch.manual_seed(6)
    parent = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    model = FinePacketModel(parent)
    rows = load_marked(8, "fit_spk", 6)
    old = load_items(40, .01, 8, "fit_spk", 6)
    for left, right in zip(rows, old):
        for i in (0, 1, 2):
            assert np.array_equal(left[i], right[i])
        assert left[3:5] == right[3:5]
        assert np.array_equal(np.bincount(left[5], minlength=len(left[0])), left[2])
        assert np.all(left[7] <= left[1][left[5]]+1e-7)
    _, _, _, assignment, fine, _ = marked_packets(np.zeros(700), np.arange(700))
    addresses = (np.arange(700)*40//700)*SLOTS+fine
    assert len(np.unique(addresses)) == 700
    # Mark augmentation consumes exactly the legacy RNG draws and packets.
    from e122_shd_continuation import augment
    rng1, rng2 = np.random.default_rng(128), np.random.default_rng(128)
    for row, previous in zip(rows, old):
        new, baseline = augment_marked(row, rng1), augment(previous, rng2)
        for i in (0, 1, 2):
            assert np.array_equal(new[i], baseline[i])
        assert np.all(new[7] <= new[1][new[5]]+1e-7)
        assert np.array_equal(np.bincount(new[5], minlength=len(new[0])), new[2])
    model.eval()
    x = marked_batch(rows[:4])
    with torch.no_grad():
        reference, _, ref_stats, ref_traces = model.core(*batch(old[:4])[:4], 4,
            continuous=torch.zeros((len(x[0]), 32)), trace=True)
        actual, _, stats, traces = model(x, trace=True)
    assert torch.equal(reference, actual)
    assert stats["packets"] == ref_stats["packets"]
    for current, previous in zip(traces, ref_traces):
        assert torch.equal(current["winner"], previous["winner"])
        assert torch.equal(current["times"], previous["times"])
    opt = warm_optimizer(model, parent, .00007, .0003)
    model.train()
    opt.zero_grad(set_to_none=True)
    before = F.cross_entropy(model(x)[0], x[4])
    before.backward()
    gradient = model.fine.weight.grad.detach().clone()
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    # Smooth interior finite-difference check, with actual hard schedules.
    direction = gradient/gradient.norm()
    eps = 1e-5
    original = model.fine.weight.detach().clone()
    with torch.no_grad():
        model.fine.weight.copy_(original-eps*direction)
        after = F.cross_entropy(model(x)[0], x[4])
        model.fine.weight.copy_(original)
    measured = float(after-before.detach())
    predicted = -eps*float(gradient.norm())
    result = {"status": "completed", "checkpoint": a.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("experiments/e139_fine_packet_model.py"),
                Path("sleeping_machines/shared_event.py"))},
        "coarse_packets_match": True, "augmentation_packets_match": True,
        "zero_adapter_predictions_and_races_match": True,
        "distinct_original_channel_addresses": len(np.unique(addresses)),
        "causal_source_assignment": True, "new_parameters": model.fine.weight.numel(),
        "source_gradient_norm": float(gradient.norm()),
        "fit_loss": float(before.detach()), "fit_probe_loss_change": measured,
        "predicted_loss_change": predicted, "probe_step_l2": eps,
        "warm_optimizer_core_groups": len(opt.param_groups)-1,
        "work": {k: v for k, v in stats.items() if k.startswith("source_")},
        "scope": "Eight fitting utterances; no official test; no retained probe update. Nonzero source credit is not a quality gain."}
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
