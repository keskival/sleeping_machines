"""E120 shared-core regression and causal/objective contracts; guarded runner only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_query import ObservedPrefix, pack_queries
from sleeping_machines.objectives import hazard_nll
from sleeping_machines.evidence_memory import RelativeRouteMemory
from e118_race_carrier_shd import contracts, evaluate
from e119_scan_audit import contracts as scan_contracts
from e117_serial_event_shd import batch, load_items
from e61_race_attention import RaceAttention, sample, make_perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = Path("experiments/results/e120")/(args.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.manual_seed(6)
    net = SharedEventModel()
    result = {"scan": scan_contracts(), "race": contracts(net)}
    # Compare against the committed, pre-extraction implementation, not a
    # second import alias of the new class. Pin the revision in the output.
    revision = "de6a45a"
    source = Path("experiments/reference/e118_pre_shared.py").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "29823725167d69de7ac990cd32c56228276fc61cfaed48864a9c1a24fc9fe909"
    legacy = types.ModuleType("e118_committed_reference")
    legacy.__file__ = str(Path("experiments/e118_race_carrier_shd.py").resolve())
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    checkpoint = Path("experiments/results/e119/race_d8_linear_n1024_e8_s6.pt")
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    old = legacy.RaceNet(memory_backend="linear")
    for model in (old, net):
        model.load_state_dict(saved["state_dict"], strict=True)
    dev = load_items(40, .01, 256, "val_spk", 7)
    inputs = batch(dev[:4])
    outputs, grads, winners = [], [], []
    for model in (old, net):
        model.train()
        model.zero_grad(set_to_none=True)
        z, _, _, trace = model(*inputs[:4], 4, trace=True)
        torch.nn.functional.cross_entropy(z, inputs[-1]).backward()
        outputs.append(z.detach())
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        winners.append(torch.stack([t["winner"] for t in trace]))
    result["extraction"] = {"reference_revision": revision,
        "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "logit_max_error": float((outputs[0]-outputs[1]).abs().max()),
        "gradient_max_error": float((grads[0]-grads[1]).abs().max()),
        "winner_disagreements": int((winners[0] != winners[1]).sum())}
    assert result["extraction"]["logit_max_error"] == 0
    assert result["extraction"]["gradient_max_error"] == 0
    assert result["extraction"]["winner_disagreements"] == 0
    result["shd_dev"] = evaluate(net, dev, 4)
    assert result["shd_dev"]["correct"] == 151
    result["shd_note"] = "Existing frozen E119 checkpoint; train speakers 3/6 held out; no new training or official test"
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prefix = ObservedPrefix(np.array([0, 1, 2]), np.array([.01, .02, .03]), np.ones(3), .03)
    packed = pack_queries([prefix])
    net.eval()
    with torch.no_grad():
        one = net(**packed)[0]
        two = net(**pack_queries([prefix, prefix]))[0]
    err = float((two-one).abs().max())
    assert err < 1e-5
    try:
        pack_queries([ObservedPrefix(prefix.channels, prefix.times, prefix.counts, .02)])
    except ValueError:
        pass
    else:
        raise AssertionError("Future event was accepted")
    # Exact hazard gradient is exposure*intensity minus the observed count.
    z = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    exposure = torch.tensor([[.01, .04, .3], [.01, .02, 0]], dtype=torch.float64)
    bucket, labels = torch.tensor([2, 1]), torch.tensor([1, 2])
    actual, = torch.autograd.grad(hazard_nll(z, labels, bucket, exposure, "sum"), z)
    expected = exposure[:, :, None]*z.detach().exp()
    expected[torch.arange(2), bucket, labels] -= 1
    assert torch.equal(actual, expected)
    result["query_objective"] = {"batch_separability_max_error": err,
                                 "future_event_rejected": True, "hazard_gradient_max_error": 0.}
    # E61's pointer update is preserved by the extracted generic primitive.
    rng = np.random.default_rng(6)
    perm = make_perm(32, rng)
    before, after = RaceAttention(32, 1., .5), RelativeRouteMemory(32, 64)
    for _ in range(256):
        seq, q, y = sample(32, 8, perm, rng)
        assert before.forward(seq, q)[0] == after.read(seq, q)[0]-32
        before.teach(seq, q, y)
        after.observe(seq, q, y+32)
    assert np.array_equal(before.Z, after.weights)
    result["pointer_extraction_exact"] = True
    for settings in ({"groups": 0}, {"bands": 0}, {"classes": 0}, {"evidence_count": -1}):
        try:
            SharedEventModel(**settings)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid dimensions accepted")
    result["invalid_dimensions_rejected"] = True
    result["status"] = "completed"
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), *Path("sleeping_machines").glob("*.py")]}
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("scan", "shd_dev", "source_sha256")}))
    print("SHD preserved:", result["shd_dev"]["correct"], "/", result["shd_dev"]["n"])


if __name__ == "__main__":
    main()
