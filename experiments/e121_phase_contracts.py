"""Guarded checks for affine phase state and its shared readout contract."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.phase_memory import PhaseMemory
from sleeping_machines.shared_event import SharedEventModel
from e121_phase_probe import contracts
from e120_shared_tasks import modular
from e120_shared_bench import inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    out = Path("experiments/results/e121")/(a.tag+".json")
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    result = {"phase_transport": contracts()}
    rng = np.random.default_rng(121)
    sequences = [rng.integers(0, 51, n) for n in (1, 2, 3, 7, 32)]
    symbols = torch.from_numpy(np.concatenate(sequences))
    ids = torch.repeat_interleave(torch.arange(len(sequences)),
                                 torch.tensor([len(x) for x in sequences]))
    for sign in (-1, 1):
        phase = PhaseMemory(51, 17, 17, reflection=sign)
        score, state = phase(symbols, ids, len(sequences))
        for i, seq in enumerate(sequences):
            pred, ref = phase.serial(seq)
            assert abs((float(state[i])-ref+8.5)%17-8.5) < 1e-10
            assert pred == int(score[i].argmax())
        # Integer chart shifts must preserve every clock race.
        phase.phase.add_(17)
        shifted, _ = phase(symbols, ids, len(sequences))
        assert torch.allclose(score, shifted, atol=1e-10, rtol=0)
    phase = PhaseMemory(51, 17, 17, sigma=0)
    repeated = np.array([2, 2, 2])  # signed occurrence sum = +1
    pred, phi = phase.serial(repeated)
    label = (pred+1)%17
    before = phase.phase.clone()
    gap = (float(phase.centers[label])-phi-.5+8.5)%17-8.5
    phase.teach(repeated, label, rng)
    assert abs(float(phase.phase[2]-before[2])-.05*np.sign(gap)) < 1e-12
    assert torch.equal(phase.phase[:2], before[:2])
    assert torch.equal(phase.phase[3:], before[3:])
    result["batch_chart_repeated_credit"] = "passed"
    task = modular(16, 32, 6)
    torch.manual_seed(6)
    config = {**task.config, "depth": 2, "phase_period": 17}
    model = SharedEventModel(**config).eval()
    packed = inputs(task.dev)
    for bound in (0., .25, 2.):
        model.phase_correction_bound = bound
        for strength in (1., 100.):
            with torch.no_grad():
                model.head.weight.normal_(std=strength)
                model.head.bias.normal_(std=strength)
                combined, _, stats, _ = model(**packed)
                pure = model(**packed, expert_mode="memory")[0]
            assert (combined-pure).abs().max() <= bound+1e-5
            certified = np.array(stats["phase_margin"]) > 2*bound
            changed = combined.argmax(-1).numpy() != pure.argmax(-1).numpy()
            assert not np.any(certified & changed)
    model.phase_correction_bound = .25
    model.phase_margin_guard = True
    with torch.no_grad():
        combined, _, stats, _ = model(**packed)
        pure = model(**packed, expert_mode="memory")[0]
    assert torch.equal(combined.argmax(-1), pure.argmax(-1))
    assert stats["phase_certified"] == len(task.dev)
    model.phase_margin_guard = False
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer); buffer.seek(0)
    loaded = SharedEventModel(**config).eval()
    loaded.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(model(**packed)[0], loaded(**packed)[0])
    result.update(status="completed", margin_certificate="passed",
                  adaptive_margin_guard="passed",
                  checkpoint_roundtrip="exact", phase_parameters=69,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), Path("sleeping_machines/phase_memory.py"),
                                   Path("sleeping_machines/shared_event.py"))})
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
