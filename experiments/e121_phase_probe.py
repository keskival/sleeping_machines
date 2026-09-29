"""Test the generic phase primitive on the existing E120 modular split."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.phase_memory import PhaseMemory
from e120_shared_tasks import modular


def evaluate(model, rows):
    symbols = torch.from_numpy(np.concatenate([r.prefix.channels for r in rows]))
    ids = torch.repeat_interleave(torch.arange(len(rows)), 3)
    scores, phase = model(symbols, ids, len(rows))
    pred = scores.argmax(-1).tolist()
    y = torch.tensor([r.label for r in rows])
    return {"n": len(rows), "correct": sum(p == r.label for p,r in zip(pred,rows)),
            "accuracy": sum(p == r.label for p,r in zip(pred,rows))/len(rows),
            "nll": float(torch.nn.functional.cross_entropy(scores, y)), "predictions": pred}


def contracts():
    rng = np.random.default_rng(121)
    net = PhaseMemory(51, 17, 17.)
    max_error = 0.
    for n in (1, 2, 3, 8, 64):
        symbols = rng.integers(0, 51, n)
        pred, phi = net.serial(symbols)
        score, state = net(torch.from_numpy(symbols), torch.zeros(n, dtype=torch.long), 1)
        delta = abs((float(state[0])-phi+8.5)%17-8.5)
        max_error = max(max_error, delta)
        assert delta < 1e-10 and pred == int(score.argmax(-1)[0])
    # Every phase coordinate has unit local derivative, up to reflection.
    symbols = torch.tensor([0, 20, 40])
    ids = torch.zeros(3, dtype=torch.long)
    phase = net(symbols, ids, 1)[1]
    eps = 1e-6
    derivatives = []
    for symbol in symbols:
        net.phase[symbol] += eps
        shifted = net(symbols, ids, 1)[1]
        net.phase[symbol] -= eps
        derivatives.append(float(((shifted-phase+8.5)%17-8.5)/eps))
    assert np.allclose(derivatives, [1., -1., 1.], atol=1e-7)
    return {"serial_scan_phase_max_error": max_error, "phase_derivatives": derivatives}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=6)
    a = p.parse_args()
    out = Path("experiments/results/e121")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(1)
    checks = contracts()
    task = modular(1473, 256, a.seed)
    model = PhaseMemory(51, 17, 17, seed=a.seed)
    rng = np.random.default_rng(a.seed+10)
    started = time.perf_counter()
    result = {"status": "running", "args": vars(a), "contracts": checks,
              "protocol": task.protocol, "fit_ids": [r.identity for r in task.fit],
              "dev_ids": [r.identity for r in task.dev], "curve": [],
              "initial": {"fit": evaluate(model,task.fit),"dev": evaluate(model,task.dev)}}
    for epoch in range(1, a.epochs+1):
        mistakes = 0
        for i in rng.permutation(len(task.fit)):
            row = task.fit[i]
            mistakes += model.teach(row.prefix.channels, row.label, rng)
        if epoch == 1 or epoch%10 == 0 or epoch == a.epochs:
            row = {"epoch": epoch, "mistakes": mistakes,
                   "fit": evaluate(model,task.fit), "dev": evaluate(model,task.dev)}
            result["curve"].append(row)
            print(json.dumps({"epoch":epoch,"fit":row["fit"]["accuracy"],"dev":row["dev"]["accuracy"]}),flush=True)
    result.update(status="completed", final=result["curve"][-1], wall_s=time.perf_counter()-started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  local_phase_updates=int(model.updates),phase_parameters=69,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__),Path("sleeping_machines/phase_memory.py"))})
    out.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"state_dict":model.state_dict(),"args":vars(a),"rng":rng.bit_generator.state},out.with_suffix(".pt"))


if __name__ == "__main__": main()
