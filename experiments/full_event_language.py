"""Resumable full 10M-character benchmark of the learned persistent event model.

No count/copy/word experts. Fixed-budget development selection; cold-context
official test scored once. Run through a unique one-job guarded queue only.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from e120_shared_tasks import text_slice
from sleeping_machines.stream_language import StreamingEventLanguageModel


@torch.no_grad()
def evaluate(model, tokens):
    model.eval()
    state, total = model.new_state(), 0.
    for i in range(len(tokens) - 1):
        logits = model.consume(tokens[i], state)
        total += float(F.cross_entropy(logits[None], tokens[i + 1:i + 2]))
    return dict(n=len(tokens) - 1, bpc=total / (len(tokens) - 1) / math.log(2),
                event_deliveries=state.deliveries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fit", type=int, default=10_000_000)
    parser.add_argument("--dev", type=int, default=200_000)
    parser.add_argument("--test", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract", action="store_true", help="Use development data for final pipeline check; no official test")
    args = parser.parse_args()
    out = ROOT / "experiments/results/full_event" / (args.tag + ".json")
    running = out.with_suffix(".running.json")
    checkpoint = out.with_suffix(".progress.pt")
    if Path(args.tag).name != args.tag or min(args.fit, args.dev, args.test) < 2 or min(args.epochs, args.chunk) < 1:
        raise ValueError("Positive data/fit settings and a plain tag required")
    if args.fit > 90_000_000 or args.dev > 5_000_000 or args.test > 5_000_000:
        raise ValueError("Reserved data split exceeded")
    if out.exists() and json.loads(out.read_text()).get("status") == "completed":
        raise ValueError("Completed results must never be rerun or overwritten")
    if (out.exists() or running.exists()) and not (args.resume and checkpoint.exists()):
        raise ValueError("Existing run requires its resumable checkpoint")
    out.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train = torch.tensor(text_slice(0, args.fit))
    dev = torch.tensor(text_slice(90_000_000, args.dev))
    model = StreamingEventLanguageModel(width=args.width, modes=args.modes, depth=args.depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    sources = [Path(__file__), ROOT / "sleeping_machines/stream_language.py", ROOT / "sleeping_machines/event_state.py",
               ROOT / "sleeping_machines/rotating_memory.py", ROOT / "sleeping_machines/event_memory.py",
               ROOT / "experiments/e120_shared_tasks.py"]
    result = dict(status="running", args=vars(args), curve=[], parameters=sum(p.numel() for p in model.parameters()),
        protocol=dict(fitting=[0, args.fit], validation=[90_000_000, 90_000_000 + args.dev],
            test=([90_001_024, 90_001_024 + args.test] if args.contract else [95_000_000, 95_000_000 + args.test]),
            official_test_read=not args.contract, excluded_first_target=True, cold_context=True,
            weights_frozen_on_test=True, statistical_experts=[], tokenizer="fixed 27-character text8 alphabet",
            credit_truncation=args.chunk, resets="each epoch, development stream and test stream",
            selection="minimum development bpc over the fixed epoch budget"),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        fitting_data_sha256=hashlib.sha256(train.numpy().astype("uint8").tobytes()).hexdigest(),
        hardware=dict(platform=platform.platform(), torch=torch.__version__, device="cpu", threads=1),
        scope=("Small execution contract, not benchmark evidence" if args.contract else
               "Full declared character benchmark of proper learned event model; historical character representation, one seed, no frontier claim"))
    epoch, next_start, total, steps, best_loss, best_state = 1, 0, 0., 0, float("inf"), None
    state, previous_wall = model.new_state(), 0.
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("tag", "fit", "dev", "test", "epochs", "chunk", "seed", "width", "modes", "depth", "contract"):
            if saved["result"]["args"][key] != vars(args)[key]:
                raise ValueError("Changed resumed settings")
        if saved["result"]["source_sha256"] != result["source_sha256"]:
            raise ValueError("Changed resumed source")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        result = saved["result"]
        epoch, next_start, total, steps, best_loss, best_state, state = (saved[k] for k in
            ("epoch", "next_start", "total", "steps", "best_loss", "best_state", "stream_state"))
        torch.set_rng_state(saved["torch_rng"])
        previous_wall = result["wall_s"]

    def persist(save=False):
        result.update(wall_s=previous_wall + time.perf_counter() - started,
                      max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target = out if result["status"] == "completed" else running
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(target)
        if result["status"] == "completed":
            running.unlink(missing_ok=True)
        if save:
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), result=result,
                epoch=epoch, next_start=next_start, total=total, steps=steps, best_loss=best_loss,
                best_state=best_state, stream_state=state.detach(), torch_rng=torch.get_rng_state()), temporary)
            temporary.replace(checkpoint)
    persist()
    print(json.dumps(dict(started=args.tag, fit=args.fit, parameters=result["parameters"])), flush=True)
    while epoch <= args.epochs:
        model.train()
        for start in range(next_start, len(train) - 1, args.chunk):
            end = min(start + args.chunk, len(train) - 1)
            optimizer.zero_grad(set_to_none=True)
            logits, state = model.forward_chunk(train[start:end], state)
            loss = F.cross_entropy(logits, train[start + 1:end + 1])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite language loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * (end - start)
            steps += 1
            next_start, state = end, state.detach()
            if steps % 256 == 0:
                result["progress"] = dict(epoch=epoch, targets=end, steps=steps,
                    fitting_online_bpc=total / end / math.log(2), event_deliveries=state.deliveries)
                persist(True)
                print(json.dumps(result["progress"]), flush=True)
        score = evaluate(model, dev)
        result["curve"].append(dict(epoch=epoch, fitting_online_bpc=total / (len(train) - 1) / math.log(2),
            optimizer_steps=steps, training_event_deliveries=state.deliveries, dev=score))
        if score["bpc"] < best_loss:
            best_loss, best_state = score["bpc"], copy.deepcopy(model.state_dict())
            result["selected_epoch"] = epoch
        print(json.dumps(result["curve"][-1]), flush=True)
        epoch, next_start, total, steps, state = epoch + 1, 0, 0., 0, model.new_state()
        persist(True)
    model.load_state_dict(best_state)
    result["final"] = dict(dev=evaluate(model, dev))
    result["final"]["contract_score" if args.contract else "official_test"] = evaluate(model,
        torch.tensor(text_slice(90_001_024 if args.contract else 95_000_000, args.test)))
    result["status"] = "completed"
    persist(True)
    print(json.dumps(result["final"]), flush=True)


if __name__ == "__main__":
    main()
