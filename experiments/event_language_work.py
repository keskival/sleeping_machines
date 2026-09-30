"""Audit the proper E176 learned event model's fitting and inference arithmetic.

One saved-checkpoint chunk calibrates whole-run estimates. No new quality run,
data selection or checkpoint write; the audit must use the guarded queue.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from e120_shared_tasks import text_slice
from sleeping_machines.operation_audit import OperationAudit
from sleeping_machines.stream_language import StreamingEventLanguageModel


def capture(action):
    with OperationAudit() as audit:
        value = action()
    return value, audit.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    output = ROOT / "experiments/results/event_language_work" / (args.tag + ".json")
    if Path(args.tag).name != args.tag or output.exists():
        raise ValueError("Unique plain tag required")
    started = time.perf_counter()
    torch.set_num_threads(1)
    source = ROOT / "experiments/results/e176/stream_language_d8_20260930.json"
    record = json.loads(source.read_text())
    if record["status"] != "completed" or record["protocol"]["experts"]:
        raise ValueError("A completed evidence-free model is required")
    checkpoint = source.with_suffix(f".epoch{record['final']['epoch']}.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    chunk = record["args"]["chunk"]
    tokens = torch.tensor(text_slice(0, chunk + 32))
    model = StreamingEventLanguageModel(depth=record["args"]["depth"])
    model.load_state_dict(saved["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.load_state_dict(saved["optimizer"])
    model.train()
    state = model.new_state()
    with torch.no_grad():
        _, warm = capture(lambda: model.forward_chunk(tokens[:31], state))
    def forward():
        logits, _ = model.forward_chunk(tokens[31:-1], state)
        return F.cross_entropy(logits, tokens[32:])
    loss, forward_trace = capture(forward)
    _, backward = capture(loss.backward)
    _, clipping = capture(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    _, adam = capture(optimizer.step)
    # Restore the unchanged quality checkpoint before inference audit.
    model.load_state_dict(saved["state_dict"])
    model.eval()
    inference_state = model.new_state()
    with torch.no_grad():
        model.forward_chunk(tokens[:31], inference_state)
        _, inference = capture(lambda: F.cross_entropy(
            model.forward_chunk(tokens[31:-1], inference_state)[0], tokens[32:]))
    traces = dict(warmup=warm, forward_and_loss=forward_trace, backward=backward,
                  gradient_clipping=clipping, optimizer=adam, inference_and_scoring=inference)
    if not all(t["formula_coverage_complete"] for t in traces.values()):
        raise ValueError({k: t["unsupported_floating_operators"] for k, t in traces.items()
                          if not t["formula_coverage_complete"]})
    targets = sum(record["args"]["fit"] for _ in record["curve"])
    steps = sum(r["optimizer_steps"] for r in record["curve"])
    if targets != steps * chunk:
        raise ValueError("This audit requires the recorded full equal-size chunks")
    stages = {k: traces[k]["arithmetic_flops"] * steps for k in
              ("forward_and_loss", "backward", "gradient_clipping", "optimizer")}
    stages["warmup"] = warm["arithmetic_flops"] * len(record["curve"])
    result = dict(status="completed", args=vars(args),
        model="Proper learned persistent Sleeping Machines event language model; no statistical experts",
        quality_result=str(source.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parameters=record["parameters"], fitting_targets=targets, optimizer_steps=steps,
        training_stages=stages, total_training_arithmetic_flops=sum(stages.values()),
        training_special_function_evaluations=sum(traces[k]["special_function_evaluations"] * steps
            for k in ("forward_and_loss", "backward", "gradient_clipping", "optimizer"))
            + warm["special_function_evaluations"] * len(record["curve"]),
        inference_arithmetic_flops_per_character=inference["arithmetic_flops"] / chunk,
        inference_special_functions_per_character=inference["special_function_evaluations"] / chunk,
        stages=traces,
        convention="Logical event arithmetic, two FLOPs/MAC; exp/log/rotations/etc separate; no simulator dispatch/allocation charge",
        scope="Representative saved-parameter chunk estimate for completed E176, not a whole-run trace. Includes fitting warmup, loss, backward, clipping and warm Adam. Inference includes NLL scoring; initial/epoch evaluations, search, index/queue work and memory traffic excluded. No matched neural quality or event-hardware claim.",
        hardware=dict(platform=platform.platform(), torch=torch.__version__, device="cpu", threads=1),
        wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), source, ROOT / "sleeping_machines/stream_language.py",
                      ROOT / "sleeping_machines/event_state.py", ROOT / "sleeping_machines/operation_audit.py")})
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("total_training_arithmetic_flops",
          "inference_arithmetic_flops_per_character", "inference_special_functions_per_character")}), flush=True)


if __name__ == "__main__":
    main()
