"""Causal online mixing ablation from an existing E173 checkpoint.

This updates only the local expert mixing weights. Counts remain frozen and
the causal copy cache is identical in both arms. It is not neural-backbone TTT.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import e173_causal_language as native


def predict_then_update(weights, expert_logp, target, rate, update):
    logits = weights @ expert_logp
    logits -= logits.max()
    probability = np.exp(logits)
    normalizer = probability.sum()
    probability /= normalizer
    loss = (np.log(normalizer) - logits[target]) / np.log(2)
    if update:
        new_weights = weights + rate * (expert_logp[:, target] - expert_logp @ probability)
    else:
        new_weights = weights
    if not np.isfinite(loss) or not np.isfinite(new_weights).all():
        raise FloatingPointError("Nonfinite online score or update")
    return probability, float(loss), new_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default="experiments/results/e173/causal_language_pilot_20260930.json")
    parser.add_argument("--offset", type=int, default=90_032_768)
    parser.add_argument("--n", type=int, default=8192)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.n < 2:
        raise ValueError("Unique plain tag and at least two positions required")
    path = ROOT / args.checkpoint
    saved = json.loads(path.read_text())
    if saved["status"] != "completed":
        raise ValueError("A completed checkpoint is required")
    if not 90_000_000 <= args.offset < args.offset + args.n <= 95_000_000:
        raise ValueError("Use development characters; official test is reserved")
    fitted_end = saved["protocol"]["validation"][1]
    if args.offset < fitted_end:
        raise ValueError("New scoring positions must follow the mixing-weight fitting split")
    output = ROOT / "experiments/results/online_language" / (args.tag + ".json")
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    count_started = time.perf_counter()
    train = native.text_slice(0, saved["args"]["D"])
    stream = native.text_slice(args.offset, args.n)
    order_count = saved["args"]["K"]
    orders = [native.S1.Order(train, k) for k in range(order_count + 1)]
    word = native.CausalWordOrder(train)
    preparation_seconds = time.perf_counter() - count_started
    with np.load(path.with_suffix(".weights.npz")) as checkpoint:
        weights = {(arm, mode): checkpoint[arm].copy()
                   for arm in saved["arms"] for mode in ("frozen", "online")}
    initial = {key: value.copy() for key, value in weights.items()}
    totals = {key: 0. for key in weights}
    blocks, online_seconds, frozen_seconds = [], 0., 0.
    for offset, distributions, selectors in native.batches(train, stream, orders, word, order_count, 1024):
        block = {key: 0. for key in weights}
        for arm, record in saved["arms"].items():
            experts = distributions if arm == "with_causal_word" else np.delete(distributions, order_count + 2, axis=1)
            for mode in ("frozen", "online"):
                begin = time.perf_counter()
                for j, expert_logp in enumerate(experts):
                    selector, target = int(selectors[j]), int(stream[offset + j])
                    _, loss, next_weights = predict_then_update(weights[arm, mode][selector],
                        expert_logp, target, record["selected_learning_rate"], mode == "online")
                    weights[arm, mode][selector] = next_weights
                    if offset + j > 0:
                        block[arm, mode] += loss
                elapsed = time.perf_counter() - begin
                if mode == "online": online_seconds += elapsed
                else: frozen_seconds += elapsed
        for key in totals: totals[key] += block[key]
        scored = len(distributions) - (offset == 0)
        blocks.append(dict(start=offset, end=offset + len(distributions), n=scored,
            bpc={arm + ":" + mode: value / scored for (arm, mode), value in block.items()}))
    result = dict(status="completed", args=vars(args),
        protocol=dict(split="new development window", predict_before_target_update=True,
            counts_frozen=True, copy_cache_identical=True, adaptation="expert mixing weights only",
            official_test_read=False, excluded_first_position=True,
            learning_rate_source="selected on the checkpoint's earlier validation split",
            inherited_count_training_characters=saved["args"]["D"],
            inherited_mixing_training_characters=saved["args"]["validation"]),
        rows=[dict(arm=arm, frozen_bpc=totals[arm, "frozen"] / (args.n - 1),
            online_bpc=totals[arm, "online"] / (args.n - 1), online_updates=args.n,
            rate=record["selected_learning_rate"],
            weight_change_l2=float(np.linalg.norm(weights[arm, "online"] - initial[arm, "online"])),
            extra_update_arithmetic_flops=args.n * (2 * len(record["experts"]) * 27 + 2 * len(record["experts"])))
            for arm, record in saved["arms"].items()], blocks=blocks,
        checkpoint_sha256=hashlib.sha256(path.with_suffix(".weights.npz").read_bytes()).hexdigest(),
        stream_sha256=hashlib.sha256(stream.astype(np.uint8).tobytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "experiments/e173_causal_language.py", ROOT / "experiments/e79_race_mixer.py")},
        hardware=dict(platform=platform.platform(), device="cpu", threads=1),
        count_reconstruction_wall_s=preparation_seconds, frozen_scoring_wall_s=frozen_seconds,
        online_scoring_wall_s=online_seconds, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Exploratory one-checkpoint, one-window ablation; no neural-backbone TTT or frontier comparison")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
