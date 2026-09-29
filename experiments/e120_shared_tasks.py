"""Bounded data adapters for the shared event model, with explicit prefix queries.

Every target is stored separately from the observed events. Local evidence
memories are fitted on disjoint examples before neural calibration/training.
All real-data scores here are development scores, not official-test claims.
"""
from dataclasses import dataclass, field
import csv
import gzip
import io
import itertools
from pathlib import Path
import zipfile
import numpy as np
from sleeping_machines.event_query import ObservedPrefix
from sleeping_machines.evidence_memory import ConditionalEvidence, RelativeRouteMemory


@dataclass
class Example:
    prefix: ObservedPrefix
    label: int
    identity: str
    evidence: np.ndarray | None = None
    bucket: int | None = None
    exposure: np.ndarray | None = None


@dataclass
class Task:
    config: dict
    fit: list
    dev: list
    protocol: dict
    memory: object = None
    extra: dict = field(default_factory=dict)


def prefix(symbols, times=None, counts=None, marks=None):
    symbols = np.asarray(symbols, dtype=np.int64)
    times = np.arange(len(symbols))*.01 if times is None else np.asarray(times, dtype=np.float64)
    counts = np.ones(len(symbols)) if counts is None else np.asarray(counts)
    return ObservedPrefix(symbols, times, counts, float(times[-1]), marks)


def text_slice(start, n):
    with Path("data/text8/text8").open("rb") as f:
        f.seek(start)
        raw = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)
    if len(raw) != n or not np.all((raw == 32) | ((raw >= 97) & (raw <= 122))):
        raise ValueError("Missing or invalid text8 data")
    return np.where(raw == 32, 0, raw-96)


def language(nfit, ndev, seed, memory_size=32768):
    context, orders = 32, (0, 1, 2, 3)
    memories = [ConditionalEvidence(27) for _ in orders]
    raw = text_slice(0, memory_size)
    for i in range(context, len(raw)):
        for order, memory in zip(orders, memories):
            memory.observe(tuple(raw[i-order:i]), int(raw[i]))
    def examples(offset, n):
        raw = text_slice(offset, n+context)
        rows = []
        for i in range(context, len(raw)):
            obs = raw[i-context:i]
            evidence = np.stack([m.scores(tuple(obs[len(obs)-o:])) for o, m in zip(orders, memories)])
            rows.append(Example(prefix(obs), int(raw[i]), str(offset+i), evidence))
        return rows
    return Task({"bands": 27, "classes": 27, "groups": 1, "readout": "last", "evidence_count": 4},
        examples(memory_size, nfit), examples(90_000_000, ndev),
        {"dataset": "text8", "memory_fit_bytes": [0, memory_size],
         "neural_fit_bytes": [memory_size+context, memory_size+context+nfit],
         "dev_bytes": [90_000_000+context, 90_000_000+context+ndev], "context": context,
         "official_test_read": False, "time_unit": "one character = 10 ms (declared encoding)",
         "evidence": "frozen KT counts, orders 0..3; disjoint memory/neural fit; no copy/word expert",
         "cost_scope": "each next-character query replays its 32-character prefix"}, memories)


def recall(nfit, ndev, seed, memory_size=4000):
    from e61_race_attention import make_perm, sample
    rng = np.random.default_rng(seed)
    perm = make_perm(32, rng)
    memory = RelativeRouteMemory(32, 64)
    for _ in range(memory_size):
        seq, q, y = sample(32, 8, perm, rng)
        memory.observe(seq, q, y+32)
    def examples(n, pairs, rng, split):
        rows = []
        for i in range(n):
            seq, q, y = sample(32, pairs, perm, rng)
            value, _ = memory.read(seq, q)
            # A smoothed categorical message from the winning pointer only.
            probs = np.full(32, .01/31)
            if 32 <= value < 64:
                probs[value-32] = .99
            else:
                probs[:] = 1/32
            rows.append(Example(prefix(np.r_[seq, 64+q]), y, f"{split}:{i}", np.log(probs)[None]))
        return rows
    return Task({"bands": 96, "classes": 32, "groups": 1, "readout": "last", "evidence_count": 1},
        examples(nfit, 8, rng, "fit"), examples(ndev, 8, np.random.default_rng(seed+1000), "dev"),
        {"dataset": "E61 associative recall", "K": 32, "pairs_fit": 8, "pairs_extrapolation": 32,
         "memory_fit_examples": memory_size, "permutation": perm.tolist(),
         "evidence": "frozen E61 relative pointer; disjoint pointer and neural training examples",
         "time_unit": "one symbol = 10 ms", "candidate_search": "all valid source/offset routes"}, memory,
        {"context4x": examples(ndev, 32, np.random.default_rng(seed+2000), "context4x")})


def temporal(nfit, ndev, seed):
    from e28_routing import make_task, sample
    motifs, classes = make_task(24, 6, 6, np.random.default_rng(seed))
    def examples(n, rng, split):
        rows = []
        for i in range(n):
            times, y = sample(motifs, classes, 24, 12., .15, rng)
            active = np.flatnonzero(np.isfinite(times))
            # The explicit observation-end marker handles an empty negative.
            channels, t = np.r_[active, 24], np.r_[times[active]/10, 1.2]
            order = np.argsort(t, kind="stable")
            rows.append(Example(prefix(channels[order], t[order]), y, f"{split}:{i}"))
        return rows
    return Task({"bands": 25, "classes": 7, "groups": 1},
        examples(nfit, np.random.default_rng(seed+1), "fit"),
        examples(ndev, np.random.default_rng(seed+2), "dev"),
        {"dataset": "E28 shared-motif composition", "channels": 24, "motifs": motifs,
         "classes": classes, "time_unit": "E28 time divided by 10", "q": .15,
         "note": "same generator, smaller configuration; no handcrafted motif detector"})


def modular(nfit, ndev, seed):
    p = 17
    tuples = np.array(list(itertools.product(range(p), repeat=3)))
    order = np.random.default_rng(seed).permutation(len(tuples))
    ntrain = int(.3*len(tuples))
    def examples(indices):
        return [Example(prefix(tuples[i]+np.array([0, p, 2*p])), int(tuples[i].sum()%p), str(i)) for i in indices]
    return Task({"bands": 51, "classes": 17, "groups": 1},
        examples(order[:min(nfit, ntrain)]), examples(order[ntrain:ntrain+ndev]),
        {"dataset": "three-operand modular addition", "modulus": p, "split_fraction": .3,
         "time_unit": "operand position = 10 ms", "seed": seed,
         "note": "unseen tuples; no arithmetic features or rhythm primitive; short run is not a grokking test"})


def mnist(nfit, ndev, seed):
    # Read just a bounded training prefix; no download and no official test.
    total = nfit+ndev
    if total > 60_000:
        raise ValueError("MNIST train size exceeded")
    with gzip.open("data/train-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        pixels = np.frombuffer(f.read(total*784), dtype=np.uint8).reshape(total, 28, 28)/255.
    with gzip.open("data/train-labels-idx1-ubyte.gz", "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(total), dtype=np.uint8)
    # Fixed 2x2 pooling is an input representation, not a trained dense layer.
    pixels = pixels.reshape(total, 14, 2, 14, 2).mean((2, 4)).reshape(total, 196)
    rows = []
    for i in range(total):
        active = np.flatnonzero(pixels[i] > .1)
        t = (1-pixels[i, active])*.8
        order = np.argsort(t, kind="stable")
        rows.append(Example(prefix(active[order], t[order]), int(labels[i]), str(i)))
    return Task({"bands": 196, "classes": 10, "groups": 4}, rows[:nfit], rows[nfit:],
        {"dataset": "MNIST training prefix", "neural_fit_indices": [0, nfit], "dev_indices": [nfit, total],
         "encoding": "2x2 average pool, intensity > 0.1, latency = 0.8*(1-intensity) seconds",
         "official_test_read": False})


def market(nfit, ndev, seed):
    from e42_when import decision_points
    bounds = np.array([.01, .05, .2, 1., 5.])
    edges = np.r_[0., bounds, np.inf]
    context, raw_limit = 32, 200_000
    def trades(day):
        path = Path(f"data/binance/BTCUSDT-aggTrades-{day}.zip")
        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.namelist()[0]) as f:
                rows = [(int(r[5]), float(r[1]), float(r[2]), r[6].lower() == "true")
                        for r in itertools.islice(csv.reader(io.TextIOWrapper(f)), raw_limit)]
        t, p, q, sell = map(np.array, zip(*rows))
        if t[0] < 10**14:
            t = t*1000
        return t, p, q, sell
    first = trades("2026-08-25")
    threshold = float(np.quantile(first[2], .99))
    def events(raw):
        t, p, q, sell = raw
        dp = decision_points(t, p)
        move = np.where(np.diff(np.log(p[dp])) > 0, 0, 1)
        big = np.flatnonzero(q >= threshold)
        times = np.r_[t[dp[1:]], t[big]].astype(np.float64)/1e6
        types = np.r_[move, np.where(sell[big], 3, 2)]
        order = np.argsort(times, kind="stable")
        return times[order], types[order]
    memories = [ConditionalEvidence(4, bins=6) for _ in range(3)]
    def target(times, types, i):
        gap = max(float(times[i]-times[i-1]), 1e-6)
        return int(types[i]), int(np.searchsorted(bounds, gap)), np.clip(gap-edges[:-1], 0, np.diff(edges))
    times, types = events(first)
    for i in range(context, len(types)):
        y, k, span = target(times, types, i)
        for order, mem in enumerate(memories):
            mem.observe(tuple(types[i-order:i]), y, k, span)
    def examples(day, n):
        times, types = events(trades(day))
        if len(types) < context+n:
            raise ValueError("Bounded market prefix has too few events; lower --fit/--dev")
        rows = []
        for i in range(context, context+n):
            obs = types[i-context:i]
            t = times[i-context:i]-times[i-context]
            scores = np.stack([mem.scores(tuple(obs[len(obs)-o:])).ravel() for o, mem in enumerate(memories)])
            y, k, span = target(times, types, i)
            rows.append(Example(prefix(obs, t), y, f"{day}:{i}", scores, k, span))
        return rows
    return Task({"bands": 4, "classes": 24, "groups": 1, "readout": "last", "evidence_count": 3},
        examples("2026-08-26", nfit), examples("2026-08-29", ndev),
        {"dataset": "BTCUSDT event TPP", "memory_day": "2026-08-25", "fit_day": "2026-08-26",
         "dev_day": "2026-08-29", "raw_trade_limit_per_day": raw_limit, "size_threshold": threshold,
         "threshold_fit": "first training day prefix only", "gap_edges_seconds": bounds.tolist(),
         "time_unit": "physical seconds", "objective": "exact marked hazard NLL",
         "official_test_read": False, "context": context,
         "note": "bounded prefixes, not full E48/E52 day protocol; next gap/type are targets only"}, memories)


BUILDERS = {"language": language, "recall": recall, "market": market, "temporal": temporal,
            "mnist": mnist, "modular": modular}
