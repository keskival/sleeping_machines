"""Learn raw event marks before packet coalescing, preserving a warm SHD core.

Each original channel has a distinct (coarse band, fine slot) address. Only
nonempty 10 ms packets enter the deep hard-race core. The source payload is a
learned fine embedding times the original event's causal time features, summed
within the packet. Zero fine weights recover the parent computation exactly.
This extends the existing common continuous-mark interface; no new race rule.
"""
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import e51_shd_world as S
from e117_serial_event_shd import batch as coarse_batch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel

BANDS, SLOTS, WINDOW = 40, 18, .01


def marked_packets(times, units):
    times = np.asarray(times, np.float64)
    units = np.asarray(units, np.int64)
    if not len(times) or np.any(units < 0) or np.any(units >= 700):
        raise ValueError("Expected nonempty SHD events with channels in 0..699")
    bands = units * BANDS // 700
    bins = np.floor(times / WINDOW).astype(np.int64)
    pairs, assignment, counts = np.unique(np.stack((bins, bands), axis=1),
        axis=0, return_inverse=True, return_counts=True)
    # Integer arithmetic makes this address injective on all 700 input units.
    fine = ((units * BANDS - bands * 700) * SLOTS // 700).astype(np.int64)
    return (pairs[:, 1].astype(np.int64),
        ((pairs[:, 0]+1)*WINDOW).astype(np.float32), counts.astype(np.float32),
        assignment.astype(np.int64), fine, times.astype(np.float32))


def load_marked(limit, part, seed):
    if part not in ("fit_spk", "val_spk") or limit < 1:
        raise ValueError("Only existing fitting/development partitions are allowed")
    with h5py.File(Path(S.ROOT)/"shd_train.h5", "r") as f:
        held = np.isin(np.asarray(f["extra"]["speaker"]), S.VAL_SPEAKERS)
        eligible = np.flatnonzero(held if part == "val_spk" else ~held)
        order = np.random.default_rng(seed).permutation(len(eligible))[:limit]
        rows = []
        for identity in order:
            index = int(eligible[identity])
            t = np.asarray(f["spikes"]["times"][index], np.float64)
            u = np.asarray(f["spikes"]["units"][index], np.int64)
            sort = np.argsort(t, kind="stable")
            b, ts, c, assignment, fine, raw_t = marked_packets(t[sort], u[sort])
            rows.append((b, ts, c, int(f["labels"][index]), int(identity),
                         assignment, fine, raw_t, index))
    return rows


def augment_marked(item, rng):
    b, t, c, y, identity, assignment, fine, raw_t, original_id = item
    shift = int(rng.integers(-2, 3))
    scale = float(np.exp(rng.uniform(-.15, .15)))
    shifted = b+shift
    keep = (shifted >= 0) & (shifted < BANDS)
    if not keep.any():
        shifted, keep = b, np.ones(len(b), dtype=bool)
    remap = np.cumsum(keep)-1
    raw_keep = keep[assignment]
    return (shifted[keep], (t[keep]*scale).astype(np.float32), c[keep], y,
            identity, remap[assignment[raw_keep]], fine[raw_keep],
            (raw_t[raw_keep]*scale).astype(np.float32), original_id)


def marked_batch(rows):
    base = coarse_batch(rows)
    lengths = np.array([len(r[0]) for r in rows])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    assignments = np.concatenate([r[5]+offset for r, offset in zip(rows, offsets)])
    source = np.concatenate([r[0][r[5]]*SLOTS+r[6] for r in rows])
    raw_t = np.concatenate([r[7] for r in rows])
    return (*base, torch.from_numpy(assignments), torch.from_numpy(source),
            torch.from_numpy(raw_t))


class FinePacketModel(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.core = SharedEventModel(depth=8, memory_backend="linear", continuous_dim=32)
        # Identity injects already projected marks with no dense projection work.
        self.core.continuous = nn.Identity()
        self.core.load_state_dict(parent["state_dict"], strict=True)
        self.fine = nn.Embedding(BANDS*SLOTS, self.core.dim//4)
        nn.init.zeros_(self.fine.weight)

    def forward(self, packed, trace=False):
        b, t, c, ids, labels, assignment, source, raw_t = packed
        phi = torch.cat((raw_t.new_ones((len(raw_t), 1)),
            torch.exp(-raw_t[:, None]/self.core.time_constants)), -1)
        # Contrast coordinates remove the redundant coarse-band embedding.
        # This is an 18-address local coupling, not a dense event-pair map.
        table = self.fine.weight.reshape(BANDS, SLOTS, -1)
        table = (table-table.mean(1, keepdim=True)).flatten(0, 1)
        raw = (F.embedding(source, table)[:, :, None]*phi[:, None, :]).flatten(1)
        marks = raw.new_zeros((len(b), self.core.dim)).index_add(0, assignment, raw)/c[:, None]
        logits, payload, stats, traces = self.core(b, t, c, ids, len(labels),
            continuous=marks, trace=trace)
        stats.update(source_events=len(raw_t), source_lookup_scalars=len(raw_t)*8,
            source_payload_multiplies=len(raw_t)*32,
            source_payload_additions=len(raw_t)*32,
            source_packet_divisions=len(b)*32,
            source_time_exponentials=len(raw_t)*3,
            source_contrast_table_additions=BANDS*SLOTS*8*2,
            fine_source_parameters=self.fine.weight.numel())
        return logits, payload, stats, traces


def warm_optimizer(model, parent, learning_rate, fine_rate):
    named = dict(model.core.named_parameters())
    groups = parent.get("optimizer_parameter_groups") or [list(named)]
    if set(sum(groups, [])) != set(named):
        raise ValueError("Parent optimizer grouping does not match common core")
    opt = torch.optim.Adam([named[n] for n in groups[0]], lr=learning_rate)
    for group in groups[1:]:
        opt.add_param_group({"params": [named[n] for n in group], "lr": learning_rate})
    opt.load_state_dict(parent["optimizer"])
    for group in opt.param_groups:
        group["lr"] = learning_rate
    opt.add_param_group({"params": model.fine.parameters(), "lr": fine_rate})
    return opt
