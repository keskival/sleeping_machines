"""DVS128 Gesture: parse AEDAT 3.1 recordings into labelled gesture samples (events x, y, polarity, t in µs).

Official split: users 1-23 train, users 24-29 test (the dataset's trials_to_train/test lists). Each recording has a
labels CSV (class, startTime_usec, endTime_usec); a sample is the events between start and end, times relative to start.
Cached as one npz per recording (all its samples) under data/dvsgesture/cache/.
"""
import csv
import glob
import os
import struct

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsgesture")


def read_aedat31(path):
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"#!END-HEADER\r\n") + len(b"#!END-HEADER\r\n")
    pos, xs, ys, ps, ts = end, [], [], [], []
    while pos + 28 <= len(raw):
        etype, _src, esize, _tsoff, tsover, ecap, enum, _valid = struct.unpack_from("<hhiiiiii", raw, pos)
        pos += 28
        n = ecap * esize
        if etype == 1 and esize == 8 and enum > 0:
            ev = np.frombuffer(raw, dtype=np.uint32, count=2 * enum, offset=pos).reshape(-1, 2)
            data, t = ev[:, 0], ev[:, 1].astype(np.int64) | (np.int64(tsover) << 31)
            ok = (data & 1) == 1
            xs.append(((data >> 17) & 0x1FFF)[ok]); ys.append(((data >> 2) & 0x1FFF)[ok])
            ps.append(((data >> 1) & 1)[ok]); ts.append(t[ok])
        pos += n
    return (np.concatenate(xs).astype(np.int16), np.concatenate(ys).astype(np.int16),
            np.concatenate(ps).astype(np.int8), np.concatenate(ts))


def labels(path):
    with open(path) as f:
        return [(int(r["class"]), int(r["startTime_usec"]), int(r["endTime_usec"])) for r in csv.DictReader(f)]


def recording_samples(aedat):
    """[(label 0-10, x, y, p, t_rel_us)] for one recording, cached."""
    cache = os.path.join(ROOT, "cache", os.path.basename(aedat).replace(".aedat", ".npz"))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["samples"])
    x, y, p, t = read_aedat31(aedat)
    out = []
    for c, s, e in labels(aedat.replace(".aedat", "_labels.csv")):
        m = (t >= s) & (t < e)
        out.append((c - 1, x[m], y[m], p[m], t[m] - s))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, samples=np.array(out, dtype=object))
    return out


def split(which):
    """all samples of the official train (users 1-23) or test (users 24-29) split: [(label, x, y, p, t, user)]."""
    d = os.path.join(ROOT, "DvsGesture")
    with open(os.path.join(d, f"trials_to_{which}.txt")) as f:           # the official recording lists
        names = [l.strip() for l in f if l.strip().endswith(".aedat")]
    res = []
    for n in names:
        a = os.path.join(d, n)
        u = int(n[4:6])
        for s in recording_samples(a):
            res.append((*s, u))
    return res


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    x, y, p, t = read_aedat31(path)
    print("events", len(t), "x", x.min(), x.max(), "y", y.min(), y.max(), "p", np.bincount(p), "dur s", (t[-1] - t[0]) / 1e6)
    for c, s, e in labels(path.replace(".aedat", "_labels.csv")):
        m = (t >= s) & (t < e)
        print(" class", c, "dur", round((e - s) / 1e6, 2), "s events", int(m.sum()))
