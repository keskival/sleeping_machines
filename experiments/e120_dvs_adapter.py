"""Bounded-memory DVS Gesture adapter: read AEDAT packets, retain count packets.

Only the first second of each labelled gesture is observed. Training users
1..19 and development users 20..23 are disjoint; official test is not opened.
No complete raw recording or dense time grid is allocated.
"""
import csv
from pathlib import Path
import struct
import numpy as np
from e120_shared_tasks import Example, Task, prefix


def recording(path, observation_seconds=1.0):
    with path.with_name(path.stem+"_labels.csv").open() as f:
        labels = [(int(r["class"])-1, int(r["startTime_usec"]), int(r["endTime_usec"]))
                  for r in csv.DictReader(f)]
    counts = [{} for _ in labels]
    raw_counts = np.zeros(len(labels), dtype=np.int64)
    with path.open("rb") as f:
        while True:
            line = f.readline(4096)
            if line.strip() == b"#!END-HEADER":
                break
            if not line or not line.startswith(b"#"):
                raise ValueError("Invalid AEDAT header")
        while header := f.read(28):
            if len(header) != 28:
                raise ValueError("Truncated AEDAT packet")
            kind, source, size, tsoffset, overflow, capacity, number, valid = struct.unpack("<hhiiiiii", header)
            if min(size, capacity, number) < 0 or number > capacity or size*capacity > 32*1024*1024:
                raise ValueError("Invalid or over-budget AEDAT packet")
            raw = f.read(size*capacity)
            if len(raw) != size*capacity:
                raise ValueError("Truncated AEDAT payload")
            if kind != 1 or size != 8 or not number:
                continue
            ev = np.frombuffer(raw, dtype="<u4", count=2*number).reshape(-1, 2)
            data, t = ev[:, 0], ev[:, 1].astype(np.int64) | (np.int64(overflow) << 31)
            ok = (data & 1) == 1
            x, y, polarity = (data >> 17) & 0x1FFF, (data >> 2) & 0x1FFF, (data >> 1) & 1
            for j, (_, start, end) in enumerate(labels):
                cutoff = end if observation_seconds is None else min(end, start+int(observation_seconds*1_000_000))
                selected = ok & (t >= start) & (t < cutoff) & (x < 128) & (y < 128)
                if not selected.any():
                    continue
                raw_counts[j] += int(selected.sum())
                # 4x4 spatial cells, two polarities, 50 ms packet closures.
                channel = ((x[selected]//32)*4+y[selected]//32)*2+polarity[selected]
                key = ((t[selected]-start)//50_000)*32+channel
                unique, number = np.unique(key, return_counts=True)
                for k, n in zip(unique, number):
                    counts[j][int(k)] = counts[j].get(int(k), 0)+int(n)
    rows = []
    for j, ((label, start, end), count) in enumerate(zip(labels, counts)):
        keys = np.array(sorted(count), dtype=np.int64)
        b, t = keys%32, (keys//32+1)*.05
        c = np.array([count[int(k)] for k in keys], dtype=np.float64)
        # A declared query/end marker ensures even an empty prefix is valid.
        # Full packet closure may be up to 50 ms beyond a short segment end;
        # that is compute/input buffering delay, not an additional observation.
        duration = (end-start)/1e6 if observation_seconds is None else observation_seconds
        close = max(duration, float(t[-1]) if len(t) else 0.)
        obs = prefix(np.r_[b, 32], np.r_[t, close], np.r_[c, 1.])
        rows.append(Example(obs, label, f"{path.name}:{j}"))
    return rows, int(raw_counts.sum())


def dvs(nfit, ndev, seed):
    root = Path("data/dvsgesture/DvsGesture")
    names = [n.strip() for n in (root/"trials_to_train.txt").read_text().splitlines() if n.endswith(".aedat")]
    def collect(n, dev):
        selected = [name for name in names if
                    ((20 <= int(name[4:6]) <= 23) if dev else (1 <= int(name[4:6]) <= 19))]
        # Interleave users so a small fit prefix covers more than one person.
        selected.sort(key=lambda name: (name[7:], int(name[4:6])))
        rows, raw, files = [], 0, []
        for name in selected:
            part, count = recording(root/name)
            rows.extend(part)
            raw += count
            files.append(name)
            if len(rows) >= n:
                return rows[:n], raw, files
        raise ValueError("Requested more DVS samples than this split contains")
    fit, fitraw, fitfiles = collect(nfit, False)
    dev, devraw, devfiles = collect(ndev, True)
    return Task({"bands": 33, "classes": 11, "groups": 1}, fit, dev,
        {"dataset": "DVS128 Gesture", "fit_users": "1..19", "dev_users": "20..23",
         "official_test_read": False, "observation": "first 1 second of each gesture",
         "encoding": "4x4 cells, polarity, count packets released at 50 ms closure, explicit end marker",
         "fit_recordings": fitfiles, "dev_recordings": devfiles,
         "raw_events_in_loaded_prefixes": {"fit": fitraw, "dev": devraw},
         "time_unit": "physical seconds", "scope": "small prefix-recognition screen, not full-gesture benchmark"})
