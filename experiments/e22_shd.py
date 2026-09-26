"""E22 data: the Spiking Heidelberg Digits (Cramer et al. 2020) as input spike times for the race network.

    python experiments/e22_shd.py            # writes data/shd/shd_700.npz

SHD: spoken digits 0–9 in English and German (20 classes) as spike trains on 700 cochlear channels.
Encoding: the 700 channels are grouped into 70 bands of 10 adjacent channels, and each band's spike train is
split into 10 slots over the first second; input (band, slot) carries the *first spike time* in its slot
(inf if silent), in units of seconds (the horizon is 1). Each input spikes at most once per utterance, which the
simulator's weight updates assume; a slot acts like a separate synapse (a delay-line tap) on that band. The
network sees real spike times, not counts. Train: 8156 utterances; test: 2264 (speakers partly unseen).
"""
import os

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "shd")
BANDS, SLOTS, WIDTH = 70, 10, 1.0


def encode(path):
    with h5py.File(path, "r") as f:
        times, units, labels = f["spikes"]["times"], f["spikes"]["units"], np.array(f["labels"])
        X = np.full((len(labels), BANDS * SLOTS), np.inf, np.float32)
        for i in range(len(labels)):
            t, u = np.asarray(times[i]), np.asarray(units[i]).astype(np.int64)
            keep = t < WIDTH
            t, u = t[keep], u[keep]
            col = (u * BANDS // 700) * SLOTS + np.minimum((t / WIDTH * SLOTS).astype(np.int64), SLOTS - 1)
            order = np.argsort(t, kind="stable")                   # first spike per (band, slot)
            col, t = col[order], t[order]
            first = np.unique(col, return_index=True)[1]
            X[i, col[first]] = t[first] / WIDTH
    return X, labels.astype(np.int64)


def main():
    Xtr, ytr = encode(os.path.join(ROOT, "shd_train.h5"))
    Xte, yte = encode(os.path.join(ROOT, "shd_test.h5"))
    np.savez_compressed(os.path.join(ROOT, "shd_700.npz"), Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    act = np.isfinite(Xtr).mean()
    print(f"train {Xtr.shape} test {Xte.shape}; active inputs per utterance {act * Xtr.shape[1]:.0f} of "
          f"{Xtr.shape[1]}; classes {len(np.unique(ytr))}", flush=True)


if __name__ == "__main__":
    main()
