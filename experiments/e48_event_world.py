"""E48: the semi-Markov world model built as an asynchronous event network, verified against its table form (E44).

Nodes (no clock anywhere):
  state nodes      one per (second-to-last type, last type): armed by an event, disarmed by the next one (§75)
  window nodes     a delay line from the last event emits expiry events at 0.01, 0.05, 0.2, 1 and 5 s; the window node
                   currently open says which gap bucket the present is in (hold windows, §61)
  detectors        one hazard detector and one type detector per event type; between two events of the network (real
                   or expiry) their rates are constant, set by the synapses from the armed state node and the open window
                   node; each detector integrates its exposure (rate × time) at the next event
Learning (at synapses, events only): at a real event of type y, the synapse (state, window) -> y's type detector gains a
count, and the hazard synapse (state, window) gains an event count; exposure time is added to the hazard synapses of
every window the gap passed through (delivered by the expiry events themselves).
Output: the prequential log-likelihood of each real event, which must equal the table model's exactly, and the number
of network events per real event (1 real + expiries crossed) and synaptic operations per real event.
"""
import argparse
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, SemiMarkov, NT  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e48")
GAPS = SemiMarkov.GAPS


class EventWorld:
    def __init__(self):
        nb = len(GAPS) + 1
        self.hN = np.full((NT, NT, nb), 0.5); self.hE = np.full((NT, NT, nb), 1.0)   # hazard synapses
        self.tW = np.ones((NT, NT, nb, NT))                                          # type synapses
        self.state = None; self.window = 0; self.t_last = None; self.t_prev_evt = None
        self.expo = np.zeros(nb)                                   # exposure per window since the last real event
        self.queue = []; self.net_events = 0; self.syn_ops = 0; self.learn = True

    def _advance(self, t):
        """deliver expiry events up to time t; each closes one window and opens the next."""
        while self.queue and self.queue[0][0] <= t:
            te, k = heapq.heappop(self.queue)
            self.expo[self.window] += te - self.t_prev_evt        # the open window's exposure node integrates
            self.t_prev_evt = te; self.window = k; self.net_events += 1; self.syn_ops += 1
        self.expo[self.window] += t - self.t_prev_evt; self.t_prev_evt = t

    def real_event(self, t, y):
        self.net_events += 1
        if self.state is None:
            self.state = (0, y); self._arm(t)
            return None
        self._advance(t)
        ctx = self.state; k = self.window
        h = self.hN[ctx] / self.hE[ctx]
        comp = float((h * self.expo).sum()); self.syn_ops += len(GAPS) + 1
        p = self.tW[ctx][k] / self.tW[ctx][k].sum(); self.syn_ops += NT
        ll = float(np.log(h[k] * p[y]) - comp)
        if self.learn:                                             # synaptic counts (events only)
            self.hN[ctx][k] += 1.0; self.hE[ctx] += self.expo; self.tW[ctx][k][y] += 1.0
            self.syn_ops += 2 + int((self.expo > 0).sum())
        self.state = (ctx[1], y); self._arm(t)
        return ll

    def _arm(self, t):
        self.queue = [(t + g, i + 1) for i, g in enumerate(GAPS)]  # delay line from this event
        heapq.heapify(self.queue)
        self.window = 0; self.t_last = self.t_prev_evt = t; self.expo[:] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--freeze_after", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    net, ref = EventWorld(), SemiMarkov(order=2)
    rows = []
    for di, day in enumerate(PILOT[:a.ndays]):
        if a.freeze_after and di == a.freeze_after:
            net.learn = ref.learn = False
        T, Y = day_events(day, big_q)
        e0, s0 = net.net_events, net.syn_ops
        ll_net = ll_ref = 0.0; n = 0; maxdiff = 0.0
        for t, y in zip(T, Y):
            a1 = net.real_event(t, y); a2 = ref.step(t, y)
            if a1 is not None:
                ll_net += a1; ll_ref += a2; n += 1; maxdiff = max(maxdiff, abs(a1 - a2))
        row = {"day": str(day), "frozen": bool(a.freeze_after and di >= a.freeze_after), "events": n,
               "ll_event_network": ll_net / n, "ll_table": ll_ref / n, "max_abs_diff": maxdiff,
               "network_events_per_real_event": (net.net_events - e0) / n, "synaptic_ops_per_real_event": (net.syn_ops - s0) / n}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"event_world_f{a.freeze_after}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
