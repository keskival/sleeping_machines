"""Local sufficient statistics and hard pointer races shared by task adapters.

Tables allocate only visited contexts. Call scores BEFORE observe for
prequential use, or freeze the table before evaluating held-out examples.
These are outcome-dense local tables; key lookup is sparse, not the readout.
"""
import numpy as np


class ConditionalEvidence:
    def __init__(self, outcomes, bins=0, alpha=.5, prior_exposure=.5, max_contexts=100_000):
        self.outcomes, self.bins = outcomes, bins
        self.alpha, self.prior_exposure = alpha, prior_exposure
        self.max_contexts = max_contexts
        self.counts, self.exposure = {}, {}
        self.lookups, self.updates = 0, 0

    def scores(self, key):
        self.lookups += 1
        shape = (self.bins, self.outcomes) if self.bins else (self.outcomes,)
        counts = self.counts.get(key, np.zeros(shape))
        if self.bins:
            spans = self.exposure.get(key, np.zeros(self.bins))
            return np.log((counts+self.alpha)/(spans[:, None]+self.prior_exposure))
        return np.log((counts+self.alpha)/(counts.sum()+self.alpha*self.outcomes))

    def observe(self, key, label, bucket=None, exposure=None):
        if not 0 <= label < self.outcomes:
            raise ValueError("Invalid outcome")
        if self.bins and (bucket is None or not 0 <= bucket < self.bins or
                          np.shape(exposure) != (self.bins,) or
                          not np.isfinite(exposure).all() or np.any(np.asarray(exposure) < 0)):
            raise ValueError("Invalid event exposure")
        if key not in self.counts:
            if len(self.counts) >= self.max_contexts:
                raise MemoryError("Evidence context budget exhausted")
            shape = (self.bins, self.outcomes) if self.bins else (self.outcomes,)
            self.counts[key] = np.zeros(shape)
            if self.bins:
                self.exposure[key] = np.zeros(self.bins)
        if self.bins:
            self.counts[key][bucket, label] += 1
            self.exposure[key] += exposure
        else:
            self.counts[key][label] += 1
        self.updates += 1

    def work(self):
        return {"contexts": len(self.counts), "lookups": self.lookups, "updates": self.updates,
                "state_bytes": sum(v.nbytes for v in self.counts.values()) +
                               sum(v.nbytes for v in self.exposure.values()),
                "bytes_note": "numeric arrays only; Python mapping overhead excluded"}


class RelativeRouteMemory:
    """E61 relative pointer primitive, independent of the meaning of symbols.

    A query chooses one (observed source symbol, relative offset) route. It
    reads exactly one destination symbol; losers receive mistake-only credit.
    Search still enumerates all present source/offset candidates: O(E R).
    """
    def __init__(self, queries, symbols, offsets=(-2, -1, 1, 2), alpha=1., beta=.5):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.alpha, self.beta = alpha, beta
        self.weights = np.full((queries, symbols, len(offsets)), 1/(symbols*len(offsets)))
        self.candidates, self.mistakes = 0, 0

    def read(self, symbols, query):
        destination = np.arange(len(symbols))[:, None] + self.offsets[None, :]
        valid = (destination >= 0) & (destination < len(symbols))
        if not valid.any():
            raise ValueError("No valid pointer routes")
        score = np.where(valid, self.weights[query, symbols], -1.)
        i, k = np.unravel_index(int(score.argmax()), score.shape)
        self.candidates += int(valid.sum())
        return int(symbols[destination[i, k]]), (int(symbols[i]), k)

    def observe(self, symbols, query, target_symbol):
        output, winner = self.read(symbols, query)
        if output == target_symbol:
            return
        self.mistakes += 1
        for p in np.flatnonzero(symbols == target_symbol):
            for k, offset in enumerate(self.offsets):
                source = p-offset
                if 0 <= source < len(symbols):
                    self.weights[query, symbols[source], k] *= 1+self.alpha
        self.weights[query, winner[0], winner[1]] *= 1-self.beta
        self.weights[query] /= self.weights[query].sum()
