"""Exact first-crossing readout for sparse event-to-class evidence updates.

This small reference implementation accepts class-logit increments emitted by
an upstream event model. It maintains the maximum logit and log-normalizer in
indexed trees, so a sparse update can check a softmax-confidence threshold
without rescanning all classes. On emission it returns the full posterior
vector; that materialization is necessarily O(number_of_classes).

The class probabilities are only calibrated posteriors if the upstream scores
and temperature have been calibrated for prefixes under the deployed stopping
policy. This module computes confidence exactly from its current logits; it
does not establish statistical calibration. It also deliberately has no
per-class time decay: such a transform would invalidate sparse point updates.
It is a portable reference, not a performance-optimized device kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


@dataclass(frozen=True)
class Decision:
    """One immutable output, with the complete probability-vector payload."""

    class_id: int
    time: float
    posterior: tuple[float, ...]
    reason: str  # "threshold" or "end_of_stream"


class SparseAnytimeReadout:
    """Maintain exact softmax confidence under sparse additive logit updates.

    Call ``update(time, deltas)`` when evidence changes. ``deltas`` is an
    iterable of ``(class_id, logit_increment)`` and may represent observed
    messages or a scheduled no-event/survival update from a point-process
    model. Repeated classes are combined with ``math.fsum`` before the state
    changes, so message serialization order does not change the intended update.
    The first update whose posterior maximum reaches ``threshold`` returns a
    ``Decision`` immediately. If no update crosses, ``finish(time)`` returns
    the terminal posterior as a full-coverage fallback. The caller must
    schedule any between-event threshold crossing caused by class-dependent
    silence evidence; this head does not schedule future crossings itself.

    Costs: O(r log C) per update affecting r distinct classes, O(C) to emit
    the full posterior payload, and O(C) state. Confidence is the exact
    temperature-scaled softmax confidence of the accumulated logits, subject
    to floating-point arithmetic.
    """

    def __init__(self, n_classes: int, threshold: float, temperature: float = 1.0,
                 initial_logits: Iterable[float] | None = None):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not math.isfinite(threshold) or not (1.0 / n_classes < threshold <= 1.0):
            raise ValueError("threshold must be in (1 / n_classes, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        if initial_logits is None:
            scores = [0.0] * n_classes
        else:
            scores = [float(x) for x in initial_logits]
            if len(scores) != n_classes or any(not math.isfinite(x) for x in scores):
                raise ValueError("initial_logits must contain n_classes finite values")
        if any(not math.isfinite(x / temperature) for x in scores):
            raise ValueError("temperature-scaled initial logits must remain finite")

        self.n_classes = int(n_classes)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self._scores = scores
        self._size = 1 << (n_classes - 1).bit_length()
        self._max_tree = [-math.inf] * (2 * self._size)
        self._lse_tree = [-math.inf] * (2 * self._size)
        for c, score in enumerate(scores):
            leaf = self._size + c
            self._max_tree[leaf] = score
            self._lse_tree[leaf] = score / self.temperature
        for node in range(self._size - 1, 0, -1):
            left, right = 2 * node, 2 * node + 1
            self._max_tree[node] = max(self._max_tree[left], self._max_tree[right])
            self._lse_tree[node] = _logaddexp(self._lse_tree[left], self._lse_tree[right])

        self._last_time = -math.inf
        self._decision: Decision | None = None
        if (self.threshold < 1.0 and
                self._max_tree[1] / self.temperature - self._lse_tree[1]
                >= math.log(self.threshold)):
            # A sufficiently concentrated prior is already an answer at t=0.
            self._last_time = 0.0
            self._make_decision(0.0, "threshold")

    @property
    def decision(self) -> Decision | None:
        """The immutable output, if the prior or an event has crossed threshold."""
        return self._decision

    def _update_leaf(self, class_id: int) -> None:
        node = self._size + class_id
        self._max_tree[node] = self._scores[class_id]
        self._lse_tree[node] = self._scores[class_id] / self.temperature
        node //= 2
        while node:
            left, right = 2 * node, 2 * node + 1
            self._max_tree[node] = max(self._max_tree[left], self._max_tree[right])
            self._lse_tree[node] = _logaddexp(self._lse_tree[left], self._lse_tree[right])
            node //= 2

    def posterior(self) -> tuple[float, ...]:
        """Materialize the current full class-probability vector in O(C)."""
        log_z = self._lse_tree[1]
        return tuple(math.exp(score / self.temperature - log_z) for score in self._scores)

    def confidence(self) -> float:
        """Return the exact maximum class probability in O(1), without materializing it."""
        return math.exp(self._max_tree[1] / self.temperature - self._lse_tree[1])

    def _make_decision(self, time: float, reason: str) -> Decision:
        posterior = self.posterior()
        # max() returns the first class on a tie, giving stable class IDs.
        decision = Decision(max(range(self.n_classes), key=posterior.__getitem__),
                            time, posterior, reason)
        self._decision = decision
        return decision

    def update(self, time: float, deltas: Iterable[tuple[int, float]]) -> Decision | None:
        """Apply all simultaneous sparse evidence increments at ``time``."""
        t = float(time)
        if not math.isfinite(t) or t < self._last_time:
            raise ValueError("event times must be finite and nondecreasing")
        if self._decision is not None:
            return self._decision

        by_class: dict[int, list[float]] = {}
        for class_id, delta in deltas:
            c, d = int(class_id), float(delta)
            if c < 0 or c >= self.n_classes:
                raise ValueError(f"class_id {c} is outside [0, {self.n_classes})")
            if not math.isfinite(d):
                raise ValueError("logit increments must be finite")
            by_class.setdefault(c, []).append(d)

        staged = []
        for class_id, values in by_class.items():
            updated = self._scores[class_id] + math.fsum(values)
            if not math.isfinite(updated) or not math.isfinite(updated / self.temperature):
                raise ValueError("updated logits must remain finite")
            staged.append((class_id, updated))

        # Validate the complete update before mutating state, so bad input
        # cannot leave a partially applied simultaneous event batch.
        for class_id, updated in staged:
            self._scores[class_id] = updated
            self._update_leaf(class_id)
        self._last_time = t

        log_max_probability = (self._max_tree[1] / self.temperature
                               - self._lse_tree[1])
        if self.threshold < 1.0 and log_max_probability >= math.log(self.threshold):
            return self._make_decision(t, "threshold")
        return None

    def finish(self, time: float | None = None) -> Decision:
        """Emit the terminal posterior if the stream ended before a crossing."""
        t = (0.0 if self._last_time == -math.inf else self._last_time) if time is None else float(time)
        if not math.isfinite(t) or t < self._last_time:
            raise ValueError("end time must be finite and no earlier than the last event")
        if self._decision is not None:
            return self._decision
        self._last_time = t
        return self._make_decision(t, "end_of_stream")
