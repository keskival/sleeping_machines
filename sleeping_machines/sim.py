"""Discrete-event engine for the Sleeping Machines experiments.

Time jumps straight to the next pending event; there is no global clock.
Pending events are part of machine state and can be cancelled before they fire.

Every operation that would cost energy on event-driven hardware is counted in
`Engine.work`, so dynamic-work results are measured from the run itself rather
than derived from a closed-form estimate.
"""
import heapq
import itertools
from collections import Counter


class Event:
    __slots__ = ("t", "seq", "kind", "data", "alive")

    def __init__(self, t, seq, kind, data):
        self.t, self.seq, self.kind, self.data, self.alive = t, seq, kind, data, True

    def __lt__(self, other):
        return (self.t, self.seq) < (other.t, other.seq)


class Engine:
    def __init__(self):
        self.now = 0.0
        self.work = Counter()
        self._queue = []
        self._seq = itertools.count()
        self._handlers = {}

    def on(self, kind, handler):
        self._handlers[kind] = handler

    def schedule(self, delay, kind, data=None):
        """Put a future event into the pending pool; returns a handle for cancellation."""
        if delay < 0:
            raise ValueError("causality: an event cannot be scheduled in the past")
        event = Event(self.now + delay, next(self._seq), kind, data)
        heapq.heappush(self._queue, event)
        self.work["scheduled"] += 1
        return event

    def cancel(self, event):
        """Remove a pending event from the future. Its record stays with whoever holds the handle."""
        if event.alive:
            event.alive = False
            self.work["cancelled"] += 1

    def count(self, key, n=1):
        self.work[key] += n

    def run(self, until=float("inf")):
        while self._queue and self._queue[0].t <= until:
            event = heapq.heappop(self._queue)
            if not event.alive:
                continue
            event.alive = False
            self.now = event.t
            self.work["fired"] += 1
            self._handlers[event.kind](event.data)
        if until != float("inf"):
            self.now = max(self.now, until)
