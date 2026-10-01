"""Timing-only label pairs with identical marks, addresses and event order."""
import numpy as np

from experiments.native_event_tasks import Event


def timing_label(marks, query, times):
    age = query - times
    return int(np.sum(marks * (np.exp(-age/.7) - .6*np.exp(-age/4.))) > 0)


def paired_timing_episodes(sources, targets, seed):
    if sources < 1 or targets < 2*sources or targets % (2*sources):
        raise ValueError('Whole short/long pairs of source populations required')
    rng = np.random.default_rng(seed)
    rows = []
    offsets = np.arange(sources) * (.05/sources)
    for _ in range(targets // (2*sources)):
        marks = []
        for source in range(sources):
            times = np.arange(3, dtype=float) + offsets[source]
            for attempt in range(10000):
                candidate = rng.uniform(-1., 1., 3)
                short = timing_label(candidate, 2.2+offsets[source], times)
                long = timing_label(candidate, 8.2+offsets[source], times)
                if short != long:
                    marks.append(candidate)
                    break
            else:
                raise RuntimeError('Bounded opposite-label rejection exhausted')
        for query in (2.2, 8.2):
            events = []
            for source in range(sources):
                times = np.arange(3, dtype=float)+offsets[source]
                for index, timestamp in enumerate(times):
                    events.append(Event(source, float(timestamp), (float(marks[source][index]), 0.)))
                q = query+offsets[source]
                events.append(Event(source, float(q), (0., 1.), timing_label(marks[source], q, times)))
            rows.append(sorted(events, key=lambda e:e.time))
    return rows


def pair_bootstrap(episode_accuracy, seed=419):
    """Cluster whole timing pairs; do not treat their members as independent."""
    scores = np.asarray(episode_accuracy, dtype=float)
    if len(scores) % 2:
        raise ValueError('Paired episodes required')
    scores = scores.reshape(-1, 2).mean(1)
    if len(scores) < 2:
        return None
    draws = np.random.default_rng(seed).choice(scores, (1000, len(scores)), replace=True).mean(1)
    return np.quantile(draws, (.025, .975)).tolist()
