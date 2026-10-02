"""Joint text + irregular-event task on one persistent address (THEORY §394).

One stream per episode; every event has a physical timestamp and a 32-wide content vector:
characters 0..26 (text8 alphabet), marks 27..30 (four event types), query flag 31.

  phase A   background marked events
  phase T   spelled question "[filler ]is <word> recent" (word names a mark), characters at irregular
            intervals, with some background events interleaved
  phase B   the last event of the named mark, then k in {0..3} distractor events of other marks
  query     label 1 iff (query time - last named-mark time) < DELTA

Offsets: [.3, .95]*DELTA for label 1 and [1.05, 2.5]*DELTA for label 0; labels balanced; k and text placement
are drawn independently of the label, so time-blind and rank-only information is at chance by construction.
Targets never enter inputs.
"""
import hashlib
import json

import numpy as np

from native_event_tasks import Event

WORDS = ('red', 'green', 'blue', 'gold')
FILLERS = ('now', 'tell', 'me', 'say')
DELTA = 4.0
WIDTH = 32
MARK0, QUERY = 27, 31


def char_index(c):
    return 0 if c == ' ' else ord(c) - 96


def one_hot(i):
    v = [0.] * WIDTH; v[i] = 1.
    return tuple(v)


def episode(rng, label=None):
    label = int(rng.integers(2)) if label is None else label
    named = int(rng.integers(4))
    events, t = [], 0.
    for _ in range(int(rng.integers(2, 6))):                       # phase A: background marks
        t += rng.uniform(.3, 2.); events.append(Event(0, t, one_hot(MARK0 + int(rng.integers(4)))))
    words = [str(rng.choice(FILLERS))] if rng.random() < .5 else []
    text = ' '.join(words + ['is', WORDS[named], 'recent'])
    for c in text:                                                  # phase T: characters, some events between
        t += rng.uniform(.05, .3); events.append(Event(0, t, one_hot(char_index(c))))
        if rng.random() < .08:
            t += rng.uniform(.02, .2); events.append(Event(0, t, one_hot(MARK0 + int(rng.integers(4)))))
    t += rng.uniform(.3, 2.)                                        # phase B: last named mark, then distractors
    last = t; events.append(Event(0, t, one_hot(MARK0 + named)))
    offset = DELTA * (rng.uniform(.3, .95) if label else rng.uniform(1.05, 2.5))
    k = int(rng.integers(4))
    others = [m for m in range(4) if m != named]
    for u in np.sort(rng.uniform(0, 1, k)):
        events.append(Event(0, last + offset * (.05 + .9 * u), one_hot(MARK0 + int(rng.choice(others)))))
    events.append(Event(0, last + offset, one_hot(QUERY), label))
    return events


def episodes(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % 2)                     # exactly balanced
    return [episode(rng, int(y)) for y in labels]


def data_hash(rows):
    return hashlib.sha256(json.dumps([[vars(e) for e in row] for row in rows], sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def blind_features(row):
    """time-blind, rank-level features a table could use: named word, last mark overall, events after named."""
    marks = [i for i, e in enumerate(row) if any(e.mark[MARK0:MARK0 + 4])]
    text = ''.join(' ' if e.mark[0] else chr(96 + e.mark.index(1.)) for e in row if 1. in e.mark[:27])
    named = next(w for w in WORDS if f' {w} ' in f' {text} ')
    last_mark = row[marks[-1]].mark.index(1.) - MARK0
    named_index = max(i for i in marks if row[i].mark.index(1.) - MARK0 == WORDS.index(named))
    after = sum(1 for i in marks if i > named_index)
    return (named, last_mark, after, len(row))
