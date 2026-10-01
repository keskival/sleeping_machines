"""Causal interleaved streams; targets never enter model inputs."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Event:
    source: int
    time: float
    mark: tuple
    target: int | None = None


def episodes(task, sources, targets, seed, stretch=1.):
    if task not in ('order','timing') or targets < sources or targets % sources or stretch <= 0:
        raise ValueError('Known task, whole source populations and positive time scale required')
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(targets//sources):
        values = rng.uniform(-1.,1.,(sources,3)) if task=='timing' else rng.choice((-1.,1.),(sources,3))
        times = (np.arange(3)[None,:]+.1+rng.uniform(0,.25,(sources,3)))*stretch
        queries = (3.1+rng.uniform(0,2.,sources))*stretch
        events = []
        for source in range(sources):
            for j in range(3):
                events.append(Event(source,float(times[source,j]),(float(values[source,j]),0.)))
            if task=='order':
                target = int(values[source,1]>0)*2+int(values[source,2]>0)
            else:
                ages = queries[source]-times[source]
                latent = (values[source]*(np.exp(-ages/.7)-.6*np.exp(-ages/4.))).sum()
                target = int(latent>0)
            events.append(Event(source,float(queries[source]),(0.,1.),target))
        rows.append(sorted(events,key=lambda e:e.time))
    return rows


def data_hash(rows):
    return hashlib.sha256(json.dumps([[vars(e) for e in row] for row in rows],
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()
