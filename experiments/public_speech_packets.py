"""Causal SHD packet adapter; standard library until explicit HDF5 loading.

No targets/IDs/speakers are arguments to packetize. Counts and first time moments
preserve every channel, but not all within-packet timing information. Times are
integer microseconds, packet contents are delivered at closure, silence emits
no periodic packet and the query deadline is fixed independently of the sample.
"""
import hashlib
import json
import math
from pathlib import Path
import random

CHANNELS = 700
CONTENT_DIM = 2 * CHANNELS + 1
DEADLINE_US = 2_000_000
DEV_SPEAKERS = (3, 6)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def packetize(times_us, channels, packet_us=16000, deadline_us=DEADLINE_US):
    if (type(packet_us) is not int or type(deadline_us) is not int or
            min(packet_us, deadline_us) <= 0 or deadline_us % packet_us):
        raise ValueError('Positive integer packet width dividing the fixed deadline required')
    if len(times_us) != len(channels):
        raise ValueError('One channel per observed spike required')
    bins = {}
    for stamp, channel in zip(times_us, channels):
        if type(stamp) is not int or not 0 <= stamp < deadline_us:
            raise ValueError('Out-of-range timestamp: no silent cropping')
        if type(channel) is not int or not 0 <= channel < CHANNELS:
            raise ValueError('700 distinct integer channel IDs required')
        bucket, offset = divmod(stamp, packet_us)
        counts, offsets = bins.setdefault(bucket, ([0] * CHANNELS, [0] * CHANNELS))
        counts[channel] += 1
        offsets[channel] += offset
    events = []
    for bucket, (counts, offsets) in sorted(bins.items()):
        # Shared immutable zero avoids allocating 700 new float objects for
        # each sparse packet; it does not change content or numerical work.
        mark = [math.log1p(n) if n else 0. for n in counts]
        mark += [offsets[j] / (n * packet_us) - .5 if n else 0.
                 for j, n in enumerate(counts)]
        mark += [0.]
        events.append((float(bucket + 1), mark))
    events.append((float(deadline_us // packet_us), [0.] * (CONTENT_DIM - 1) + [1.]))
    return events


def split_ids(speakers, limit_fit=None, limit_dev=None, data_seed=1201):
    """Speaker-disjoint train-only sets, fixed across optimizer seeds; no labels."""
    fit = [i for i, speaker in enumerate(speakers) if int(speaker) not in DEV_SPEAKERS]
    dev = [i for i, speaker in enumerate(speakers) if int(speaker) in DEV_SPEAKERS]
    random.Random(data_seed).shuffle(fit)
    random.Random(data_seed + 1).shuffle(dev)
    selected = []
    for indices, limit in ((fit, limit_fit), (dev, limit_dev)):
        if limit is not None and (type(limit) is not int or not 1 <= limit <= len(indices)):
            raise ValueError('Requested split size is unavailable')
        selected.append(indices if limit is None else indices[:limit])
    return tuple(selected)


def load_rows(path, fit, dev, packet_us, expected_sha):
    """Only the named official TRAIN file can be opened. Call in a guarded job."""
    path = Path(path)
    if path.name != 'shd_train.h5' or file_sha(path) != expected_sha:
        raise ValueError('Frozen official training file required')
    import h5py  # deliberately deferred; importing this module is stdlib-only
    rows = []
    with h5py.File(path, 'r') as data:
        if len(data['labels']) != 8156:
            raise ValueError('Expected the complete official SHD training population')
        speakers = [int(x) for x in data['extra']['speaker']]
        ids = split_ids(speakers, fit, dev)
        for population in ids:
            converted = []
            for index in population:
                times = [float(x) for x in data['spikes']['times'][index]]
                channels = [int(x) for x in data['spikes']['units'][index]]
                if any(not math.isfinite(t) or not 0 <= t < DEADLINE_US / 1e6 for t in times):
                    raise ValueError('Out-of-deadline/nonfinite raw spike; no silent filtering')
                target = int(data['labels'][index])
                if not 0 <= target < 20:
                    raise ValueError('SHD class must be in 0..19')
                events = packetize([int(math.floor(t * 1e6)) for t in times], channels, packet_us)
                converted.append(dict(index=index, target=target, speaker=speakers[index],
                                      raw_spikes=len(times), events=events))
            rows.append(converted)
    encoded = hashlib.sha256(json.dumps(rows, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return *rows, dict(train_sha256=expected_sha, fit_ids=ids[0], dev_ids=ids[1],
                      encoded_sha256=encoded, packet_us=packet_us, deadline_us=DEADLINE_US,
                      classes=20, content_dim=CONTENT_DIM, official_test_read=False,
                      temporal_precision='Raw seconds floored to integer microsecond (less than 1us quantization)',
                      input='700 log-counts, 700 occupied-channel time centroids, query flag',
                      scope='Lossy causal packets; speaker3/6 development reused, not untouched confirmation')
