import struct

import numpy as np

from experiments.e120_dvs_adapter import recording


def test_full_gesture_keeps_evidence_after_first_second(tmp_path):
    path = tmp_path / "user01_test.aedat"
    path.with_name(path.stem + "_labels.csv").write_text(
        "class,startTime_usec,endTime_usec\n1,0,2000000\n")
    events = np.array([[1, 100000], [1, 1200000]], dtype="<u4")
    header = struct.pack("<hhiiiiii", 1, 0, 8, 4, 0, 2, 2, 2)
    path.write_bytes(b"#!AER-DAT3.1\n#!END-HEADER\n" + header + events.tobytes())
    short, _ = recording(path)
    full, _ = recording(path, observation_seconds=None)
    def observed(row):
        return row.prefix.counts[row.prefix.channels != 32].sum()
    assert observed(short[0]) == 1
    assert observed(full[0]) == 2
    assert full[0].prefix.times[-1] == 2
