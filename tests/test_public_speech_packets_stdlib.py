"""Input information, split separation and pre-import admission contracts."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from public_speech_packets import CONTENT_DIM, load_rows, packetize, split_ids
import public_speech_admission as admission
from public_speech_admission import preflight


class PacketContracts(unittest.TestCase):
    def test_every_channel_and_count_are_preserved(self):
        events = packetize([0] * 700 + [15999, 16000], list(range(700)) + [699, 0])
        self.assertEqual(len(events), 3)
        first, second, query = events
        self.assertEqual(first[0], 1.)
        self.assertEqual(second[0], 2.)
        self.assertEqual(len(first[1]), CONTENT_DIM)
        self.assertEqual([round(math.expm1(x)) for x in first[1][:700]], [1] * 699 + [2])
        self.assertEqual(round(math.expm1(second[1][0])), 1)
        self.assertAlmostEqual(first[1][1399], 15999 / (2 * 16000) - .5)
        self.assertEqual(query[1], [0.] * 1400 + [1.])

    def test_causal_prefix_is_invariant_to_future_spikes(self):
        before = packetize([1, 15000], [0, 699])
        after = packetize([1, 15000, 16000, 1900000], [0, 699, 16, 17])
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[-1], after[-1])

    def test_no_empty_periodic_packets_or_sample_dependent_deadline(self):
        self.assertEqual(len(packetize([], [])), 1)
        self.assertEqual(packetize([], [])[0][0], 125.)
        self.assertEqual([t for t, _ in packetize([0, 1999999], [1, 2])], [1., 125., 125.])

    def test_centroid_distinguishes_timing_but_not_higher_moments(self):
        self.assertNotEqual(packetize([100], [4]), packetize([10000], [4]))
        self.assertEqual(packetize([100, 900], [4, 4]), packetize([300, 700], [4, 4]))

    def test_no_silent_crop_or_channel_coalescing(self):
        cases = [([-1], [0]), ([2000000], [0]), ([1.], [0]), ([True], [0]),
                 ([1], [-1]), ([1], [700]), ([1], [False]), ([1, 2], [0])]
        for times, channels in cases:
            with self.subTest(times=times, channels=channels), self.assertRaises(ValueError):
                packetize(times, channels)
        with self.assertRaises(ValueError):
            packetize([], [], 17000)

    def test_train_speaker_separation_and_fixed_sampling(self):
        speakers = [0, 3, 1, 6, 2, 3, 4, 6]
        fit, dev = split_ids(speakers, 3, 3)
        self.assertTrue(all(speakers[i] not in (3, 6) for i in fit))
        self.assertTrue(all(speakers[i] in (3, 6) for i in dev))
        self.assertFalse(set(fit) & set(dev))
        self.assertEqual((fit, dev), split_ids(speakers, 3, 3))
        with self.assertRaises(ValueError):
            split_ids(speakers, 5, 3)

    def test_official_test_and_changed_training_file_reject_before_hdf5_import(self):
        with self.assertRaises(ValueError):
            load_rows('/does/not/exist/shd_test.h5', 1, 1, 16000, 'bad')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'shd_train.h5'
            path.write_bytes(b'not a numerical dataset')
            with self.assertRaises(ValueError):
                load_rows(path, 1, 1, 16000, 'bad')
        self.assertNotIn('h5py', sys.modules)


class AdmissionContracts(unittest.TestCase):
    def manifest(self):
        return dict(status='prepared_unrun', source_sha256={}, stages={
            'contracts': dict(tag='contracts', output='unused.json', requires=[]),
            'smoke': dict(tag='smoke', output='unused2.json', requires=['contracts'])})

    def test_mutated_manifest_and_source_reject_without_numerical_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            record = self.manifest()
            path.write_text(json.dumps(record))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            preflight(path, 'contracts', digest)
            record['stages']['contracts']['tag'] = 'changed-settings'
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                preflight(path, 'contracts', digest)
            record['source_sha256'] = {'experiments/public_speech_packets.py': 'bad'}
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                preflight(path, 'contracts')
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_missing_prerequisite_and_bad_stage_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            path.write_text(json.dumps(self.manifest()))
            with self.assertRaises(FileNotFoundError):
                preflight(path, 'smoke')
            with self.assertRaises(ValueError):
                preflight(path, 'full')

    def test_unshared_container_rejects_before_any_numerical_import(self):
        record = self.manifest()
        record['admission'] = {'reject_unshared_container': True}
        with patch.object(admission, 'preflight', return_value=(record, {})), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(sys, 'argv', ['admission', '--manifest', 'unused', '--manifest-sha256',
                                         'unused', '--stage', 'contracts']):
            with self.assertRaisesRegex(ValueError, 'no physical-host reservation'):
                admission.main()
        self.assertNotIn('torch', sys.modules)

    def test_default_or_weakened_memory_guard_rejects(self):
        cfg = dict(min_available_mb=8192, timeout_s=600, rss_cap_kb=2000000, vms_cap_kb=6000000)
        with patch.dict(admission.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'frozen run_safe'):
                admission.require_guard_environment(cfg)
        with patch.dict(admission.os.environ, dict(MIN_AVAIL_MB='6000', JOB_TIMEOUT_S='600',
                MEM_CAP_RSS_KB='2000000', MEM_CAP_KB='6000000'), clear=True):
            with self.assertRaisesRegex(ValueError, 'frozen run_safe'):
                admission.require_guard_environment(cfg)


if __name__ == '__main__':
    unittest.main()
