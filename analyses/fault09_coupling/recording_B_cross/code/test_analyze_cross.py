"""Small synthetic checks of pairing and inferential arithmetic; no study data."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_cross as ac


def fixture_rows():
    rows = []
    for role in ac.ROLES:
        for i, base in enumerate((30, 150, 600)):
            case = ("N" if role == ac.ROLES[0] else "F") + str(i)
            for estimator, view, pool in ac.ARM_ORDER:
                offset = (-12 if estimator == "batch" else 0) + (6 if view == "snapshot" else 0)
                offset += 9 if estimator == "batch" and view == "snapshot" else 0
                offset += -3 if pool == "moment" else 0
                loss = base + offset
                normal = role == ac.ROLES[0]
                rows.append(dict(case_id=case, role=role, estimator=estimator,
                                 recording=view, pool=pool, seed_u32=100 + i + (10 if normal else 0),
                                 normal_alarm=0 if normal else 1, pre_alarm=0,
                                 clean_120=0 if normal else int(loss <= 120),
                                 clean_480=0 if normal else int(loss <= 480),
                                 clean_2400=0 if normal else 1,
                                 loss_minutes=2400 if normal else loss))
    return rows


def write_fixture(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "synthetic_events.csv"
        write_fixture(self.path, fixture_rows())
        self.ids, self.arrays, _ = ac.load_events(self.path, expected_n=3)
        rng = np.random.default_rng(451)
        self.indices = {role: rng.integers(0, 3, size=(2000, 3)) for role in ac.ROLES}

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_arms_have_exact_zero_interval(self):
        arrays = {role: {k: np.repeat(v[:, :1], 8, axis=1) for k, v in self.arrays[role].items()}
                  for role in ac.ROLES}
        for row in ac.contrast_rows(arrays, self.indices):
            self.assertEqual(row["estimate"], 0.)
            self.assertEqual(row["ci95_low"], 0.)
            self.assertEqual(row["ci95_high"], 0.)

    def test_known_constant_loss_shift_and_interaction(self):
        rows = ac.contrast_rows(self.arrays, self.indices)
        loss = {r["contrast"]: r for r in rows if r["endpoint"] == "loss_minutes"}
        for key, truth in (("estimator_native_mean", -12.),
                           ("C2_original_snapshot", -3.),
                           ("interaction_recording_by_estimator_mean", 9.)):
            r = loss[key]
            self.assertAlmostEqual(r["estimate"], truth, places=12)
            self.assertAlmostEqual(r["ci95_low"], truth, places=12)
            self.assertAlmostEqual(r["ci95_high"], truth, places=12)

    def test_csv_order_does_not_change_pairing_or_results(self):
        rows = fixture_rows()
        np.random.default_rng(179).shuffle(rows)
        write_fixture(self.path, rows)
        ids, arrays, _ = ac.load_events(self.path, expected_n=3)
        self.assertEqual(ids, self.ids)
        for role in ac.ROLES:
            for k in ac.ENDPOINTS:
                np.testing.assert_array_equal(arrays[role][k], self.arrays[role][k])
        self.assertEqual(ac.contrast_rows(arrays, self.indices),
                         ac.contrast_rows(self.arrays, self.indices))

    def test_missing_case_in_one_arm_is_rejected(self):
        rows = fixture_rows()
        rows.pop(5)
        write_fixture(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Case identities differ"):
            ac.load_events(self.path, expected_n=3)

    def test_same_id_with_conflicting_seed_is_rejected(self):
        rows = fixture_rows()
        rows[5]["seed_u32"] += 1
        write_fixture(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Seed differs"):
            ac.load_events(self.path, expected_n=3)

    def test_duplicate_label_identity_is_rejected(self):
        rows = fixture_rows()
        rows.append(rows[5].copy())
        write_fixture(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate case/arm"):
            ac.load_events(self.path, expected_n=3)

    def test_contradictory_clean_flag_is_rejected(self):
        rows = fixture_rows()
        rows[-1]["pre_alarm"] = 1
        write_fixture(self.path, rows)
        with self.assertRaisesRegex(ValueError, "Clean detection with prefault alarm"):
            ac.load_events(self.path, expected_n=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
