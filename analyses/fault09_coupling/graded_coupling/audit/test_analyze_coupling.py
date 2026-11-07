"""Synthetic-only tests; no archived scientific score files are read."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import analyze_coupling as a


class CouplingAnalysisTests(unittest.TestCase):
    def test_constant_and_subreference_scores(self):
        values = np.zeros((3, 960, 6))
        values[0] = .05
        values[1] = -.05
        values[2] = .2
        charts = a.charts(values)
        self.assertTrue(np.array_equal(charts[:2], np.zeros_like(charts[:2])))
        # Closed-form exact-arithmetic value versus 960 floating additions/subtractions.
        rounding_bound = 960 * np.finfo(np.float64).eps * 150
        np.testing.assert_allclose(charts[2], np.arange(1, 961)[:, None] * .15 * np.ones((1, 6)), atol=rounding_bound, rtol=0)

    def test_event_boundaries_and_penalties(self):
        c = np.zeros((8, 960, 6))
        c[0, 160, :] = 2.          # first affected record => 0 min
        c[1, 320, :] = 2.          # inclusive 480 min
        c[2, 321, :] = 2.          # 483 min => outside 8 h
        c[3, 200, :] = 2.          # inclusive 120 min
        c[4, 159, :] = 2.          # prefault alarm disqualifies
        c[4, 160, :] = 2.
        c[5, :, :] = 1.            # equality never alarms
        c[6, 959, :] = 2.          # last observed postfault record
        c[7, 159, :] = 2.          # prefault only
        e = a.events(c, np.ones(6))
        np.testing.assert_array_equal(e["clean40"][:, 0], [1, 1, 1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(e["clean8"][:, 0], [1, 1, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(e["clean2"][:, 0], [1, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(e["loss_minutes"][:, 0], [0, 480, 483, 120, 2400, 2400, 2397, 2400])
        np.testing.assert_array_equal(e["pre_alarm"][:, 0], [0, 0, 0, 0, 1, 0, 0, 1])

    def test_paired_bootstrap_has_common_indices_and_correct_units(self):
        normal = {"any_alarm": np.zeros((3, 6), bool), "clean40": np.ones((3, 6), bool)}
        fault = {"clean40": np.ones((4, 6), bool), "clean8": np.zeros((4, 6), bool),
                 "clean2": np.zeros((4, 6), bool), "loss_minutes": np.tile(np.arange(6), (4, 1))}
        normal["any_alarm"][:, 0] = True
        fault["clean40"][:, 0] = False
        result = a.bootstrap(normal, fault, draws=37, seed=901)
        rng = np.random.default_rng(901)
        np.testing.assert_array_equal(result["normal_indices"], rng.integers(0, 3, (37, 3), dtype=np.int32))
        np.testing.assert_array_equal(result["fault_indices"], rng.integers(0, 4, (37, 4), dtype=np.int32))
        np.testing.assert_array_equal(result["distributions"][:, 0, 0], np.full(37, -100.))
        np.testing.assert_array_equal(result["distributions"][:, 0, 3], np.full(37, -4.))
        np.testing.assert_array_equal(result["distributions"][:, 0, 4], np.full(37, 100.))
        # Identical arms for all binary endpoints must have exactly zero paired uncertainty.
        np.testing.assert_array_equal(result["bounds"][:, 1, [0, 1, 2, 4, 5]], np.zeros((2, 5)))

    def test_pooled_variance_uses_all_samples_and_ddof1(self):
        normal = {"any_alarm": np.zeros((2, 6), bool), "clean40": np.zeros((2, 6), bool)}
        fault = {k: np.zeros((3, 6), bool) for k in ("clean40", "clean8", "clean2", "pre_alarm")}
        fault["loss_minutes"] = np.full((3, 6), 2400)
        scores = np.repeat(np.arange(1920, dtype=np.float64).reshape(2, 960, 1), 6, axis=2)
        rows = a.main_rows(normal, fault, scores, np.ones(6))
        self.assertEqual(rows[0]["normal_score_mean"], 959.5)
        expected_variance = 1920 * 1921 / 12  # unbiased variance of integers 0..1919
        self.assertAlmostEqual(rows[0]["normal_score_variance_ddof1"], expected_variance)
        self.assertEqual(rows[5]["rho"], "")

    def test_wilson_extremes(self):
        low, high = a.wilson(0, 200)
        self.assertEqual(low, 0.)
        self.assertAlmostEqual(high, 0.01884532637726658)
        low, high = a.wilson(200, 200)
        self.assertAlmostEqual(low, 1 - 0.01884532637726658)
        self.assertEqual(high, 1.)

    def test_calibration_lock_and_input_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="coupling-analysis-synthetic-") as temp:
            root = Path(temp)
            (root / "scores").mkdir()
            (root / "PROTOCOL.json").write_text('{"synthetic_fixture":true}', encoding="utf-8")
            (root / "CODE_LOCK.json").write_text('{"synthetic_fixture":true}', encoding="utf-8")
            rows = []
            ordinal = 0
            for role, count in a.COUNTS.items():
                for index in range(1, count + 1):
                    ordinal += 1
                    run_id = f"{role}_{index:03}"
                    rows.append({"run_id": run_id, "role": role, "fault_id": 9 if role == "fault_evaluation" else 0,
                                 "index": index, "ordinal": ordinal, "key_hex": f"0x{ordinal:032x}"})
                    if role == "normal_calibration":
                        # The 190th ascending maximum is the file with index 190.
                        score = .05 + index * .0001
                        a.write_npz(root / "scores" / (run_id + ".npz"), scores=np.full((960, 6), score), arm_names=np.asarray(a.ARMS))
            a.write_csv(root / "MANIFEST.csv", rows)
            with self.assertRaisesRegex(ValueError, "Calibrate and lock"):
                a.evaluate(root, make_figure=False)
            lock = a.calibrate(root)
            np.testing.assert_allclose(lock["thresholds"], np.full(6, 960 * .019), atol=5e-13, rtol=0)
            self.assertEqual(lock, a.calibrate(root))
            self.assertEqual(lock, a.verify_lock(root))
            file = root / "scores" / "normal_calibration_001.npz"
            original = file.read_bytes()
            file.write_bytes(original + b"altered")
            with self.assertRaisesRegex(ValueError, "changed after locking"):
                a.verify_lock(root)
            file.write_bytes(original)
            self.assertEqual(lock, a.verify_lock(root))
            lock_path = root / "results" / "THRESHOLDS_LOCK.json"
            lock_original = lock_path.read_bytes()
            altered_lock = json.loads(lock_original)
            altered_lock["thresholds"][0] += 1
            a.write_json(lock_path, altered_lock)
            with self.assertRaisesRegex(ValueError, "values differ from the verified calibration"):
                a.verify_lock(root)
            altered_lock["thresholds"][0] = float("inf")
            lock_path.write_text(json.dumps(altered_lock))
            with self.assertRaisesRegex(ValueError, "six finite thresholds"):
                a.verify_lock(root)
            altered_lock = json.loads(lock_original)
            altered_lock["calibration_score_files"][1]["run_id"] = altered_lock["calibration_score_files"][0]["run_id"]
            a.write_json(lock_path, altered_lock)
            with self.assertRaisesRegex(ValueError, "case order or identities"):
                a.verify_lock(root)
            lock_path.write_bytes(lock_original)
            self.assertEqual(lock, a.verify_lock(root))
            protocol = root / "PROTOCOL.json"
            protocol.write_text('{"synthetic_fixture":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding differs"):
                a.verify_lock(root)

    def test_original_calibration_column_mapping_and_exact_rejection(self):
        with tempfile.TemporaryDirectory(prefix="coupling-replay-synthetic-") as temp:
            root = Path(temp)
            (root / "reference").mkdir()
            # This fixture exercises the formal replay branch with invented arrays only.
            (root / "PROTOCOL.json").write_text('{"study_id":"SYNTHETIC_REPLAY"}')
            rows = [{"run_id": f"n{i:03}"} for i in range(199)]
            old_maxima = np.arange(199 * 22, dtype=np.float64).reshape(199, 22)
            old_h = np.sort(old_maxima, axis=0)[189]
            a.write_npz(root / "reference" / "CALIBRATION_MAXIMA.npz", maxima=old_maxima)
            a.write_json(root / "reference" / "THRESHOLDS_LOCK.json", {
                "n_calibration": 199, "order_1based": 190,
                "calibration_run_ids": [r["run_id"] for r in rows],
                "thresholds": old_h.tolist(),
                "calibration_maxima_sha256": a.sha(root / "reference" / "CALIBRATION_MAXIMA.npz")})
            new_peaks = np.zeros((199, 6))
            new_peaks[:, 4] = old_maxima[:, 21]
            new_peaks[:, 5] = old_maxima[:, 20]
            new_h = np.sort(new_peaks, axis=0)[189]
            self.assertEqual(a.original_calibration_replay(root, rows, new_peaks, new_h)["status"], "PASS")
            new_peaks[0, 4] = np.nextafter(new_peaks[0, 4], np.inf)
            with self.assertRaisesRegex(ValueError, "exact archived replay"):
                a.original_calibration_replay(root, rows, new_peaks, new_h)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CouplingAnalysisTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "synthetic_only": True,
              "real_scientific_scores_read": False, "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "analysis_code_sha256": a.sha(Path(a.__file__))}
    a.write_json(Path(__file__).with_name("SYNTHETIC_ANALYSIS_TESTS.json"), report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
