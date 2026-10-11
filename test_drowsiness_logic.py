import unittest

from drowsiness_logic import ClosureTracker, eye_aspect_ratio


class EyeAspectRatioTests(unittest.TestCase):
    def test_open_eye_has_higher_ratio_than_closed_eye(self):
        open_eye = [(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)]
        closed_eye = [(0, 0), (1, -0.1), (3, -0.1), (4, 0), (3, 0.1), (1, 0.1)]

        self.assertGreater(eye_aspect_ratio(open_eye), eye_aspect_ratio(closed_eye))

    def test_rejects_invalid_landmark_count(self):
        with self.assertRaises(ValueError):
            eye_aspect_ratio([(0, 0)] * 5)


class ClosureTrackerTests(unittest.TestCase):
    def test_alert_requires_continuous_closure(self):
        tracker = ClosureTracker(ear_threshold=0.2, closed_seconds=1.0)

        self.assertFalse(tracker.update(0.1, 10.0))
        self.assertFalse(tracker.update(0.1, 10.9))
        self.assertTrue(tracker.update(0.1, 11.0))

    def test_open_eye_resets_closure_timer(self):
        tracker = ClosureTracker(ear_threshold=0.2, closed_seconds=1.0)
        tracker.update(0.1, 10.0)

        self.assertFalse(tracker.update(0.3, 10.8))
        self.assertFalse(tracker.update(0.1, 11.5))

    def test_duration_is_zero_when_eyes_are_open(self):
        tracker = ClosureTracker()
        tracker.update(0.1, 4.0)
        tracker.update(0.3, 4.5)

        self.assertEqual(tracker.duration(5.0), 0.0)


if __name__ == "__main__":
    unittest.main()