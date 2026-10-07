"""Check support-board initialization against the bundled P6 photograph."""
import importlib.util
from pathlib import Path
import unittest

import cv2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p6_measurement', ROOT / 'easy/P6/evaluator/measure_backend.py')
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class P6FirstFrameTests(unittest.TestCase):
    def test_initial_board_edges_match_photograph_and_not_background_table(self):
        image = cv2.imread(str(ROOT / 'easy/P6/first_frame.png'))
        rois, source = measurement.load_rois()
        self.assertEqual(source, 'annotated')
        h, w = image.shape[:2]
        roi = measurement.roi_pixels(rois['board'], w, h)
        # Check the supplied PNG and a lossy encoding as used in generated clips.
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(ok)
        for frame in (image, cv2.imdecode(encoded, cv2.IMREAD_COLOR)):
            board = measurement.fit_board(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), roi)
            self.assertIsNotNone(board)
            self.assertAlmostEqual(board['top_intercept'], 460, delta=6)
            self.assertAlmostEqual(board['bottom_intercept'], 479, delta=6)
            self.assertLess(board['thickness_px'], 28)


if __name__ == '__main__':
    unittest.main()
