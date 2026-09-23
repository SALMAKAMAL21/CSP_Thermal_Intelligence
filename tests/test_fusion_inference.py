"""Contrats ROI, normalisation et sorties ordinales sans dépendance réseau."""
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from src.inference.fusion_anomaly import FusionPredictor, crop_box, temperature_delta

ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.predictor = FusionPredictor(ROOT / "src/models/best_fusion_model.pt", ROOT / "configs/fusion.json", "cpu")

    def test_signed_delta_and_missing_values(self):
        self.assertEqual(temperature_delta(70, 100), -30)
        for value in (None, math.nan, math.inf):
            with self.assertRaises(ValueError):
                temperature_delta(value, 100)

    def test_crop_bounds_no_mask_no_padding(self):
        image = np.arange(10 * 20 * 3).reshape(10, 20, 3)
        np.testing.assert_array_equal(crop_box(image, [-2, 2, 8, 15]), image[2:10, 0:8])
        for bbox in ([5, 5, 1, 1], [0, 0, math.nan, 8], None, [40, 0, 50, 10]):
            self.assertIsNone(crop_box(image, bbox))

    def test_missing_pair_is_not_normal(self):
        result = self.predictor.predict(np.zeros((50, 100, 3), dtype=np.uint8), [], 100, 70)
        self.assertEqual(result["status"], "insufficient_tubes")
        self.assertNotIn("level", result)

    def test_input_order_normalization_and_ordinal_output(self):
        # Spy captures the actual tensors supplied to the network.
        class Recorder(torch.nn.Module):
            def forward(self, test, ref, delta):
                self.inputs = (test, ref, delta)
                return torch.tensor([[4., 2., -1., -2.]])
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        image[:, :50] = 255
        detections = [
            {"class_name": "tube_test", "confidence": 0.9, "bbox": [0, 0, 50, 50]},
            {"class_name": "tube_ref", "confidence": 0.9, "bbox": [50, 0, 100, 50]},
        ]
        model = self.predictor.model
        recorder = Recorder()
        self.predictor.model = recorder
        try:
            result = self.predictor.predict(image, detections, 70, 100)
        finally:
            self.predictor.model = model
        test, ref, delta = recorder.inputs
        self.assertEqual(tuple(test.shape), (1, 3, 224, 224))
        self.assertGreater(test.mean().item(), ref.mean().item())
        self.assertAlmostEqual(delta.item(), (-30 - self.predictor.mean) / self.predictor.std, places=5)
        self.assertEqual(result["level"], 2)
        self.assertEqual(result["delta_t"], -30)
        self.assertEqual(len(result["cumulative_probabilities"]), 4)


if __name__ == "__main__":
    unittest.main()
