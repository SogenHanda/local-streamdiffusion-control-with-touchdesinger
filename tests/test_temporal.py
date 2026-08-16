from __future__ import annotations

import unittest

import torch
from PIL import Image

from ergonomics_diffusion.config import AppConfig
from ergonomics_diffusion.engine import StreamDiffusionEngine


class TemporalFeedbackTests(unittest.TestCase):
    def make_engine(self, **overrides: float) -> StreamDiffusionEngine:
        config = AppConfig(**overrides)
        return StreamDiffusionEngine(config, lambda _message: None)

    def test_temporal_input_uses_camera_history_not_generated_output(self) -> None:
        engine = self.make_engine(temporal_feedback=0.4, scene_cut_threshold=0.35)
        current = Image.new("RGB", (32, 32), "black")
        engine._previous_input = current.copy()
        engine._previous_output = Image.new("RGB", (32, 32), "red")

        blended = engine._apply_temporal_feedback(current)

        self.assertAlmostEqual(engine.last_motion_score, 0.0)
        self.assertAlmostEqual(engine.last_temporal_feedback, 0.4)
        self.assertEqual(blended.getpixel((0, 0)), (0, 0, 0))

    def test_similar_input_blends_previous_camera_frame(self) -> None:
        engine = self.make_engine(temporal_feedback=0.4, scene_cut_threshold=1.0)
        current = Image.new("RGB", (32, 32), (100, 100, 100))
        engine._previous_input = Image.new("RGB", (32, 32), (120, 120, 120))
        engine._previous_output = Image.new("RGB", (32, 32), "red")

        blended = engine._apply_temporal_feedback(current)

        self.assertGreater(engine.last_temporal_feedback, 0.35)
        red, green, blue = blended.getpixel((0, 0))
        self.assertEqual(red, green)
        self.assertEqual(green, blue)
        self.assertGreater(red, 100)

    def test_scene_change_disables_feedback_and_smoothing(self) -> None:
        engine = self.make_engine(
            temporal_feedback=0.4,
            temporal_smoothing=0.2,
            scene_cut_threshold=0.1,
        )
        engine._previous_input = Image.new("RGB", (32, 32), "black")
        engine._previous_output = Image.new("RGB", (32, 32), "black")
        current = Image.new("RGB", (32, 32), "white")

        temporal_input = engine._apply_temporal_feedback(current)
        smoothed = engine._smooth_output(current)

        self.assertEqual(temporal_input.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(smoothed.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(engine.last_temporal_feedback, 0.0)

    def test_output_smoothing_blends_previous_result(self) -> None:
        engine = self.make_engine(temporal_smoothing=0.2)
        engine._previous_output = Image.new("RGB", (32, 32), "white")
        current = Image.new("RGB", (32, 32), "black")

        smoothed = engine._smooth_output(current)

        self.assertEqual(smoothed.getpixel((0, 0)), (51, 51, 51))

    def test_generated_latent_history_morphs_without_rgb_feedback(self) -> None:
        engine = self.make_engine(
            latent_morph_strength=0.5,
            latent_history_frames=3,
            scene_cut_threshold=1.0,
        )
        engine.last_motion_score = 0.0

        first = engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))
        second = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))
        third = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))

        self.assertAlmostEqual(first.item(), 0.0)
        self.assertAlmostEqual(second.item(), 0.5)
        # History stores raw 0 and 1 features. Recency-weighted reference is 2/3.
        self.assertAlmostEqual(third.item(), 5 / 6, places=5)
        self.assertAlmostEqual(engine.last_latent_morph, 0.5)

    def test_latent_morph_is_motion_adaptive_and_history_is_bounded(self) -> None:
        engine = self.make_engine(
            latent_morph_strength=0.6,
            latent_history_frames=2,
            scene_cut_threshold=0.5,
        )
        engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))
        engine.last_motion_score = 0.25
        result = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))
        engine._stabilize_generated_latent(torch.full((1, 1, 1, 1), 2.0))

        self.assertAlmostEqual(engine.last_latent_morph, 0.45)
        self.assertAlmostEqual(result.item(), 0.55)
        self.assertEqual(len(engine._latent_history), 2)

    def test_scene_change_discards_generated_latent_history(self) -> None:
        engine = self.make_engine(latent_morph_strength=0.6, latent_history_frames=3)
        engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))
        engine._scene_cut_detected = True

        result = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))

        self.assertAlmostEqual(result.item(), 1.0)
        self.assertAlmostEqual(engine.last_latent_morph, 0.0)
        self.assertEqual(len(engine._latent_history), 1)

if __name__ == "__main__":
    unittest.main()
