from __future__ import annotations

import unittest

import torch
from PIL import Image

from streamdiffusion_bridge.config import AppConfig
from streamdiffusion_bridge.engine import StreamDiffusionEngine


class TemporalFeedbackTests(unittest.TestCase):
    def make_engine(self, **overrides: float) -> StreamDiffusionEngine:
        config = AppConfig(**overrides)
        return StreamDiffusionEngine(config, lambda _message: None)

    def test_temporal_input_uses_camera_history_not_generated_output(self) -> None:
        engine = self.make_engine(temporal_feedback=0.4, scene_cut_threshold=0.35)
        current = Image.new("RGB", (32, 32), "black")
        engine._previous_input = current.copy()

        blended = engine._apply_temporal_feedback(current)

        self.assertAlmostEqual(engine.last_motion_score, 0.0)
        self.assertAlmostEqual(engine.last_temporal_feedback, 0.4)
        self.assertEqual(blended.getpixel((0, 0)), (0, 0, 0))

    def test_similar_input_blends_previous_camera_frame(self) -> None:
        engine = self.make_engine(temporal_feedback=0.4, scene_cut_threshold=1.0)
        current = Image.new("RGB", (32, 32), (100, 100, 100))
        engine._previous_input = Image.new("RGB", (32, 32), (120, 120, 120))

        blended = engine._apply_temporal_feedback(current)

        self.assertGreater(engine.last_temporal_feedback, 0.35)
        red, green, blue = blended.getpixel((0, 0))
        self.assertEqual(red, green)
        self.assertEqual(green, blue)
        self.assertGreater(red, 100)

    def test_large_motion_reduces_feedback_without_blending_a_new_image(self) -> None:
        engine = self.make_engine(
            temporal_feedback=0.4,
            temporal_smoothing=0.2,
            scene_cut_threshold=0.1,
        )
        engine._previous_input = Image.new("RGB", (32, 32), "black")
        current = Image.new("RGB", (32, 32), "white")

        temporal_input = engine._apply_temporal_feedback(current)
        smoothed = engine._smooth_output(current)

        self.assertEqual(temporal_input.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(smoothed.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(engine.last_temporal_feedback, 0.0)

    def test_zero_motion_threshold_is_valid_and_disables_moving_history(self) -> None:
        engine = self.make_engine(
            temporal_feedback=0.4,
            latent_morph_strength=1.0,
            scene_cut_threshold=0.0,
        )
        engine._previous_input = Image.new("RGB", (32, 32), "black")
        current = Image.new("RGB", (32, 32), "white")

        result = engine._apply_temporal_feedback(current)

        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(engine.last_temporal_feedback, 0.0)

    def test_output_smoothing_rejects_large_changes_to_prevent_ghosts(self) -> None:
        engine = self.make_engine(temporal_smoothing=0.5)
        engine._previous_raw_output = Image.new("RGB", (32, 32), "white")
        current = Image.new("RGB", (32, 32), "black")

        smoothed = engine._smooth_output(current)

        self.assertEqual(smoothed.getpixel((0, 0)), (0, 0, 0))

    def test_output_smoothing_reduces_small_flicker(self) -> None:
        engine = self.make_engine(temporal_smoothing=0.5)
        engine._previous_raw_output = Image.new("RGB", (32, 32), (110, 110, 110))
        current = Image.new("RGB", (32, 32), (100, 100, 100))

        smoothed = engine._smooth_output(current)

        value = smoothed.getpixel((0, 0))[0]
        self.assertGreater(value, 100)
        self.assertLess(value, 110)

    def test_generated_latent_history_morphs_without_rgb_feedback(self) -> None:
        engine = self.make_engine(
            latent_morph_strength=0.5,
            latent_history_frames=3,
            scene_cut_threshold=1.0,
        )
        engine.last_motion_score = 0.0

        first = engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))
        second = engine._stabilize_generated_latent(torch.full((1, 1, 1, 1), 0.05))
        third = engine._stabilize_generated_latent(torch.full((1, 1, 1, 1), 0.05))

        self.assertAlmostEqual(first.item(), 0.0)
        self.assertGreater(second.item(), 0.0)
        self.assertLess(second.item(), 0.05)
        self.assertGreater(third.item(), second.item())
        self.assertLess(third.item(), 0.05)
        self.assertAlmostEqual(engine.last_latent_morph, 0.5)

    def test_latent_history_morphs_a_structural_change_when_camera_is_still(self) -> None:
        engine = self.make_engine(
            latent_morph_strength=0.8,
            latent_history_frames=8,
            scene_cut_threshold=1.0,
        )
        engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))

        result = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))

        # A changed generation now transitions even when its latent delta is large.
        self.assertGreater(result.item(), 0.5)
        self.assertLess(result.item(), 0.9)

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

        self.assertAlmostEqual(engine.last_latent_morph, 0.3)
        self.assertGreater(result.item(), 0.8)
        self.assertLessEqual(len(engine._latent_history), 64)

    def test_large_motion_does_not_discard_generated_latent_history(self) -> None:
        engine = self.make_engine(
            latent_morph_strength=0.6,
            latent_history_frames=3,
            scene_cut_threshold=0.1,
        )
        engine._stabilize_generated_latent(torch.zeros((1, 1, 1, 1)))
        engine.last_motion_score = 1.0

        result = engine._stabilize_generated_latent(torch.ones((1, 1, 1, 1)))

        self.assertAlmostEqual(result.item(), 1.0)
        self.assertAlmostEqual(engine.last_latent_morph, 0.0)
        self.assertEqual(len(engine._latent_history), 2)

    def test_manual_temporal_reset_does_not_zero_stream_pipeline_buffer(self) -> None:
        engine = self.make_engine()

        class StreamBuffer:
            x_t_latent_buffer = torch.ones((1, 1, 1, 1))
            stock_noise = torch.ones((1, 1, 1, 1))

        engine.stream = StreamBuffer()
        engine.reset_temporal(log=False)

        self.assertEqual(engine.stream.x_t_latent_buffer.item(), 1.0)
        self.assertEqual(engine.stream.stock_noise.item(), 1.0)

if __name__ == "__main__":
    unittest.main()
