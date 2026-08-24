from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergonomics_diffusion.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            expected = AppConfig(
                model_profile="sd-turbo",
                prompt="a test prompt",
                denoise_index=12,
                lcm_steps=2,
                temporal_feedback=0.42,
                temporal_smoothing=0.18,
                latent_morph_strength=0.4,
                latent_history_frames=4,
                spout_sample_fps=24,
                acceleration_backend="xformers",
            )
            expected.save(path)
            actual = AppConfig.load(path)
            self.assertEqual(actual.prompt, expected.prompt)
            self.assertEqual(actual.model_profile, expected.model_profile)
            self.assertEqual(actual.denoise_index, expected.denoise_index)
            self.assertEqual(actual.lcm_steps, expected.lcm_steps)
            self.assertEqual(actual.temporal_feedback, expected.temporal_feedback)
            self.assertEqual(actual.temporal_smoothing, expected.temporal_smoothing)
            self.assertEqual(actual.latent_morph_strength, expected.latent_morph_strength)
            self.assertEqual(actual.latent_history_frames, expected.latent_history_frames)
            self.assertEqual(actual.spout_sample_fps, expected.spout_sample_fps)
            self.assertEqual(actual.acceleration_backend, expected.acceleration_backend)

    def test_spout_names_must_be_different(self) -> None:
        config = AppConfig(spout_input="same", spout_output="same")
        with self.assertRaises(ValueError):
            config.validate()

    def test_resolution_must_be_multiple_of_eight(self) -> None:
        config = AppConfig(width=511)
        with self.assertRaises(ValueError):
            config.validate()

    def test_temporal_values_are_bounded(self) -> None:
        config = AppConfig(temporal_feedback=0.9)
        with self.assertRaises(ValueError):
            config.validate()

    def test_latent_history_values_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig(latent_morph_strength=1.01).validate()
        with self.assertRaises(ValueError):
            AppConfig(latent_history_frames=9).validate()

    def test_realtime_temporal_controls_accept_full_normalized_range(self) -> None:
        AppConfig(
            temporal_smoothing=1.0,
            latent_morph_strength=1.0,
            scene_cut_threshold=0.0,
        ).validate()
        with self.assertRaises(ValueError):
            AppConfig(temporal_smoothing=1.01).validate()
        with self.assertRaises(ValueError):
            AppConfig(scene_cut_threshold=-0.01).validate()

    def test_sd_turbo_cannot_use_sd15_lcm_lora(self) -> None:
        config = AppConfig(model_path="models/sd-turbo", use_lcm_lora=True)
        with self.assertRaises(ValueError):
            config.validate()

    def test_lcm_steps_must_be_one_or_two(self) -> None:
        config = AppConfig(lcm_steps=3)
        with self.assertRaises(ValueError):
            config.validate()

    def test_acceleration_and_spout_sampling_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig(acceleration_backend="unknown").validate()
        with self.assertRaises(ValueError):
            AppConfig(spout_sample_fps=0).validate()


if __name__ == "__main__":
    unittest.main()
