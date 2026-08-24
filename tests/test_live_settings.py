from __future__ import annotations

import unittest
from typing import Any

import torch
from PIL import Image

from ergonomics_diffusion.config import AppConfig
from ergonomics_diffusion.engine import StreamDiffusionEngine


class FakeImageProcessor:
    def preprocess(self, _image: Image.Image, height: int, width: int) -> torch.Tensor:
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


class FakeStream:
    def __init__(self) -> None:
        self.t_list = [32, 45]
        self.prepare_calls: list[dict[str, Any]] = []
        self.denoising_steps_num = 2
        self.image_processor = FakeImageProcessor()
        self.height = 8
        self.width = 8
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.x_t_latent_buffer = torch.ones((1, 4, 1, 1))
        self.prime_calls = 0

    def prepare(self, **kwargs: Any) -> None:
        self.prepare_calls.append(kwargs)
        self.x_t_latent_buffer.zero_()

    def encode_image(self, _image: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, 4, 1, 1))

    def predict_x0_batch(self, latent: torch.Tensor) -> torch.Tensor:
        self.prime_calls += 1
        self.x_t_latent_buffer.copy_(latent)
        return latent


class LiveSettingsTests(unittest.TestCase):
    def make_engine(self) -> tuple[StreamDiffusionEngine, FakeStream]:
        config = AppConfig(lcm_steps=2, denoise_index=32, seed=2)
        engine = StreamDiffusionEngine(config, lambda _message: None)
        stream = FakeStream()
        engine.stream = stream
        engine._torch = torch
        return engine, stream

    def test_temporal_and_fps_values_update_without_preparing_stream(self) -> None:
        engine, stream = self.make_engine()

        engine.update_live_settings(
            {
                "target_fps": 24,
                "temporal_feedback": 0.45,
                "temporal_smoothing": 0.2,
                "latent_morph_strength": 0.4,
                "latent_history_frames": 5,
                "scene_cut_threshold": 0.4,
            }
        )

        self.assertEqual(engine.config.target_fps, 24)
        self.assertEqual(engine.config.temporal_feedback, 0.45)
        self.assertEqual(engine.config.temporal_smoothing, 0.2)
        self.assertEqual(engine.config.latent_morph_strength, 0.4)
        self.assertEqual(engine.config.latent_history_frames, 5)
        self.assertEqual(engine.config.scene_cut_threshold, 0.4)
        self.assertEqual(stream.prepare_calls, [])

    def test_denoise_and_seed_reprepare_without_reloading_model(self) -> None:
        engine, stream = self.make_engine()
        engine._previous_input = Image.new("RGB", (8, 8), "black")
        engine._latent_history.append(torch.full((1, 4, 1, 1), 0.25))

        engine.update_live_settings({"denoise_index": 10, "seed": 99})

        self.assertEqual(engine.config.denoise_index, 10)
        self.assertEqual(engine.config.seed, 99)
        self.assertEqual(stream.t_list, [10, 23])
        self.assertEqual(len(stream.prepare_calls), 1)
        self.assertEqual(stream.prepare_calls[0]["seed"], 99)
        self.assertEqual(stream.prime_calls, 1)
        self.assertTrue(torch.all(stream.x_t_latent_buffer == 1))
        self.assertEqual(len(engine._latent_history), 1)

    def test_restart_only_setting_is_rejected(self) -> None:
        engine, _stream = self.make_engine()

        with self.assertRaises(ValueError):
            engine.update_live_settings({"width": 768})


if __name__ == "__main__":
    unittest.main()
