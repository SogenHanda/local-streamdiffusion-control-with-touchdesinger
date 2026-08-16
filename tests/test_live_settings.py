from __future__ import annotations

import unittest
from typing import Any

from ergonomics_diffusion.config import AppConfig
from ergonomics_diffusion.engine import StreamDiffusionEngine


class FakeStream:
    def __init__(self) -> None:
        self.t_list = [32, 45]
        self.prepare_calls: list[dict[str, Any]] = []

    def prepare(self, **kwargs: Any) -> None:
        self.prepare_calls.append(kwargs)


class LiveSettingsTests(unittest.TestCase):
    def make_engine(self) -> tuple[StreamDiffusionEngine, FakeStream]:
        config = AppConfig(lcm_steps=2, denoise_index=32, seed=2)
        engine = StreamDiffusionEngine(config, lambda _message: None)
        stream = FakeStream()
        engine.stream = stream
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

        engine.update_live_settings({"denoise_index": 10, "seed": 99})

        self.assertEqual(engine.config.denoise_index, 10)
        self.assertEqual(engine.config.seed, 99)
        self.assertEqual(stream.t_list, [10, 23])
        self.assertEqual(len(stream.prepare_calls), 1)
        self.assertEqual(stream.prepare_calls[0]["seed"], 99)

    def test_restart_only_setting_is_rejected(self) -> None:
        engine, _stream = self.make_engine()

        with self.assertRaises(ValueError):
            engine.update_live_settings({"width": 768})


if __name__ == "__main__":
    unittest.main()
