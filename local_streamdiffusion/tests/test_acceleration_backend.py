from __future__ import annotations

import unittest
from unittest.mock import patch

from streamdiffusion_bridge.config import AppConfig
from streamdiffusion_bridge.engine import StreamDiffusionEngine
from streamdiffusion_bridge.tensorrt_backend import TensorRTStatus


class FakePipe:
    def __init__(self) -> None:
        self.xformers_enabled = False

    def enable_xformers_memory_efficient_attention(self) -> None:
        self.xformers_enabled = True


class AccelerationBackendTests(unittest.TestCase):
    def test_auto_falls_back_to_xformers_when_engine_is_missing(self) -> None:
        logs: list[str] = []
        engine = StreamDiffusionEngine(AppConfig(acceleration_backend="auto"), logs.append)
        engine.pipe = FakePipe()
        engine.stream = object()
        with patch(
            "streamdiffusion_bridge.engine.inspect_tensorrt",
            return_value=TensorRTStatus("missing", "engine missing"),
        ):
            engine._configure_acceleration_backend()

        self.assertEqual(engine.active_backend, "xFormers")
        self.assertTrue(engine.pipe.xformers_enabled)
        self.assertTrue(any("フォールバック" in message for message in logs))

    def test_explicit_tensorrt_does_not_silently_fallback(self) -> None:
        engine = StreamDiffusionEngine(
            AppConfig(acceleration_backend="tensorrt"),
            lambda _message: None,
        )
        engine.pipe = FakePipe()
        engine.stream = object()
        with patch(
            "streamdiffusion_bridge.engine.inspect_tensorrt",
            return_value=TensorRTStatus("missing", "engine missing"),
        ):
            with self.assertRaises(RuntimeError):
                engine._configure_acceleration_backend()


if __name__ == "__main__":
    unittest.main()
