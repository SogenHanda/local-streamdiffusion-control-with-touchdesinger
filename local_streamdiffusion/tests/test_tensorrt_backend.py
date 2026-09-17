from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from streamdiffusion_bridge.config import AppConfig
from streamdiffusion_bridge.tensorrt_backend import cache_for, inspect_tensorrt


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "Test RTX"

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (8, 9)


class FakeTorch:
    cuda = FakeCuda()
    version = type("Version", (), {"cuda": "12.1"})()


class TensorRTCacheTests(unittest.TestCase):
    def make_config(self, directory: str) -> AppConfig:
        root = Path(directory)
        model = root / "model"
        lora = root / "lora"
        (model / "unet").mkdir(parents=True)
        lora.mkdir()
        (model / "model_index.json").write_text("{}", encoding="utf-8")
        (model / "unet" / "config.json").write_text("{}", encoding="utf-8")
        (model / "unet" / "weights.safetensors").write_bytes(b"weights")
        (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
        return AppConfig(
            model_path=str(model),
            lcm_lora_path=str(lora),
            tensorrt_engine_root=str(root / "engines"),
            width=512,
            height=512,
            lcm_steps=2,
        )

    def test_cache_changes_with_resolution_and_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory)
            first = cache_for(config, FakeTorch)
            second = cache_for(config, FakeTorch)
            resized = cache_for(
                AppConfig(**{**config.to_public_dict(), "width": 640, "height": 640}),
                FakeTorch,
            )

            self.assertEqual(first.directory, second.directory)
            self.assertNotEqual(first.directory, resized.directory)
            self.assertIn("512x512-b2", first.directory.name)

    def test_status_only_accepts_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory)
            with patch(
                "streamdiffusion_bridge.tensorrt_backend.missing_tensorrt_modules",
                return_value=(),
            ):
                cache = cache_for(config, FakeTorch)
                cache.directory.mkdir(parents=True)
                cache.engine_path.write_bytes(b"engine")
                cache.metadata_path.write_text(
                    json.dumps({"spec": asdict(cache.spec)}),
                    encoding="utf-8",
                )
                ready = inspect_tensorrt(config, FakeTorch)
                self.assertTrue(ready.ready)

                metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
                metadata["spec"]["width"] = 768
                cache.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                invalid = inspect_tensorrt(config, FakeTorch)
                self.assertEqual(invalid.state, "incompatible")


if __name__ == "__main__":
    unittest.main()
