from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergonomics_diffusion.model_catalog import (
    BUILTIN_MODEL_PROFILES,
    discover_model_profiles,
    find_profile,
    infer_performance_preset,
)


class ModelCatalogTests(unittest.TestCase):
    def test_builtin_profiles_cover_fast_and_quality_models(self) -> None:
        keys = {profile.key for profile in BUILTIN_MODEL_PROFILES}
        self.assertIn("dreamshaper-8", keys)
        self.assertIn("absolute-reality-1.81", keys)
        self.assertIn("lcm-dreamshaper-v7", keys)
        self.assertIn("realistic-vision-v5.1", keys)
        self.assertIn("epicrealism", keys)
        self.assertIn("sd-turbo", keys)

    def test_discovers_additional_diffusers_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "my-model"
            model.mkdir()
            (model / "model_index.json").write_text("{}", encoding="utf-8")

            profiles = discover_model_profiles(Path(directory))

            custom = next(profile for profile in profiles if profile.key == "custom:my-model")
            self.assertIn("my-model", custom.model_path)

    def test_finds_profile_by_path_when_key_is_missing(self) -> None:
        profiles = list(BUILTIN_MODEL_PROFILES)
        profile = find_profile(profiles, "missing", "models/sd-turbo")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.key, "sd-turbo")

    def test_infers_balanced_preset(self) -> None:
        preset = infer_performance_preset(2, True)
        self.assertEqual(preset.key, "balanced")


if __name__ == "__main__":
    unittest.main()
