from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import APP_DIR


@dataclass(frozen=True, slots=True)
class ModelProfile:
    key: str
    label: str
    model_path: str
    acceleration: str
    description: str
    recommended_preset: str

    @property
    def use_lcm_lora(self) -> bool:
        return self.acceleration == "lcm_lora"

    @property
    def installed(self) -> bool:
        path = APP_DIR / self.model_path
        return path.is_dir() and (path / "model_index.json").is_file()

    @property
    def display_label(self) -> str:
        return f"{self.label}  {'【導入済み】' if self.installed else '【未導入】'}"


@dataclass(frozen=True, slots=True)
class PerformancePreset:
    key: str
    label: str
    steps: int
    use_tiny_vae: bool
    description: str


BUILTIN_MODEL_PROFILES = (
    ModelProfile(
        key="dreamshaper-8",
        label="[バランス] DreamShaper 8 + LCM",
        model_path="models/dreamshaper-8",
        acceleration="lcm_lora",
        description="汎用・製品・コンセプト表現向け。現在の標準モデル。",
        recommended_preset="balanced",
    ),
    ModelProfile(
        key="absolute-reality-1.81",
        label="[バランス] Absolute Reality 1.81 + LCM",
        model_path="models/absolute-reality-1.81",
        acceleration="lcm_lora",
        description="人間の質感を残しやすいSD 1.5モデル。写実と作品表現の中間向け。",
        recommended_preset="balanced",
    ),
    ModelProfile(
        key="realistic-vision-v5.1",
        label="[高品質] Realistic Vision 5.1 + LCM",
        model_path="models/realistic-vision-v5.1",
        acceleration="lcm_lora",
        description="写真・素材感・実在感を優先するSD 1.5モデル。",
        recommended_preset="balanced",
    ),
    ModelProfile(
        key="epicrealism",
        label="[高品質] epiCRealism + LCM",
        model_path="models/epicrealism",
        acceleration="lcm_lora",
        description="皮膚・布・素材の写真質感を重視。2-stepで比較したい高品質候補。",
        recommended_preset="quality",
    ),
    ModelProfile(
        key="lcm-dreamshaper-v7",
        label="[高速] LCM DreamShaper v7",
        model_path="models/lcm-dreamshaper-v7",
        acceleration="native_lcm",
        description="少ステップ用に直接蒸留されたLCM。1-step品質の比較候補。",
        recommended_preset="realtime",
    ),
    ModelProfile(
        key="sd-turbo",
        label="[最速] SD-Turbo",
        model_path="models/sd-turbo",
        acceleration="turbo",
        description="速度優先の1-stepモデル。画質比較用の高速基準。",
        recommended_preset="realtime",
    ),
)


PERFORMANCE_PRESETS = (
    PerformancePreset(
        key="realtime",
        label="Realtime — 1-step + TinyVAE",
        steps=1,
        use_tiny_vae=True,
        description="速度優先。1-step向けモデル同士の比較に使用。",
    ),
    PerformancePreset(
        key="balanced",
        label="Balanced — 2-step + TinyVAE",
        steps=2,
        use_tiny_vae=True,
        description="UNetは2-step、VAEだけ高速化する推奨折衷設定。",
    ),
    PerformancePreset(
        key="quality",
        label="Quality — 2-step + Full VAE",
        steps=2,
        use_tiny_vae=False,
        description="画質優先。512pxでもFPSは大きく低下します。",
    ),
)


def discover_model_profiles(models_directory: Path | None = None) -> list[ModelProfile]:
    profiles = list(BUILTIN_MODEL_PROFILES)
    known_paths = {profile.model_path.lower() for profile in profiles}
    directory = models_directory or APP_DIR / "models"
    if not directory.exists():
        return profiles

    for model_directory in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
        if not model_directory.is_dir() or not (model_directory / "model_index.json").is_file():
            continue
        try:
            relative = model_directory.relative_to(APP_DIR).as_posix()
        except ValueError:
            relative = model_directory.as_posix()
        if relative.lower() in known_paths:
            continue
        profiles.append(
            ModelProfile(
                key=f"custom:{model_directory.name}",
                label=f"{model_directory.name}（ローカル）",
                model_path=relative,
                acceleration="lcm_lora",
                description="自動検出したSD 1.5 Diffusersモデル。LCM互換性を確認してください。",
                recommended_preset="balanced",
            )
        )
    return profiles


def find_profile(
    profiles: list[ModelProfile],
    key: str,
    model_path: str,
) -> ModelProfile | None:
    for profile in profiles:
        if profile.key == key:
            return profile
    normalized = model_path.replace("\\", "/").lower().rstrip("/")
    for profile in profiles:
        if profile.model_path.lower().rstrip("/") == normalized:
            return profile
    return None


def find_performance_preset(key: str) -> PerformancePreset | None:
    return next((preset for preset in PERFORMANCE_PRESETS if preset.key == key), None)


def infer_performance_preset(steps: int, use_tiny_vae: bool) -> PerformancePreset:
    for preset in PERFORMANCE_PRESETS:
        if preset.steps == steps and preset.use_tiny_vae == use_tiny_vae:
            return preset
    return PERFORMANCE_PRESETS[0]
