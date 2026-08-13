from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"


@dataclass(slots=True)
class AppConfig:
    model_profile: str = "dreamshaper-8"
    model_path: str = "models/dreamshaper-8"
    lcm_lora_path: str = "models/lcm-lora-sdv1-5"
    tiny_vae_path: str = "models/taesd"
    spout_input: str = "TD_Camera"
    spout_output: str = "AI_Output"
    prompt: str = "cinematic portrait, soft studio light, detailed, natural colors"
    width: int = 512
    height: int = 512
    denoise_index: int = 32
    seed: int = 2
    target_fps: float = 30.0
    use_lcm_lora: bool = True
    lcm_steps: int = 1
    use_tiny_vae: bool = True
    temporal_feedback: float = 0.35
    temporal_smoothing: float = 0.15
    scene_cut_threshold: float = 0.35
    flip_input: bool = False
    flip_output: bool = False
    offline_mode: bool = True
    preview_size: int = 256

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        config = cls(**filtered)
        config.validate()
        return config

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def validate(self) -> None:
        if not self.model_path.strip():
            raise ValueError("モデルのパスを入力してください。")
        if self.use_lcm_lora and not self.lcm_lora_path.strip():
            raise ValueError("LCM-LoRAのパスを入力してください。")
        if self.use_lcm_lora and "sd-turbo" in self.model_path.lower():
            raise ValueError(
                "SD-TurboにはSD 1.5用LCM-LoRAを適用できません。"
                "LCM-LoRAを無効化するか、models/dreamshaper-8を選択してください。"
            )
        if self.lcm_steps not in (1, 2):
            raise ValueError("LCM推論ステップは1または2を指定してください。")
        if self.use_tiny_vae and not self.tiny_vae_path.strip():
            raise ValueError("TinyVAEのパスを入力してください。")
        if not self.spout_input.strip():
            raise ValueError("Spout入力名を入力してください。")
        if not self.spout_output.strip():
            raise ValueError("Spout出力名を入力してください。")
        if self.spout_input.strip() == self.spout_output.strip():
            raise ValueError("Spout入力名と出力名は別の名前にしてください。")
        if self.width < 64 or self.height < 64:
            raise ValueError("解像度は64px以上にしてください。")
        if self.width % 8 or self.height % 8:
            raise ValueError("解像度は8の倍数にしてください。")
        if not 0 <= self.denoise_index <= 49:
            raise ValueError("変換の強さは0〜49の範囲で指定してください。")
        if not 0.1 <= self.target_fps <= 240:
            raise ValueError("FPS上限は0.1〜240の範囲で指定してください。")
        if not 0.0 <= self.temporal_feedback <= 0.8:
            raise ValueError("入力フレーム保持は0〜0.8の範囲で指定してください。")
        if not 0.0 <= self.temporal_smoothing <= 0.8:
            raise ValueError("出力平滑化は0〜0.8の範囲で指定してください。")
        if not 0.05 <= self.scene_cut_threshold <= 1.0:
            raise ValueError("シーン変化リセットは0.05〜1.0の範囲で指定してください。")
        if not self.prompt.strip():
            raise ValueError("プロンプトを入力してください。")

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (APP_DIR / path).resolve()

    @property
    def resolved_model_path(self) -> Path:
        return self.resolve_path(self.model_path)

    @property
    def resolved_lcm_lora_path(self) -> Path:
        return self.resolve_path(self.lcm_lora_path)

    @property
    def resolved_tiny_vae_path(self) -> Path:
        return self.resolve_path(self.tiny_vae_path)

    def to_public_dict(self) -> dict[str, Any]:
        """Return settings safe to display in the UI and logs."""
        return asdict(self)
