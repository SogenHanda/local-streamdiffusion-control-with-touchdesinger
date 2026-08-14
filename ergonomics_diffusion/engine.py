from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .config import AppConfig


LogCallback = Callable[[str], None]


class StreamDiffusionEngine:
    """Continuous img2img pipeline with motion-aware temporal stabilization."""

    def __init__(self, config: AppConfig, log: LogCallback) -> None:
        self.config = config
        self.log = log
        self.pipe = None
        self.stream = None
        self._postprocess_image = None
        self._torch = None
        self._warmed_up = False
        self._previous_input: Image.Image | None = None
        self._previous_output: Image.Image | None = None
        self._scene_cut_detected = False
        self.last_motion_score = 0.0
        self.last_temporal_feedback = 0.0

    def load(self) -> None:
        model_path = self.config.resolved_model_path
        if self.config.offline_mode and not model_path.exists():
            raise FileNotFoundError(
                f"モデルがありません: {model_path}\n"
                "先に download_models.ps1 を実行してください。"
            )
        lcm_lora_path = self.config.resolved_lcm_lora_path
        if self.config.use_lcm_lora and self.config.offline_mode and not lcm_lora_path.exists():
            raise FileNotFoundError(
                f"LCM-LoRAがありません: {lcm_lora_path}\n"
                "先に download_models.ps1 を実行してください。"
            )

        self.log("CUDAとStreamDiffusionを初期化しています…")
        import torch
        from diffusers import AutoencoderTiny, DiffusionPipeline
        from streamdiffusion import StreamDiffusion
        from streamdiffusion.image_utils import postprocess_image

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA対応GPUを認識できません。NVIDIAドライバーとCUDA版PyTorchを確認してください。"
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self._torch = torch
        self._stream_class = StreamDiffusion
        self._postprocess_image = postprocess_image

        load_target = str(model_path) if model_path.exists() else self.config.model_path
        self.log(f"モデルを読み込んでいます: {load_target}")
        load_options = dict(
            torch_dtype=torch.float16,
            local_files_only=self.config.offline_mode,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        if model_path.exists() and any(model_path.rglob("*.fp16.safetensors")):
            load_options["variant"] = "fp16"
        self.pipe = DiffusionPipeline.from_pretrained(
            load_target,
            **load_options,
        ).to(device=torch.device("cuda"), dtype=torch.float16)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.log("xFormersメモリ効率化を有効にしました。")
        except Exception as exc:  # xFormers availability depends on the local CUDA stack.
            self.log(f"xFormersを有効化できないため通常推論を使用します: {exc}")

        if self.config.use_tiny_vae:
            tiny_vae_path = self.config.resolved_tiny_vae_path
            if self.config.offline_mode and not tiny_vae_path.exists():
                raise FileNotFoundError(
                    f"TinyVAEがありません: {tiny_vae_path}\n"
                    "先に download_models.ps1 を実行してください。"
                )
            tiny_vae_target = (
                str(tiny_vae_path) if tiny_vae_path.exists() else self.config.tiny_vae_path
            )
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                tiny_vae_target,
                torch_dtype=torch.float16,
                local_files_only=self.config.offline_mode,
            ).to(device=torch.device("cuda"), dtype=torch.float16)
            self.log("TinyVAEを有効にしました。")

        self._build_stream()
        gpu_name = torch.cuda.get_device_name(0)
        self.log(f"モデル準備完了: {gpu_name}")

    def _build_stream(self) -> None:
        assert self.pipe is not None
        assert self._torch is not None

        t_index_list = self._t_index_list()

        self.stream = self._stream_class(
            self.pipe,
            t_index_list=t_index_list,
            torch_dtype=self._torch.float16,
            width=self.config.width,
            height=self.config.height,
            do_add_noise=True,
            use_denoising_batch=True,
            frame_buffer_size=1,
            cfg_type="none",
        )
        if self.config.use_lcm_lora:
            lcm_lora_path = self.config.resolved_lcm_lora_path
            lcm_lora_target = (
                str(lcm_lora_path) if lcm_lora_path.exists() else self.config.lcm_lora_path
            )
            self.log(f"高品質LCM-LoRAを読み込んでいます: {lcm_lora_target}")
            self.stream.load_lcm_lora(lcm_lora_target)
            self.stream.fuse_lora()
            mode = "リアルタイム1-step" if self.config.lcm_steps == 1 else "高品質2-step"
            self.log(f"LCM {mode}推論を有効にしました: t_index={t_index_list}")
        else:
            self.log(f"モデル内蔵の少ステップ推論を使用します: t_index={t_index_list}")
        self._prepare_stream()
        self._warmed_up = False
        self.reset_temporal(log=False)

    def _t_index_list(self) -> list[int]:
        first_step = min(self.config.denoise_index, 48)
        t_index_list = [first_step]
        if self.config.lcm_steps == 2:
            t_index_list.append(min(first_step + 13, 49))
        return t_index_list

    def _prepare_stream(self) -> None:
        assert self.stream is not None
        self.stream.prepare(
            prompt=self.config.prompt,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.0,
            seed=self.config.seed,
        )

    def update_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or self.stream is None:
            return
        self.stream.update_prompt(prompt)
        self.config.prompt = prompt
        self.reset_temporal(log=False)
        self.log("プロンプトを更新し、時間履歴をリセットしました。")

    def update_live_settings(self, settings: dict[str, Any]) -> None:
        """Apply values that do not require model, VAE, Spout, or resolution reloads."""
        allowed = {
            "denoise_index",
            "seed",
            "target_fps",
            "temporal_feedback",
            "temporal_smoothing",
            "scene_cut_threshold",
        }
        unknown = set(settings) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"実行中に変更できない設定です: {names}")

        normalized: dict[str, int | float] = {}
        if "denoise_index" in settings:
            normalized["denoise_index"] = int(round(float(settings["denoise_index"])))
        if "seed" in settings:
            normalized["seed"] = int(settings["seed"])
        for name in (
            "target_fps",
            "temporal_feedback",
            "temporal_smoothing",
            "scene_cut_threshold",
        ):
            if name in settings:
                normalized[name] = float(settings[name])

        candidate = replace(self.config, **normalized)
        candidate.validate()
        requires_prepare = any(
            getattr(candidate, name) != getattr(self.config, name)
            for name in ("denoise_index", "seed")
        )
        changed = [
            name
            for name, value in normalized.items()
            if value != getattr(self.config, name)
        ]
        if not changed:
            return

        for name in changed:
            setattr(self.config, name, getattr(candidate, name))

        if requires_prepare:
            if self.stream is None:
                return
            # The step count and tensor shapes stay unchanged, so replacing the
            # timestep list and preparing the cached stream is much cheaper than
            # reloading the model. Commands run between frames on the worker thread.
            self.stream.t_list = self._t_index_list()
            self._prepare_stream()
            self.reset_temporal(log=False)

        labels = {
            "denoise_index": "変換強度",
            "seed": "Seed",
            "target_fps": "FPS上限",
            "temporal_feedback": "入力保持",
            "temporal_smoothing": "出力平滑化",
            "scene_cut_threshold": "シーン変化",
        }
        self.log("即時設定を更新しました: " + ", ".join(labels[name] for name in changed))

    def reset_temporal(self, log: bool = True) -> None:
        self._previous_input = None
        self._previous_output = None
        self._scene_cut_detected = False
        self.last_motion_score = 0.0
        self.last_temporal_feedback = 0.0
        if log:
            self.log("時間安定化の履歴をリセットしました。")

    def process(self, image: Image.Image) -> tuple[Image.Image, float]:
        if self.stream is None or self._postprocess_image is None:
            raise RuntimeError("推論エンジンが初期化されていません。")

        fitted = self._fit_image(image).convert("RGB")
        if not self._warmed_up:
            self.log("最初の入力映像でウォームアップしています…")
            for _ in range(3):
                self.stream(fitted)
            self._warmed_up = True
            self.log("ウォームアップ完了。")

        temporal_input = self._apply_temporal_feedback(fitted)
        started = time.perf_counter()
        output_tensor = self.stream(temporal_input)
        raw_output = self._postprocess_image(output_tensor, output_type="pil")[0].convert("RGB")
        output = self._smooth_output(raw_output)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._previous_input = fitted.copy()
        self._previous_output = output.copy()
        return output, elapsed_ms

    def _apply_temporal_feedback(self, current: Image.Image) -> Image.Image:
        self._scene_cut_detected = False
        self.last_temporal_feedback = 0.0
        if self._previous_input is None:
            self.last_motion_score = 0.0
            return current

        motion = self._motion_score(current, self._previous_input)
        self.last_motion_score = motion
        if motion >= self.config.scene_cut_threshold:
            self._scene_cut_detected = True
            return current

        motion_ratio = min(motion / self.config.scene_cut_threshold, 1.0)
        feedback = self.config.temporal_feedback * (1.0 - motion_ratio)
        self.last_temporal_feedback = feedback
        if feedback <= 0.0:
            return current
        # Feed camera history, never a generated image, back into the model.
        # Recursive generated-image feedback amplifies bright edges and eventually
        # produces the white-line artifacts seen at high retention values.
        return Image.blend(current, self._previous_input, feedback)

    def _smooth_output(self, current: Image.Image) -> Image.Image:
        if (
            self._previous_output is None
            or self._scene_cut_detected
            or self.config.temporal_smoothing <= 0.0
        ):
            return current
        return Image.blend(current, self._previous_output, self.config.temporal_smoothing)

    @staticmethod
    def _motion_score(current: Image.Image, previous: Image.Image) -> float:
        size = (64, 64)
        current_small = current.convert("L").resize(size, Image.Resampling.BILINEAR)
        previous_small = previous.convert("L").resize(size, Image.Resampling.BILINEAR)
        difference = ImageChops.difference(current_small, previous_small)
        return float(ImageStat.Stat(difference).mean[0] / 255.0)

    def _fit_image(self, image: Image.Image) -> Image.Image:
        target_ratio = self.config.width / self.config.height
        source_ratio = image.width / image.height
        if source_ratio > target_ratio:
            crop_width = round(image.height * target_ratio)
            left = (image.width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, image.height))
        elif source_ratio < target_ratio:
            crop_height = round(image.width / target_ratio)
            top = (image.height - crop_height) // 2
            image = image.crop((0, top, image.width, top + crop_height))
        return image.resize(
            (self.config.width, self.config.height),
            Image.Resampling.LANCZOS,
        )

    def close(self) -> None:
        self.reset_temporal(log=False)
        self.stream = None
        self.pipe = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
