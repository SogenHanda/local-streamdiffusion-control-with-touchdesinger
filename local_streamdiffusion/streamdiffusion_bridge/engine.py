from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .config import AppConfig
from .image_processing import decoded_tensor_to_pil
from .tensorrt_backend import (
    activate_unet_engine,
    build_unet_engine,
    inspect_tensorrt,
)


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
        self._previous_raw_output: Image.Image | None = None
        self._latent_history: deque[Any] = deque()
        self._original_decode_image = None
        self._vae_graph_calls: dict[str, Any] = {}
        self._cuda_stage_events: dict[str, tuple[Any, Any]] = {}
        self._cuda_stage_event_pool: dict[str, tuple[Any, Any]] = {}
        self._processed_frames = 0
        self._measure_cuda_stages = False
        self._last_prompt_length_warning: tuple[str, int, int] | None = None
        self.active_backend = "未初期化"
        self.engine_status = "-"
        self.last_stage_metrics: dict[str, float] = {
            "preprocess_ms": 0.0,
            "vae_encode_ms": 0.0,
            "unet_ms": 0.0,
            "vae_decode_ms": 0.0,
            "postprocess_ms": 0.0,
            "diffusion_ms": 0.0,
        }
        self.last_motion_score = 0.0
        self.last_temporal_feedback = 0.0
        self.last_latent_morph = 0.0

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
        # StreamDiffusion exposes the denoised latent immediately before VAE
        # decoding. Stabilize at that point so history morphs generated features
        # instead of overlaying RGB frames and producing trails/bright edges.
        self._original_decode_image = self.stream.decode_image
        self.stream.decode_image = self._decode_with_latent_history
        self.log(
            "生成latentモーフを有効化しました: "
            f"{self.config.latent_morph_strength * 100:.0f}% / "
            f"{self.config.latent_history_frames}フレーム"
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
        self._configure_acceleration_backend()
        if self.active_backend == "TensorRT FP16" and self.config.tensorrt_cuda_graph and self.config.use_tiny_vae:
            from .cuda_graph import CapturedVAECall

            for name in ("encode", "decode"):
                captured = CapturedVAECall(getattr(self.stream.vae, name))
                self._vae_graph_calls[name] = captured
                setattr(self.stream.vae, name, captured)
            self.log("TinyVAEの固定バッファCUDA Graphを有効にしました。")
        self._prepare_stream()
        self._install_stage_timers()
        self._warmed_up = False
        self.reset_temporal(log=False)

    def _configure_acceleration_backend(self) -> None:
        assert self.stream is not None
        requested = self.config.acceleration_backend
        if requested in {"auto", "tensorrt"}:
            status = inspect_tensorrt(self.config, self._torch)
            self.engine_status = status.message
            if status.ready:
                activate_unet_engine(self.stream, self.config, self.log)
                self.active_backend = "TensorRT FP16"
                return
            if requested == "tensorrt":
                raise RuntimeError(
                    status.message
                    + "\nsetup_tensorrt.ps1の実行後、build_tensorrt_engine.cmdで"
                    "現在のモデル用エンジンを作成してください。"
                )
            self.log(f"Auto: {status.message}。xFormersへフォールバックします。")

        if requested in {"auto", "xformers"}:
            try:
                assert self.pipe is not None
                self.pipe.enable_xformers_memory_efficient_attention()
                self.active_backend = "xFormers"
                if requested == "xformers":
                    self.engine_status = "xFormersを明示選択"
                self.log("xFormersメモリ効率化を有効にしました。")
                return
            except Exception as exc:
                if requested == "xformers":
                    raise RuntimeError(f"xFormersを有効化できません: {exc}") from exc
                self.log(f"xFormersを有効化できないためPyTorchへ戻します: {exc}")

        self.active_backend = "PyTorch"
        if requested == "pytorch":
            self.engine_status = "PyTorchを明示選択"
        elif self.engine_status == "-":
            self.engine_status = "xFormers/TensorRTを使用していません"

    def build_tensorrt(self, *, force: bool = False) -> None:
        """Build the current fused UNet; used by the standalone build command."""
        if self.stream is None:
            raise RuntimeError("モデルが読み込まれていません。")
        build_unet_engine(self.stream, self.config, self.log, force=force)

    def _install_stage_timers(self) -> None:
        """Measure CUDA stages without adding per-stage synchronize calls."""
        if self.stream is None or self._torch is None or not self._torch.cuda.is_available():
            return
        self._cuda_stage_event_pool = {
            name: (
                self._torch.cuda.Event(enable_timing=True),
                self._torch.cuda.Event(enable_timing=True),
            )
            for name in ("vae_encode_ms", "unet_ms", "vae_decode_ms")
        }
        original_encode = self.stream.encode_image
        original_predict = self.stream.predict_x0_batch

        def timed_encode(*args: Any, **kwargs: Any) -> Any:
            return self._timed_cuda_call("vae_encode_ms", original_encode, *args, **kwargs)

        def timed_predict(*args: Any, **kwargs: Any) -> Any:
            return self._timed_cuda_call("unet_ms", original_predict, *args, **kwargs)

        self.stream.encode_image = timed_encode
        self.stream.predict_x0_batch = timed_predict

    def _timed_cuda_call(
        self,
        name: str,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert self._torch is not None
        if not self._measure_cuda_stages:
            return callback(*args, **kwargs)
        start, end = self._cuda_stage_event_pool[name]
        start.record()
        result = callback(*args, **kwargs)
        end.record()
        self._cuda_stage_events[name] = (start, end)
        return result

    def _collect_cuda_stage_metrics(self) -> None:
        for name, (start, end) in self._cuda_stage_events.items():
            try:
                self.last_stage_metrics[name] = float(start.elapsed_time(end))
            except RuntimeError:
                self.last_stage_metrics[name] = 0.0
        self._cuda_stage_events.clear()

    def _t_index_list(self) -> list[int]:
        first_step = min(self.config.denoise_index, 48)
        t_index_list = [first_step]
        if self.config.lcm_steps == 2:
            t_index_list.append(min(first_step + 13, 49))
        return t_index_list

    def _prepare_stream(self) -> None:
        assert self.stream is not None
        self._report_prompt_length(self.config.prompt)
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
        self._report_prompt_length(prompt)
        self.stream.update_prompt(prompt)
        self.config.prompt = prompt
        # Keep the bounded/clamped history so a prompt change morphs instead of
        # cutting. StreamDiffusion's own delayed latent also remains valid.
        self.log("プロンプトを更新しました（時間履歴を保ったまま遷移します）。")

    def _report_prompt_length(self, prompt: str) -> None:
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            return
        try:
            encoded = tokenizer(prompt, truncation=False, add_special_tokens=True)
            token_count = len(encoded["input_ids"])
            limit = int(tokenizer.model_max_length)
        except Exception:
            return
        if 0 < limit < 10_000 and token_count > limit:
            warning_key = (prompt, token_count, limit)
            if warning_key == self._last_prompt_length_warning:
                return
            self._last_prompt_length_warning = warning_key
            self.log(
                f"注意: プロンプトは{token_count} tokensで上限{limit}を超えています。"
                f"末尾{token_count - limit} tokensは生成へ反映されません。"
            )
        else:
            self._last_prompt_length_warning = None

    def update_live_settings(self, settings: dict[str, Any]) -> None:
        """Apply values that do not require model, VAE, Spout, or resolution reloads."""
        allowed = {
            "denoise_index",
            "seed",
            "target_fps",
            "temporal_feedback",
            "temporal_smoothing",
            "latent_morph_strength",
            "latent_history_frames",
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
            "latent_morph_strength",
            "scene_cut_threshold",
        ):
            if name in settings:
                normalized[name] = float(settings[name])
        if "latent_history_frames" in settings:
            normalized["latent_history_frames"] = int(
                round(float(settings["latent_history_frames"]))
            )

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
            # prepare() initializes the second-step frame buffer with zero. Prime
            # it from the latest camera image before publishing another frame;
            # otherwise the next frame can become a flat brown/grey image.
            primed = self._prime_stream_frame_buffer()
            if primed:
                self.log("変換設定変更後の内部バッファを現在フレームで再同期しました。")

        labels = {
            "denoise_index": "変換強度",
            "seed": "Seed",
            "target_fps": "FPS上限",
            "temporal_feedback": "入力保持",
            "temporal_smoothing": "出力平滑化",
            "latent_morph_strength": "生成特徴モーフ",
            "latent_history_frames": "特徴履歴フレーム",
            "scene_cut_threshold": "動き追従しきい値",
        }
        self.log("即時設定を更新しました: " + ", ".join(labels[name] for name in changed))

    def reset_temporal(self, log: bool = True) -> None:
        self._previous_input = None
        self._previous_raw_output = None
        self._latent_history.clear()
        self.last_motion_score = 0.0
        self.last_temporal_feedback = 0.0
        self.last_latent_morph = 0.0
        if log:
            self.log("時間安定化の履歴をリセットしました。")

    def _prime_stream_frame_buffer(self) -> bool:
        """Fill StreamDiffusion's denoising batch without decoding an output.

        StreamDiffusion uses a one-frame pipeline in 2-step batch mode. Its
        prepare() method resets the delayed latent to zero, so the first decoded
        output after a live denoise/seed update is invalid. Feeding the latest
        camera image through encode + UNet once restores a valid delayed latent
        while leaving our visible RGB/latent temporal history untouched.
        """
        stream = self.stream
        if (
            stream is None
            or self._torch is None
            or self._previous_input is None
            or int(getattr(stream, "denoising_steps_num", 1)) <= 1
        ):
            return False
        required = ("image_processor", "height", "width", "device", "dtype")
        if any(not hasattr(stream, name) for name in required):
            return False
        try:
            with self._torch.inference_mode():
                image_tensor = stream.image_processor.preprocess(
                    self._previous_input,
                    stream.height,
                    stream.width,
                ).to(device=stream.device, dtype=stream.dtype)
                x_t_latent = stream.encode_image(image_tensor)
                stream.predict_x0_batch(x_t_latent)
            self._cuda_stage_events.clear()
            return True
        except Exception as exc:
            # Do not stop a running show because an optional transition prime
            # failed. Preserve the previous visible output and report the cause.
            self.log(f"内部バッファの再同期を省略しました: {exc}")
            return False

    def process(self, image: Image.Image) -> tuple[Image.Image, float]:
        if self.stream is None or self._postprocess_image is None:
            raise RuntimeError("推論エンジンが初期化されていません。")

        preprocess_started = time.perf_counter()
        fitted = self._fit_image(image).convert("RGB")
        if not self._warmed_up:
            self.log("最初の入力映像でウォームアップしています…")
            for _ in range(3):
                self.stream(fitted)
            self._warmed_up = True
            self.log("ウォームアップ完了。")
            self._cuda_stage_events.clear()

        temporal_input = self._apply_temporal_feedback(fitted)
        self._processed_frames += 1
        # CUDA Events are accurate but their per-frame allocation has measurable
        # overhead in a realtime loop. Sample stages periodically and keep the
        # latest values visible between samples.
        self._measure_cuda_stages = self._processed_frames % 10 == 1
        self.last_stage_metrics["preprocess_ms"] = (
            time.perf_counter() - preprocess_started
        ) * 1000.0
        started = time.perf_counter()
        output_tensor = self.stream(temporal_input)
        diffusion_completed = time.perf_counter()
        self.last_stage_metrics["diffusion_ms"] = (
            diffusion_completed - started
        ) * 1000.0
        raw_output = decoded_tensor_to_pil(output_tensor)[0].convert("RGB")
        output = self._smooth_output(raw_output)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_stage_metrics["postprocess_ms"] = (
            time.perf_counter() - diffusion_completed
        ) * 1000.0
        if self._measure_cuda_stages:
            self._collect_cuda_stage_metrics()
        self._previous_input = fitted.copy()
        self._previous_raw_output = raw_output.copy()
        return output, elapsed_ms

    def _decode_with_latent_history(self, current_latent: Any) -> Any:
        """Blend short generated-latent history, then run the original VAE decode."""
        if self._original_decode_image is None:
            raise RuntimeError("VAEデコーダーが初期化されていません。")
        stabilized = self._stabilize_generated_latent(current_latent)
        if self._torch is not None and self._torch.cuda.is_available():
            return self._timed_cuda_call(
                "vae_decode_ms",
                self._original_decode_image,
                stabilized,
            )
        return self._original_decode_image(stabilized)

    def _stabilize_generated_latent(self, current: Any) -> Any:
        """Stabilize small latent flicker while rejecting stale structures."""
        self.last_latent_morph = 0.0

        strength = self.config.latent_morph_strength
        frames = self.config.latent_history_frames
        history = list(self._latent_history)[-frames:]

        stabilized = current
        if strength > 0.0 and history:
            # Exponential recency weighting makes an 8-frame history useful for
            # noise estimation without equally overlaying eight old structures.
            # History always stores raw generated latents, never filtered output.
            weighted = history[-1].float()
            total_weight = 1.0
            decay = 0.55
            weight = decay
            for latent in reversed(history[:-1]):
                weighted = weighted + latent.float() * weight
                total_weight += weight
                weight *= decay
            reference = weighted / total_weight

            threshold = max(self.config.scene_cut_threshold, 1e-6)
            motion_ratio = min(self.last_motion_score / threshold, 1.0)
            # Smoothly reduce history influence as the camera moves. There is no
            # hard scene-cut reset; the installation input changes continuously.
            motion_gate = 1.0 - motion_ratio * motion_ratio * (3.0 - 2.0 * motion_ratio)
            effective = strength * motion_gate
            if effective > 0.0:
                current_float = current.float()
                delta = reference - current_float
                # A history-clamped correction behaves like temporal antialiasing:
                # small stochastic changes are smoothed, but a moved edge or new
                # shape cannot drag an unrestricted old latent into this frame.
                spatial_dims = tuple(range(2, len(current.shape)))
                if spatial_dims:
                    scale = current_float.var(
                        dim=spatial_dims,
                        keepdim=True,
                        unbiased=False,
                    ).add(1e-6).sqrt()
                else:
                    scale = current_float.abs().mean().reshape(
                        (1,) * len(current.shape)
                    )
                correction_limit = scale * 0.35 + 0.04
                correction = correction_limit * (delta / correction_limit).tanh()
                stabilized = (current_float + correction * effective).to(
                    dtype=current.dtype
                )
                self.last_latent_morph = effective

        self._latent_history.append(current.detach().clone())
        while len(self._latent_history) > frames:
            self._latent_history.popleft()
        return stabilized

    def _apply_temporal_feedback(self, current: Image.Image) -> Image.Image:
        self.last_temporal_feedback = 0.0
        if self._previous_input is None:
            self.last_motion_score = 0.0
            return current

        motion = self._motion_score(current, self._previous_input)
        self.last_motion_score = motion
        threshold = max(self.config.scene_cut_threshold, 1e-6)
        motion_ratio = min(motion / threshold, 1.0)
        motion_gate = 1.0 - motion_ratio * motion_ratio * (3.0 - 2.0 * motion_ratio)
        feedback = self.config.temporal_feedback * motion_gate
        self.last_temporal_feedback = feedback
        if feedback <= 0.0:
            return current
        # Feed camera history, never a generated image, back into the model.
        # Recursive generated-image feedback amplifies bright edges and eventually
        # produces the white-line artifacts seen at high retention values.
        return self._detail_preserving_blend(
            current,
            self._previous_input,
            feedback,
            difference_cutoff=48,
        )

    def _smooth_output(self, current: Image.Image) -> Image.Image:
        if self._previous_raw_output is None or self.config.temporal_smoothing <= 0.0:
            return current
        # Blend against the previous *raw* generation rather than the already
        # smoothed output. This bounds history to one frame and prevents an
        # infinite feedback tail. The difference mask protects moving edges.
        return self._detail_preserving_blend(
            current,
            self._previous_raw_output,
            self.config.temporal_smoothing,
            difference_cutoff=96,
        )

    @staticmethod
    def _detail_preserving_blend(
        current: Image.Image,
        history: Image.Image,
        strength: float,
        *,
        difference_cutoff: int,
    ) -> Image.Image:
        """Blend stable pixels and reject history where image content moved."""
        strength = max(0.0, min(float(strength), 1.0))
        if strength <= 0.0:
            return current
        difference = ImageChops.difference(current, history).convert("L")
        cutoff = max(int(difference_cutoff), 1)
        mask_lut = []
        for value in range(256):
            confidence = max(0.0, 1.0 - value / cutoff)
            history_weight = strength * confidence * confidence
            mask_lut.append(round(history_weight * 255.0))
        history_mask = difference.point(mask_lut)
        return Image.composite(history, current, history_mask)

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
        import gc

        for name, captured in self._vae_graph_calls.items():
            captured.close()
            if self.stream is not None:
                setattr(self.stream.vae, name, captured.callback)
        self._vae_graph_calls.clear()
        if self.stream is not None:
            close_unet = getattr(getattr(self.stream, "unet", None), "close", None)
            if callable(close_unet):
                close_unet()
        self.reset_temporal(log=False)
        self._original_decode_image = None
        self._cuda_stage_events.clear()
        self._cuda_stage_event_pool.clear()
        self.stream = None
        self.pipe = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
