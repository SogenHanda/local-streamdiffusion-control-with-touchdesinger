from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig


LogCallback = Callable[[str], None]
TRT_REQUIRED_MODULES = (
    "tensorrt",
    "polygraphy",
    "onnx",
    "onnx_graphsurgeon",
    "cuda",
)


@dataclass(frozen=True, slots=True)
class TensorRTSpec:
    schema: int
    model: str
    lora: str
    width: int
    height: int
    batch_size: int
    gpu_name: str
    gpu_capability: str
    torch_cuda: str
    tensorrt_version: str
    fp16: bool = True
    cfg_type: str = "none"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class TensorRTCache:
    directory: Path
    engine_path: Path
    metadata_path: Path
    onnx_dir: Path
    spec: TensorRTSpec


@dataclass(frozen=True, slots=True)
class TensorRTStatus:
    state: str
    message: str
    cache: TensorRTCache | None = None
    missing_modules: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.state == "ready"


def missing_tensorrt_modules() -> tuple[str, ...]:
    return tuple(
        name for name in TRT_REQUIRED_MODULES if importlib.util.find_spec(name) is None
    )


def _small_file_digest(path: Path) -> str:
    try:
        if path.stat().st_size <= 4 * 1024 * 1024:
            return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        pass
    return "-"


def _path_signature(path: Path, include_weights: bool = True) -> str:
    """Create a fast identity without hashing multi-gigabyte model weights."""
    if not path.exists():
        return f"missing:{path.as_posix()}"
    if path.is_file():
        stat = path.stat()
        return f"file:{path.name}:{stat.st_size}:{stat.st_mtime_ns}:{_small_file_digest(path)}"

    patterns = ("*.json", "*.safetensors", "*.bin") if include_weights else ("*.json",)
    entries: list[str] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for file_path in sorted(path.rglob(pattern)):
            if file_path in seen or not file_path.is_file():
                continue
            seen.add(file_path)
            stat = file_path.stat()
            relative = file_path.relative_to(path).as_posix()
            digest = _small_file_digest(file_path) if file_path.suffix == ".json" else "-"
            entries.append(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}:{digest}")
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_identity(torch_module: Any | None = None) -> tuple[str, str, str, str]:
    if torch_module is None:
        import torch as torch_module

    gpu_name = "no-cuda"
    capability = "-"
    if torch_module.cuda.is_available():
        gpu_name = str(torch_module.cuda.get_device_name(0))
        major, minor = torch_module.cuda.get_device_capability(0)
        capability = f"{major}.{minor}"
    torch_cuda = str(torch_module.version.cuda or "-")
    try:
        import tensorrt

        tensorrt_version = str(tensorrt.__version__)
    except Exception:
        tensorrt_version = "missing"
    return gpu_name, capability, torch_cuda, tensorrt_version


def build_spec(config: AppConfig, torch_module: Any | None = None) -> TensorRTSpec:
    gpu_name, capability, torch_cuda, tensorrt_version = _runtime_identity(torch_module)
    lora = "disabled"
    if config.use_lcm_lora:
        lora = _path_signature(config.resolved_lcm_lora_path)
    return TensorRTSpec(
        schema=1,
        model=_path_signature(config.resolved_model_path),
        lora=lora,
        width=config.width,
        height=config.height,
        batch_size=config.lcm_steps,
        gpu_name=gpu_name,
        gpu_capability=capability,
        torch_cuda=torch_cuda,
        tensorrt_version=tensorrt_version,
    )


def cache_for(config: AppConfig, torch_module: Any | None = None) -> TensorRTCache:
    spec = build_spec(config, torch_module)
    model_name = Path(config.model_path).name or config.model_profile or "model"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-_") or "model"
    directory = config.resolved_tensorrt_engine_root / (
        f"{slug}-{config.width}x{config.height}-b{config.lcm_steps}-{spec.fingerprint}"
    )
    return TensorRTCache(
        directory=directory,
        engine_path=directory / "unet.engine",
        metadata_path=directory / "engine.json",
        onnx_dir=directory / "onnx",
        spec=spec,
    )


def inspect_tensorrt(config: AppConfig, torch_module: Any | None = None) -> TensorRTStatus:
    missing = missing_tensorrt_modules()
    if missing:
        return TensorRTStatus(
            state="dependencies_missing",
            message="TensorRT環境が未導入: " + ", ".join(missing),
            missing_modules=missing,
        )
    try:
        cache = cache_for(config, torch_module)
    except Exception as exc:
        return TensorRTStatus(state="unavailable", message=f"GPU情報を取得できません: {exc}")
    if not cache.engine_path.is_file() or not cache.metadata_path.is_file():
        return TensorRTStatus(
            state="missing",
            message=(
                f"現在のモデル/GPU用エンジンは未ビルド ({cache.spec.fingerprint})"
            ),
            cache=cache,
        )
    try:
        metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return TensorRTStatus(
            state="invalid",
            message=f"TensorRTメタデータを読めません: {exc}",
            cache=cache,
        )
    if metadata.get("spec") != asdict(cache.spec):
        return TensorRTStatus(
            state="incompatible",
            message="TensorRTエンジンの条件が現在の設定と一致しません。再ビルドしてください。",
            cache=cache,
        )
    return TensorRTStatus(
        state="ready",
        message=f"TensorRT FP16使用可能 ({cache.spec.fingerprint})",
        cache=cache,
    )


def build_unet_engine(
    stream: Any,
    config: AppConfig,
    log: LogCallback,
    *,
    force: bool = False,
) -> TensorRTCache:
    missing = missing_tensorrt_modules()
    if missing:
        raise RuntimeError(
            "TensorRTビルド環境がありません。先に setup_tensorrt.ps1 を実行してください。"
            f"\n不足: {', '.join(missing)}"
        )

    cache = cache_for(config)
    current_status = inspect_tensorrt(config)
    if current_status.ready and not force:
        log(f"TensorRTエンジンは作成済みです: {cache.engine_path}")
        return cache

    from streamdiffusion.acceleration.tensorrt import UNet, compile_unet, create_onnx_path

    cache.onnx_dir.mkdir(parents=True, exist_ok=True)
    unet = stream.unet
    text_encoder = stream.text_encoder
    model_data = UNet(
        fp16=True,
        device=stream.device,
        max_batch_size=config.lcm_steps,
        min_batch_size=config.lcm_steps,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
    log(
        "TensorRT UNetをビルドします。初回は10〜30分程度かかる場合があります: "
        f"{cache.directory}"
    )
    rebuild_engine = True
    compile_unet(
        unet,
        model_data,
        create_onnx_path("unet", str(cache.onnx_dir), opt=False),
        create_onnx_path("unet", str(cache.onnx_dir), opt=True),
        str(cache.engine_path),
        opt_batch_size=config.lcm_steps,
        engine_build_options={
            "opt_image_height": config.height,
            "opt_image_width": config.width,
            "min_image_resolution": min(config.width, config.height),
            "max_image_resolution": max(config.width, config.height),
            "build_static_batch": True,
            "build_dynamic_shape": False,
            "build_all_tactics": True,
            "force_engine_build": rebuild_engine,
            "force_onnx_export": force,
            "force_onnx_optimize": force,
        },
    )
    cache.metadata_path.write_text(
        json.dumps(
            {
                "format": "ergonomics-streamdiffusion-tensorrt-unet",
                "spec": asdict(cache.spec),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    removed_bytes = 0
    for intermediate in cache.onnx_dir.glob("*.onnx"):
        try:
            removed_bytes += intermediate.stat().st_size
            intermediate.unlink()
        except OSError as exc:
            log(f"中間ONNXを削除できませんでした: {intermediate} ({exc})")
    try:
        cache.onnx_dir.rmdir()
    except OSError:
        pass
    if removed_bytes:
        log(f"ビルド中間ONNXを削除しました: {removed_bytes / (1024**3):.2f} GB")
    log(f"TensorRTエンジンを保存しました: {cache.engine_path}")
    return cache


def activate_unet_engine(
    stream: Any,
    config: AppConfig,
    log: LogCallback,
) -> TensorRTCache:
    status = inspect_tensorrt(config)
    if not status.ready or status.cache is None:
        raise RuntimeError(status.message)

    import gc
    import torch
    from polygraphy import cuda
    from polygraphy.backend.trt import util as trt_util
    from streamdiffusion.acceleration.tensorrt import UNet2DConditionModelEngine

    # StreamDiffusion 0.1.1 was written for Polygraphy 0.47. NVIDIA's public
    # PyPI now starts at 0.48, where this small helper was removed. TensorRT 9
    # still exposes both values needed to preserve the exact old behavior.
    if not hasattr(trt_util, "get_bindings_per_profile"):
        trt_util.get_bindings_per_profile = lambda engine: (
            engine.num_bindings // max(engine.num_optimization_profiles, 1)
        )

    polygraphy_stream = cuda.Stream()
    wrapper = UNet2DConditionModelEngine(
        str(status.cache.engine_path),
        polygraphy_stream,
        # StreamDiffusion 0.1.1 reallocates TensorRT buffers on every call.
        # Replaying a graph captured with the old addresses returns corrupted
        # noise after the first frame, so quality takes priority here.
        use_cuda_graph=False,
    )
    # The wrapper needs the Polygraphy stream to outlive every inference call.
    wrapper._ergonomics_cuda_stream = polygraphy_stream

    original_unet = stream.unet
    original_unet.to(torch.device("cpu"))
    try:
        del stream.pipe.unet
    except AttributeError:
        pass
    stream.unet = wrapper
    del original_unet
    gc.collect()
    torch.cuda.empty_cache()
    if config.tensorrt_cuda_graph:
        log("CUDA Graphは連続フレームのバッファ互換性を優先して無効化しました。")
    log("TensorRT FP16 UNetを有効化しました。")
    return status.cache
