from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


APP_DIR = Path(__file__).resolve().parent
QUALITY_MODEL_ID = "Lykon/dreamshaper-8"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
SD_TURBO_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
TAESD_REVISION = "9565f9356505d7c3537d0e15c558520ed95142ee"
SD_TURBO_FILES = (
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
QUALITY_MODEL_FILES = (
    "README.md",
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
FULL_PRECISION_MODEL_FILES = (
    "README.md",
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
TAESD_FILES = (
    "README.md",
    "config.json",
    "diffusion_pytorch_model.safetensors",
)
LCM_LORA_FILES = (
    "README.md",
    "pytorch_lora_weights.safetensors",
)
SAFETENSORS_ONLY_IGNORE = (
    "*.bin",
    "*.ckpt",
    "*.msgpack",
    "*.onnx",
    "*.onnx_data",
)
MODEL_PRESETS = {
    "dreamshaper-8": {
        "repo_id": QUALITY_MODEL_ID,
        "directory": "dreamshaper-8",
        "patterns": QUALITY_MODEL_FILES,
    },
    "absolute-reality-1.81": {
        "repo_id": "Lykon/absolute-reality-1.81",
        "directory": "absolute-reality-1.81",
        "patterns": QUALITY_MODEL_FILES,
    },
    "lcm-dreamshaper-v7": {
        "repo_id": "SimianLuo/LCM_Dreamshaper_v7",
        "directory": "lcm-dreamshaper-v7",
        "patterns": FULL_PRECISION_MODEL_FILES,
    },
    "realistic-vision-v5.1": {
        "repo_id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "directory": "realistic-vision-v5.1",
        "patterns": FULL_PRECISION_MODEL_FILES,
    },
    "epicrealism": {
        "repo_id": "emilianJR/epiCRealism",
        "directory": "epicrealism",
        "patterns": FULL_PRECISION_MODEL_FILES,
    },
    "sd-turbo": {
        "repo_id": "stabilityai/sd-turbo",
        "directory": "sd-turbo",
        "revision": SD_TURBO_REVISION,
        "patterns": SD_TURBO_FILES,
    },
}


def download(
    repo_id: str,
    destination: Path,
    revision: str | None = None,
    allow_patterns: tuple[str, ...] | None = None,
    ignore_patterns: tuple[str, ...] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {destination}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=4,
    )
    print(f"Ready: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for offline inference")
    parser.add_argument(
        "--preset",
        choices=tuple(MODEL_PRESETS),
        help="Download one model profile used by the UI",
    )
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--model-id", default=QUALITY_MODEL_ID)
    parser.add_argument("--model-directory")
    parser.add_argument("--model-revision")
    parser.add_argument("--lcm-lora-id", default=LCM_LORA_ID)
    parser.add_argument("--lcm-lora-directory", default="lcm-lora-sdv1-5")
    parser.add_argument("--lcm-lora-revision")
    parser.add_argument("--skip-lcm-lora", action="store_true")
    parser.add_argument("--tiny-vae-id", default="madebyollin/taesd")
    parser.add_argument("--tiny-vae-revision")
    parser.add_argument("--skip-tiny-vae", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        for name, preset in MODEL_PRESETS.items():
            print(f"{name}: {preset['repo_id']}")
        return

    if args.preset:
        preset = MODEL_PRESETS[args.preset]
        download(
            str(preset["repo_id"]),
            APP_DIR / "models" / str(preset["directory"]),
            revision=preset.get("revision"),
            allow_patterns=preset.get("patterns"),
        )
        print(f"Model preset is ready: {args.preset}")
        return

    if args.model_id == "stabilityai/sd-turbo":
        model_patterns = SD_TURBO_FILES
    elif args.model_id == QUALITY_MODEL_ID:
        model_patterns = QUALITY_MODEL_FILES
    else:
        model_patterns = None
    model_ignore_patterns = None if model_patterns else SAFETENSORS_ONLY_IGNORE
    model_revision = args.model_revision or (
        SD_TURBO_REVISION if args.model_id == "stabilityai/sd-turbo" else None
    )
    model_directory = args.model_directory or (
        "sd-turbo" if args.model_id == "stabilityai/sd-turbo" else args.model_id.rsplit("/", 1)[-1].lower()
    )
    download(
        args.model_id,
        APP_DIR / "models" / model_directory,
        revision=model_revision,
        allow_patterns=model_patterns,
        ignore_patterns=model_ignore_patterns,
    )
    if not args.skip_lcm_lora:
        download(
            args.lcm_lora_id,
            APP_DIR / "models" / args.lcm_lora_directory,
            revision=args.lcm_lora_revision,
            allow_patterns=LCM_LORA_FILES if args.lcm_lora_id == LCM_LORA_ID else None,
        )
    if not args.skip_tiny_vae:
        tiny_vae_patterns = TAESD_FILES if args.tiny_vae_id == "madebyollin/taesd" else None
        tiny_vae_revision = args.tiny_vae_revision or (
            TAESD_REVISION if args.tiny_vae_id == "madebyollin/taesd" else None
        )
        download(
            args.tiny_vae_id,
            APP_DIR / "models" / "taesd",
            revision=tiny_vae_revision,
            allow_patterns=tiny_vae_patterns,
        )

    print("All models are stored locally. The inference UI can now run offline.")


if __name__ == "__main__":
    main()
