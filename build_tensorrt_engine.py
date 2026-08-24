from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from ergonomics_diffusion.config import AppConfig, DEFAULT_CONFIG_PATH
from ergonomics_diffusion.engine import StreamDiffusionEngine
from ergonomics_diffusion.tensorrt_backend import inspect_tensorrt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-shape TensorRT FP16 UNet for the current settings."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="Re-export ONNX and rebuild")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(args.config.resolve())
    # Never activate an old engine while exporting the fused PyTorch UNet.
    build_config = replace(config, acceleration_backend="pytorch")
    build_config.validate()

    print("=" * 72)
    print("TensorRT FP16 UNet build")
    print(f"Model      : {build_config.resolved_model_path}")
    print(f"Resolution : {build_config.width} x {build_config.height}")
    print(f"Batch/steps: {build_config.lcm_steps}")
    print("=" * 72)

    engine = StreamDiffusionEngine(build_config, print)
    try:
        engine.load()
        engine.build_tensorrt(force=args.force)
    finally:
        engine.close()

    status = inspect_tensorrt(config)
    print(status.message)
    return 0 if status.ready else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("TensorRT build cancelled.", file=sys.stderr)
        raise SystemExit(130)
