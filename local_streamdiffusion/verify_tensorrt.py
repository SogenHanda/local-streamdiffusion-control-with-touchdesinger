from __future__ import annotations

import importlib
import sys


def main() -> int:
    failures: list[str] = []
    modules = (
        "torch",
        "tensorrt",
        "polygraphy",
        "onnx",
        "onnx_graphsurgeon",
        "cuda",
        "streamdiffusion",
    )
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(loaded[name], '__version__', 'installed')}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"[NG] {name}: {exc}")

    torch = loaded.get("torch")
    if torch is not None:
        if not torch.cuda.is_available():
            failures.append("PyTorch CUDA is unavailable")
        else:
            print(f"[OK] CUDA: {torch.version.cuda} / {torch.cuda.get_device_name(0)}")

    try:
        from streamdiffusion.acceleration.tensorrt import (
            UNet,
            UNet2DConditionModelEngine,
            compile_unet,
        )

        _ = (UNet, UNet2DConditionModelEngine, compile_unet)
        print("[OK] StreamDiffusion TensorRT bridge")
    except Exception as exc:
        failures.append(f"StreamDiffusion TensorRT bridge: {exc}")

    if failures:
        print("\nTensorRT verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nTensorRT environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
