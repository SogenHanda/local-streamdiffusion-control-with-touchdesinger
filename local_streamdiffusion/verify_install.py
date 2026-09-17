from __future__ import annotations

import importlib
import sys


MODULES = (
    "torch",
    "torchvision",
    "xformers",
    "diffusers",
    "streamdiffusion",
    "SpoutGL",
    "OpenGL",
    "PIL",
    "pynvml",
)


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    failures: list[str] = []
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "OK")
            print(f"{module_name}: {version}")
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")

    if failures:
        print("\nImport errors:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    import torch
    import numpy

    if int(numpy.__version__.split(".")[0]) >= 2:
        raise SystemExit(
            f"NumPy {numpy.__version__} is incompatible with PyTorch 2.1.0. "
            "Run: python -m pip install --force-reinstall numpy==1.26.4"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available from PyTorch.")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Installation check passed.")


if __name__ == "__main__":
    main()
