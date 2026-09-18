"""Image conversions that keep floating-point work on the inference device."""
from __future__ import annotations

from typing import Any

from PIL import Image


def decoded_tensor_to_pil(image: Any) -> list[Image.Image]:
    """Match StreamDiffusion's FP16 normalization and float32 uint8 rounding."""
    import torch

    normalized = (image / 2 + 0.5).clamp(0, 1)
    pixels = normalized.float().mul(255).round().to(torch.uint8)
    pixels = pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [Image.fromarray(frame.squeeze(-1) if frame.shape[-1] == 1 else frame)
            for frame in pixels]
