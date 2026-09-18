"""Finite, time-weighted latent morphing with camera-guided motion rejection."""
from __future__ import annotations

from collections import deque
import math
from typing import Any


class LatentMorph:
    """Keep raw features only, so settled images have no recursive blur tail."""

    def __init__(self) -> None:
        self.history: deque[tuple[float, Any, Any]] = deque(maxlen=64)
        self._key = None

    def reset(self) -> None:
        self.history.clear()
        self._key = None

    def apply(self, current: Any, *, now: float, strength: float, frames: int,
              motion_threshold: float, guide: Any = None) -> Any:
        import torch
        import torch.nn.functional as functional

        key = (tuple(current.shape), current.dtype, current.device)
        if key != self._key or (self.history and now < self.history[-1][0]):
            self.reset()
            self._key = key
        # Existing TD history control now denotes a duration at the 30fps
        # reference rate: 2 -> 33ms, 8 -> 233ms. Actual FPS does not set the tail.
        window = max(1, frames - 1) / 30.0
        while self.history and now - self.history[0][0] >= window:
            self.history.popleft()
        if strength <= 0.0:
            self.history.append((now, current.detach().clone(),
                                 guide.detach().clone() if guide is not None else None))
            return current

        current_float = current.float()
        target = current_float
        if self.history:
            weights = torch.tensor([
                0.5 * (1.0 + math.cos(math.pi * max(0.0, now - stamp) / window))
                for stamp, _, _ in self.history
            ], device=current.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
            if guide is not None:
                # Reject stale positions locally, even when a small moving
                # object barely changes the global camera-motion score.
                guides = torch.stack([old if old is not None else guide
                                      for _, _, old in self.history])
                difference = (guide.unsqueeze(0) - guides).abs().mean(dim=2)
                difference = functional.max_pool2d(difference, 3, stride=1, padding=1)
                cutoff = 0.06 + 0.18 * motion_threshold
                weights = weights / (1.0 + (difference.unsqueeze(2) / cutoff).pow(4))
            latents = torch.stack([latent for _, latent, _ in self.history]).float()
            target = (current_float + (latents * weights).sum(dim=0)) / (1.0 + weights.sum(dim=0))
        output = torch.lerp(current_float, target, strength).to(dtype=current.dtype)
        # Clones own storage independently of TensorRT's borrowed output buffer.
        self.history.append((now, current.detach().clone(),
                             guide.detach().clone() if guide is not None else None))
        return output


def camera_guide(image: Any, latent: Any) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    pixels = np.array(image.resize((latent.shape[-1], latent.shape[-2]),
                                  Image.Resampling.BILINEAR), copy=True)
    guide = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    guide = guide.to(device=latent.device, dtype=torch.float32).div_(255.0)
    return functional.avg_pool2d(guide, 3, stride=1, padding=1, count_include_pad=False)
