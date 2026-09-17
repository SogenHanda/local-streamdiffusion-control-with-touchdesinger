from __future__ import annotations

import array
import time
from dataclasses import dataclass
from itertools import repeat

from PIL import Image


@dataclass(slots=True)
class ReceivedFrame:
    image: Image.Image
    width: int
    height: int
    received_at: float
    receive_ms: float = 0.0
    source_fps: float = 0.0


class SpoutInput:
    """CPU image receiver backed by SpoutGL's receiveImage API."""

    def __init__(self, sender_name: str, flip_vertical: bool = False) -> None:
        import SpoutGL

        self._spout_module = SpoutGL
        self._receiver = SpoutGL.SpoutReceiver()
        self._receiver.setReceiverName(sender_name)
        self._sender_name = sender_name
        self._flip_vertical = flip_vertical
        self._buffer: array.array[int] | None = None
        self._width = 0
        self._height = 0
        self._saw_non_empty = False
        self._last_sender_frame: int | None = None
        self._last_sender_frame_at = 0.0
        self._source_fps_ema = 0.0

    def receive(self) -> ReceivedFrame | None:
        from OpenGL import GL

        started = time.perf_counter()
        result = self._receiver.receiveImage(
            self._buffer,
            GL.GL_RGBA,
            self._flip_vertical,
            0,
        )

        if self._receiver.isUpdated():
            width = int(self._receiver.getSenderWidth())
            height = int(self._receiver.getSenderHeight())
            if width > 0 and height > 0:
                self._width = width
                self._height = height
                self._buffer = array.array("B", repeat(0, width * height * 4))
                self._saw_non_empty = False
            return None

        if not result or self._buffer is None or self._width <= 0 or self._height <= 0:
            return None

        # ReceiveImage can return the currently held texture more than once. Only
        # forward frames whose Spout frame counter actually advanced.
        if self._receiver.isFrameCountEnabled() and not self._receiver.isFrameNew():
            return None

        if not getattr(self, "_saw_non_empty", False):
            if self._spout_module.helpers.isBufferEmpty(self._buffer):
                return None
            self._saw_non_empty = True

        rgba = Image.frombuffer(
            "RGBA",
            (self._width, self._height),
            self._buffer,
            "raw",
            "RGBA",
            0,
            1,
        )
        source_sampled_at = time.perf_counter()
        source_fps = float(getattr(self, "_source_fps_ema", 0.0))
        get_sender_frame = getattr(self._receiver, "getSenderFrame", None)
        if callable(get_sender_frame):
            try:
                sender_frame = int(get_sender_frame())
                last_frame = getattr(self, "_last_sender_frame", None)
                last_at = float(getattr(self, "_last_sender_frame_at", 0.0))
                if (
                    last_frame is not None
                    and sender_frame >= last_frame
                    and source_sampled_at > last_at
                ):
                    instantaneous = (sender_frame - last_frame) / (source_sampled_at - last_at)
                    previous = float(getattr(self, "_source_fps_ema", 0.0))
                    source_fps = instantaneous if previous <= 0.0 else previous * 0.8 + instantaneous * 0.2
                    self._source_fps_ema = source_fps
                self._last_sender_frame = sender_frame
                self._last_sender_frame_at = source_sampled_at
            except (TypeError, ValueError, RuntimeError):
                pass
        rgb = rgba.convert("RGB")
        received_at = time.perf_counter()
        return ReceivedFrame(
            image=rgb,
            width=self._width,
            height=self._height,
            received_at=received_at,
            receive_ms=(received_at - started) * 1000.0,
            source_fps=source_fps,
        )

    def close(self) -> None:
        try:
            self._receiver.releaseReceiver()
        except (AttributeError, RuntimeError):
            pass

    def __enter__(self) -> "SpoutInput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SpoutOutput:
    """CPU image sender backed by SpoutGL's sendImage API."""

    def __init__(self, sender_name: str, flip_vertical: bool = False) -> None:
        import SpoutGL

        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(sender_name)
        self._sender_name = sender_name
        self._flip_vertical = flip_vertical

    def send(self, image: Image.Image) -> bool:
        from OpenGL import GL

        rgba = image.convert("RGBA")
        width, height = rgba.size
        pixels = rgba.tobytes()
        result = bool(
            self._sender.sendImage(
                pixels,
                height,
                width,
                GL.GL_RGBA,
                self._flip_vertical,
                0,
            )
        )
        if result:
            self._sender.setFrameSync(self._sender_name)
        return result

    def close(self) -> None:
        try:
            self._sender.releaseSender()
        except (AttributeError, RuntimeError):
            pass

    def __enter__(self) -> "SpoutOutput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
