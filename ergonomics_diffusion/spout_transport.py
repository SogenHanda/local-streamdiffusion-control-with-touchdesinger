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

    def receive(self) -> ReceivedFrame | None:
        from OpenGL import GL

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
            return None

        if not result or self._buffer is None or self._width <= 0 or self._height <= 0:
            return None

        # ReceiveImage can return the currently held texture more than once. Only
        # forward frames whose Spout frame counter actually advanced.
        if self._receiver.isFrameCountEnabled() and not self._receiver.isFrameNew():
            return None

        if self._spout_module.helpers.isBufferEmpty(self._buffer):
            return None

        rgba = Image.frombuffer(
            "RGBA",
            (self._width, self._height),
            self._buffer,
            "raw",
            "RGBA",
            0,
            1,
        ).copy()
        return ReceivedFrame(
            image=rgba.convert("RGB"),
            width=self._width,
            height=self._height,
            received_at=time.perf_counter(),
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
