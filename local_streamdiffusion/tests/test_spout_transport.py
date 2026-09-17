from __future__ import annotations

import array
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from streamdiffusion_bridge.spout_transport import SpoutInput, SpoutOutput


class FakeReceiver:
    def __init__(self, frame_new: bool = True) -> None:
        self.frame_new = frame_new

    def receiveImage(self, *_: object) -> bool:
        return True

    def isUpdated(self) -> bool:
        return False

    def isFrameCountEnabled(self) -> bool:
        return True

    def isFrameNew(self) -> bool:
        return self.frame_new


class FakeSender:
    def __init__(self) -> None:
        self.arguments: tuple[object, ...] | None = None
        self.frame_sync_name: str | None = None

    def sendImage(self, *arguments: object) -> bool:
        self.arguments = arguments
        return True

    def setFrameSync(self, name: str) -> None:
        self.frame_sync_name = name


class SpoutTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        gl = types.SimpleNamespace(GL_RGBA=6408)
        self.modules = {
            "OpenGL": types.SimpleNamespace(GL=gl),
            "OpenGL.GL": gl,
        }

    def test_receiver_ignores_duplicate_frame(self) -> None:
        receiver = SpoutInput.__new__(SpoutInput)
        receiver._receiver = FakeReceiver(frame_new=False)
        receiver._buffer = array.array("B", [255, 0, 0, 255])
        receiver._width = 1
        receiver._height = 1
        receiver._flip_vertical = False
        receiver._spout_module = types.SimpleNamespace(
            helpers=types.SimpleNamespace(isBufferEmpty=lambda _: False)
        )
        with patch.dict(sys.modules, self.modules):
            self.assertIsNone(receiver.receive())

    def test_receiver_returns_rgb_image(self) -> None:
        receiver = SpoutInput.__new__(SpoutInput)
        receiver._receiver = FakeReceiver(frame_new=True)
        receiver._buffer = array.array("B", [255, 0, 0, 255])
        receiver._width = 1
        receiver._height = 1
        receiver._flip_vertical = False
        receiver._spout_module = types.SimpleNamespace(
            helpers=types.SimpleNamespace(isBufferEmpty=lambda _: False)
        )
        with patch.dict(sys.modules, self.modules):
            frame = receiver.receive()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.image.mode, "RGB")
        self.assertEqual(frame.image.getpixel((0, 0)), (255, 0, 0))

    def test_sender_uses_spoutgl_height_width_order(self) -> None:
        sender = SpoutOutput.__new__(SpoutOutput)
        sender._sender = FakeSender()
        sender._sender_name = "AI_Output"
        sender._flip_vertical = False
        image = Image.new("RGB", (2, 3), "red")
        with patch.dict(sys.modules, self.modules):
            self.assertTrue(sender.send(image))
        assert sender._sender.arguments is not None
        self.assertEqual(sender._sender.arguments[1:3], (3, 2))
        self.assertEqual(sender._sender.frame_sync_name, "AI_Output")


if __name__ == "__main__":
    unittest.main()

