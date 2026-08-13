from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from ergonomics_diffusion.runtime import LatestFrameReceiver
from ergonomics_diffusion.spout_transport import ReceivedFrame


class FakeSpoutInput:
    def __init__(self, _sender_name: str, _flip_vertical: bool) -> None:
        self.index = 0

    def receive(self) -> ReceivedFrame | None:
        if self.index >= 3:
            time.sleep(0.001)
            return None
        self.index += 1
        return ReceivedFrame(
            image=Image.new("RGB", (16, 16), (self.index, 0, 0)),
            width=16,
            height=16,
            received_at=time.perf_counter(),
        )

    def __enter__(self) -> "FakeSpoutInput":
        return self

    def __exit__(self, *_args: object) -> None:
        pass


class LatestFrameReceiverTests(unittest.TestCase):
    def test_receiver_keeps_newest_frame_and_counts_all_inputs(self) -> None:
        stop_event = threading.Event()
        with patch("ergonomics_diffusion.runtime.SpoutInput", FakeSpoutInput):
            receiver = LatestFrameReceiver("camera", False, stop_event)
            receiver.start()
            deadline = time.perf_counter() + 1.0
            while receiver.snapshot().frames < 3 and time.perf_counter() < deadline:
                time.sleep(0.005)

            frame, sequence = receiver.latest_after(0)
            snapshot = receiver.snapshot()
            receiver.close()

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(sequence, 3)
        self.assertEqual(frame.image.getpixel((0, 0)), (3, 0, 0))
        self.assertEqual(snapshot.frames, 3)
        self.assertEqual(snapshot.resolution, "16 × 16")
if __name__ == "__main__":
    unittest.main()
