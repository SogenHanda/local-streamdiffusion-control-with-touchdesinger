from __future__ import annotations

import queue
import socket
import struct
import unittest

from ergonomics_diffusion.config import AppConfig
from ergonomics_diffusion.model_catalog import BUILTIN_MODEL_PROFILES
from ergonomics_diffusion.osc_control import (
    OSCEvent,
    OSCMessage,
    OSCUDPServer,
    apply_osc_config,
    decode_control_message,
    decode_osc_packet,
)


def osc_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return encoded + b"\0" * ((-len(encoded)) % 4)


def osc_message(address: str, tags: str, *values: object) -> bytes:
    packet = osc_string(address) + osc_string("," + tags)
    for tag, value in zip(tags, values):
        if tag == "i":
            packet += struct.pack(">i", int(value))
        elif tag == "f":
            packet += struct.pack(">f", float(value))
        elif tag == "s":
            packet += osc_string(str(value))
        else:
            raise AssertionError(f"unsupported test tag: {tag}")
    return packet


class OSCDecodeTests(unittest.TestCase):
    def test_decodes_touchdesigner_types(self) -> None:
        packet = osc_message(
            "/ergonomics/live/prompt",
            "sif",
            "organic chair",
            4,
            0.75,
        )
        message = decode_osc_packet(packet)[0]
        self.assertEqual(message.address, "/ergonomics/live/prompt")
        self.assertEqual(message.args[:2], ("organic chair", 4))
        self.assertAlmostEqual(message.args[2], 0.75)

    def test_decodes_bundle(self) -> None:
        first = osc_message("/ergonomics/config/width", "i", 768)
        second = osc_message("/ergonomics/config/height", "i", 512)
        bundle = (
            b"#bundle\0"
            + b"\0" * 8
            + struct.pack(">i", len(first))
            + first
            + struct.pack(">i", len(second))
            + second
        )
        messages = decode_osc_packet(bundle)
        self.assertEqual([message.args[0] for message in messages], [768, 512])


class OSCControlTests(unittest.TestCase):
    def test_strength_maps_zero_to_input_and_one_to_generation(self) -> None:
        zero = decode_control_message(
            OSCMessage("/ergonomics/live/strength", (0.0,))
        )
        one = decode_control_message(
            OSCMessage("/ergonomics/live/strength", (1.0,))
        )
        self.assertIsNotNone(zero)
        self.assertIsNotNone(one)
        self.assertEqual(zero.value, 49)
        self.assertEqual(one.value, 0)

    def test_button_falling_edge_does_not_trigger(self) -> None:
        command = decode_control_message(
            OSCMessage("/ergonomics/system/start", (0,))
        )
        self.assertIsNone(command)

    def test_temporal_controls_accept_zero_to_one(self) -> None:
        cases = (
            ("/ergonomics/live/output_smoothing", 1.0),
            ("/ergonomics/live/latent_morph", 1.0),
            ("/ergonomics/live/motion_threshold", 0.0),
        )
        for address, value in cases:
            with self.subTest(address=address, value=value):
                command = decode_control_message(OSCMessage(address, (value,)))
                self.assertIsNotNone(command)
                self.assertEqual(command.value, value)
        with self.assertRaises(ValueError):
            decode_control_message(
                OSCMessage("/ergonomics/live/output_smoothing", (1.01,))
            )

    def test_rejects_invalid_resolution(self) -> None:
        with self.assertRaises(ValueError):
            decode_control_message(
                OSCMessage("/ergonomics/config/width", (513,))
            )

    def test_applies_indices_and_python_owned_settings(self) -> None:
        base = AppConfig(
            flip_input=True,
            flip_output=True,
            offline_mode=False,
            tensorrt_cuda_graph=True,
        )
        config = apply_osc_config(
            base,
            {
                "model_index": 2,
                "backend_index": 2,
                "performance_index": 1,
                "width": 768,
                "height": 512,
                "spout_input": "TD_Camera_Main",
                "spout_output": "AI_Output_Main",
                "spout_sample_fps": 60.0,
            },
        )
        self.assertEqual(config.model_profile, BUILTIN_MODEL_PROFILES[2].key)
        self.assertEqual(config.acceleration_backend, "xformers")
        self.assertEqual((config.lcm_steps, config.use_tiny_vae), (2, True))
        self.assertEqual((config.width, config.height), (768, 512))
        self.assertFalse(config.flip_input)
        self.assertFalse(config.flip_output)
        self.assertTrue(config.offline_mode)
        self.assertFalse(config.tensorrt_cuda_graph)


class OSCServerTests(unittest.TestCase):
    def test_receives_udp_message(self) -> None:
        events: "queue.Queue[OSCEvent]" = queue.Queue()
        server = OSCUDPServer(events, "127.0.0.1", 0)
        server.start()
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sender.sendto(
                    osc_message("/ergonomics/live/target_fps", "f", 30.0),
                    ("127.0.0.1", server.port),
                )
            finally:
                sender.close()
            event = events.get(timeout=1.0)
            self.assertEqual(event.kind, "message")
            self.assertIsInstance(event.data, OSCMessage)
            self.assertEqual(event.data.address, "/ergonomics/live/target_fps")
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
