from __future__ import annotations

import queue
import socket
import struct
import threading
from dataclasses import dataclass, replace
from typing import Any

from .config import AppConfig
from .model_catalog import BUILTIN_MODEL_PROFILES, PERFORMANCE_PRESETS


OSC_HOST = "127.0.0.1"
OSC_PORT = 13001
OSC_CONFIG_DEBOUNCE_MS = 350
OSC_MONITOR_HOST = "127.0.0.1"
OSC_MONITOR_PORT = 9001
OSC_MONITOR_PREFIX = "/streamdiffusion/monitor"

BACKEND_INDEX = ("auto", "tensorrt", "xformers", "pytorch")


@dataclass(frozen=True, slots=True)
class OSCMessage:
    address: str
    args: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class OSCEvent:
    kind: str
    data: OSCMessage | str


@dataclass(frozen=True, slots=True)
class OSCCommand:
    category: str
    name: str
    value: Any = None


class OSCDecodeError(ValueError):
    pass


def _encode_padded_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return encoded + b"\0" * ((-len(encoded)) % 4)


def encode_osc_message(message: OSCMessage) -> bytes:
    """Encode one OSC 1.0 message using TouchDesigner-compatible types."""
    if not message.address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {message.address!r}")

    tags: list[str] = []
    payload = bytearray()
    for value in message.args:
        if isinstance(value, bool):
            tags.append("T" if value else "F")
        elif value is None:
            tags.append("N")
        elif isinstance(value, int):
            if -(2**31) <= value < 2**31:
                tags.append("i")
                payload.extend(struct.pack(">i", value))
            else:
                tags.append("h")
                payload.extend(struct.pack(">q", value))
        elif isinstance(value, float):
            tags.append("f")
            payload.extend(struct.pack(">f", value))
        elif isinstance(value, str):
            tags.append("s")
            payload.extend(_encode_padded_string(value))
        else:
            raise TypeError(f"Unsupported OSC value type: {type(value).__name__}")

    return (
        _encode_padded_string(message.address)
        + _encode_padded_string("," + "".join(tags))
        + bytes(payload)
    )


def encode_osc_bundle(messages: list[OSCMessage] | tuple[OSCMessage, ...]) -> bytes:
    """Encode messages as one immediately applicable OSC bundle."""
    packet = bytearray(b"#bundle\0")
    packet.extend(struct.pack(">Q", 1))  # OSC immediate timetag.
    for message in messages:
        encoded = encode_osc_message(message)
        packet.extend(struct.pack(">i", len(encoded)))
        packet.extend(encoded)
    return bytes(packet)


def _resolution_components(value: Any) -> tuple[int, int]:
    normalized = str(value or "").lower().replace("×", "x")
    parts = [part.strip() for part in normalized.split("x", 1)]
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def monitor_messages(metrics: dict[str, Any]) -> list[OSCMessage]:
    """Convert the monitor cards into stable OSC addresses and raw values."""
    state = str(metrics.get("state", "停止"))
    prefix = OSC_MONITOR_PREFIX
    messages = [
        OSCMessage(f"{prefix}/state", (state,)),
        OSCMessage(f"{prefix}/status", (str(metrics.get("status", "")),)),
        OSCMessage(f"{prefix}/running", (int(state in {"起動中", "実行中"}),)),
    ]

    float_fields = (
        "input_fps",
        "source_fps",
        "output_fps",
        "inference_fps",
        "inference_ms",
        "end_to_end_ms",
        "gpu_utilization",
        "vram_used_mb",
        "vram_total_mb",
        "gpu_temperature",
        "spout_receive_ms",
        "spout_send_ms",
        "input_age_ms",
        "preprocess_ms",
        "vae_encode_ms",
        "unet_ms",
        "vae_decode_ms",
        "postprocess_ms",
        "motion_score",
        "temporal_feedback",
        "latent_morph",
    )
    messages.extend(
        OSCMessage(f"{prefix}/{name}", (float(metrics.get(name, 0.0) or 0.0),))
        for name in float_fields
    )
    messages.append(
        OSCMessage(
            f"{prefix}/active_backend",
            (str(metrics.get("active_backend", "-")),),
        )
    )

    for label in ("input", "output"):
        value = str(metrics.get(f"{label}_resolution", "-") or "-")
        width, height = _resolution_components(value)
        messages.extend(
            (
                OSCMessage(f"{prefix}/{label}_resolution", (value,)),
                OSCMessage(f"{prefix}/{label}_width", (width,)),
                OSCMessage(f"{prefix}/{label}_height", (height,)),
            )
        )
    return messages


class OSCUDPSender:
    """Small reusable UDP sender for OSC messages and bundles."""

    def __init__(
        self,
        host: str = OSC_MONITOR_HOST,
        port: int = OSC_MONITOR_PORT,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._socket: socket.socket | None = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self._socket.settimeout(0.05)
        self._lock = threading.Lock()

    def send_bundle(self, messages: list[OSCMessage] | tuple[OSCMessage, ...]) -> None:
        packet = encode_osc_bundle(messages)
        with self._lock:
            sock = self._socket
            if sock is None:
                return
            sock.sendto(packet, (self.host, self.port))

    def close(self) -> None:
        with self._lock:
            sock = self._socket
            self._socket = None
            if sock is not None:
                sock.close()


def _read_padded_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\0", offset)
    if end < 0:
        raise OSCDecodeError("OSC文字列の終端がありません。")
    try:
        value = data[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OSCDecodeError("OSC文字列がUTF-8ではありません。") from exc
    next_offset = (end + 4) & ~3
    if next_offset > len(data):
        raise OSCDecodeError("OSC文字列のpaddingが不足しています。")
    return value, next_offset


def _decode_message(data: bytes) -> OSCMessage:
    address, offset = _read_padded_string(data, 0)
    if not address.startswith("/"):
        raise OSCDecodeError(f"OSC addressが不正です: {address!r}")
    if offset == len(data):
        return OSCMessage(address, ())

    tags, offset = _read_padded_string(data, offset)
    if not tags.startswith(","):
        raise OSCDecodeError("OSC type tagがありません。")

    args: list[Any] = []
    for tag in tags[1:]:
        if tag == "i":
            if offset + 4 > len(data):
                raise OSCDecodeError("OSC int32が途中で終了しています。")
            args.append(struct.unpack_from(">i", data, offset)[0])
            offset += 4
        elif tag == "f":
            if offset + 4 > len(data):
                raise OSCDecodeError("OSC float32が途中で終了しています。")
            args.append(struct.unpack_from(">f", data, offset)[0])
            offset += 4
        elif tag == "s":
            value, offset = _read_padded_string(data, offset)
            args.append(value)
        elif tag == "d":
            if offset + 8 > len(data):
                raise OSCDecodeError("OSC doubleが途中で終了しています。")
            args.append(struct.unpack_from(">d", data, offset)[0])
            offset += 8
        elif tag == "h":
            if offset + 8 > len(data):
                raise OSCDecodeError("OSC int64が途中で終了しています。")
            args.append(struct.unpack_from(">q", data, offset)[0])
            offset += 8
        elif tag == "T":
            args.append(True)
        elif tag == "F":
            args.append(False)
        elif tag in {"N", "I"}:
            args.append(None)
        else:
            raise OSCDecodeError(f"未対応のOSC type tagです: {tag}")
    return OSCMessage(address, tuple(args))


def decode_osc_packet(data: bytes) -> list[OSCMessage]:
    """Decode one OSC message or all messages in an OSC bundle."""
    if data.startswith(b"#bundle\0"):
        if len(data) < 16:
            raise OSCDecodeError("OSC bundleが短すぎます。")
        messages: list[OSCMessage] = []
        offset = 16  # '#bundle\0' followed by the 64-bit timetag.
        while offset < len(data):
            if offset + 4 > len(data):
                raise OSCDecodeError("OSC bundle element sizeがありません。")
            size = struct.unpack_from(">i", data, offset)[0]
            offset += 4
            if size <= 0 or offset + size > len(data):
                raise OSCDecodeError("OSC bundle element sizeが不正です。")
            messages.extend(decode_osc_packet(data[offset : offset + size]))
            offset += size
        return messages
    return [_decode_message(data)]


def _one_arg(message: OSCMessage) -> Any:
    if len(message.args) != 1:
        raise ValueError(
            f"{message.address} は値を1つ送ってください（受信: {len(message.args)}個）。"
        )
    return message.args[0]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} は数値で送ってください。")
    return float(value)


def _integer(value: Any, label: str) -> int:
    number = _number(value, label)
    rounded = int(round(number))
    if abs(number - rounded) > 1e-5:
        raise ValueError(f"{label} は整数で送ってください。")
    return rounded


def _ranged(value: Any, label: str, minimum: float, maximum: float) -> float:
    number = _number(value, label)
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} は {minimum}〜{maximum} の範囲で送ってください。")
    return number


def _triggered(message: OSCMessage) -> bool:
    if not message.args:
        return True
    if len(message.args) != 1:
        raise ValueError(f"{message.address} はtrigger値を1つ送ってください。")
    value = message.args[0]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    raise ValueError(f"{message.address} のtriggerは0または1で送ってください。")


def decode_control_message(message: OSCMessage) -> OSCCommand | None:
    """Validate an OSC message and convert it into an application command."""
    address = message.address

    if address == "/streamdiffusion/live/prompt":
        value = _one_arg(message)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("promptは空でない文字列を送ってください。")
        return OSCCommand("live", "prompt", value.strip())
    if address == "/streamdiffusion/live/strength":
        strength = _ranged(_one_arg(message), "strength", 0.0, 1.0)
        # The StreamDiffusion t-index runs in the opposite direction from the
        # artist-facing 0=input / 1=full generation strength.
        return OSCCommand("live", "denoise_index", int(round((1.0 - strength) * 49)))
    if address == "/streamdiffusion/live/seed":
        value = _integer(_one_arg(message), "seed")
        if not -1 <= value <= 2_147_483_647:
            raise ValueError("seedは -1〜2147483647 の範囲で送ってください。")
        return OSCCommand("live", "seed", value)
    if address == "/streamdiffusion/live/target_fps":
        value = _ranged(_one_arg(message), "target_fps", 0.1, 240.0)
        return OSCCommand("live", "target_fps", value)
    if address == "/streamdiffusion/live/input_feedback":
        value = _ranged(_one_arg(message), "input_feedback", 0.0, 0.8)
        return OSCCommand("live", "temporal_feedback", value)
    if address == "/streamdiffusion/live/output_smoothing":
        value = _ranged(_one_arg(message), "output_smoothing", 0.0, 1.0)
        return OSCCommand("live", "temporal_smoothing", value)
    if address == "/streamdiffusion/live/latent_morph":
        value = _ranged(_one_arg(message), "latent_morph", 0.0, 1.0)
        return OSCCommand("live", "latent_morph_strength", value)
    if address == "/streamdiffusion/live/history_frames":
        value = _integer(_one_arg(message), "history_frames")
        if not 2 <= value <= 8:
            raise ValueError("history_framesは2〜8の範囲で送ってください。")
        return OSCCommand("live", "latent_history_frames", value)
    if address == "/streamdiffusion/live/motion_threshold":
        value = _ranged(_one_arg(message), "motion_threshold", 0.0, 1.0)
        return OSCCommand("live", "scene_cut_threshold", value)

    if address == "/streamdiffusion/config/model_index":
        value = _integer(_one_arg(message), "model_index")
        if not 0 <= value < len(BUILTIN_MODEL_PROFILES):
            raise ValueError(f"model_indexは0〜{len(BUILTIN_MODEL_PROFILES) - 1}です。")
        return OSCCommand("config", "model_index", value)
    if address == "/streamdiffusion/config/backend_index":
        value = _integer(_one_arg(message), "backend_index")
        if not 0 <= value < len(BACKEND_INDEX):
            raise ValueError(f"backend_indexは0〜{len(BACKEND_INDEX) - 1}です。")
        return OSCCommand("config", "backend_index", value)
    if address in {"/streamdiffusion/config/width", "/streamdiffusion/config/height"}:
        name = address.rsplit("/", 1)[1]
        value = _integer(_one_arg(message), name)
        if value < 64 or value % 8:
            raise ValueError(f"{name}は64以上かつ8の倍数で送ってください。")
        return OSCCommand("config", name, value)
    if address == "/streamdiffusion/config/performance_index":
        value = _integer(_one_arg(message), "performance_index")
        if not 0 <= value < len(PERFORMANCE_PRESETS):
            raise ValueError(f"performance_indexは0〜{len(PERFORMANCE_PRESETS) - 1}です。")
        return OSCCommand("config", "performance_index", value)
    if address in {
        "/streamdiffusion/config/spout_input",
        "/streamdiffusion/config/spout_output",
    }:
        value = _one_arg(message)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{address.rsplit('/', 1)[1]}は空でない文字列を送ってください。")
        return OSCCommand("config", address.rsplit("/", 1)[1], value.strip())
    if address == "/streamdiffusion/config/spout_sample_fps":
        value = _ranged(_one_arg(message), "spout_sample_fps", 1.0, 240.0)
        return OSCCommand("config", "spout_sample_fps", value)

    operations = {
        "/streamdiffusion/config/apply": "apply",
        "/streamdiffusion/system/start": "start",
        "/streamdiffusion/system/stop": "stop",
        "/streamdiffusion/system/shutdown": "shutdown",
        "/streamdiffusion/temporal/reset": "reset_temporal",
    }
    if address in operations:
        if not _triggered(message):
            return None
        return OSCCommand("system", operations[address])

    raise ValueError(f"未対応のOSC addressです: {address}")


def apply_osc_config(base: AppConfig, updates: dict[str, Any]) -> AppConfig:
    """Build a validated restart configuration from staged OSC values."""
    values: dict[str, Any] = {}
    if "model_index" in updates:
        profile = BUILTIN_MODEL_PROFILES[int(updates["model_index"])]
        values.update(
            model_profile=profile.key,
            model_path=profile.model_path,
            use_lcm_lora=profile.use_lcm_lora,
        )
    if "backend_index" in updates:
        values["acceleration_backend"] = BACKEND_INDEX[int(updates["backend_index"])]
    if "performance_index" in updates:
        preset = PERFORMANCE_PRESETS[int(updates["performance_index"])]
        values.update(lcm_steps=preset.steps, use_tiny_vae=preset.use_tiny_vae)
    for name in ("width", "height", "spout_input", "spout_output", "spout_sample_fps"):
        if name in updates:
            values[name] = updates[name]

    # These are intentionally owned by Python rather than TouchDesigner.
    values.update(
        lcm_lora_path="models/lcm-lora-sdv1-5",
        tiny_vae_path="models/taesd",
        tensorrt_engine_root="engines/tensorrt",
        tensorrt_cuda_graph=False,
        flip_input=False,
        flip_output=False,
        offline_mode=True,
    )
    config = replace(base, **values)
    config.validate()
    return config


class OSCUDPServer:
    """Small OSC 1.0 UDP receiver with no third-party runtime dependency."""

    def __init__(
        self,
        events: "queue.Queue[OSCEvent]",
        host: str = OSC_HOST,
        port: int = OSC_PORT,
    ) -> None:
        self.events = events
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        try:
            sock.bind((self.host, self.port))
        except Exception:
            sock.close()
            raise
        self.port = int(sock.getsockname()[1])
        self._socket = sock
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="streamdiffusion-osc-receiver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            sock.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        self._thread = None

    def _put(self, event: OSCEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                break
            try:
                packet, _sender = sock.recvfrom(65_535)
            except socket.timeout:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    self._put(OSCEvent("error", "OSC受信socketが終了しました。"))
                break
            try:
                for message in decode_osc_packet(packet):
                    self._put(OSCEvent("message", message))
            except (OSCDecodeError, ValueError) as exc:
                self._put(OSCEvent("error", f"OSC packetを解釈できません: {exc}"))
