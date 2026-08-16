from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from .config import AppConfig
from .engine import StreamDiffusionEngine
from .spout_transport import ReceivedFrame, SpoutInput, SpoutOutput


@dataclass(slots=True)
class RuntimeMetrics:
    state: str = "停止"
    status: str = "待機中"
    input_fps: float = 0.0
    output_fps: float = 0.0
    inference_fps: float = 0.0
    inference_ms: float = 0.0
    input_resolution: str = "-"
    output_resolution: str = "-"
    input_frames: int = 0
    output_frames: int = 0
    send_errors: int = 0
    gpu_utilization: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    gpu_temperature: float = 0.0
    motion_score: float = 0.0
    temporal_feedback: float = 0.0
    latent_morph: float = 0.0


@dataclass(slots=True)
class WorkerEvent:
    kind: str
    data: Any


class RateMeter:
    def __init__(self, window_seconds: float = 1.5) -> None:
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()

    def tick(self, timestamp: float | None = None) -> None:
        now = timestamp if timestamp is not None else time.perf_counter()
        self.timestamps.append(now)
        self._trim(now)

    def value(self, timestamp: float | None = None) -> float:
        now = timestamp if timestamp is not None else time.perf_counter()
        self._trim(now)
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / duration if duration > 0 else 0.0

    def _trim(self, now: float) -> None:
        boundary = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < boundary:
            self.timestamps.popleft()


@dataclass(slots=True)
class InputSnapshot:
    fps: float
    frames: int
    resolution: str


class LatestFrameReceiver:
    """Receive Spout independently and keep only the newest camera frame."""

    def __init__(
        self,
        sender_name: str,
        flip_vertical: bool,
        parent_stop_event: threading.Event,
    ) -> None:
        self._sender_name = sender_name
        self._flip_vertical = flip_vertical
        self._parent_stop_event = parent_stop_event
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._latest_frame: ReceivedFrame | None = None
        self._sequence = 0
        self._frames = 0
        self._rate = RateMeter()
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="spout-input-receiver",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            with SpoutInput(self._sender_name, self._flip_vertical) as spout_input:
                while not self._should_stop():
                    frame = spout_input.receive()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    with self._condition:
                        self._latest_frame = frame
                        self._sequence += 1
                        self._frames += 1
                        self._rate.tick(frame.received_at)
                        self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def latest_after(
        self,
        sequence: int,
        timeout: float = 0.02,
    ) -> tuple[ReceivedFrame | None, int]:
        with self._condition:
            if (
                self._sequence <= sequence
                and self._error is None
                and not self._should_stop()
            ):
                self._condition.wait(timeout)
            if self._error is not None:
                raise RuntimeError(f"Spout入力の受信に失敗しました: {self._error}") from self._error
            if self._sequence <= sequence:
                return None, sequence
            return self._latest_frame, self._sequence

    def snapshot(self, now: float | None = None) -> InputSnapshot:
        with self._condition:
            frame = self._latest_frame
            resolution = f"{frame.width} × {frame.height}" if frame is not None else "-"
            return InputSnapshot(
                fps=self._rate.value(now),
                frames=self._frames,
                resolution=resolution,
            )

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _should_stop(self) -> bool:
        return self._stop_event.is_set() or self._parent_stop_event.is_set()


class GpuTelemetry:
    def __init__(self) -> None:
        self._pynvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None
            self._handle = None

    def sample(self) -> dict[str, float]:
        if self._pynvml is None or self._handle is None:
            return {}
        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temperature = self._pynvml.nvmlDeviceGetTemperature(
                self._handle,
                self._pynvml.NVML_TEMPERATURE_GPU,
            )
            return {
                "gpu_utilization": float(utilization.gpu),
                "vram_used_mb": memory.used / (1024 * 1024),
                "vram_total_mb": memory.total / (1024 * 1024),
                "gpu_temperature": float(temperature),
            }
        except Exception:
            return {}

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class DiffusionWorker:
    def __init__(self, events: "queue.Queue[WorkerEvent]") -> None:
        self.events = events
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._commands: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, config: AppConfig) -> None:
        if self.running:
            raise RuntimeError("推論はすでに実行中です。")
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(config,),
            name="stream-diffusion-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def update_prompt(self, prompt: str) -> None:
        self._commands.put(("prompt", prompt))

    def update_live_settings(self, settings: dict[str, Any]) -> None:
        """Queue settings that are safe to change without reloading the model."""
        self._commands.put(("live_settings", dict(settings)))

    def reset_temporal(self) -> None:
        self._commands.put(("reset_temporal", None))

    def _publish(self, kind: str, data: Any) -> None:
        event = WorkerEvent(kind=kind, data=data)
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

    def _log(self, message: str) -> None:
        self._publish("log", message)

    def _run(self, config: AppConfig) -> None:
        metrics = RuntimeMetrics(state="起動中", status="モデル読込中")
        output_rate = RateMeter()
        inference_rate = RateMeter()
        telemetry = GpuTelemetry()
        engine: StreamDiffusionEngine | None = None
        receiver: LatestFrameReceiver | None = None
        last_metrics_at = 0.0
        last_preview_at = 0.0
        last_inference_started_at = 0.0
        last_input_sequence = 0
        failed = False

        self._publish("state", "起動中")
        self._publish("metrics", asdict(metrics))
        try:
            engine = StreamDiffusionEngine(config, self._log)
            engine.load()
            metrics.state = "実行中"
            metrics.status = f"Spout入力待ち: {config.spout_input}"
            self._publish("state", "実行中")

            receiver = LatestFrameReceiver(
                config.spout_input,
                config.flip_input,
                self._stop_event,
            )
            receiver.start()
            with SpoutOutput(config.spout_output, config.flip_output) as spout_output:
                self._log(
                    f"Spout接続: {config.spout_input} → 推論 → {config.spout_output}"
                )
                self._log("Spout受信を推論スレッドから分離しました（常に最新フレームを使用）。")
                while not self._stop_event.is_set():
                    self._drain_commands(engine)
                    frame, sequence = receiver.latest_after(last_input_sequence)
                    now = time.perf_counter()
                    self._copy_input_metrics(metrics, receiver.snapshot(now))

                    if frame is None:
                        if now - last_metrics_at >= 0.2:
                            self._update_metrics(
                                metrics,
                                output_rate,
                                inference_rate,
                                telemetry,
                                now,
                            )
                            self._publish("metrics", asdict(metrics))
                            last_metrics_at = now
                        continue

                    last_input_sequence = sequence
                    minimum_interval = 1.0 / config.target_fps
                    if now - last_inference_started_at < minimum_interval:
                        continue

                    last_inference_started_at = now
                    metrics.status = "推論中"
                    output, inference_ms = engine.process(frame.image)
                    inference_rate.tick()
                    metrics.inference_ms = inference_ms
                    metrics.motion_score = engine.last_motion_score
                    metrics.temporal_feedback = engine.last_temporal_feedback
                    metrics.latent_morph = engine.last_latent_morph
                    metrics.output_resolution = f"{output.width} × {output.height}"

                    if spout_output.send(output):
                        output_rate.tick()
                        metrics.output_frames += 1
                        metrics.status = "送受信中"
                    else:
                        metrics.send_errors += 1
                        metrics.status = "Spout出力エラー"

                    completed_at = time.perf_counter()
                    self._copy_input_metrics(metrics, receiver.snapshot(completed_at))
                    if completed_at - last_preview_at >= 0.15:
                        self._publish(
                            "preview",
                            {
                                "input": self._preview(frame.image, config.preview_size),
                                "output": self._preview(output, config.preview_size),
                            },
                        )
                        last_preview_at = completed_at

                    if completed_at - last_metrics_at >= 0.2:
                        self._update_metrics(
                            metrics,
                            output_rate,
                            inference_rate,
                            telemetry,
                            completed_at,
                        )
                        self._publish("metrics", asdict(metrics))
                        last_metrics_at = completed_at

        except Exception as exc:
            failed = True
            metrics.state = "エラー"
            metrics.status = str(exc).splitlines()[0]
            self._publish("state", "エラー")
            self._publish("error", str(exc))
            self._log(traceback.format_exc())
        finally:
            metrics.state = "エラー" if failed else "停止"
            if not failed:
                metrics.status = "停止しました"
            self._publish("metrics", asdict(metrics))
            if receiver is not None:
                receiver.close()
            if engine is not None:
                engine.close()
            telemetry.close()
            self._publish("state", "エラー" if failed else "停止")
            self._log("推論処理はエラーで終了しました。" if failed else "推論処理を停止しました。")

    def _drain_commands(self, engine: StreamDiffusionEngine) -> None:
        prompt: str | None = None
        live_settings: dict[str, Any] = {}
        reset_temporal = False
        while True:
            try:
                command, value = self._commands.get_nowait()
            except queue.Empty:
                break
            if command == "prompt":
                prompt = str(value)
            elif command == "live_settings":
                live_settings.update(dict(value))
            elif command == "reset_temporal":
                reset_temporal = True

        # Apply only the latest value of each control. A slow 2-step frame can
        # otherwise leave several slider updates queued and feel delayed.
        if prompt is not None:
            engine.update_prompt(prompt)
        if live_settings:
            engine.update_live_settings(live_settings)
        if reset_temporal:
            engine.reset_temporal()

    @staticmethod
    def _update_metrics(
        metrics: RuntimeMetrics,
        output_rate: RateMeter,
        inference_rate: RateMeter,
        telemetry: GpuTelemetry,
        now: float,
    ) -> None:
        metrics.output_fps = output_rate.value(now)
        metrics.inference_fps = inference_rate.value(now)
        for key, value in telemetry.sample().items():
            setattr(metrics, key, value)

    @staticmethod
    def _copy_input_metrics(metrics: RuntimeMetrics, snapshot: InputSnapshot) -> None:
        metrics.input_fps = snapshot.fps
        metrics.input_frames = snapshot.frames
        metrics.input_resolution = snapshot.resolution

    @staticmethod
    def _preview(image: Image.Image, size: int) -> Image.Image:
        preview = image.copy()
        preview.thumbnail((size, size), Image.Resampling.BILINEAR)
        return preview
