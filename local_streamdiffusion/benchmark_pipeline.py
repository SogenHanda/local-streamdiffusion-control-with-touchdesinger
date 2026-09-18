from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from streamdiffusion_bridge.config import AppConfig, DEFAULT_CONFIG_PATH
from streamdiffusion_bridge.engine import StreamDiffusionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one inference backend without Spout.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--backend",
        choices=("auto", "tensorrt", "xformers", "pytorch"),
        default="auto",
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-frame", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> int:
    args = parse_args()
    if args.frames < 2 or args.warmup < 0:
        raise ValueError("--frames must be >= 2 and --warmup must be >= 0")
    config = replace(
        AppConfig.load(args.config.resolve()),
        acceleration_backend=args.backend,
        target_fps=240.0,
    )
    config.validate()
    if args.input:
        frame = Image.open(args.input).convert("RGB")
    else:
        frame = Image.effect_noise((config.width, config.height), 48.0).convert("RGB")

    engine = StreamDiffusionEngine(config, print)
    timings: list[float] = []
    frame_timings: list[float] = []
    stages: dict[str, list[float]] = {}
    last_output: Image.Image | None = None
    try:
        engine.load()
        for _ in range(args.warmup):
            engine.process(frame)
        for _ in range(args.frames):
            frame_started = time.perf_counter()
            last_output, elapsed_ms = engine.process(frame)
            frame_timings.append((time.perf_counter() - frame_started) * 1000.0)
            timings.append(elapsed_ms)
            for name, value in engine.last_stage_metrics.items():
                stages.setdefault(name, []).append(value)

        torch = engine._torch
        gpu_name = torch.cuda.get_device_name(0) if torch is not None else "-"
        assert last_output is not None
        grayscale = last_output.convert("L")
        histogram = grayscale.histogram()
        pixel_count = max(last_output.width * last_output.height, 1)
        result: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "requested_backend": args.backend,
            "active_backend": engine.active_backend,
            "engine_status": engine.engine_status,
            "gpu": gpu_name,
            "model": str(config.resolved_model_path),
            "resolution": [config.width, config.height],
            "steps": config.lcm_steps,
            "tiny_vae": config.use_tiny_vae,
            "frames": args.frames,
            "mean_ms": statistics.fmean(timings),
            "p50_ms": statistics.median(timings),
            "p95_ms": percentile(timings, 0.95),
            "fps": 1000.0 / statistics.fmean(timings),
            "frame_mean_ms": statistics.fmean(frame_timings),
            "frame_fps": 1000.0 / statistics.fmean(frame_timings),
            "frame_p95_ms": percentile(frame_timings, 0.95),
            "cuda_graph": config.tensorrt_cuda_graph,
            "temporal_filter": "finite_window_camera_guided_v1",
            "stage_mean_ms": {
                name: statistics.fmean(values) for name, values in stages.items()
            },
            "output_mean_rgb": [round(value, 3) for value in ImageStat.Stat(last_output).mean],
            "output_black_ratio": sum(histogram[:5]) / pixel_count,
            "output_white_ratio": sum(histogram[251:]) / pixel_count,
        }
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            last_output.save(args.save_frame)
    finally:
        engine.close()

    output_path = args.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("benchmarks") / f"{args.backend}-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
