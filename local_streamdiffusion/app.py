from __future__ import annotations

import argparse
from pathlib import Path

from streamdiffusion_bridge.config import DEFAULT_CONFIG_PATH
from streamdiffusion_bridge.ui import run_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="TouchDesigner Spout local diffusion bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="Start generation automatically after the UI is ready",
    )
    window_mode = parser.add_mutually_exclusive_group()
    window_mode.add_argument(
        "--hidden",
        dest="hidden",
        action="store_true",
        help="Keep the Tk monitor running without showing its window",
    )
    window_mode.add_argument(
        "--show-monitor",
        dest="hidden",
        action="store_false",
        help="Show the Python monitor window for debugging",
    )
    parser.set_defaults(hidden=False)
    args = parser.parse_args()
    run_ui(
        config_path=args.config.resolve(),
        autostart=args.autostart,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
