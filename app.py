from __future__ import annotations

import argparse
from pathlib import Path

from ergonomics_diffusion.config import DEFAULT_CONFIG_PATH
from ergonomics_diffusion.ui import run_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="TouchDesigner Spout local diffusion bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()
    run_ui(config_path=args.config.resolve())


if __name__ == "__main__":
    main()

