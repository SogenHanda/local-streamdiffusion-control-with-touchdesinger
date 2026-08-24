"""TouchDesigner-side launcher for the local inference application.

Paste or drag this file into a Text DAT named ``local_inference_launcher``.
Use ``startup()``, ``set_run(value)``, and ``shutdown()`` from Execute DATs.
Set APP_DIR when the repository is installed somewhere else.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
from pathlib import Path


APP_DIR = Path(
    r"C:\Users\sogen\Documents\client works\ergonomics_shinhanagata\local_inference"
)
OSC_HOST = "127.0.0.1"
OSC_PORT = 13001
USE_TENSORRT = True

# Keep the DAT compatible with TouchDesigner builds that still embed Python 3.9.
_process = None


def is_running() -> bool:
    return _process is not None and _process.poll() is None


def _osc_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return encoded + b"\0" * ((-len(encoded)) % 4)


def _send_trigger(address: str) -> None:
    """Send a single OSC int trigger without depending on a TouchDesigner OP."""
    packet = _osc_string(address) + _osc_string(",i") + struct.pack(">i", 1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(packet, (OSC_HOST, OSC_PORT))
    finally:
        sender.close()


def start(autostart: bool = True) -> bool:
    """Launch the monitoring UI and optionally start generation."""
    global _process

    if is_running():
        if autostart:
            _send_trigger("/ergonomics/system/start")
        print("Local inference is already running.")
        return False

    environment_name = ".venv-trt" if USE_TENSORRT else ".venv"
    python_exe = APP_DIR / environment_name / "Scripts" / "python.exe"
    app_py = APP_DIR / "app.py"
    if not python_exe.is_file():
        raise FileNotFoundError(
            f"Python environment was not found: {python_exe}\n"
            + (
                "Run setup_tensorrt.ps1 in local_inference first."
                if USE_TENSORRT
                else "Run setup.ps1 in local_inference first."
            )
        )
    if not app_py.is_file():
        raise FileNotFoundError(f"Application was not found: {app_py}")

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HOME": str(APP_DIR / ".cache" / "huggingface"),
            "PYTHONUTF8": "1",
        }
    )
    command = [str(python_exe), str(app_py)]
    if autostart:
        command.append("--autostart")

    _process = subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    print(f"Local inference launched (PID {_process.pid}).")
    return True


def startup(run_value=1) -> bool:
    """Launch Python and generate by default when the .toe opens."""
    should_generate = bool(round(float(run_value)))
    return start(autostart=should_generate)


def set_run(value) -> None:
    """Connect a TouchDesigner Run value (0/1) to generation start/stop."""
    should_run = bool(round(float(value)))
    if should_run:
        if is_running():
            _send_trigger("/ergonomics/system/start")
            print("Generation start sent to local inference.")
        else:
            # --autostart is reliable even before the OSC listener has opened.
            start(autostart=True)
    elif is_running():
        _send_trigger("/ergonomics/system/stop")
        print("Generation stop sent to local inference.")


def shutdown() -> bool:
    """Close the Python process when the TouchDesigner project exits."""
    return stop()


def stop() -> bool:
    """Stop the process started by this TouchDesigner session."""
    global _process

    if not is_running():
        _process = None
        print("Local inference is not running.")
        return False

    _process.terminate()
    _process = None
    print("Local inference stopped.")
    return True
