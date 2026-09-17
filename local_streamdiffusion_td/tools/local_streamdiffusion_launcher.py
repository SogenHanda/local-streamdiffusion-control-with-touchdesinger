"""TouchDesigner-side launcher for the local inference application.

Paste or drag this file into a Text DAT named ``local_streamdiffusion_launcher``.
Use ``startup()``, ``set_run(value)``, and ``shutdown()`` from Execute DATs.
The sibling ``local_streamdiffusion`` directory is detected automatically.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from pathlib import Path


def _resolve_app_dir() -> Path:
    override = os.environ.get("LOCAL_STREAMDIFFUSION_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    td_project = globals().get("project")
    project_folder = getattr(td_project, "folder", "") if td_project else ""
    if project_folder:
        return (Path(project_folder).resolve().parent / "local_streamdiffusion")

    script_file = globals().get("__file__")
    if script_file:
        return Path(script_file).resolve().parents[2] / "local_streamdiffusion"

    return Path.cwd().resolve().parent / "local_streamdiffusion"


APP_DIR = _resolve_app_dir()
OSC_HOST = "127.0.0.1"
OSC_PORT = 13001
USE_TENSORRT = True
HIDE_MONITOR = False

# Keep the DAT compatible with TouchDesigner builds that still embed Python 3.9.
_process = None


def is_running() -> bool:
    return (
        (_process is not None and _process.poll() is None)
        or _controller_is_listening()
    )


def _controller_is_listening() -> bool:
    """Detect an existing app even after this Text DAT has been reloaded."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((OSC_HOST, OSC_PORT))
    except OSError:
        return True
    finally:
        probe.close()
    return False


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
    """Launch the controller process and optionally start generation."""
    global _process

    if is_running():
        print("Local inference is already running; duplicate launch blocked.")
        return False

    environment_name = ".venv-trt" if USE_TENSORRT else ".venv"
    python_exe = APP_DIR / environment_name / "Scripts" / "python.exe"
    app_py = APP_DIR / "app.py"
    if not python_exe.is_file():
        raise FileNotFoundError(
            f"Python environment was not found: {python_exe}\n"
            + (
                "Run setup_tensorrt.ps1 in local_streamdiffusion first."
                if USE_TENSORRT
                else "Run setup.ps1 in local_streamdiffusion first."
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
    if HIDE_MONITOR:
        command.append("--hidden")
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


def startup(run_value=0) -> bool:
    """Launch only when the saved/current TouchDesigner Run value is 1."""
    should_generate = bool(round(float(run_value)))
    if not should_generate:
        print("Run is 0; local inference was not launched.")
        return False
    return start(autostart=True)


def set_run(value) -> None:
    """Run=1 launches once. Run=0 intentionally does not stop inference."""
    should_run = bool(round(float(value)))
    if should_run:
        start(autostart=True)


def start_generation() -> None:
    """Launch and start generation unless an instance already exists."""
    start(autostart=True)


def stop_generation() -> None:
    """Fully close inference so the next Run starts from a clean state."""
    if not _controller_is_listening():
        print("Local inference is not running.")
        return
    _send_trigger("/streamdiffusion/system/stop")
    print("Generation Stop sent; local inference is shutting down.")


def shutdown() -> bool:
    """Close the Python process when the TouchDesigner project exits."""
    return stop()


def stop() -> bool:
    """Close the one application instance and its full Windows process tree."""
    global _process

    controller_running = _controller_is_listening()
    local_running = _process is not None and _process.poll() is None
    if not controller_running and not local_running:
        _process = None
        print("Local inference is not running.")
        return False

    if controller_running:
        _send_trigger("/streamdiffusion/system/shutdown")
        deadline = time.perf_counter() + 3.0
        while _controller_is_listening() and time.perf_counter() < deadline:
            time.sleep(0.05)

    # A Windows venv python.exe can be a redirector process. /T also closes its
    # real Python child, preventing an orphan from continuing to publish Spout.
    if _process is not None and _process.poll() is None:
        subprocess.run(
            ["taskkill", "/PID", str(_process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    _process = None
    print("Local inference stopped.")
    return True
