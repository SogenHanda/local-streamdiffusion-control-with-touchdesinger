# Local StreamDiffusion + TouchDesigner

A Windows-first, fully local realtime image-generation system. TouchDesigner
sends camera frames to a Python StreamDiffusion backend through Spout, controls
generation over OSC, and receives the generated frames through Spout.

## Repository layout

```text
local_streamdiffusion/       Python / CUDA / StreamDiffusion backend
local_streamdiffusion_td/    TouchDesigner project and helper scripts
```

Downloaded models, virtual environments, TensorRT engines, TouchDesigner
backups, and local media are intentionally excluded from Git.
Machine-specific runtime settings are stored in the ignored
`local_streamdiffusion/config.json`; `config.example.json` is the reusable
template tracked by the repository.

## Quick start

1. Clone this repository on Windows.
2. Open PowerShell in `local_streamdiffusion`.
3. Run `setup.ps1`, or `setup_tensorrt.ps1` for the TensorRT environment.
4. Download the desired models with `download_models.ps1`.
5. Open `local_streamdiffusion_td/local_streamdiffusion_td.toe`.

The included `.toe` already embeds the generic launcher. The standalone
`local_streamdiffusion_td/tools/local_streamdiffusion_launcher.py` is kept as
the editable source for future launcher updates.

The launcher discovers the sibling Python directory automatically. Override it
only when necessary with the `LOCAL_STREAMDIFFUSION_DIR` environment variable.

OSC control uses UDP port `13001` with the `/streamdiffusion` namespace.
Monitoring data is sent to `127.0.0.1:9001`.

See [backend documentation](local_streamdiffusion/README.md) and
[TouchDesigner documentation](local_streamdiffusion_td/tools/README.md) for
the full setup and OSC address list.
