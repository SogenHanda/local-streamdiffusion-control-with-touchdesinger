# Local StreamDiffusion project structure

This Git repository contains two sibling applications:

```text
local_streamdiffusion/       Python inference backend
local_streamdiffusion_td/    TouchDesigner controller and project
```

The Python backend owns StreamDiffusion, CUDA/TensorRT, OSC receive, and Spout
input/output. The TouchDesigner project owns the operator UI and sends runtime
settings over OSC.

Large machine-specific data stays local and is excluded from Git:

- Python virtual environments and caches
- downloaded diffusion models
- TensorRT engines and benchmark output
- TouchDesigner backup files and local movie/audio assets

The active TouchDesigner project is
`local_streamdiffusion_td/local_streamdiffusion_td.toe`. Supporting Text DAT
scripts are in `local_streamdiffusion_td/tools/`.
