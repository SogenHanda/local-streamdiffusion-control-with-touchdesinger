# Codex workspace instructions

Before changing this repository, read `CODEX_HANDOFF.md`, the root `README.md`,
and the README belonging to the component you will edit.

## Active scope

- `local_streamdiffusion/` is the active Python inference backend.
- `local_streamdiffusion_td/` is the active TouchDesigner controller.
- Directories named `archive_*`, `assets_source_video_edit`, and
  `reference_tox_analysis` are local historical/reference material. They are
  outside the active product and are intentionally ignored by Git.

## Compatibility rules

- Keep the OSC namespace `/streamdiffusion` unless the user explicitly asks
  for a protocol migration.
- Preserve the default Spout names `TD_Camera` and `AI_Output`.
- The Python window is a read-only monitor; TouchDesigner owns operational
  controls and sends them over OSC.
- Run=0 does not stop inference. `/streamdiffusion/system/stop 1` fully closes
  the Python app so the next Run starts a clean process.
- Do not commit models, virtual environments, TensorRT engines, `config.json`,
  benchmarks, TouchDesigner backups, or local media.
- TensorRT engines are specific to GPU, model, resolution, step count, LoRA,
  and runtime. Build them on the target PC rather than copying them.
- The standalone `tools/local_streamdiffusion_launcher.py` is the generic
  launcher source. The embedded Text DAT may contain a path saved on another
  machine; synchronize it or update `APP_DIR` before first launch on a new PC.

## Verification

Run the Python test suite after backend changes:

```powershell
cd .\local_streamdiffusion
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For TensorRT-related changes, also run:

```powershell
.\.venv-trt\Scripts\python.exe .\verify_tensorrt.py
```

Treat `local_streamdiffusion_td/local_streamdiffusion_td.toe` as a binary
artifact. Edit it through TouchDesigner or use Derivative's official
`toeexpand` / `toecollapse` tools, and retain a recoverable backup.
