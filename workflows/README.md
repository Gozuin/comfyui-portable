# Workflow Directory

Put host-independent ComfyUI API-format workflows here if they should be
discovered automatically by `setup`. Export them from ComfyUI with
`Workflow > Export (API)`.

Only generic graph structure belongs in Git. Keep model filenames in descriptors
as loader targets, then let setup resolve the actual local filename into
`config.local.json`.

You may also pass a workflow from anywhere:

```bash
python scripts/comfyui_portable.py setup \
  --workflow txt2img="/absolute/path/to/txt2img.api.json"
```
