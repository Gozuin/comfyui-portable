# Setup

## What Setup Locks

`setup` writes a local profile containing:

- ComfyUI server URL.
- Optional local installation root, embedded Python, input directory, and output
  directory.
- Workflow file paths.
- Semantic bindings such as prompt, negative prompt, seed, steps, dimensions,
  references, and model loader inputs.
- Default parameter values.
- The exact model filename selected for each model input on this computer.

The profile is written to `config.local.json` in the skill directory. It is
excluded from Git. Do not commit it.

## Discovery Order

The installer uses this order for the installation root:

1. `--comfy-root`.
2. `COMFYUI_ROOT`.
3. `COMFYUI_HOME`.
4. The current directory and its parents.
5. Common home and drive locations.

For Python it checks `--python`, `COMFYUI_PYTHON`, an embedded Python next to the
ComfyUI portable directory, then common virtual environments.

For the server it uses `--server`, `COMFYUI_URL`, then
`http://127.0.0.1:8188`.

## Remote Or Containerized Servers

The runtime uploads references and downloads outputs over HTTP. A local ComfyUI
root is therefore optional. Use:

```bash
python scripts/comfyui_portable.py setup \
  --server http://host.docker.internal:8188 \
  --workflow txt2img="/workflows/txt2img.api.json"
```

If the API requires authentication, put headers in `config.local.json` or set
`COMFYUI_API_KEY`. Do not commit either value.

## Model Selection

When `/object_info` is reachable, setup extracts the model file options exposed
by each loader node. If the workflow's stored model filename exists on the host,
it is locked unchanged. If it is missing, interactive setup lists local
candidates. Non-interactive setup keeps the workflow value and `doctor` reports
the mismatch.

Use an explicit override when needed:

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --workflow txt2img="/workflows/txt2img.api.json" \
  --model txt2img.checkpoint="actual-model.safetensors"
```

## Unfamiliar Workflows

Run:

```bash
python scripts/comfyui_portable.py inspect "/workflows/custom.api.json"
```

The output lists node IDs, class types, detected bindings, model inputs, and
defaults. If automatic detection is incomplete, copy a descriptor from
`examples/` and add explicit bindings:

```json
{
  "bindings": {
    "prompt": {"node": "6", "input": "text"},
    "negative": {"node": "7", "input": "text"},
    "seed": {"node": "3", "input": "seed"}
  },
  "models": {
    "checkpoint": {"node": "4", "input": "ckpt_name"}
  }
}
```

Pass it during setup with `--descriptor name="/path/to/descriptor.json"`.
