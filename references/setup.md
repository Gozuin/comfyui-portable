# Setup

## What Setup Does

`setup` creates a machine-local profile and checks whether it is safe to run.
The profile stores:

- Server URL and HTTP settings.
- Optional ComfyUI root and Python paths for local diagnosis.
- Workflow and descriptor paths.
- Semantic bindings, parameter types, defaults, and required nodes.
- The model filename actually exposed by the target server.
- Workflow, descriptor, and node-schema hashes.

The profile is written to `config.local.json` by default and is ignored by Git.

## Ready And Draft

Online setup must complete all required checks before writing `ready`:

- `/system_stats` and `/object_info` are reachable.
- Every graph node class exists.
- Every model loader exposes a recognized, non-empty option list.
- Every model value is present in that list.
- Every binding is structurally valid and points to an existing node input.

If any check fails, setup returns exit code `2` and records `draft` when it is
safe to keep a partial profile. An existing `ready` profile is not replaced by
a failed online setup.

`--offline` intentionally writes `draft` without contacting ComfyUI. `run`
rejects draft profiles unless `--allow-draft` is explicitly supplied.

## Discovery

Installation root order:

1. `--comfy-root`.
2. `COMFYUI_ROOT`.
3. `COMFYUI_HOME`.
4. Current directory and parents.
5. Common home and drive locations.

Python order:

1. `--python`.
2. `COMFYUI_PYTHON`.
3. Embedded Python or common virtual environments.
4. The Python executable running the tool.

Server order:

1. `--server`.
2. `COMFYUI_URL`.
3. Existing profile.
4. `http://127.0.0.1:8188`.

The tool does not scan arbitrary ports or start ComfyUI.

## Model Selection

Model options come from known loader inputs returned by `/object_info`.
Descriptor bindings can explicitly extend this to custom loader classes.

An empty option list is `known_empty`, not `unknown`. It produces:

```json
{
  "code": "NEEDS_MODEL_SELECTION",
  "state": "draft",
  "candidates": []
}
```

An interactive terminal can list candidates. Non-interactive setup returns the
candidates in the error and never chooses one automatically. A single candidate
is still not proof that the model architecture is compatible.

## Minimal Online Example

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --workflow txt2img="/workflows/txt2img.api.json" \
  --descriptor txt2img="/workflows/txt2img.descriptor.json" \
  --model txt2img.checkpoint="actual-model.safetensors" \
  --json
```

Check the result before running:

```bash
python scripts/comfyui_portable.py doctor --json
```

## Offline Draft

Use offline mode only when server validation is not possible yet:

```bash
python scripts/comfyui_portable.py setup \
  --offline \
  --workflow txt2img="/workflows/txt2img.api.json" \
  --json
```

The result is deliberately marked `draft`. Run online setup before real
execution.

## Remote Servers

HTTP execution does not require a local ComfyUI installation:

```bash
python scripts/comfyui_portable.py setup \
  --server http://host.docker.internal:8188 \
  --workflow txt2img="/workflows/txt2img.api.json" \
  --json
```

Non-loopback plain HTTP produces a warning because credentials and job data may
be visible on the network. Put authentication headers in the local profile or
set `COMFYUI_API_KEY`; do not commit either value.
