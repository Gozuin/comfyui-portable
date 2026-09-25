# Config And Result Schema

## Local Profile

Current profile schema: `schema_version: 2`.

```json
{
  "schema_version": 2,
  "validation_state": "ready",
  "validation": {
    "ok": true,
    "errors": [],
    "warnings": []
  },
  "server": "http://127.0.0.1:8188",
  "timeout_seconds": 30,
  "poll_interval_seconds": 2,
  "max_wait_seconds": 900,
  "headers": {},
  "state_dir": "/path/to/.comfyui-portable",
  "comfyui": {
    "root": "/path/to/ComfyUI",
    "python": "/path/to/python",
    "input_dir": "/path/to/ComfyUI/input",
    "output_dir": "/path/to/ComfyUI/output"
  },
  "workflows": {
    "txt2img": {
      "file": "examples/txt2img.api.json",
      "descriptor": "examples/txt2img.descriptor.json",
      "bindings": {},
      "models": {},
      "defaults": {},
      "params": {},
      "required_nodes": [],
      "graph_sha256": "...",
      "descriptor_sha256": "...",
      "node_schema_sha256": "...",
      "task_type": "text_to_image"
    }
  }
}
```

`validation_state` is one of:

- `ready`: online validation passed and execution is allowed.
- `draft`: incomplete, stale, or offline; execution is refused by default.

Schema v1 profiles are detected as migration-required. Run setup again before
running them.

## Bindings

One role can point to one input:

```json
{"node": "6", "input": "text"}
```

Or to multiple inputs:

```json
[
  {"node": "6", "input": "text_g"},
  {"node": "6", "input": "text_l"}
]
```

Recognized roles:

- `prompt`
- `negative`
- `seed`
- `steps`
- `cfg`
- `denoise`
- `sampler_name`
- `scheduler`
- `width`
- `height`
- `batch_size`
- `resolution`
- `reference`

Invalid roles, strings in place of bindings, empty lists, malformed list members,
missing nodes, missing inputs, and duplicate targets are rejected.

## Model Bindings

```json
{
  "checkpoint": {
    "node": "4",
    "input": "ckpt_name",
    "value": "model-on-this-computer.safetensors",
    "required": true
  }
}
```

Automatic model discovery only uses known loader classes and inputs. A
descriptor can explicitly declare a custom loader target. The target input must
still expose an enum through `/object_info`; an empty enum is not treated as
unknown.

## Parameter Types

Setup derives parameter types and enums from `/object_info` and stores them in
`params`. A descriptor can override the derived shape:

```json
{
  "params": {
    "resolution": {
      "type": "string",
      "enum": ["1024x1024", "512x512"]
    }
  }
}
```

This prevents values such as `1024x1024` from being converted to an integer.

Defaults must point to a role that has a matching binding. Setup and doctor
reject “configured but never consumed” parameters instead of silently ignoring
them.

## Job Manifest

Each submitted task is stored at:

```text
.comfyui-portable/jobs/<job-id>.json
```

Main fields:

```json
{
  "schema_version": 1,
  "job_id": "4f...",
  "state": "accepted",
  "server": "http://127.0.0.1:8188",
  "workflow": "txt2img",
  "prompt_id": "abc123",
  "client_id": "c1...",
  "request_summary": {"prompt": "...", "steps": 24},
  "graph_sha256": "...",
  "uploads": [],
  "node_errors": {},
  "outputs": []
}
```

Credentials are not written to the manifest. Upload records and the final graph
hash are persisted before execution is considered submitted.

## Machine Result

Commands with `--json` use one result shape:

```json
{
  "schema_version": 1,
  "ok": true,
  "state": "completed",
  "error": null,
  "prompt_id": "abc123",
  "files": ["/path/output.png"],
  "warnings": []
}
```

Common states:

- `ready`, `draft`, `planned`
- `accepted`, `accepted_with_node_errors`
- `pending`, `running`, `completed`, `failed`
- `rejected`, `needs_configuration`, `unknown`

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | Input, profile, binding, or model configuration needs correction |
| `3` | Server unreachable |
| `4` | Execution failed |
| `5` | Submission or execution state unknown |
| `130` | User interrupted |
