---
name: comfyui-portable
description: Use when an AI agent must generate or edit media through ComfyUI across machines with different server addresses, model filenames, node schemas, or workflow IDs; also use when a ComfyUI job must be safely planned, submitted, recovered, or diagnosed without an MCP bridge.
metadata:
  short-description: Let AI call ComfyUI directly, safely, and across machines.
---

# ComfyUI Portable

## Core Rules

The main purpose is to let AI call ComfyUI easily through its HTTP API without
building an MCP bridge or hard-coding one computer's models and paths.

1. Read `config.local.json`. It is machine-local and ignored by Git.
2. If it is missing, run `setup`. If it is `draft`, stale, or drifted, run
   `doctor` or configure it again. Do not run a draft profile.
3. Expand the user's media request before generation. Preserve the subject,
   identity, clothing, composition, timing, text, colors, and aspect ratio.
   Add only execution details that do not change intent.
4. Select a configured workflow. Ask when multiple workflows match.
5. Never invent a node ID, path, model filename, or enum value.
6. Do not download models, install nodes, open ports, or modify ComfyUI
   settings unless the user explicitly asks.

## Workflow

### 1. Configure Once Per Machine

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --workflow txt2img="/path/to/txt2img.api.json" \
  --descriptor txt2img="examples/txt2img.descriptor.json" \
  --model txt2img.checkpoint="actual-model.safetensors" \
  --json
```

Online setup must report `state: "ready"`. `draft` is incomplete and `run`
rejects it. `--offline` is only for intentionally saving an unverified profile.

### 2. Inspect When Binding Is Ambiguous

```bash
python scripts/comfyui_portable.py inspect "/path/to/workflow.api.json"
python scripts/comfyui_portable.py doctor --json
```

Add descriptor bindings for multiple samplers, custom loaders, unusual text
encoders, or shared/linked parameter values.

### 3. Plan Before Submitting

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "<expanded prompt>" \
  --negative "<explicit prohibitions only>" \
  --dry-run \
  --json
```

Dry-run performs no upload, prompt submission, model unload, or interrupt.
Review its upload plan and warnings before the real run.

### 4. Submit and Recover

For long jobs, preserve task state instead of relying on one long command:

```bash
python scripts/comfyui_portable.py submit --workflow txt2img --prompt "..." --json
python scripts/comfyui_portable.py status --job JOB_ID --json
python scripts/comfyui_portable.py wait --job JOB_ID --json
python scripts/comfyui_portable.py fetch --job JOB_ID --out "./outputs" --json
```

`accepted` means ComfyUI queued the job, not that generation succeeded.
`unknown` means do not resubmit automatically. Use the saved job manifest.

## Reference

Read only what the task needs:

- [references/setup.md](references/setup.md): discovery, ready/draft, model choice.
- [references/config-schema.md](references/config-schema.md): profile, descriptor,
  task manifests, and result fields.
- [docs/workflow-authoring.md](docs/workflow-authoring.md): explicit bindings and
  validation rules.
- [docs/architecture.md](docs/architecture.md): execution flow and HTTP boundary.
- [examples/](examples): portable API workflows and descriptors.

## Failure Handling

- Missing profile or `draft`: run online `setup`.
- Model missing or empty model enum: report candidates and stop.
- Missing node or schema drift: run `doctor`; do not fabricate a class name.
- Binding missing or linked: add an explicit descriptor binding; do not overwrite
  a graph link with a parameter.
- Submission status unknown: inspect the manifest; do not blindly resubmit.
- Identity or composition drift: use a mask, inpaint, or compositing workflow.
  Do not hide the problem by repeatedly changing seeds.
