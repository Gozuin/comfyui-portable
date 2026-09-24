---
name: comfyui-portable
description: Use when an agent must generate or edit images or videos through a local ComfyUI on an unknown or newly installed machine, especially when server addresses, installation paths, model filenames, or workflow node parameters differ by host.
metadata:
  short-description: Discover a host ComfyUI once, then run its image and video workflows from a locked local profile.
---

# ComfyUI Portable

## Core Rules

Use this skill when ComfyUI may be installed differently on each computer. Never
invent a path, model filename, node ID, or workflow parameter.

1. Read `config.local.json`. If it is missing or invalid, run `doctor`. If `doctor`
   reports a missing profile, run `setup` first.
2. Expand the user's media request before generation. Preserve the requested
   subject, identity, clothing, composition, action, timing, text, colors, and
   aspect ratio. Add only execution details that do not change intent.
3. Prefer the locked workflow selected in the local profile. Do not rebuild an
   arbitrary graph by hand when a matching profile exists.
4. Use local ComfyUI first. Use another tool only when the user asks for it or the
   local server cannot complete the task after diagnosis.
5. Do not download models, install custom nodes, or modify ComfyUI `models/` or
   `custom_nodes/` unless the user explicitly asks.

## First Run

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --comfy-root "/path/to/ComfyUI" \
  --workflow txt2img="/path/to/txt2img.api.json" \
  --descriptor txt2img="examples/txt2img.descriptor.json"
```

`setup` discovers the host installation, reads `/object_info`, validates model
filenames and node classes, detects common parameter bindings, and writes
`config.local.json`. That file is intentionally ignored by Git.

If the workflow uses unfamiliar node classes or multiple samplers, run `inspect`,
copy `examples/txt2img.descriptor.json`, and provide explicit semantic bindings.
The descriptor stays portable; model filenames and paths exist only in the local
profile.

```bash
python scripts/comfyui_portable.py inspect "/path/to/workflow.api.json"
python scripts/comfyui_portable.py doctor
```

## Generate

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "<expanded prompt>" \
  --negative "<negative prompt>" \
  --out "./result.png"
```

For image-to-image or video editing, pass one or more reference files:

```bash
python scripts/comfyui_portable.py run \
  --workflow img2img \
  --prompt "<expanded edit instruction>" \
  --reference "./reference.png" \
  --out "./edited.png"
```

The runner uploads references through ComfyUI's HTTP API, submits the resolved
graph, waits for history, and downloads outputs through `/view`. It does not rely
on the local filesystem layout when the server is remote.

## Prompt Contract

Fill this before submitting any generation:

```text
Task type:
Input media and role:
Desired result:
Must preserve:
Must change:
Allowed additions:
Forbidden changes:
Style and medium:
Composition, camera, or timeline:
Output specification:
Acceptance criteria:
```

Write the positive prompt around the desired result. Put only explicit
prohibitions and common artifacts in the negative prompt. Keep exact text,
numbers, colors, ratios, directions, and durations unchanged.

## Failure Handling

- Missing profile: run `setup`.
- Missing model or node class: run `doctor`; choose an installed alternative or
  update the workflow. Do not fabricate a filename.
- Identity, clothing, or composition drift: use the workflow's mask, inpaint, or
  compositing path. Do not solve it by repeatedly changing the seed.
- Prompt ignored: strengthen preservation and prohibition clauses, then inspect
  the bound prompt node.
- Wrong output shape: fix the width, height, or resolution bindings.
- Video discontinuity: inspect first frame, last frame, motion peak, frame count,
  frame rate, and audio alignment.
- First run fails: diagnose the reported node or model, change one verified
  variable, and retry. Do not use random regeneration as diagnosis.

## Reference Files

- `references/setup.md`: installation and profile generation.
- `references/config-schema.md`: local profile and descriptor fields.
- `examples/`: portable API-workflow and descriptor examples.
- `scripts/comfyui_portable.py`: setup, doctor, inspect, and run commands.
