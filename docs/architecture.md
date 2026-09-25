# Architecture

## Goal

让 AI 直接调用 ComfyUI，同时把“可分享的工作流”和“某台电脑如何执行它”分开。

```text
Agent
  |
  v
comfyui-portable skill
  |
  +-- config.local.json (machine-local, ignored by Git)
  +-- workflow + descriptor (portable)
  |
  v
ComfyUI HTTP API
  |
  +-- /object_info
  +-- /upload/image
  +-- /prompt
  +-- /history/{prompt_id}
  +-- /queue
  +-- /view
```

没有 MCP 常驻进程，也没有第二种执行核心。

## Layers

### Portable Layer

进入 Git 的内容：

- `SKILL.md`
- CLI implementation
- API workflows
- descriptors
- examples and documentation

这里不保存任何机器的模型名、绝对路径或鉴权信息。

### Local Layer

`config.local.json` 由 `setup` 生成，包含：

- 服务器和超时策略。
- 本地诊断路径。
- 实际模型名。
- 严格绑定和参数类型。
- 文件与节点 schema 指纹。
- `ready` 或 `draft` 状态。

### Job Layer

`.comfyui-portable/jobs/<job-id>.json` 保存任务状态。`submit` 在提交前后持续
更新它，使客户端重启后可以继续 `status`、`wait` 和 `fetch`。

## Execution Flow

### Run

```text
load profile and workflow
  -> verify ready / hashes / node schema
  -> build a pure plan
  -> validate all bindings and references
  -> if dry-run: emit plan only
  -> upload references with unique names
  -> persist upload records and final graph hash
  -> POST /prompt
  -> persist prompt_id immediately
  -> poll /history and /queue
  -> fetch outputs through /view
```

`/free` is not part of the default path.

### Submit / Recover

```text
submit -> manifest(prompt_id)
status -> read manifest + /history + /queue
wait   -> poll without resubmitting
fetch  -> download the already accepted task
```

If a POST to `/prompt` loses its response, the state is `unknown`. The client
does not automatically submit again.

## Validation Boundary

Setup and doctor share the same rules:

- Required nodes are the graph classes plus descriptor extra requirements.
- Model options are known loader inputs plus explicit descriptor extensions.
- Empty enums, unknown model inputs, missing nodes, bad roles, malformed
  bindings, and unconsumed defaults cannot produce `ready`.
- Graph, descriptor, and required-node schema hashes detect drift.

Server-side custom `VALIDATE_INPUTS` can still reject a graph. Static validation
does not claim to prove media quality or full node compatibility.

## Security Boundary

- No model download or custom-node installation.
- No shell execution of workflow values.
- No credentials in job manifests.
- Upload filenames are sanitized and task-scoped.
- Output paths are derived from response filenames and never joined with
  arbitrary server paths.
- Outputs are written through `.part` files.
- Non-loopback plain HTTP produces a warning.
