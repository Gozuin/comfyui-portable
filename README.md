# ComfyUI Portable Skill

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dependencies: standard library](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#技术特性)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **最主要的目的：让 AI 能轻松、直接、可靠地调用你电脑上的 ComfyUI。**
>
> 不需要再搭一层 MCP 服务，不需要把某台电脑的模型名和路径写死。
> 这个仓库本身就是一个持续更新的项目，新能力会继续在当前仓库发布。

## 这个项目解决什么问题

AI 要真正使用 ComfyUI，难点通常不是工作流本身，而是下面这些机器差异：

- ComfyUI 装在哪里，服务地址是什么。
- 当前电脑实际安装了哪些模型。
- 工作流里的节点编号、参数位置是否仍然匹配。
- 参考图上传后叫什么名字，任务中断后还能否继续查询。
- 哪些操作只是预演，哪些操作真的会修改 ComfyUI。

ComfyUI Portable Skill 把这些差异整理成一套面向 Agent 的执行协议：

1. 通过标准 HTTP API 直接连接 ComfyUI。
2. 用一次在线 `setup` 验证服务器、节点和模型，生成本机 `ready` 配置。
3. 用可分享的 API 工作流和 descriptor 描述语义。
4. 用稳定状态、任务清单和机器可读结果把执行过程交给 AI。

它不是模型下载器，也不会自动安装节点。换电脑后需要重新 `setup`，因为模型、节点和服务地址属于机器本地信息。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 直接 HTTP 调用 | 使用 `/prompt`、`/history`、`/view`、`/upload/image` 等接口，无 MCP 常驻进程。 |
| 可执行配置 | `setup` 区分 `ready` 和 `draft`；缺模型、缺节点或空模型列表不会伪装成功。 |
| 本机绑定 | 实际模型名存进被 Git 忽略的 `config.local.json`，共享工作流不绑定某台电脑。 |
| 严格校验 | 校验所有绑定、节点、模型枚举和参数类型；旧 profile 会提示重新配置。 |
| 安全预演 | `run --dry-run` 不上传、不提交、不卸载模型，只输出计划。 |
| 任务恢复 | `submit` 保存 `prompt_id`；`status`、`wait`、`fetch` 可在进程重启后继续。 |
| 防覆盖 | 上传使用任务命名空间和唯一文件名；输出按节点与序号保存，默认不覆盖已有文件。 |
| 稳定机器合同 | 支持机器输出的命令只向 stdout 输出一个 JSON；日志和错误都有稳定状态与退出码。 |

## 安装

把仓库放到 Agent 的技能目录。常见位置：

| Agent | 技能目录 |
| --- | --- |
| Codex | `~/.codex/skills/comfyui-portable` |
| 跨运行时别名 | `~/.agents/skills/comfyui-portable` |
| Claude Code | `~/.claude/skills/comfyui-portable` |

也可以只放在某个项目里供当前项目使用。

## 快速开始

### 1. 导出 API 工作流

在 ComfyUI 中选择：

```text
Workflow > Export (API)
```

普通 UI 工作流不能直接提交到 `/prompt`。

### 2. 在线配置并验证

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --workflow txt2img="/path/to/txt2img.api.json" \
  --descriptor txt2img="examples/txt2img.descriptor.json" \
  --model txt2img.checkpoint="actual-model.safetensors" \
  --json
```

在线 `setup` 会连接 `/system_stats` 和 `/object_info`，验证图中的节点类和模型枚举。

- 输出 `state: ready`：可以执行。
- 输出 `state: draft`：配置已保存，但缺模型、缺节点或未在线验证，默认不能运行。
- 只有显式使用 `--offline` 时，才接受未验证的 draft。

### 3. 检查配置

```bash
python scripts/comfyui_portable.py doctor --json
```

`doctor` 会检查 profile 版本、服务器、节点、模型、绑定、工作流哈希和节点 schema 漂移。

### 4. 预演

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "A red apple on a clean white table" \
  --negative "blurry, low quality" \
  --steps 24 \
  --dry-run \
  --json
```

预演不会上传参考文件，也不会访问 `/prompt`、`/free` 或 `/interrupt`。

### 5. 执行

短任务可以继续使用组合命令：

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "A red apple on a clean white table" \
  --negative "blurry, low quality" \
  --steps 24 \
  --out "./apple.png" \
  --json
```

长任务建议拆分，避免客户端超时后丢失任务上下文：

```bash
python scripts/comfyui_portable.py submit \
  --workflow txt2img \
  --prompt "A red apple on a clean white table" \
  --json

python scripts/comfyui_portable.py status --job JOB_ID --json
python scripts/comfyui_portable.py wait --job JOB_ID --json
python scripts/comfyui_portable.py fetch --job JOB_ID --out "./outputs" --json
```

`submit` 成功只表示 ComfyUI 已接受任务，不代表已经生成完成。最终状态以 `status` 或 `wait` 的结果为准。

## 命令

| 命令 | 用途 |
| --- | --- |
| `setup` | 探测本机 ComfyUI，验证后生成 `ready` 或 `draft` profile。 |
| `doctor` | 在线复核服务器、节点、模型、绑定和漂移。 |
| `inspect` | 分析 API 工作流并输出自动绑定、歧义和模型加载器。 |
| `run` | 组合执行：提交、等待并下载。 |
| `submit` | 只提交并立即保存任务清单。 |
| `status` | 查询任务状态，不重复提交。 |
| `wait` | 等待已提交任务完成。 |
| `fetch` | 下载已完成任务的结果。 |

查看帮助：

```bash
python scripts/comfyui_portable.py run --help
```

## Descriptor

descriptor 只描述节点语义，不保存本机模型名：

```json
{
  "name": "txt2img",
  "task_type": "text_to_image",
  "bindings": {
    "prompt": {"node": "2", "input": "text"},
    "negative": {"node": "3", "input": "text"},
    "steps": {"node": "4", "input": "steps"},
    "reference": [
      {"node": "8", "input": "image"},
      {"node": "9", "input": "image"}
    ]
  },
  "models": {
    "checkpoint": {
      "node": "1",
      "input": "ckpt_name",
      "required": true
    }
  },
  "params": {
    "resolution": {
      "type": "string",
      "enum": ["1024x1024", "512x512"]
    }
  },
  "required_nodes": [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "EmptyLatentImage",
    "VAEDecode",
    "SaveImage"
  ]
}
```

`required_nodes` 是额外要求，不会替换工作流实际包含的节点类。

支持的语义角色：

`prompt`、`negative`、`seed`、`steps`、`cfg`、`denoise`、`sampler_name`、`scheduler`、`width`、`height`、`batch_size`、`resolution`、`reference`。

## 结果合同

每个支持 `--json` 的命令，其 stdout 都是一个可直接解析的 JSON 文档：

```json
{
  "schema_version": 1,
  "ok": false,
  "state": "needs_configuration",
  "error": {
    "code": "NEEDS_MODEL_SELECTION",
    "message": "The configured checkpoint is not installed.",
    "retryable": false,
    "candidates": ["model-a.safetensors"]
  },
  "prompt_id": null,
  "files": [],
  "warnings": []
}
```

退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 命令成功。 |
| `2` | 输入、配置或绑定需要修正。 |
| `3` | ComfyUI 服务不可达。 |
| `4` | ComfyUI 执行失败。 |
| `5` | 提交或执行状态未知，不应盲目重提。 |
| `130` | 用户中断。 |

## 本机数据

`config.local.json` 默认被 `.gitignore` 排除。当前版本的 state schema 是 `2`。

任务清单保存在 profile 同目录的：

```text
.comfyui-portable/jobs/<job-id>.json
```

清单不保存凭据，包含：

- 服务器地址和工作流名。
- `prompt_id`、状态、请求摘要。
- 上传文件与服务器返回路径。
- 最终提交图的 SHA-256。
- 下载产物清单。

## 技术特性

- Python 3.10 及以上。
- 仅使用 Python 标准库。
- 支持鉴权请求头与 `COMFYUI_API_KEY`。
- 支持常见模型加载器、采样器、文本编码和参考文件输入。
- 支持同一角色绑定多个输入，例如 SDXL 的 `text_g` 与 `text_l`。
- 支持显式 descriptor 扩展自定义加载器和参数类型。
- 默认不使用 `/free`，避免在共享服务上产生全局副作用。
- 非回环 HTTP 地址会给出明文传输警告。

## 已知边界

- 只接受 ComfyUI API 格式工作流。
- 自动识别覆盖常见节点；多个采样器或自定义节点应提供 descriptor。
- 默认在线 `setup` 必须验证成功才能生成 `ready`。
- `--offline` 生成 draft，`run` 默认拒绝 draft。
- 本地路径只用于诊断和本机发现；远程执行仍走 HTTP。
- 视频或特定模型是否可用，取决于目标 ComfyUI 的节点、模型和真实 smoke 结果。
- 不自动下载模型、不自动安装自定义节点、不修改系统设置。
- 当前没有 MCP 包装层；只有客户端只能使用 MCP 时，才值得另做薄包装并复用同一核心。

## 开发与测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
ruff check scripts tests
ruff format --check scripts tests
```

CI 覆盖 Windows/Linux 与 Python 3.10/3.13。测试使用模拟 ComfyUI HTTP 服务，不接触真实模型，也不执行 GPU 生成。

## 更新记录

见 [CHANGELOG.md](CHANGELOG.md)。

## License

[MIT](LICENSE)
