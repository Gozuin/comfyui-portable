# ComfyUI Portable Skill

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dependencies: standard library](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#技术特性)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**一句话：让 Codex、Claude 这类 AI 助手直接会用你的 ComfyUI，换一台电脑也不用改代码。**

## 先说人话：它到底是干什么的

平时用 ComfyUI，最烦的不是画图本身，而是每换一台电脑就要重新找：

- ComfyUI 装在哪个盘？
- 服务开在哪个端口？
- 模型文件到底叫什么名字？
- 工作流里的节点编号变没变？
- 哪些参数是这台电脑专用的？

这个项目就是把这些麻烦事包起来。

你把技能装到某台电脑，执行一次初始化，它会自己去问 ComfyUI：

> “你现在装在哪？你能用哪些模型？这个工作流需要哪些节点？参数怎么接？”

然后它把这些答案写进电脑本地的配置文件。以后 AI 助手只负责说“我要生成什么、怎么改”，底层路径、模型名和节点参数都交给技能自动处理。

换电脑时，GitHub 上的技能代码不用改；到了新电脑重新初始化一次，它就会认新电脑的环境。

## 举个大白话的例子

你在电脑 A 上做了一个文生图工作流，模型叫：

```text
SDXL_A.safetensors
```

电脑 B 上装的是：

```text
SDXL_B_quantized.safetensors
```

如果写死代码，电脑 B 一跑就报错。

用这个技能，工作流本身只说明“这里需要一个 checkpoint”，初始化时它会自动发现电脑 B 上真正的模型名，并把 `SDXL_B_quantized.safetensors` 锁进电脑 B 的本地配置。

所以：

- 工作流可以公开分享。
- 技能可以重复安装。
- 模型名和本机路径不会互相污染。
- 私人的电脑配置不会上传到 GitHub。

## 项目定位

ComfyUI Portable Skill 不是另一个 ComfyUI 客户端，也不是模型下载器。它是一层面向 Agent 的稳定执行协议：

- 对 Agent 暴露统一的 `setup`、`doctor`、`inspect`、`run` 命令。
- 对 ComfyUI 使用标准 HTTP API，不依赖界面自动化。
- 对工作流使用语义描述文件，不把模型文件名和节点编号写死在源码里。
- 对每台电脑生成本地配置，自动锁定实际路径、模型和工作流参数。
- 对远程或容器化 ComfyUI 保留 HTTP-only 运行能力。

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 自动探测 ComfyUI | 从命令行、环境变量、当前目录和常见安装位置发现 ComfyUI 根目录。 |
| 自动探测服务器 | 默认连接 `http://127.0.0.1:8188`，也支持局域网、Docker 和远程 API。 |
| 自动读取模型列表 | 通过 `/object_info` 获取各加载节点暴露的真实模型文件。 |
| 自动锁定模型名 | 把工作流中的占位模型替换为目标电脑实际安装的模型，并写入本地配置。 |
| 节点和参数绑定 | 自动识别常见文本编码、采样器、Latent 尺寸和模型加载节点。 |
| 自定义工作流支持 | 使用 `descriptor.json` 显式绑定任意节点输入，不受固定节点编号限制。 |
| 文生图 | 通过配置好的文生图工作流提交提示词、负向提示词和采样参数。 |
| 图生图 | 通过 `/upload/image` 上传参考图，再执行配置好的图像编辑工作流。 |
| 视频和高级工作流 | 可绑定首帧、参考视频、时序节点和其他自定义输入。 |
| 运行前诊断 | `doctor` 检查服务、CUDA、节点类、模型文件、工作流路径和参数绑定。 |
| 工作流检查 | `inspect` 输出节点结构、自动识别结果、模型输入和默认参数。 |
| 结果下载 | 通过 `/view` 下载图像、GIF 或视频输出，不强依赖 ComfyUI 输出目录。 |
| 本机与远程兼容 | 本地安装缺失时仍可使用远程 HTTP API，参考文件和输出都走网络。 |
| 零第三方依赖 | 仅使用 Python 标准库，不需要 `requests`、`comfy-cli` 或额外 SDK。 |
| Agent 跨运行时 | 目录结构兼容 Codex、Claude Code 和常见 `skills` 约定。 |

## 工作方式

```text
Claude Code / Codex / other agent
              |
              v
      comfyui-portable skill
              |
     setup -> config.local.json
              |
      semantic bindings
              |
              v
   ComfyUI HTTP API /prompt
              |
       /upload/image
       /history/{id}
       /view
              |
              v
       image / gif / video
```

技能内部分成三层：

1. **可移植层**：`SKILL.md`、命令实现、通用工作流描述和示例。
2. **机器配置层**：`config.local.json` 保存路径、服务地址、实际模型名和锁定参数。
3. **工作流层**：ComfyUI API 格式图，按语义角色被 Agent 调用。

机器配置层默认被 `.gitignore` 排除，因此公开仓库不会包含某个人的本机路径或模型名。

## 技术特性

- Python 3.10 及以上。
- 纯标准库实现，包含 HTTP、Multipart 上传、JSON 配置和轮询。
- 支持鉴权请求头，也可使用 `COMFYUI_API_KEY` 环境变量。
- 支持 `CheckpointLoaderSimple`、`UNETLoader`、`CLIPLoader`、`VAELoader`、
  `LoraLoader`、ControlNet 和常见自定义加载节点。
- 支持 `KSampler`、`KSamplerAdvanced`、`SamplerCustom` 等常见采样节点。
- 支持 `CLIPTextEncode` 和同节点正负提示词结构，例如
  `TextEncodeQwenImage21`。
- 支持 `EmptyLatentImage`、`EmptySD3LatentImage` 和带 `resolution`
  参数的模型。
- 支持一个工作流绑定多个参考文件。

## 安装

克隆或下载仓库后，把 `comfyui-portable` 目录放进 Agent 的技能目录。

常见位置：

| Agent | 技能目录 |
| --- | --- |
| Codex | `~/.codex/skills/comfyui-portable` |
| 跨运行时别名 | `~/.agents/skills/comfyui-portable` |
| Claude Code | `~/.claude/skills/comfyui-portable` |

也可以把技能目录放入项目，仅让当前项目使用。

## 快速开始

### 1. 导出 ComfyUI 工作流

在 ComfyUI 界面中选择：

```text
Workflow > Export (API)
```

必须使用 **API 格式** JSON，普通 UI 工作流不能直接提交到 `/prompt`。

### 2. 配置目标电脑

```bash
cd comfyui-portable

python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --comfy-root "/path/to/ComfyUI" \
  --workflow txt2img="/path/to/txt2img.api.json" \
  --descriptor txt2img="examples/txt2img.descriptor.json"
```

`setup` 会：

1. 探测 ComfyUI 根目录和 Python。
2. 连接 `/system_stats` 和 `/object_info`。
3. 读取目标电脑真正可用的模型列表。
4. 校验工作流需要的节点类。
5. 自动识别参数和模型绑定。
6. 生成 `config.local.json`。

如果本机模型名与工作流默认值不同，交互终端会列出候选模型供选择。自动化环境可显式指定：

```bash
python scripts/comfyui_portable.py setup \
  --server http://127.0.0.1:8188 \
  --workflow txt2img="/workflows/txt2img.api.json" \
  --model txt2img.checkpoint="actual-model.safetensors" \
  --non-interactive
```

### 3. 检查配置

```bash
python scripts/comfyui_portable.py doctor
```

检查内容包括服务器可达性、CUDA 设备、节点注册表、模型文件、工作流文件和语义绑定。

### 4. 生成图片

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "A red apple on a clean white table" \
  --negative "blurry, low quality" \
  --steps 24 \
  --out "./apple.png"
```

### 5. 图生图或视频编辑

```bash
python scripts/comfyui_portable.py run \
  --workflow img2img \
  --prompt "Keep the person and clothing unchanged; replace the background with a clean studio" \
  --reference "./person.jpg" \
  --out "./person-studio.png"
```

## 命令速查

| 命令 | 用途 |
| --- | --- |
| `setup` | 探测本机 ComfyUI，并生成本地配置。 |
| `doctor` | 验证服务器、节点、模型和工作流绑定。 |
| `inspect` | 分析任意 API 工作流的节点和自动绑定。 |
| `run` | 解析工作流并提交到 ComfyUI。 |

查看单个命令帮助：

```bash
python scripts/comfyui_portable.py run --help
```

## 工作流描述文件

描述文件是“可移植工作流”和“本机模型配置”之间的桥梁。它只描述节点语义，不写本机模型名。

```json
{
  "bindings": {
    "prompt": {"node": "2", "input": "text"},
    "negative": {"node": "3", "input": "text"},
    "seed": {"node": "4", "input": "seed"},
    "steps": {"node": "4", "input": "steps"},
    "width": {"node": "5", "input": "width"},
    "height": {"node": "5", "input": "height"},
    "reference": {"node": "8", "input": "image"}
  },
  "models": {
    "checkpoint": {
      "node": "1",
      "input": "ckpt_name",
      "required": true
    }
  },
  "required_nodes": [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "SaveImage"
  ]
}
```

支持的语义角色：

`prompt`、`negative`、`seed`、`steps`、`cfg`、`denoise`、`sampler_name`、
`scheduler`、`width`、`height`、`batch_size`、`resolution`、`reference`。

仓库提供完整示例：

- `examples/txt2img.api.json`
- `examples/txt2img.descriptor.json`
- `examples/img2img.api.json`
- `examples/img2img.descriptor.json`

## 任意节点覆盖

当工作流有特殊参数时，可以不修改技能代码：

```bash
python scripts/comfyui_portable.py run \
  --workflow custom \
  --prompt "A city at night" \
  --set "17.strength=0.72" \
  --set "21.enabled=true" \
  --out "./city.png"
```

只解析不提交，用于排查工作流：

```bash
python scripts/comfyui_portable.py run \
  --workflow txt2img \
  --prompt "A city at night" \
  --dry-run \
  --graph-out "./resolved-graph.json"
```

## 远程和容器 ComfyUI

HTTP 执行器不要求本机存在 ComfyUI 安装目录：

```bash
python scripts/comfyui_portable.py setup \
  --server http://host.docker.internal:8188 \
  --workflow txt2img="/workflows/txt2img.api.json"
```

需要鉴权时，把请求头写入本地配置，或设置：

```text
COMFYUI_API_KEY=your-token
```

本地配置和 Token 都应保持私密，不要提交到 Git。

## 目录结构

```text
comfyui-portable/
  SKILL.md
  README.md
  LICENSE
  VERSION
  agents/
    openai.yaml
  docs/
    architecture.md
    workflow-authoring.md
  examples/
    txt2img.api.json
    txt2img.descriptor.json
    img2img.api.json
    img2img.descriptor.json
  references/
    config-schema.md
    setup.md
  scripts/
    comfyui_portable.py
    setup.ps1
    setup.sh
  tests/
    test_comfyui_portable.py
  workflows/
    README.md
```

## 安全设计

- 不接收 ComfyUI 界面密码，不执行浏览器自动化。
- 不存储 GitHub 或 ComfyUI 账号密码。
- `config.local.json` 默认被 Git 忽略。
- 不自动下载模型。
- 不自动安装或修改 `custom_nodes/`。
- 不删除 ComfyUI `models/` 中的任何文件。
- 工作流写入 `input/` 仅通过官方 `/upload/image` 接口发生。
- 远端响应和输出文件名在写入前会按文件名提取，避免路径穿越。

## 兼容性

| 项目 | 支持情况 |
| --- | --- |
| 操作系统 | Windows、Linux、macOS |
| Python | 3.10+ |
| ComfyUI | 提供 HTTP `/prompt`、`/history`、`/view`、`/upload/image` 的版本 |
| 工作流 | ComfyUI API 格式 JSON |
| Agent | Codex、Claude Code 及读取 `SKILL.md` 的运行时 |

## 开发与测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

测试覆盖：

- 常见文生图节点自动识别。
- Qwen 风格同节点正负提示词识别。
- 配置生成与模型锁定。
- `doctor` 的节点和模型校验。
- 文生图端到端提交和结果下载。
- 图生图参考文件上传。
- 缺失模型时拒绝通过诊断。

## 已知边界

- ComfyUI 必须提供 API 格式工作流；UI 格式需要先在界面导出为 API 格式。
- 自动识别覆盖常见节点。高度定制的第三方节点建议提供描述文件。
- 工作流的语义角色由描述文件决定，不依赖节点在画布上的视觉位置。
- 视频处理能力取决于目标电脑已安装的 ComfyUI 视频节点和模型。

## 贡献

提交前请运行全部测试，并阅读 `CONTRIBUTING.md`。新增工作流示例时，请同时提供可移植的 `descriptor.json`，不要把本机模型文件名写入示例。

## License

[MIT](LICENSE)
