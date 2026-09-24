# Architecture

## Design Goal

目标是把“工作流是什么”和“这台电脑上如何执行它”分开。

可移植工作流描述节点关系和语义角色，本地配置描述真实服务器、路径、模型和参数。

## Layers

### Portable Layer

仓库中可公开的部分：

- `SKILL.md`
- `scripts/comfyui_portable.py`
- API 工作流
- `.descriptor.json`
- 文档和示例

这一层不包含任何特定电脑的模型文件名或绝对路径。

### Local Layer

`config.local.json` 由 `setup` 生成，包含：

- 服务器 URL。
- ComfyUI 根目录和 Python。
- 输入、输出目录。
- 实际模型文件名。
- 节点和参数绑定。
- 锁定默认值。

这一层默认不进入 Git。

### Runtime Layer

`run` 按以下顺序解析：

```text
load graph
  -> apply locked model values
  -> apply workflow defaults
  -> apply CLI parameters
  -> upload references
  -> apply explicit --set overrides
  -> submit /prompt
  -> poll /history/{prompt_id}
  -> download /view
```

显式 `--set` 放在最后，保证 Agent 可以处理描述文件未覆盖的特殊节点。

## Discovery

`setup` 按顺序探测：

1. 命令行参数。
2. 环境变量。
3. 当前目录及父目录。
4. 用户目录和常见安装位置。

ComfyUI 的模型信息来自 `/object_info`，不是扫描文件系统。这样本地和远程服务器使用相同逻辑。

## HTTP API

核心端点：

| 端点 | 用途 |
| --- | --- |
| `/system_stats` | 服务状态和设备 |
| `/object_info` | 节点类、输入定义和模型列表 |
| `/upload/image` | 上传参考图或视频 |
| `/prompt` | 提交 API 工作流 |
| `/history/{id}` | 查询执行状态和输出 |
| `/view` | 下载输出 |
| `/free` | 请求释放显存 |

## Security Boundary

执行器把 ComfyUI 当作受信任的本地或远程服务，但仍做以下限制：

- 不从响应中拼接任意输出路径。
- 不使用 Shell 执行工作流参数。
- 不在源码中保存凭据。
- 不自动安装节点或下载模型。
