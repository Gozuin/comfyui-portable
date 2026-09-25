# Security Policy

## Sensitive Data

不要提交以下内容：

- `config.local.json`
- ComfyUI API Token
- GitHub Token
- SSH 私钥
- 模型文件
- 含隐私内容的输入图片或视频
- 生成结果
- `.comfyui-portable/jobs/` 中的本机任务清单

`config.local.json` 和 `*.local.json` 已在 `.gitignore` 中排除。
任务清单不保存 API Token，但仍包含本机路径、提示词和任务标识，不应公开提交。

## Reporting

如果发现以下问题，请不要创建公开 Issue：

- 任意文件读写。
- 路径穿越。
- 命令注入。
- Token 泄露。
- 远程 ComfyUI 请求伪造。

请通过仓库所有者公开的安全联系方式报告，并提供最小复现步骤。

## Supported Versions

安全修复默认面向最新发布版本。升级前请先运行：

```bash
python scripts/comfyui_portable.py doctor
python -m unittest discover -s tests -v
```
