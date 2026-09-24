# Contributing

感谢你改进 ComfyUI Portable Skill。

## 开发环境

要求：

- Python 3.10 或更高版本。
- 能运行 ComfyUI 的测试实例，或使用仓库中的模拟测试。

本项目没有第三方运行依赖。开发时只需：

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

## 提交原则

- 不提交 `config.local.json`。
- 不提交本机绝对路径。
- 不提交模型文件、输出图片或视频。
- 示例工作流使用占位模型名，实际模型名在 `setup` 时解析。
- 新增节点支持时，同时增加单元测试或模拟服务测试。
- 保持 Python 标准库依赖，除非有明确且必要的理由。

## 添加工作流

1. 在 ComfyUI 中导出 API 格式 JSON。
2. 把通用图放入 `workflows/` 或 `examples/`。
3. 添加同名 `.descriptor.json`。
4. 模型加载节点只记录节点和输入，不记录某个人的实际模型文件名。
5. 运行 `inspect`，确认语义绑定正确。
6. 更新 README 的功能或示例列表。

## Pull Request

PR 描述请包含：

- 修改目的。
- 影响的命令或工作流类型。
- 已运行的测试。
- 是否存在兼容性变化。

请避免在同一个 PR 中混入无关重构、格式化或元数据修改。
