# Workflow Authoring

## Required Format

只接受 ComfyUI API 格式 JSON：

```text
Workflow > Export (API)
```

每个节点至少包含：

```json
{
  "class_type": "KSampler",
  "inputs": {
    "seed": 1,
    "steps": 20
  }
}
```

## Auto Detection

`inspect` 会尝试识别：

- 单个采样器的 seed、steps、cfg、denoise、sampler 和 scheduler。
- 采样器 positive/negative 链接对应的文本节点。
- 同一文本节点上的 prompt/negative_prompt。
- SDXL 文本编码器的 `text_g` 与 `text_l` 双通道。
- Latent 节点的 width、height 和 batch size。
- 字面量 `resolution`。
- LoadImage 或 LoadVideo 参考输入。
- 已知加载器的模型枚举。

发现多个采样器时会输出 `MULTIPLE_SAMPLERS`，不会静默选择第一个。

## Explicit Descriptor

以下情况必须提供 descriptor：

- 多个采样器。
- 第三方节点类不包含常见输入名。
- 参数由条件节点或自定义节点控制。
- 参考输入不是常见的 LoadImage/LoadVideo。
- 模型加载器不在内置注册表中。
- 参数是字符串枚举，需要阻止自动整数转换。

示例：

```json
{
  "name": "custom",
  "task_type": "image_edit",
  "bindings": {
    "prompt": {"node": "12", "input": "prompt"},
    "negative": {"node": "12", "input": "negative_prompt"},
    "seed": {"node": "30", "input": "noise_seed"},
    "steps": {"node": "30", "input": "steps"},
    "resolution": {"node": "12", "input": "resolution"},
    "reference": [
      {"node": "4", "input": "image"},
      {"node": "7", "input": "image"}
    ]
  },
  "models": {
    "unet": {"node": "1", "input": "unet_name"},
    "clip": {"node": "2", "input": "clip_name"}
  },
  "params": {
    "resolution": {
      "type": "string",
      "enum": ["1024x1024", "512x512"]
    }
  },
  "defaults": {
    "steps": 28
  },
  "required_nodes": [
    "CustomLoader"
  ]
}
```

`required_nodes` 只增加要求，不能删除工作流中实际存在的节点类。

## References

一个角色可以绑定多个输入：

```json
{
  "reference": [
    {"node": "4", "input": "image"},
    {"node": "7", "input": "image"}
  ]
}
```

运行时按 `--reference` 出现顺序上传。每个输入会得到独立、唯一的服务器文件名；
不同任务使用独立子目录。

## Models

descriptor 只声明加载节点：

```json
{
  "models": {
    "checkpoint": {
      "node": "1",
      "input": "ckpt_name",
      "required": true
    }
  }
}
```

本机实际值来自在线 `setup` 或显式的 `--model`。模型值必须出现在目标服务的
对应枚举中。

## Links And Constants

如果绑定目标当前连接到上游节点，普通参数不会覆盖这个连线。需要改变常量时，
应把 binding 指向实际常量 Primitive 节点，或通过明确的 `--set` 和描述文件
说明处理方式。

不要为了“能跑”而把连线永久替换成旧默认值。这会让共享工作流在不同机器上
产生不同结构。

## Validation

```bash
python scripts/comfyui_portable.py inspect "/path/to/workflow.api.json"

python scripts/comfyui_portable.py setup \
  --workflow custom="/path/to/workflow.api.json" \
  --descriptor custom="/path/to/workflow.descriptor.json" \
  --json

python scripts/comfyui_portable.py doctor --json
```
