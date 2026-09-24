# Workflow Authoring

## Required Format

只接受 ComfyUI API 格式 JSON：

```text
Workflow > Export (API)
```

每个节点是：

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

- 采样器的 seed、steps、cfg、denoise、sampler 和 scheduler。
- 采样器 positive 和 negative 链接对应的文本节点。
- 同一文本节点输出的正负提示词。
- Latent 节点的 width、height 和 batch size。
- `resolution` 参数。
- LoadImage 或 LoadVideo 参考输入。
- `ckpt_name`、`unet_name`、`clip_name`、`vae_name`、`lora_name` 等模型输入。

## Explicit Descriptor

以下情况请编写描述文件：

- 多个采样器，且默认选择的不是主采样器。
- 第三方节点类不包含常见输入名。
- 参数由开关、条件节点或自定义节点控制。
- 参考输入不是 `LoadImage` 或 `LoadVideo`。
- 需要固定特殊默认值。

示例：

```json
{
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
    "clip": {"node": "2", "input": "clip_name"},
    "vae": {"node": "3", "input": "vae_name"}
  },
  "defaults": {
    "steps": 28,
    "cfg": 4.5,
    "resolution": 1024
  }
}
```

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

运行时按 `--reference` 的出现顺序上传并绑定。

## Models

描述文件只声明加载节点：

```json
{
  "models": {
    "checkpoint": {
      "node": "1",
      "input": "ckpt_name"
    }
  }
}
```

实际值由 `setup --model workflow.checkpoint=...` 写入本地配置。

## Validation

完成后运行：

```bash
python scripts/comfyui_portable.py inspect "/path/to/workflow.api.json"

python scripts/comfyui_portable.py setup \
  --workflow custom="/path/to/workflow.api.json" \
  --descriptor custom="/path/to/workflow.descriptor.json"

python scripts/comfyui_portable.py doctor
```
