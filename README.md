# ComfyUI KohakuMOSS Full Pipeline

在 ComfyUI 中直接运行 **KohakuMossTrainer 的完整推理管线**：`MossPipeline.from_pretrained()` → `MossPipeline.generate()`。模型、LoRA、文本编码、采样、注意力、权重流式传输及 VAE 都由训练器执行，输出标准 ComfyUI `IMAGE`。

**状态：实验性初版。本次版本按要求未运行测试、未出图、未测速；没有已验证的性能提升声明。**

这是一个独立节点包，使用另行安装的训练器及其 Python 环境。无需安装旧 Precompile 或旧 Trainer 节点，也不修改 ComfyUI 的 PyTorch。

## 安装

在 ComfyUI 的 `custom_nodes` 目录中克隆：

```bash
git clone https://github.com/buxinzi2233/ComfyUI-KohakuMOSS-FullPipeline.git
```

准备自己的 [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) checkout，并按训练器文档安装其依赖。此节点不会自动下载模型、安装训练器、替换 Torch 或安装 GPU 工具链。

当前源码对接的训练器接口来自 revision `0536e62b6062039104d8f9575635b5d07bf573d3`。更新训练器可能改变接口和数值行为。训练器环境必须包含可用于目标 GPU 的 Torch、TileLang、Triton 及编译工具链；V100 需要支持 SM70 的环境，不能照搬新显卡的 CUDA 13 配置。

重启 ComfyUI，在 `KohakuMOSS/Full Pipeline` 分类添加节点。

## 工作流

```text
Full Pipeline Loader → Full Pipeline LoRA → Full Pipeline Sampler → Save Image
                                                   └─ report → Preview Any
```

LoRA 节点可串联，也可以绕过它直接采样。它加载训练器支持的 LoRA，默认 `bypass`。该类型是 `KOHAKU_MOSS_PIPELINE`，不连接普通 `MODEL`、`CLIP`、`KSampler` 或 `VAE Decode`；完整采样器已经输出解码后的图片。

`examples/` 中提供可导入的 UI 工作流及 API 工作流，包含一个 LoRA 加载器。示例使用占位 LoRA `example.safetensors`，请改为自己的文件。

## Loader 设置

先选择本地模型文件，再设置高级字段 `trainer_root` 为训练器的绝对路径。也可在启动 ComfyUI 前设置：

```bash
export KOHAKU_MOSS_TRAINER_ROOT=/path/to/KohakuMossTrainer
```

`python_path` 留空时选择 `trainer_root/.venv/bin/python`；Windows 环境请显式填写训练器 Python 的路径。保持虚拟环境可执行文件路径，不要替换成其符号链接指向的基础 Python。

| 参数 | 作用 |
|---|---|
| `precision` | 训练器 DiT 计算精度；`auto` 交给训练器，在 V100 上选择 FP16 |
| `memory_profile` | 训练器 DiT 显存策略，`auto/speed/balanced/low/minimal` |
| `text_memory` | 训练器文本编码器放置策略 |
| `vae_precision` | `auto/fp16/bf16/fp32`；`auto` 交给训练器，在当前接口的 V100 配置上选择 FP16 |
| `vae_memory` | `auto/resident/offload/ring/cpu`；`resident` 请求 VAE 常驻，`auto` 按可用资源选择 |
| `tune` | `off` 使用已有决策或默认候选；`table` 会测量缺失的内核选择并缓存 |
| `adapter_mode` | `bypass` 低秩旁路，或 `merge` 合并 LoRA；变更适配器可能重载管线，不承诺热切换 |
| `fp16_fix` | 训练器原生 FP16 修正；已修正权重保留其元数据，未修正权重需要合适的显式设置 |
| `cuda_home` / `cxx` | 仅配置训练器子进程的 CUDA 工具链及 C++ 编译器；留空继承环境 |
| `worker_ram_gib` | Linux systemd 子进程 RAM 上限，默认 12 GiB，并限制 2 GiB swap；0 明确关闭限制 |

RAM 限制需要可用的 `systemd-run --user`。非 systemd 或 Windows 环境须显式将该值设为 0；这些平台没有本版运行验证。

## 采样设置

- 使用训练器原生 Euler 采样、shift 后吸附到训练离散 sigma 网格；请求步数与实际去重后的步数可能不同，报告记录每个批次的实际步数。
- `cfg_mode` 支持 `packed` 和 `two_pass`。
- `skip_layers=default` 使用训练器模型默认的负分支跳层；`none` 禁用，或填写逗号分隔的层号。
- 每张图片使用训练器 CUDA RNG，批次种子为 `seed + image_index`，不保证与 ComfyUI 原生噪声逐像素一致。
- `keep_loaded=true` 保留兼容管线；模型、LoRA、精度或放置配置改变，或者需要更大前向预算时会重载。
- `force_run=true` 会实际重新采样，避免同种子被节点结果缓存跳过。设为 false 可使用 ComfyUI 正常结果缓存。

## 管线范围

完整入口调用训练器本身的加载器、模型前向、OpSet/registry、varlen attention、融合候选、采样器和 VAE。具体后端由训练器根据 GPU、精度和已有调优表选择，不保证所有算子都走 TileLang，也不会强制融合一个实际更慢的候选。

它不复刻训练任务，也不会加载优化器、梯度或训练数据。训练时离线预编码并释放文本编码器的流程没有另行实现；这里遵循训练器交互式 `generate()`，重复请求仍会编码文本。`full_precompile` 式 TE/VAE 整图编译不属于本入口。

## 运行与释放

ComfyUI 与训练器通过本地 stdio 和临时 NumPy 文件通信。训练器在自己的 Python 进程中运行，权重及数据不上传。设置 Hugging Face 离线模式，不触发自动模型下载。

采样前释放 ComfyUI 的已加载 GPU 模型。执行不含本采样器的后续工作流时请求释放 worker；也可使用 `Full Pipeline Release` 节点。ComfyUI 中断会终止 worker，在 POSIX 系统中同时终止其编译子进程组。

在同一个图中混用其他 GPU 采样分支时，将 `keep_loaded=false`，让本节点返回图片前释放训练器。并存的其他训练器插件管理各自 worker，应避免在同一图中混用。

报告包含训练器 revision、Torch/CUDA/GPU、真实执行 API、模型准备与生成耗时、实际步数、VAE dtype/placement、权重流式状态及训练器原始 step stats。首次加载、编译和调优可能明显慢于后续请求，不能用其耗时表示稳定采样速度。

## English

This standalone package runs the complete `kohakumoss.inference.MossPipeline` in an existing trainer Python environment. It includes loader, chainable LoRA, sampler and release nodes. The sampler returns decoded ComfyUI images and a runtime report; it does not use KSampler. Set `trainer_root` or `KOHAKU_MOSS_TRAINER_ROOT`, select your model files, and edit the example LoRA. The current initial release is **unverified: no tests or benchmarks were run**. No weights, trainer implementation or precompiled GPU binaries are bundled.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE). KohakuMossTrainer and the model weights are separate dependencies and retain their own terms. This package keeps Kohaku in its primary name and does not redistribute the third-party Precompile implementation.
