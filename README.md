# llama2-7b 本地推理与加速实验

GitHub 项目：llama2-7b-lora-opt-exp — Record how to optimize the inference of llama2-7b model with merged lora.

完整执行顺序与可复制命令见 **[全套运行手册](docs/runbook.md)**：编译 → 下载 → 合并 → BF16 测试/bench → Q4 测试/bench → PPL 对比。

当前已定：Llama 2 7B Base 的 `modelscope/Llama-2-7b-ms` 镜像 + `iamshnoo/alpaca-2-7b-chinese`，CPU 合并 BF16，再本地 Q4_K_M。镜像配置和 tokenizer 已核对；与官方权重逐字节一致性及真实合并尚未验证。最新下载说明见运行手册，环境与验收进度见 [状态记录](docs/status.md)。

目标：将指定 LoRA 与其原始 Llama 2 7B 基座合并为 BF16，使用 llama.cpp 部署，并比较 Q4、投机解码与二者组合。

当前交付是可执行工程框架；尚未提供模型，也尚未执行真实合并、量化或推理。设计与验收见 [docs/design.md](docs/design.md)，环境状态见 [docs/status.md](docs/status.md)。

## 1. 配置环境

以下命令在本项目根目录执行，适用于 PowerShell。Python 3.11+、Git、CMake 和 C++ 编译器为前提。Windows CUDA 构建还需要受支持的 CUDA Toolkit 和 Visual Studio C++ 工具链；nvidia-smi 的 CUDA 版本不表示已安装 Toolkit。

```powershell
(& E:\Software\Anaconda\Scripts\conda.exe shell.powershell hook) | Out-String | Invoke-Expression
conda activate llama2-7b
python -m pip check
if (-not (Test-Path config.local.json)) { Copy-Item config.example.json config.local.json }
```

首次 bootstrap 使用配置的 llama_ref，记录实际提交到 llama.lock.json；随后始终使用该提交。应将两个 lock 文件纳入 Git。未生成锁之前不宣称依赖已经固定。更严格复现可在第一次构建前指定完整 commit SHA。默认 CPU 后端可先验证流程；准备好 CUDA 工具链后，将 backend 改为 cuda 并重新 bootstrap。

## 2. 放置模型并转换

在 config.local.json 填写本地目录：base_model 是 LoRA 训练时使用的准确基座，adapter 是 PEFT 适配器目录，tokenizer 是训练时使用的 tokenizer 目录。默认使用 models/base、models/lora。若已有完整合并 BF16 checkpoint，直接把 merged_model 指向它并跳过 merge。

```powershell
python scripts/pipeline.py merge
python scripts/pipeline.py convert
python scripts/pipeline.py quantize
```

合并采用 CPU BF16 并启用 safe_merge，避免 8GB 显存溢出；主机内存仍需预留权重和合并临时空间。扩词表或特殊适配器结构需要另行验证，脚本发现词表变化会停止。中途失败可能留下不完整输出，检查后换新输出路径再运行。

## 3. 运行实验

```powershell
# 每次仅启动一组；在另一个终端运行 benchmark，结束后 Ctrl+C 停止服务。
python scripts/pipeline.py serve --precision bf16 --spec none
python scripts/pipeline.py serve --precision q4 --spec none
python scripts/pipeline.py serve --precision bf16 --spec ngram-simple
python scripts/pipeline.py serve --precision q4 --spec ngram-simple
# 需要先配置兼容的 draft_gguf
python scripts/pipeline.py serve --precision bf16 --spec draft
python scripts/pipeline.py serve --precision q4 --spec draft

python scripts/benchmark.py --label q4-none --output results/q4-none.jsonl
```

label 必须与实际运行配置一致；每组输出使用新文件名。接口为 http://127.0.0.1:8080，benchmark 使用原生 /completion，不自动套聊天模板。请将 bench/prompts.jsonl 替换为符合该 LoRA 训练格式的固定评测集。

gpu_layers 默认 0。CUDA 环境下先将 Q4 设置为 16 层，根据空闲显存调整；全量卸载使用 99 并核对日志，32 不一定包含输出层。BF16 从较少层数开始。每次调整保存配置与对应结果。Flash Attention 单独作为变量比较。

所有 stage 均支持 --dry-run（不执行命令）；无需本地配置时附加 --config config.example.json。模型、构建文件、本地配置和实验结果不入 Git。不会自动下载模型或访问 Hugging Face 私有仓库。

仓库代码采用 MIT 许可（见 LICENSE）；模型权重和 LoRA 分别遵守各自上游许可证。
