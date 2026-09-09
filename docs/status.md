# 初始落地状态

## 最新更新：专用环境与执行手册

已创建 Conda `llama2-7b`，位于 `E:\Software\Anaconda\envs\llama2-7b`。已验证 Python 3.11、torch 2.11.0+cpu、transformers 4.57.6、peft 0.18.0；pip check 通过。随机小型 Llama 的 CPU BF16 前向、非零 LoRA 合并、safetensors 保存与重新加载检查通过。实际转换器 --help 已运行。

项目源码路径已改为 `llama.cpp`，锁定 `b31b71f3a076bfc4278daad442203a9c51c6e676`；构建目标包含 server、completion、quantize、bench、perplexity。完整命令见 [运行手册](runbook.md)。实际 7B 下载/合并/转换/推理未执行。下面及决策记录中的旧环境状态作为历史快照保留。

专用 Conda 环境已安装 CUDA Toolkit 12.9.2（nvcc 12.9.86）及 Ninja 1.13.2。`nvcc --version`、pip check 通过；`scripts/build_cuda.cmd configure` 已使用 VS2022 MSVC 19.44 完成 CUDA 编译器 ABI 检查和 CMake 配置，目标架构为 RTX 4060 的 89。完整 llama.cpp 构建尚未执行。构建脚本在 Conda 激活后显式加载 VS2022，产物目录为 `llama.cpp/build-cuda/bin`。Conda 包精确版本记录在 `conda-win64.lock.txt`（不包含 pip 包，pip 版本另见 `environment.lock.txt`）。

以下为首次建项快照。2026-09-09 后续已确定模型与 LoRA，并完成 PowerShell 7 用户级 Python 修复；最新环境、选型和待验收状态见 [决策记录](decision-2026-09-09.md)。

2026-09-09，本地 Windows / Python 3.11.10 / Git 2.48.1 / CMake 4.1.0。

- 检测到 RTX 4060 系列 GPU，8188 MiB 总显存；检查时已使用约 2726 MiB。
- 未在 PATH 检测到 nvcc；主机 RAM 查询被操作系统拒绝，尚未确认。
- 终端 git ls-remote 无法连接 GitHub 443；未拉取或构建 llama.cpp，尚未生成源码锁。
- 尚未提供基座、LoRA、训练 tokenizer、草稿模型和质量评测集。
- 已实现配置、转换/量化/启动脚本、原始基准记录、设计及运行说明。
- 没有真实推理成绩，也没有量化产物。后续需安装依赖并执行 README 步骤。
