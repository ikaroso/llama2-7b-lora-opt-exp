# 项目协作约定

## 目标与入口

- 本项目验证 Llama 2 7B Base + LoRA → CPU 合并 BF16 → GGUF → 本地 Q4_K_M，并比较推理、吞吐、PPL 和投机解码组合。
- 开始工作先读 `docs/runbook.md` 和 `docs/status.md`；设计背景见 `docs/design.md`。旧决策记录属于历史，实际进度以文件、命令输出和实验日志为准。
- 当前底座源为 `modelscope/Llama-2-7b-ms`，LoRA 为 `iamshnoo/alpaca-2-7b-chinese`。不要将 Chat、中文扩词表或已量化模型直接替代底座；调整选型时同步核验适配器、tokenizer 和来源记录。

## 环境与实现

- 用户命令、文档和新增 Windows 脚本统一使用 **CMD**：`set "NAME=value"`、`%NAME%`、`type`；不混入 PowerShell 语法。路径加双引号，从项目根目录执行。
- 使用 Conda `llama2-7b`，不要依赖系统 Python。合并用 CPU PyTorch；llama.cpp 的 CUDA 后端独立于 PyTorch CUDA。
- llama.cpp 位于 `llama.cpp/`，版本以 `llama.lock.json` 为准。CUDA 构建入口为 `scripts/build_cuda.cmd`，产物在 `llama.cpp/build-cuda/bin`；不要无故升级依赖或修改第三方源码。
- 优先复用 `scripts/` 中现有入口，保持实现简单。修改模型源或依赖时同步更新相关锁文件与手册，不放宽哈希、词表及输出覆盖检查来掩盖错误。

## 验证与实验

- 优化实验统一输入 512、输出 256 token；llama-bench 使用 p512/n256，其他工具使用可追溯的真实文本或问题，并核对实际 token 数。最终确认至少五轮，记录逐轮结果与波动；初筛不得冒充确认。
- 关闭提示缓存，避免重复请求或退化输出夸大投机收益。固定长度实验若忽略 EOS，必须说明；质量、输出差异、负收益和未满足的适配条件如实记录。

- 按改动范围验证：文档核对命令和变量；脚本先做语法与相关参数检查，适用时使用 `--dry-run`；依赖修改后执行 `python -m pip check`。不因小改动重复跑完整 7B 实验。
- 区分静态检查、小模型测试、真实 7B 合并和真实推理。只报告实际完成的验证；失败也记录，不把配置成功当成全链路跑通。
- 实验通过 `scripts/run_logged.py` 保存命令、退出码和原始输出。每轮用新的结果前缀，同组比较沿用相同 RUN；不覆盖旧结果。
- 对比固定模型来源、语料、上下文、KV 类型和运行参数，明确变化项。BF16/Q4 的纯量化对比使用同一后端；GPU、Flash Attention 和投机解码收益单独说明。
- `llama-bench` 不衡量投机解码收益；投机实验通过 server 和对应请求测量。GPU 实验先查空闲显存并核对实际卸载日志，避免并行启动多个大模型进程。

## 文件与交付

- 保持 `.gitignore` 的边界：模型、数据、构建产物、实验结果、本地配置和凭据不提交；模型与 LoRA 许可证独立于仓库代码许可证。
- 保留用户未提交的改动；提交仅包含本次相关文件。Git 推送按用户授权执行，不强推或改写已有历史。
- 流程变化同步更新 `docs/runbook.md`；完成真实实验后更新 `docs/status.md`，引用具体结果。交付简述改了什么、验证了什么、还有什么未验证。
