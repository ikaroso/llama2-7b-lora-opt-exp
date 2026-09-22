# 项目协作约定

## 目标与完成条件

优化 **Llama 2 7B + LoRA 合并模型**的单请求 prefill、decode 和 TTFT。原始基线是 Merge BF16，无量化、无投机；Q4 GPU 无投机只是投机实验对照。当前底座为 `modelscope/Llama-2-7b-ms`，LoRA 为 `iamshnoo/alpaca-2-7b-chinese`。

在用户授权范围内完成实现、必要验证和留档；本地修复、复测无需逐步确认。实验任务按约定的方法范围与停止条件收尾，不在首次结果后提前交付，也不无限扩展实验。只做文档整理时不重跑模型。用户新的明确要求优先于本文件默认约定。

## 按需查阅

- 结论、基线、收益与未完成项：[总报告](docs/final-report.md)；阶段进度：[状态记录](docs/status.md)。历史文字与现有日志冲突时核查证据，不照抄旧结论。
- 下载、合并、构建、量化与 PPL：[运行手册](docs/runbook.md)；改变模型选型或架构时参考 [设计](docs/design.md)。
- 复现实验：[优化实验](docs/optimization-20260912.md)、[BF16 补测](docs/bf16-baseline-20260913.md)，对应协议与摘要在 `bench/`，原始证据在 `results/`。

## 本机环境

- 面向用户的命令、文档及 Windows 脚本统一 **CMD**，从项目根目录执行，路径加双引号；使用 Conda `llama2-7b`，不依赖系统 Python。合并用 CPU PyTorch，llama.cpp CUDA 后端独立于 PyTorch CUDA。
- `llama.cpp/` 版本以 `llama.lock.json` 为准；构建入口 `scripts/build_cuda.cmd`，CUDA 产物 `llama.cpp/build-cuda/bin/`。复用现有 `scripts/`；因任务需要变更模型源、依赖或源码时同步记录版本、兼容性与验证，不绕过哈希或词表检查。
- 本机 RTX 4060 Laptop 8GB 无法完整容纳 BF16 7B 权重。测量前核对空闲显存和实际卸载；性能实验串行运行，避免多个测试争用 CPU/GPU。

## 实验可信度

- 性能协议固定**单并发、输入 512、输出 256 token**。bench 用 p512/tg256；server 用可追溯的真实文本/问题，保存来源、提示和长度核验。PPL 等质量指标使用各自协议，不套用生成长度要求。
- 最终确认至少五轮，保留逐轮数据、统计波动与失败；区分初筛、确认及问题覆盖。关闭提示缓存；忽略 EOS、重复输出和运行状态漂移必须披露，不能用于夸大成绩。
- 固定对照条件并列出变化项：纯量化用相同后端与卸载条件，整体部署收益单列；不同日期的倍率不直接相乘。batch/ubatch 不是并发数，按当前约定不列为主优化方法。
- `llama-bench` 不提供 TTFT 或投机收益；这两项用 server 实测。接受率、返回文本一致性和回答质量分别报告；流式 token ID 不完整时不声称逐 token 一致，性能通过不等于 PPL/质量通过。
- 用 `scripts/optimize.py` 或 `scripts/run_logged.py` 等现有入口保存协议、命令、退出/完成状态和原始输出；结果使用新目录或前缀，不覆盖旧日志。汇总引用证据，未运行、失败及适配前提缺失均如实标记。

## 验证、文档与 Git

- 验证覆盖本次变化即可：文档检查链接与命令，代码运行受影响检查；修复相关失败后复测。没有新问题时不重复全套 7B 测试。
- 流程变化更新运行手册；实验结果更新总报告、状态记录及可复现摘要。`AGENTS.md` 只保留稳定约定，不堆积成绩与历史。
- 保留用户改动，合理分阶段提交本次相关文件。版本化协议、配置模板和 `bench/` 摘要可提交；模型、数据、构建产物、`results/`、`config.local.json` 与凭据遵循 `.gitignore`。推送按用户授权执行，不强推或改写历史。

维护依据：[OpenAI：Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。按任务指向必要资料，保留项目特有约束与明确完成边界。
