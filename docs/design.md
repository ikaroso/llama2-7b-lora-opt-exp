# 设计与实验规范

## 数据流与边界

准确基座 + PEFT LoRA + 训练 tokenizer → CPU 合并 BF16 HF → BF16 GGUF → Q4_K_M GGUF → llama-server → 固定请求集 → JSONL。

先合并再量化，使所有实验对应同一微调模型。Q4_K_M 是混合 K 量化方案，并非每个张量严格 4 bit。BF16 指权重存储格式，不能据此认定所有算子均以 BF16 执行。运行时挂载 LoRA 可作为未来独立实验，当前不混入主链。

基座、适配器来源、revision、训练 prompt 模板、tokenizer、权重 SHA256 应由操作者登记到模型清单。模型权重只放在忽略目录；下载与使用遵循模型许可证。架构检查不能证明模型来源确为 Llama 2，必须核对训练记录。safe_merge 的有限值检查也不能替代语义验证。

## 实验矩阵

| 编号 | 目标权重 | 投机模式 | 前提 |
|---|---|---|---|
| B0 | BF16 | none | 合并 GGUF |
| Q0 | Q4_K_M | none | 本地量化 |
| B1 | BF16 | ngram-simple | 无额外模型 |
| Q1 | Q4_K_M | ngram-simple | 无额外模型 |
| B2 | BF16 | draft | 兼容草稿模型 |
| Q2 | Q4_K_M | draft | 兼容草稿模型 |

n-gram 适合含重复片段的负载，应同时用自然语言和代码样本验证，不预设必然加速。draft 需要检查 tokenizer、token ID、BOS/EOS 和词表兼容性；模型名称或同属 Llama 系列不构成兼容证明。还要确认草稿速度与接受率，LoRA 改变分布可能降低收益。不默认指定 TinyLlama，也不假定其他模型的 EAGLE 权重可用于当前微调目标。

第一轮固定硬件、CPU 线程、上下文 2048、单并发、GPU 层数、KV 类型（llama.cpp 默认）、Flash Attention、prompt、生成上限、贪心采样。第二轮再分别调优卸载层数与 Flash Attention，记录为不同配置，区分算法收益和显存释放带来的卸载收益。draft_tokens 建议扫描 3/5/8；8GB GPU 上草稿额外显存可能迫使目标回退 CPU。

## 基准与质量验收

脚本先预热，再顺序执行每个 prompt 三次；关闭 prompt cache 复用，记录整请求耗时、完整输出以及服务返回的 timings。以实际生成 token 数与服务返回的解码速率比较，不能用请求上限冒充实际 token 数。当前为非流式基准，不测 TTFT，不将总耗时称为首 token 延迟。

正式评估扩大到代表任务分布的固定数据集，报告中位数、P95、prefill/decode 吞吐、显存/RAM 峰值、模型文件大小，并单独采集服务日志中的草稿接受统计。当前脚本保留原始数据，尚未实现系统内存采样或自动接受率汇总。短样本仅用于冒烟检查，不能推断业务质量。

比较 Q0 对 B0 的任务质量；比较 B1/B2 对 B0、Q1/Q2 对 Q0 的输出与质量。量化可能改变输出，投机实现也需实际验证，不保证跨内核逐 token 相同。速度提升必须同时满足事先约定的业务质量门槛；阈值待任务明确后设定。不要把合成接受率模式作为有效模型输出。

## 资源与复现

7B 的 BF16 权重约 14GB（十进制），加上 KV 与运行时缓冲，8GB 显存无法全量容纳。Q4 文件通常为数 GB，实际大小以产物为准。建议主机内存至少 32GB、可用磁盘至少 60GB 起步，合并峰值需实测；当前未确认主机 RAM。先采用保守 CPU 默认值，后续根据实际空闲显存调整 CUDA 卸载。

每次正式实验保存：项目 commit、llama.lock.json、pip freeze、实际服务命令、配置副本、模型哈希、工具链和 GPU 驱动版本、启动日志、原始 JSONL、数据集哈希。依赖范围用于初次安装，完整 freeze 才是当次 Python 环境记录。bootstrap 不安装编译器，不自动升级已锁源码。

## 上游依据

- [llama.cpp 构建](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [GGUF 模型转换](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)
- [量化工具](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)
- [投机解码及参数](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [服务接口](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [PEFT LoRA 合并 API](https://huggingface.co/docs/peft/package_reference/lora)

文档核对日期：2026-09-09。上游 master 可变化，正式运行以本地锁定提交及其 --help 为准。
