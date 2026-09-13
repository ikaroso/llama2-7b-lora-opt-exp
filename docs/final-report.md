# Llama2 + LoRA 本地推理优化总报告

更新日期：2026-09-13。本文件为当前结论的统一入口；分阶段报告保留逐轮数据和历史实验，不代表所有历史方案仍被推荐。

## 目标、模型与基线

目标是优化合并 LoRA 后模型的单请求 prefill throughput、decode throughput 和 TTFT。底座为 modelscope/Llama-2-7b-ms，LoRA 为 iamshnoo/alpaca-2-7b-chinese；已存在 CPU 合并后的 BF16 与由其转换、量化得到的 Q4_K_M GGUF。镜像与官方底座权重的逐字节一致性未证明。

**原始基线是 Merge BF16，无量化、无投机。** Q4 GPU 无投机仅是投机实验的对照组。硬件为 RTX 4060 Laptop 8GB、i7-14650HX；llama.cpp 固定 b31b71f3a076bfc4278daad442203a9c51c6e676。BF16 权重约 13.48 GB，Q4_K_M 约 4.08 GB；BF16 无法完整放入本机显存，本报告使用明确的 CPU 基线，未搜索最快的 BF16 部分 GPU 卸载方案。

## 测量口径

- 单并发，8 线程，context=2048，主线 batch/ubatch=128。batch 是逻辑 token 批次上限，ubatch 是物理计算微批次上限，二者不是请求并发数。按用户要求，调整这两个参数不列为主优化方法。
- server 使用 CMRC2018 官方开发集的真实问题与上下文，按数据顺序选择三个问题，模板与截断后输入严格为 512 token，输出固定 256 token。温度 0、seed=42，关闭提示缓存，忽略 EOS；忽略 EOS 的固定长度输出不代表自然停止回答质量。
- 每个最终配置五个独立轮次，每轮独立进程、独立预热、三个正式问题。报告均值及样本标准差；收益按同轮配对计算。五轮重复衡量运行波动，不等于十五个不同问题。
- TTFT 是客户端请求开始至首个非空文本事件的时间，排除模型加载。server 吞吐按各轮总 token/总时间计算，decode 使用引擎首 token 之后的计时口径。
- llama-bench 独立测 p512、tg256，各配置五个独立进程。它不提供 TTFT，也不用于测量投机收益。

## 量化与 GPU 部署：2026-09-13 五轮确认

共 45 条正式 server 请求及 15 个 bench 进程，全部完成。CPU 两组均关闭层、KV 和额外算子 GPU 卸载，FA off；GPU 组 ngl=99、FA on。以下是 server 五轮均值 ± 标准差。

| 配置 | Prefill token/s ↑ | Decode token/s ↑ | TTFT ms ↓ |
|---|---:|---:|---:|
| Merge BF16，CPU，原始基线 | 17.45 ± 0.82 | 4.38 ± 0.13 | 29690.95 ± 1401.14 |
| Q4_K_M，相同 CPU 条件 | 25.97 ± 0.80 | 8.97 ± 0.32 | 19952.42 ± 627.58 |
| Q4_K_M，全 GPU + FA | 2130.40 ± 59.39 | 47.68 ± 1.08 | 554.88 ± 15.38 |

纯量化同条件收益：prefill **1.49 倍**、decode **2.05 倍**、TTFT 降低 **32.8%**。BF16 CPU → Q4 GPU 的整体部署收益：prefill **122.19 倍**、decode **10.90 倍**、TTFT 降低 **98.1%**；这同时改变精度、GPU 卸载与 FA，不能全部归因于量化。

bench 五轮均值（pp512 / tg256，token/s）：BF16 CPU **18.22 / 4.48**；Q4 CPU **26.21 / 9.90**；Q4 GPU **2136.39 / 50.70**。bench 的同 CPU 量化配对收益为 **1.44 / 2.21 倍**，不与 server 数值混算。

## 投机解码：2026-09-12 五轮确认

以下三组各有五轮、每轮三个正式问题。日期不同，保留当天 Q4 GPU 对照，不将旧投机成绩除以新 BF16 成绩作为同轮总倍率。

| 配置 | Prefill token/s ↑ | Decode token/s ↑ | TTFT ms ↓ |
|---|---:|---:|---:|
| Q4 GPU，无投机对照 | 2037.51 ± 87.78 | 45.53 ± 0.51 | 398.67 ± 13.02 |
| Q4 GPU + ngram-simple | 1944.03 ± 102.81 | 53.93 ± 0.76 | 407.85 ± 16.35 |
| Q4 GPU + TinyLlama draft-simple，草拟上限 3 | 1703.67 ± 66.60 | 66.63 ± 4.60 | 453.32 ± 14.99 |

ngram-simple 从已有 token 序列的重复片段提出候选，由目标模型验证；配对 decode 提升约 **18%**，TTFT 略差。TinyLlama 先产生候选，目标模型批量验证；配对 decode 提升约 **46%**，但 prefill 下降约 **16%**、TTFT 增加约 **14%**。

草稿权重来自 [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)，固定 revision 52e7645ba7c309695bec7ac98f4f005b139cf465，文件 tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf。draft-simple 是固定版本 llama.cpp 的普通草稿验证方法，不是另一个模型名称。草稿没有针对本项目 LoRA 重新训练；已检查 32000 个词表映射和 BOS/EOS 兼容，并完成真实运行。

## 方法覆盖与结论边界

已初筛 FA、部分/全部 GPU 卸载、线程数、Q8 KV、多种 ngram、TinyLlama 草拟长度以及 batch 组合。部分卸载、Q8 KV、4 线程等未显示值得主推的收益；ngram-cache 在不复用提示的 WikiText 初筛中慢于对照。batch 调整与组合仅留历史记录。MTP 缺少 Llama2 原生预测头；EAGLE、DFlash、DSpark、DFlash2 没有经本项目验证且适配当前合并模型的专用权重，未伪造实测。

当前主线是 **BF16 → Q4_K_M → 全 GPU → 按需求选投机**。Q4 GPU 是已验证的部署对照；TinyLlama 草稿适合关注生成吞吐的候选用途，但会增加首字延迟。ngram 收益依赖文本重复性。

中文及英文输出均存在复制或重复，不能将吞吐当成正确答案产出速度。ngram 确认集返回文本与对照匹配 10/15；draft3 为 15/15，但更广中文初筛仅 4/6，不能声称普遍无损。当前 server 的 UTF-8 缓冲会使流式 token ID 列表不完整，长度检查依赖最终计数，一致性仅比较返回文本。PPL、正式准确率及自然停止质量验收仍未完成。

本机后台负载和时钟未严格控制。9 月 12 日 GPU 最高采样 93°C，9 月 13 日 server 最高 79°C；温度和日期变化限制跨批次比较。显存采样为整卡已用量，不是模型独占峰值。

## 证据与复现入口

| 内容 | 入口 |
|---|---|
| 下载、合并、量化和执行命令 | [运行手册](runbook.md) |
| BF16/Q4 五轮逐次值、配对收益与复现 | [基线补测报告](bf16-baseline-20260913.md) |
| 投机协议、初筛、五轮逐次值及历史组合 | [优化实验详细报告](optimization-20260912.md) |
| BF16 补测机器可读结果与原始文件哈希 | [bf16-results-20260913.json](../bench/bf16-results-20260913.json) |
| 投机五轮统计 | [optimization-confirmation-20260912.json](../bench/optimization-confirmation-20260912.json) |
| 初筛及确认逐样本、返回文本哈希 | [optimization-results-20260912.json](../bench/optimization-results-20260912.json) |
| 模型、草稿与数据哈希 | [optimization-inputs-20260912.json](../bench/optimization-inputs-20260912.json) |

原始日志在本地 results/opt-20260912-*、results/bf16-confirm-server-20260913、results/bf16-confirm-bench-20260913，按项目约定不提交大体积日志或模型。协议与结果已 Git 保存；本总报告不包含新运行的实验。
