# Llama2 + LoRA 本地推理优化总报告

更新日期：2026-09-22。原实验在 9 月 12–13 日完成，本次新增 Spec-Bench 摘要/RAG 子集的五轮投机测试。本文件为当前结论的统一入口；不同数据集和日期的结果分开报告，不混成一个倍率。

## 新增：Spec-Bench 子集投机评测

早期最终确认使用 CMRC2018 的三个中文问题加文章，WikiText-2 用于初筛。按用户选择，现已补充 Spec-Bench 摘要与 RAG 各六条真实样本，仍为单并发、512 输入 / 256 输出，三方法各五轮，共 **180 次正式请求**。数据固定版本与哈希，按确定规则取样；仅截上下文，完整保留问题/指令，不重复填充短题。模型、ngram N12/M48/min_hits1、TinyLlama 草拟上限3等配置保持不变。

| 本次配置 | Prefill token/s | Decode token/s | TTFT ms | 草稿接受率 | 返回文本完全匹配对照 |
|---|---:|---:|---:|---:|---:|
| Q4 GPU 无投机对照 | 2309.22 ± 35.81 | 49.59 ± 0.10 | 352.70 ± 9.75 | 不适用 | 60/60 |
| Q4 GPU + ngram-simple | 2314.65 ± 39.07 | 83.71 ± 0.32 | 345.35 ± 4.69 | 51.28% | 50/60 |
| Q4 GPU + TinyLlama draft3 | 2045.37 ± 18.31 | 78.87 ± 0.63 | 380.63 ± 1.80 | 73.79% | 30/60 |

均值 ± 五轮样本标准差；五轮是相同十二题的重复，不能当成六十个不同问题。相对同轮无投机对照，ngram decode **+68.8%**、请求耗时 **-38.3%**；draft3 decode **+59.0%**、请求耗时 **-34.2%**，但 prefill **-11.4%**、TTFT **+8.0%**。ngram TTFT 约 -2.0%，幅度接近计时波动，不宣称稳定的首字优势。

| 任务 | 无投机 decode | ngram decode / 配对变化 | draft3 decode / 配对变化 |
|---|---:|---:|---:|
| 摘要 | 49.60 | 70.01 / +41.1% | 76.41 / +54.1% |
| RAG | 49.58 | 104.07 / +109.9% | 81.50 / +64.4% |

单位 token/s。**摘要中 draft3 更快，RAG 中 ngram 更快**；接受率高并不必然意味着整体更快。部分输出有复制、跑题或继续生成模板，且与无投机文本存在差异；结果不能称为严格无损加速。摘要/RAG 偏长上下文与输入输出重叠，不代表其余任务、中文能力或官方完整 Spec-Bench 成绩；原始输入与自然停止协议也已改变。完整分类/逐轮/逐题结果见 [Spec-Bench 结果报告](specbench-results.md)，取样与复现见 [实验协议](specbench-subset.md)。

## 早期优化与效果总览（9 月 12–13 日）

原始基线为 **LoRA 合并后的 BF16 模型，无量化、无投机**。以下收益均由同一批次的五轮 server 配对结果计算；“2.05 倍”表示达到对照的 2.05 倍，不是增加 2.05 倍。吞吐越高越好，TTFT 越低越好。

| 已做的方法 | 对照与变化项 | Prefill 效果 | Decode 效果 | TTFT 效果 | 当前结论 |
|---|---|---:|---:|---:|---|
| 本地 Q4_K_M 量化 | BF16 CPU → Q4 CPU；其余运行条件一致 | 1.49 倍 | 2.05 倍 | 降低 32.8% | 同条件量化有性能收益；权重文件缩小约 69.7%，质量尚未验收 |
| 全层 GPU 部署 + Flash Attention | Q4 CPU → Q4 GPU + FA | 82.07 倍 | 5.32 倍 | 降低 97.2% | Q4 可全 GPU 运行；收益包含 GPU 卸载和 FA，未拆分各自贡献 |
| ngram-simple 投机解码 | 当天 Q4 GPU 无投机 → 加 ngram | 下降 4.6% | 提升 18.5% | 增加 2.3% | 生成更快，依赖文本重复性，确认集部分返回文本不同 |
| TinyLlama draft-simple | 当天 Q4 GPU 无投机 → 加 1.1B Q4 草稿，草拟上限 3 | 下降 16.2% | 提升 46.3% | 增加 13.9% | 生成收益更大，但首字更慢，需要额外草稿模型 |
| 量化 + GPU + FA 的整体部署 | BF16 CPU → Q4 GPU + FA | 122.19 倍 | 10.90 倍 | 降低 98.1% | 整体部署收益，不能称为纯量化收益，也不是相对最快 BF16 部分卸载的收益 |

ngram 与 draft-simple 都是在 Q4 + GPU + FA 基础上叠加，已经包含量化与投机组合。没有将两种投机方法互相叠加。不同日期的倍率不可直接连乘；batch/ubatch 调整按用户要求不进入主优化表。

## 目标、模型与基线

目标是优化合并 LoRA 后模型的单请求 prefill throughput、decode throughput 和 TTFT。底座为 modelscope/Llama-2-7b-ms，LoRA 为 iamshnoo/alpaca-2-7b-chinese；已存在 CPU 合并后的 BF16 与由其转换、量化得到的 Q4_K_M GGUF。镜像与官方底座权重的逐字节一致性未证明。

**原始基线是 Merge BF16，无量化、无投机。** Q4 GPU 无投机仅是投机实验的对照组。硬件为 RTX 4060 Laptop 8GB、i7-14650HX；llama.cpp 固定 b31b71f3a076bfc4278daad442203a9c51c6e676。BF16 权重约 13.48 GB，Q4_K_M 约 4.08 GB；BF16 无法完整放入本机显存，本报告使用明确的 CPU 基线，未搜索最快的 BF16 部分 GPU 卸载方案。

## 早期 CMRC 测试的测量口径

- 单并发，8 线程，context=2048，主线 batch/ubatch=128。batch 是逻辑 token 批次上限，ubatch 是物理计算微批次上限，二者不是请求并发数。按用户要求，调整这两个参数不列为主优化方法。
- server 使用 CMRC2018 官方开发集的真实问题与上下文，按数据顺序选择三个问题，模板与截断后输入严格为 512 token，输出固定 256 token。温度 0、seed=42，关闭提示缓存，忽略 EOS；忽略 EOS 的固定长度输出不代表自然停止回答质量。
- 每个最终配置五个独立轮次，每轮独立进程、独立预热、三个正式问题。报告均值及样本标准差；收益按同轮配对计算。五轮重复衡量运行波动，不等于十五个不同问题。
- TTFT 是客户端请求开始至首个非空文本事件的时间，排除模型加载。server 吞吐按各轮总 token/总时间计算，decode 使用引擎首 token 之后的计时口径。
- llama-bench 独立测 p512、tg256，各配置五个独立进程。它不提供 TTFT，也不用于测量投机收益。

### 实际输入

三组输入均为“真实问题 + 对应文章”，不是仅输入问题。来自 CMRC2018 官方开发集，不在提示中提供标注答案。

| 问题 | 附带的文章 | 保留的文章 token 数 |
|---|---|---:|
| 《战国无双3》是由哪两个公司合作开发的？ | 《战国无双3》的游戏介绍 | 467 |
| 锣鼓经是什么？ | 锣鼓经、戏曲打击乐及其用途介绍 | 478 |
| 广茂铁路全长多少公里？ | 广茂铁路路线、长度及运营介绍 | 476 |

使用 Instruction / Input / Response 模板，文章按 token 截断；加问题、模板和 BOS 后每组总计 512 token。相同三组输入重复五轮，不能视为广泛问答评测。

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

### 投机参数与命中情况

接受率 = 目标模型接受的草稿 token / 提出的草稿 token，不是回答正确率，也不是返回文本一致率。以下为正式五轮共十五条请求的合计，排除预热。

| 方法 | 实验参数 | 提出草稿 token | 接受 token | 接受率 |
|---|---|---:|---:|---:|
| ngram-simple | `--spec-type ngram-simple`；未显式调参，固定版本默认 N=12、M=48、min_hits=1 | 3740 | 715 | 19.12% |
| TinyLlama draft-simple | `--spec-type draft-simple -md <草稿GGUF> -ngld 99 --spec-draft-n-max 3` | 3540 | 2635 | 74.44% |

ngram 的 N 是查找片段长度，M 是候选续写长度设置，min_hits 是提出候选所需的最低匹配次数；M=48 不代表每次都会提出或接受 48 个 token。五轮相同问题、确定性采样下接受计数相同，重复仅体现时间波动，不增加问题覆盖。

## 方法覆盖与结论边界

已初筛 FA、部分/全部 GPU 卸载、线程数、Q8 KV、多种 ngram、TinyLlama 草拟长度以及 batch 组合。部分卸载、Q8 KV、4 线程等未显示值得主推的收益；ngram-cache 在不复用提示的 WikiText 初筛中慢于对照。batch 调整与组合仅留历史记录。MTP 缺少 Llama2 原生预测头；EAGLE、DFlash、DSpark、DFlash2 没有经本项目验证且适配当前合并模型的专用权重，未伪造实测。

| 其他已尝试项 | 观察结果 | 验证级别与处理 |
|---|---|---|
| FA 开关 | Q4 GPU bench：FA on 的 pp/tg 为 2150.21/49.64，off 为 1866.71/48.37 token/s | 三次初筛，保留 FA on；没有独立五轮 FA 因果确认 |
| 部分 GPU 卸载 | ngl=24 的 pp/tg 为 666.32/22.19，低于全卸载的 2150.21/49.64 | 三次初筛，显存能容纳 Q4 时未见收益 |
| KV cache Q8 | b512/ub512 下 pp/tg 为 2108.65/48.28，对应普通 KV 为 2244.79/49.44 | 三次初筛，未见吞吐优势 |
| CPU 线程 8 → 4 | b512/ub512 下 pp/tg 为 2177.78/49.14，对应 8 线程为 2244.79/49.44 | 三次初筛，未见性能优势 |
| 多种 ngram 与草拟长度 | map-k、map-k4v、mod、cache、短 ngram，以及 TinyLlama 草拟上限 5 已测 | WikiText 重复标题夸大收益；最终确认采用中文真实问题、ngram-simple 和 draft3 |
| batch/ubatch 调整及 ngram 组合 | 曾完成初筛和五轮确认 | 按用户要求移出主方案，历史结果不删除；两参数不是请求并发数 |
| MTP、EAGLE、DFlash、DSpark、DFlash2 | 未完成适配当前合并模型的运行验证 | 无实测成绩；不能标为已实现的优化 |

上表 bench 数值均为 p512/tg256，不能与另一日期的 BF16 严格 CPU 确认混算；旧 ngl=0 初筛没有严格关闭所有 GPU 算子，不替代 BF16 补测。

当前主线是 **BF16 → Q4_K_M → 全 GPU → 按需求选投机**。Q4 GPU 是已验证的部署对照；TinyLlama 草稿适合关注生成吞吐的候选用途，但会增加首字延迟。ngram 收益依赖文本重复性。

中文及英文输出均存在复制或重复，不能将吞吐当成正确答案产出速度。ngram 确认集返回文本与对照匹配 10/15；draft3 为 15/15，但更广中文初筛仅 4/6，不能声称普遍无损。当前 server 的 UTF-8 缓冲会使流式 token ID 列表不完整，长度检查依赖最终计数，一致性仅比较返回文本。PPL、正式准确率及自然停止质量验收仍未完成。

本机后台负载和时钟未严格控制。9 月 12 日 GPU 最高采样 93°C，9 月 13 日 server 最高 79°C；温度和日期变化限制跨批次比较。显存采样为整卡已用量，不是模型独占峰值。

## 证据与复现入口

| 内容 | 入口 |
|---|---|
| Spec-Bench 子集最新分类及五轮结果 | [结果报告](specbench-results.md)、[结构化结果](../bench/specbench-results-20260922.json) |
| Spec-Bench 固定源、选题与复现 | [实验协议](specbench-subset.md)、[源锁定](../bench/specbench-source.json) |
| 下载、合并、量化和执行命令 | [运行手册](runbook.md) |
| BF16/Q4 五轮逐次值、配对收益与复现 | [基线补测报告](bf16-baseline-20260913.md) |
| 投机协议、初筛、五轮逐次值及历史组合 | [优化实验详细报告](optimization-20260912.md) |
| BF16 补测机器可读结果与原始文件哈希 | [bf16-results-20260913.json](../bench/bf16-results-20260913.json) |
| 投机五轮统计 | [optimization-confirmation-20260912.json](../bench/optimization-confirmation-20260912.json) |
| 初筛及确认逐样本、返回文本哈希 | [optimization-results-20260912.json](../bench/optimization-results-20260912.json) |
| 模型、草稿与数据哈希 | [optimization-inputs-20260912.json](../bench/optimization-inputs-20260912.json) |

原始日志在本地 results/opt-20260912-*、results/bf16-confirm-server-20260913、results/bf16-confirm-bench-20260913、results/specbench-20260922，按项目约定不提交大体积日志或模型。Spec-Bench 目标与草稿完整哈希在计时结束后重新核验，与此前产物一致，记录在 bench/specbench-models-20260922.json。
