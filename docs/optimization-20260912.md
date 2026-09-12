# 512/256 推理优化实验

## 协议与范围

优化前保存点：baseline-before-optimization-20260912（c605740）。保持原合并模型与 llama.cpp 锁定版本，不更换目标模型。日期：2026-09-12。

- llama-bench：p512、n256，初筛重复 3 次；候选与基准复测。pp512 与 tg256 是工具的独立测试，不能据此计算服务端 TTFT。
- server：真实 WikiText-2 test 文本，经目标 tokenizer 编码，按 offset=4096*i 选择连续 512-token 片段，每轮三个不同片段。不重复同一提示，避免 ngram-cache 从前轮生成直接获益；预热使用独立 offset=65536。每个配置使用相同片段顺序。不拼造问题，不添加模板或隐式 BOS；发送 token ID 数组保证实际输入长度，日志检查 prompt_n=512。
- 数据来源：https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip 。记录本地语料 SHA256 与具体 token ID；英文文章续写负载，不代表中文问答或代码任务。
- 输出固定 256 token，temperature=0、seed=42、ignore_eos=true、cache_prompt=false，单并发，context=2048。忽略 EOS 是固定长度性能协议，不用于自然停止行为或质量打分。
- 每个 server 配置一次排除的预热，之后相同文本与轮次。TTFT 是本机 HTTP 请求发出至首个非空文本 SSE 的客户端时间；排除模型启动，不代表冷启动。原始 SSE 事件与相对时间全部保存。
- 记录服务端最终输入/输出 token 计数，长度不合格立即标记错误，不能混入汇总。输出一致性比较完整返回文本，不声称逐 token ID 相同。不同文本分别比较，再汇总，避免只挑最佳案例。
- 串行运行 GPU 工作，前后复测基准检查漂移。未控制 Windows 后台负载、笔记本功耗和散热，不能将小幅变化断言为稳定收益。未做跨机器泛化。

## 方法覆盖

初筛 Q4/F16 KV 基准、FA off、batch/ubatch 512、4 线程、Q8 KV、部分卸载、ngl=0 与 BF16 ngl=0；服务端测五种 n-gram、短匹配参数、TinyLlama 普通草稿，随后复测优选组合。

TinyLlama 候选：TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF，提交 52e7645ba7c309695bec7ac98f4f005b139cf465，文件 tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf，668788096 字节，SHA256 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0。GGUF 32000 个 token 文本映射全部一致，BOS=1、EOS=2；add_bos/add_eos 草稿未显式写入，最终以引擎运行时检查为准。

EAGLE3、DFlash、DSpark、DFlash2 需要与目标隐藏特征和词表接口匹配的专门权重；目前无已核实适配这个 Base+LoRA 的现成权重。MTP 缺少原生预测层。记录为不具备实验前置条件，不编造收益，也不通过替换目标模型假装完成。来源与说明见 speculative-plan.md。

## 执行（CMD）

先按 runbook 激活环境，每次使用新的输出目录：

```cmd
python scripts/optimize.py bench --out results/opt-20260912-bench --configs bench/optimization-bench.json --repeat 3
python scripts/optimize.py server --out results/opt-20260912-server --configs bench/optimization-server.json --repeat 2
```

过程原始日志存 results/，不纳入 Git；本轮已完成，最终可复查汇总、原始文件哈希与结论见本文件和 bench 下的摘要。

## 初筛发现与协议补充

WikiText 多个输入诱发目标模型重复生成标题，导致 n-gram 获得很高吞吐。保留这一负载结果，但不作为正常问答收益的依据。补充 CMRC2018 官方开发集的真实问题与上下文（https://github.com/ymcui/cmrc2018 ，CC-BY-SA-4.0，Cui et al., EMNLP-IJCNLP 2019）。顺序选择不同 context 的首个问题，用 Alpaca 的 Instruction/Input/Response 模板，保留完整问题与响应起始标记，仅截断原始上下文到总输入 512 token；不使用数据集答案作为提示，不按模型表现挑选样本。它是固定长度性能测试，不是官方 CMRC 准确率测评。

用户追加要求：最终确认至少五轮。最终阶段对相同三条真实输入进行五轮独立复测，每个配置/轮次重启 server，采用交错顺序，防止跨轮草稿缓存；保留每轮数据，报告均值、标准差和范围。初筛结果不能冒充五次确认。

CMRC 初筛中也发现复制上下文和重复词语，说明当前模型的固定 256-token 输出存在质量限制，不能把这些测试解释成优质回答的产出速度。无投机前后基准六个样本返回文本一致；n-gram、普通草稿和 batch 改变在部分样本中出现输出差异，需单列记录。批量验证和单 token 解码的数值路径差异是可能原因，尚未定位到逐层数值证据，不宣称已证明无损。

最终确认使用 CMRC 开发集前三个满足长度要求的 context 的首个问题（DEV_0_QUERY_0、DEV_1_QUERY_0、DEV_2_QUERY_0），不根据输出优劣筛样本。五轮分别启动原基准、batch512、ngram-simple、组合、TinyLlama draft3，共 75 条正式请求，预热不计入。统计先按每轮总处理时间计算 prefill/decode 吞吐，再汇总五轮均值、样本标准差与范围；TTFT 为每轮三个请求均值。同轮相对基准的比值另行保存。

CMRC 数据文件固定提交 7680ab3a7a1e3ac0d53ad13f7a4d6a3adff1da6a；SHA256 为 5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1。原始数据可从上述仓库的这个提交下载。

```cmd
python scripts/optimize.py server --out results/opt-20260912-confirm --configs bench/optimization-cmrc-confirm.json --repeat 1 --dataset cmrc
python scripts/optimize.py bench --out results/opt-20260912-bench-confirm --configs bench/optimization-bench-confirm.json --repeat 1
```

配置文件包含五轮不同标签，所以这里 repeat=1，不是仅测一次：server 每个标签三条相同问题，bench 每个标签一组 p512/tg256，分别在五个独立进程中复测。GPU 状态由 nvidia-smi 每秒采样，显存是整卡已用量（包含其他进程），不能视为模型独占或瞬时精确峰值。

## 五轮服务端确认结果

所有正式请求的服务端最终计数均为 512 输入 / 256 输出、cache_n=0。下表为五个独立轮次的均值 ± 样本标准差；每轮三条固定真实中文问题。吞吐按总 token / 总时间汇总，避免高速重复样本在算术平均中被放大。

最终审计发现固定版本 server-context.cpp 的 process_token 在 UTF-8 字符未完整时暂缓发送事件，而 send_partial_response 只附带当前 token ID，因此中文流式事件的 ID 总数少于实际生成计数（最终事件 tokens 为空）。汇总已修正为拼接全部 content 比较返回文本，并单列 streamed_token_ids；不把缺失的 ID 补造出来，也不把返回文本一致视为完整 token 序列一致。修正后匹配数量与性能数据不变。长度核验依赖服务端最终计数，未进行独立的完整生成 ID 审计。

| 配置 | Prefill token/s | Decode token/s | TTFT ms | 完整输出匹配原基准 |
| --- | ---: | ---: | ---: | ---: |
| reference | 2037.51 ± 87.78 | 45.52 ± 0.51 | 398.67 ± 13.02 | 15/15 |
| batch512 | 2168.42 ± 133.17 | 45.27 ± 0.67 | 376.72 ± 16.41 | 10/15 |
| ngram | 1944.03 ± 102.81 | 53.93 ± 0.76 | 407.85 ± 16.35 | 10/15 |
| combined | 2146.60 ± 115.99 | 62.91 ± 1.19 | 378.50 ± 17.10 | 10/15 |
| draft3 | 1703.67 ± 66.60 | 66.64 ± 4.60 | 453.32 ± 14.99 | 15/15 |

### 逐轮原始汇总

| 配置 | 轮次 | Prefill token/s | Decode token/s | TTFT ms | 接受/草稿 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 1 | 2150.85 | 46.39 | 398.07 | 0/0 |
| reference | 2 | 2097.50 | 45.47 | 381.67 | 0/0 |
| reference | 3 | 1983.30 | 45.47 | 402.19 | 0/0 |
| reference | 4 | 2024.54 | 45.22 | 393.95 | 0/0 |
| reference | 5 | 1931.37 | 45.07 | 417.48 | 0/0 |
| batch512 | 1 | 2354.73 | 46.17 | 356.42 | 0/0 |
| batch512 | 2 | 2138.28 | 45.04 | 374.48 | 0/0 |
| batch512 | 3 | 2181.50 | 45.66 | 366.14 | 0/0 |
| batch512 | 4 | 2185.83 | 45.06 | 393.74 | 0/0 |
| batch512 | 5 | 1981.75 | 44.41 | 392.80 | 0/0 |
| ngram | 1 | 2069.79 | 54.87 | 401.39 | 143/748 |
| ngram | 2 | 2000.79 | 53.95 | 385.26 | 143/748 |
| ngram | 3 | 1973.15 | 54.39 | 405.45 | 143/748 |
| ngram | 4 | 1837.38 | 53.54 | 420.54 | 143/748 |
| ngram | 5 | 1839.05 | 52.90 | 426.58 | 143/748 |
| combined | 1 | 2294.32 | 63.68 | 355.51 | 232/714 |
| combined | 2 | 2052.95 | 63.59 | 382.08 | 232/714 |
| combined | 3 | 2202.18 | 63.08 | 375.14 | 232/714 |
| combined | 4 | 2175.81 | 63.36 | 376.50 | 232/714 |
| combined | 5 | 2007.74 | 60.82 | 403.26 | 232/714 |
| draft3 | 1 | 1606.12 | 69.99 | 473.89 | 527/708 |
| draft3 | 2 | 1685.90 | 69.83 | 458.80 | 527/708 |
| draft3 | 3 | 1781.64 | 69.15 | 438.51 | 527/708 |
| draft3 | 4 | 1698.57 | 64.91 | 456.74 | 527/708 |
| draft3 | 5 | 1746.13 | 59.29 | 438.65 | 527/708 |

### 轮次范围

| 配置 | Prefill 范围 | Decode 范围 | TTFT 范围 ms |
| --- | ---: | ---: | ---: |
| batch512 | 1981.75–2354.73 | 44.41–46.17 | 356.42–393.74 |
| combined | 2007.74–2294.32 | 60.82–63.68 | 355.51–403.26 |
| draft3 | 1606.12–1781.64 | 59.29–69.99 | 438.51–473.89 |
| ngram | 1837.38–2069.79 | 52.90–54.87 | 385.26–426.58 |
| reference | 1931.37–2150.85 | 45.07–46.39 | 381.67–417.48 |

## 五轮 llama-bench 确认

| 配置 | pp512 均值 ± 标准差 | tg256 均值 ± 标准差 |
| --- | ---: | ---: |
| reference | 1850.01 ± 176.56 | 48.15 ± 0.35 |
| batch512 | 1946.12 ± 187.75 | 48.08 ± 0.34 |

| 配置 | 轮次 | pp512 | tg256 |
| --- | ---: | ---: | ---: |
| reference | 1 | 2090.66 | 48.72 |
| reference | 2 | 1943.80 | 47.87 |
| reference | 3 | 1851.52 | 47.88 |
| reference | 4 | 1709.90 | 48.17 |
| reference | 5 | 1654.17 | 48.09 |
| batch512 | 1 | 2143.84 | 48.54 |
| batch512 | 2 | 2115.09 | 48.24 |
| batch512 | 3 | 1699.85 | 47.69 |
| batch512 | 4 | 1833.65 | 47.82 |
| batch512 | 5 | 1938.17 | 48.11 |

bench 的 prefill 波动较大（基准范围 1654–2091 token/s），batch512 的平均优势约 5.2%，但并非每轮都赢，不能把它描述为已证明稳定的大幅收益。decode 基本持平。服务端真实问题上 prefill/TTFT 的观测优势另按上述配对结果报告，不将两种工具的吞吐混算。

## 结论、限制与停止点

- **首字延迟优先的候选：batch512。** 同轮配对平均 prefill +6.4%、TTFT -5.5%，decode -0.6%（基本持平）。收益幅度小且有波动，不能宣称强稳定优势；确认集 10/15 条与原 batch128 完整输出一致，另外五条均来自同一问题，batch 改变仍需业务质量验收。
- **生成吞吐优先的候选：TinyLlama draft3。** 同轮配对 decode +46.3%、端到端耗时 -28.6%，但 prefill -16.2%、TTFT +13.9%。五轮确认集 15/15 条与基准一致；较广的中文初筛只有 4/6 条一致、英文初筛也有差异，不能据小确认集宣称普遍无损。最终轮次 decode 从约 70 降至 59 token/s，明确存在持续负载波动。
- **n-gram 作为实验选项保留。** 单独 ngram-simple 的配对 decode +18.5%、TTFT +2.3%；batch512 组合 decode +38.2%、TTFT -5.0%。两者确认集均仅 10/15 条完整输出匹配基准；组合与同 batch 的非投机结果也只有 10/15 匹配。未确认差异根因，不启用为默认无损方案。
- 负结果：部分 GPU 卸载使初筛 decode 从约 49.6 降到 22.2 token/s；FA off 降低了初筛 prefill；Q8 KV、4 线程未显示有说服力的优势。ngram-cache 在不重复提示的 WikiText 初筛中慢于基准。以上只作初筛记录，未伪装成五轮确认。
- 英文 WikiText 的高 n-gram 倍率很大程度来自重复标题输出，中文样本也有复制和重复。256-token 固定长度且忽略 EOS 的性能结果，不等于正确答案产出速度；未做正式准确率、PPL 或自然停止条件下的质量验收。
- 确认阶段整卡已用显存采样最大值：reference 6168 MiB、batch512 6245 MiB、ngram 6191 MiB、combined 6249 MiB、draft3 6839 MiB。包含桌面及其他进程，不能视为模型独占显存。采样温度最高 93°C，有明显热/时序漂移；结论仅覆盖本机持续负载，未控制为恒温、恒频实验。
- 已覆盖量化基线、FA、卸载、线程、batch/ubatch、KV 类型、五种 n-gram、短匹配参数、普通草稿 3/5 及 n-gram 组合。EAGLE3/MTP/DFlash/DSpark/DFlash2 的缺失前置条件已记录，不把不可运行的方法填上虚构分数。完成预定方法筛选和五轮确认后停止，本轮不再扩大搜索或训练新草稿头。

## 使用候选配置（CMD）

原 config.local.json 保持不变。以下是可复用候选，不承诺对其他问题、长度或采样参数保持同样收益。模型内存及上下文按当前 8GB GPU 配置。

```cmd
cd /d E:\CodeProjects\boruiLi\llama2-7b
call E:\Software\Anaconda\condabin\conda.bat activate llama2-7b
set "LLAMA_ARG_FIT=off"
python scripts/pipeline.py serve --config config.cuda-latency.json --precision q4 --spec none
```

停止前一个服务后，吞吐优先的草稿候选：

```cmd
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf README.md --revision 52e7645ba7c309695bec7ac98f4f005b139cf465 --local-dir models/draft-tinyllama
python scripts/pipeline.py serve --config config.cuda-draft.json --precision q4 --spec draft-simple
```

正式复测仍使用 optimize.py 的固定 token 数、流式计时和真实数据协议；不要用旧 benchmark.py 的任意长度提示产生可比成绩。

## 交付与复查

- bench/optimization-results-20260912.json：全部 bench 数据、服务端逐样本摘要、返回文本哈希、首个差异字符位置、原始文件 SHA256。
- bench/optimization-confirmation-20260912.json：五轮服务端统计与配对比值。
- bench/optimization-bench-confirmation-20260912.json：五轮 bench 原始值、均值、标准差、范围。
- bench/optimization-inputs-20260912.json：两份目标 GGUF、草稿、两份数据集的完整 SHA256（计时结束后计算）。
- bench/optimization-telemetry-20260912.json：显存、温度和时钟采样摘要。
- 原始 SSE/服务器日志/配置/输入 token ID 位于 results/opt-20260912-*，仍按项目约定留在本地。测量脚本的固定 token 预算和未完成流拒绝测试通过，两个候选配置的参数映射检查通过。
