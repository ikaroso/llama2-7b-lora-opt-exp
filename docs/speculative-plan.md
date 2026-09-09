# 投机解码实验方案

## 适用对象与选择

目标为当前 Base+中文 LoRA 合并模型，优先使用 Q4_K_M、RTX 4060 Laptop 8GB、固定 2048 上下文。目标模型必须验证候选；量化误差仍以 BF16/Q4 PPL 单独评价。以下为方案，尚未执行投机测试。

| 方法 | 当前结论 | 前置条件 |
| --- | --- | --- |
| ngram-simple / ngram-map-k | 第一优先级，无需额外权重 | 验证普通中文与重复/复制负载，可能无收益或变慢 |
| draft-simple | 第二阶段 | 先筛选 tokenizer、token ID、BOS/EOS 兼容的小草稿模型，再测接受率和显存 |
| EAGLE3 | 暂不列入首轮 | 需要匹配草稿网络；Llama2-Chat 的旧 EAGLE 权重不等于当前目标的 EAGLE3 权重 |
| MTP | 当前模型不能直接开启 | Llama 2 没有原生 MTP 层，LoRA 合并不会新增 |
| DFlash / DSpark / DFlash2 | 后续研究分支 | 需要对应草稿权重和目标特征接口适配，尚未核实当前合并模型的现成权重 |

锁定源码包含 draft-eagle3、draft-mtp、draft-dflash、draft-dspark；DFlash2 在 DFlash 路径内根据模型元数据识别。引擎存在实现不代表本目标已适配。方案依据：[本地实现](../llama.cpp/common/speculative.cpp)、[EAGLE](https://github.com/SafeAILab/EAGLE)、[DFlash](https://github.com/z-lab/dflash)、[DeepSpec](https://github.com/deepseek-ai/DeepSpec)。

## 首轮实验

1. 先完成 BF16/Q4 PPL，确认量化质量变化。
2. 固定同一 Q4、GPU 卸载层数、FA、KV 类型、上下文、8 线程、单并发、贪心采样和生成长度，比较 none、ngram-simple、ngram-map-k。
3. 准备两组相同格式的 prompts：普通中文问答；重复文本/复制编辑。分别汇总，避免用重复任务收益代表普通聊天。保留语料哈希。
4. 每组预热后至少重复 3 次，记录端到端延迟、生成 token 数、生成速度、草稿/接受 token 数与接受率、峰值显存。接受率无法从响应获得时，从 server 指标或日志读取，注明统计口径。
5. 先用默认 n-gram 参数，再独立扫描其 lookup/draft 长度。普通草稿模型另扫 draft_tokens=3/5/8；不要把 draft_tokens 当成所有 n-gram 变体的长度参数。

## 执行入口（CMD）

按 runbook 第 0 节初始化环境。config.local.json 设置 backend=cuda、gpu_layers=99、flash_attention=on、context=2048、threads=8；这些是实验设置，运行前核查显存。

```cmd
python scripts/pipeline.py serve --precision q4 --spec none
```

另一个 CMD 激活同一环境，从项目目录测量：

```cmd
set "RUN=%RANDOM%-%RANDOM%"
python scripts/benchmark.py --label q4-none --output results/q4-none-%RUN%.jsonl --repeat 3 --tokens 256
```

停止服务，再分别使用 `--spec ngram-simple` 和 `--spec ngram-map-k` 启动；相应修改 benchmark 的 label 和输出名，使用相同 prompts。保存完整 server 启动命令与日志。现有 benchmark 是非流式请求，只能测端到端延迟和响应 timings，不能报告 TTFT；峰值显存和接受率需另行采集。

普通草稿模型准备好后使用 `--spec draft-simple`，并在配置填写 draft_gguf、draft_gpu_layers。旧 `--spec draft` 保留为项目别名，传给 llama.cpp 时映射为 draft-simple。

## 验收

必须有同后端、同负载的 server 无投机基线；现有 llama-bench 55.09 token/s 只能作为参考，不作为跨工具加速比分母。贪心结果核对 token 序列；不一致时保留证据并调查数值路径、终止条件或实现问题。接受率高不等于吞吐一定高；负收益同样留档。BF16+投机可作为后续扩展，不与 GPU Q4 混算算法收益。
