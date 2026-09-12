# 512/256 推理优化实验

## 协议与范围

优化前保存点：baseline-before-optimization-20260912（c605740）。保持原合并模型与 llama.cpp 锁定版本，不更换目标模型。日期：2026-09-12。

- llama-bench：p512、n256，初筛重复 3 次；候选与基准复测。pp512 与 tg256 是工具的独立测试，不能据此计算服务端 TTFT。
- server：真实 WikiText-2 test 文本，经目标 tokenizer 编码，使用 offset 0/4096/8192 起始的三个连续 512-token 片段。不拼造问题，不添加模板或隐式 BOS；发送 token ID 数组保证实际输入长度，日志检查 prompt_n=512。
- 数据来源：https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip 。记录本地语料 SHA256 与具体 token ID；英文文章续写负载，不代表中文问答或代码任务。
- 输出固定 256 token，temperature=0、seed=42、ignore_eos=true、cache_prompt=false，单并发，context=2048。忽略 EOS 是固定长度性能协议，不用于自然停止行为或质量打分。
- 每个 server 配置一次排除的预热，之后相同文本与轮次。TTFT 是本机 HTTP 请求发出至首个非空文本 SSE 的客户端时间；排除模型启动，不代表冷启动。原始 SSE 事件与相对时间全部保存。
- 记录实际输入/输出 token 数，长度不合格立即标记错误，不能混入汇总。记录输出 token 序列用于检查投机结果差异。不同文本分别比较，再汇总，避免只挑最佳案例。
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

过程原始日志存 results/，不纳入 Git；最终可复查的汇总、原始文件哈希与结论将纳入本文件和 bench 下的摘要。当前实验进行中，尚无最终结论。
