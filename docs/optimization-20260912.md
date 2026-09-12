# 512/256 推理优化实验

## 协议与范围

优化前保存点：baseline-before-optimization-20260912（c605740）。保持原合并模型与 llama.cpp 锁定版本，不更换目标模型。日期：2026-09-12。

- llama-bench：p512、n256，初筛重复 3 次；候选与基准复测。pp512 与 tg256 是工具的独立测试，不能据此计算服务端 TTFT。
- server：真实 WikiText-2 test 文本，经目标 tokenizer 编码，按 offset=4096*i 选择连续 512-token 片段，每轮三个不同片段。不重复同一提示，避免 ngram-cache 从前轮生成直接获益；预热使用独立 offset=65536。每个配置使用相同片段顺序。不拼造问题，不添加模板或隐式 BOS；发送 token ID 数组保证实际输入长度，日志检查 prompt_n=512。
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

## 初筛发现与协议补充

WikiText 多个输入诱发目标模型重复生成标题，导致 n-gram 获得很高吞吐。保留这一负载结果，但不作为正常问答收益的依据。补充 CMRC2018 官方开发集的真实问题与上下文（https://github.com/ymcui/cmrc2018 ，CC-BY-SA-4.0，Cui et al., EMNLP-IJCNLP 2019）。顺序选择不同 context 的首个问题，用 Alpaca 的 Instruction/Input/Response 模板，保留完整问题与响应起始标记，仅截断原始上下文到总输入 512 token；不使用数据集答案作为提示，不按模型表现挑选样本。它是固定长度性能测试，不是官方 CMRC 准确率测评。

用户追加要求：最终确认至少五轮。最终阶段对相同三条真实输入进行五轮独立复测，每个配置/轮次重启 server，采用交错顺序，防止跨轮草稿缓存；保留每轮数据，报告均值、标准差和范围。初筛结果不能冒充五次确认。

CMRC 初筛中也发现复制上下文和重复词语，说明当前模型的固定 256-token 输出存在质量限制，不能把这些测试解释成优质回答的产出速度。无投机前后基准六个样本逐 token 一致；n-gram、普通草稿和 batch 改变在部分样本中出现输出差异，需单列记录。批量验证和单 token 解码的数值路径差异是可能原因，尚未定位到逐层数值证据，不宣称已证明无损。

最终确认使用 CMRC 开发集前三个满足长度要求的 context 的首个问题（DEV_0_QUERY_0、DEV_1_QUERY_0、DEV_2_QUERY_0），不根据输出优劣筛样本。五轮分别启动原基准、batch512、ngram-simple、组合、TinyLlama draft3，共 75 条正式请求，预热不计入。统计先按每轮总处理时间计算 prefill/decode 吞吐，再汇总五轮均值、样本标准差与范围；TTFT 为每轮三个请求均值。同轮相对基准的比值另行保存。

```cmd
python scripts/optimize.py server --out results/opt-20260912-confirm --configs bench/optimization-cmrc-confirm.json --repeat 1 --dataset cmrc
python scripts/optimize.py bench --out results/opt-20260912-bench-confirm --configs bench/optimization-bench-confirm.json --repeat 1
```

配置文件包含五轮不同标签，所以这里 repeat=1，不是仅测一次：server 每个标签三条相同问题，bench 每个标签一组 p512/tg256，分别在五个独立进程中复测。GPU 状态由 nvidia-smi 每秒采样，显存是整卡已用量（包含其他进程），不能视为模型独占或瞬时精确峰值。
