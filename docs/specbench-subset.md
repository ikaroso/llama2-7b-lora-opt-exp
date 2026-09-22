# Spec-Bench 子集固定长度测试

本轮把投机评测从三个 CMRC2018 问题扩展到 Spec-Bench 的摘要与 RAG 两类，每类六个不同样本。原始 BF16 基线定义不变；此次只比较相同 Q4 GPU 目标的无投机、ngram-simple 和 TinyLlama draft3，隔离投机的额外效果。

## 数据与取样

- 上游：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 `fd2c1cd7d2201ef71db4c5f4e455008f017967bf`，文件 `data/spec_bench/question.jsonl`，SHA256 `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`。仓库许可及原评测代码存于本地 `data/spec-bench/`；数据来源、再分发条件遵循各原数据集，本仓库只提交协议及索引。
- 用户已选择保持 512 输入 / 256 输出，因此不使用不足长度的短问答、数学、翻译，不填充重复或无关文本。
- 每类先过滤上下文足够的样本，按 `SHA256(specbench-v1:42:category:question_id)` 排序取前六；取样不查看输出、接受率、速度或参考答案。两类交替排列，另用下一条合格摘要预热。每轮与各方法使用同一组十二条。
- 摘要保留真实 Summarize 指令，RAG 保留原文最后一行的完整问题；使用当前 LoRA 的 Instruction/Input/Response 模板，只取文章/检索上下文的前若干 token，使含 BOS 的总输入为 512。通过运行中的目标 GGUF tokenizer 构造，保存选题依据、原始上下文长度、截断长度、token ID 与可读输入。
- 裁剪检索上下文可能去掉答案证据；这是固定长度性能测量，不用于官方 QA 或摘要质量打分。两类偏长上下文及输入输出重叠，不能推广到六类完整 Spec-Bench 或中文任务。

## 实验与指标

Q4_K_M 目标、ngl=99、FA on、8 线程、b128/ub128、c2048、单并发。ngram-simple 明确 N12/M48/min_hits1；TinyLlama Q4_K_M 草稿最多 3 token、ngld99。每组五个独立 server 进程，三方法顺序跨轮轮换；每进程一条独立预热后运行十二条正式输入，共 180 正式请求。

temperature=0、seed=42、ignore_eos=true、cache_prompt=false，实际计数 512/256。记录完整 SSE、命令、完成状态、GPU 状态。TTFT 为本机请求至首个非空文本事件（排除加载）；prefill 与 decode 分别按 token/对应引擎时间，decode 遵循当前版本首 token 之后的计时。每类及总体按轮汇总，再报告五轮均值、标准差、范围与同轮配对比值。

草稿接受率为 accepted/drafted，缺失草稿计数按未提出草稿单列，不把它称为上游的平均接受长度。返回文本比较基于所有 content 事件拼接，不依赖已知缺失的中文流式 token ID；输出差异和重复现象不删除。输出一致不代表正确，也不证明所有输入均无损。

已复核固定版本源码：ngram-simple 每次在当前传入 token 序列中回扫，不持有跨请求候选缓存（`common/ngram-map.cpp` 的 `common_ngram_simple_draft`，`common/speculative.cpp` 的 ngram-simple 实现）。server 在 cache_prompt=false 时重建当前提示，因此同进程不同题目不会让该方法查询上一题或预热的输出；此结论不适用于 ngram-cache。

与上游的差异：保留本机 llama.cpp 与合并模型，裁剪输入、固定输出、忽略 EOS、使用本项目模板；不使用原平台的模型与自然长度/停止协议。上游完整调用吞吐与本项目独立 decode 指标也不同，因此结果命名为 **Spec-Bench 子集固定长度本机测试**，不宣称官方榜单成绩。原协议见固定版本的 [eval.py](https://github.com/hemingkx/Spec-Bench/blob/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/evaluation/eval.py) 与 [speed.py](https://github.com/hemingkx/Spec-Bench/blob/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/evaluation/speed.py)。

## 执行（CMD）

```cmd
cd /d E:\CodeProjects\boruiLi\llama2-7b
call E:\Software\Anaconda\condabin\conda.bat activate llama2-7b
python scripts/prepare_specbench.py
python scripts/optimize.py server --out results/specbench-20260922 --configs bench/specbench-confirm.json --repeat 1 --dataset specbench
python scripts/report_specbench.py --folder results/specbench-20260922 --output bench/specbench-results-20260922.json --report docs/specbench-results.md
```

结果目录必须为新目录。180 条正式请求与 15 个配置进程均已完成，所有输入一致、计数合格、零提示缓存，无错误记录。最终结果见 [分类与五轮结果](specbench-results.md)，不覆盖旧 CMRC/WikiText 证据。

实际合格样本：摘要 64/80、RAG 80/80。选中的摘要 ID 为 250、253、318、278、266、319；RAG 为 509、559、538、524、488、499；独立预热为摘要255。[可读输入、token与筛选审计](../results/specbench-20260922/specbench-selection.json) 保存了全部入选/未入选依据。目标 Q4 与草稿完整哈希在计时结束后复核，见 [模型指纹](../bench/specbench-models-20260922.json)。

人工抽查可见输出复制、跑题与续写模板，例如问题559询问电影 MLP，目标输出却提到 Super Troopers 2；未因此替换该题或过滤结果。这也说明接受或文本一致仅用于算法行为核对，不是正确率。正式 PPL 与质量验收仍未完成。
