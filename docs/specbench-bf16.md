# Spec-Bench：合并 BF16 纯 CPU 基线补测

本次按用户要求，补上同一 Spec-Bench 子集的原始精度基线：Llama 2 7B + `iamshnoo/alpaca-2-7b-chinese` 合并后的 BF16 GGUF，不量化、不投机，纯 CPU 推理。底座为 `modelscope/Llama-2-7b-ms`。BF16、Q4 和草稿完整文件哈希在计时前重新核验，见 [模型指纹](../bench/specbench-bf16-models-20260922.json)。

## 对照与执行条件

- 数据和选题沿用 [Spec-Bench 子集协议](specbench-subset.md)，摘要六题、RAG 六题，每条 512 输入 / 256 输出。启动时逐项核对上一批的 token ID、元数据、语料哈希和独立预热题，任一不同就中止该配置，不进入计时。
- BF16 纯 CPU：`-ngl 0 -fa off --no-kv-offload --no-op-offload`，关闭模型层、KV 与额外算子卸载；复用固定版本的 CUDA 构建，但推理计算在 CPU。线程 8、batch/ubatch 128、context 2048、单并发。
- 同批重新测 Q4 GPU 无投机、ngram-simple（N12/M48/min_hits1）、TinyLlama draft3；仍为 `-ngl 99 -fa on`。目的是提供本批对照，不能把此前批次的结果冒充同轮测量。
- 四配置各五轮，按配置文件轮换执行先后；每配置每轮独立进程和一条排除在统计外的预热，共 240 条正式请求，其中 BF16 CPU 为 60 条。不并行运行模型。
- 温度 0、seed 42、关闭提示缓存、忽略 EOS，最终严格核验输入 512 / 输出 256。Prefill/decode 用每轮总 token 数除以对应总时间；TTFT 为客户端首个非空文本事件时间，排除加载。报告五轮均值、样本标准差、范围、分类及逐题数据。

BF16 CPU 对比 Q4 GPU 的倍率是**整体部署收益**，同时包含精度、后端与 FA 的变化，不是纯量化收益；没有新增 Q4 CPU 组来分离量化影响。五轮顺序轮换不能消除 CPU/GPU 热状态、后台负载和时钟变化，也不代表 BF16 最快的部分 GPU 卸载方案。输出一致与性能不等于质量通过，PPL 仍按原状态记录。

## 复现（CMD）

```cmd
cd /d "E:\CodeProjects\boruiLi\llama2-7b"
call "E:\Software\Anaconda\condabin\conda.bat" activate llama2-7b
python scripts/optimize.py server --out results/specbench-bf16-20260922 --configs bench/specbench-bf16-confirm.json --repeat 1 --dataset specbench --prompt-reference results/specbench-20260922/reference-r1.prompts.json
python scripts/report_specbench.py --folder results/specbench-bf16-20260922 --output bench/specbench-bf16-results-20260922.json --report docs/specbench-bf16-results.md --include-bf16
```

运行目录必须为新目录，不能覆盖。输入参考文件位于本地原始记录；在其他机器复现时，先按子集协议构建同一输入并核验来源。原始请求、命令、完成/错误状态与 GPU 整卡采样保存在 `results/specbench-bf16-20260922/`，不提交模型或原始大文件。

## 状态

2026-09-22 15:55（本地时间）：四配置各五轮全部完成，无错误记录。2026-09-23 汇总校验确认 240 条正式请求均为 512/256、零提示缓存；20 份输入文件与原批次完全一致，BF16 每轮命令均满足纯 CPU 参数约束。相关代码的 15 项测试已在运行期间通过。

结果见 [完整对比表](specbench-bf16-results.md) 与 [结构化结果](../bench/specbench-bf16-results-20260922.json)。BF16 CPU 五轮平均 prefill 23.02 token/s、decode 4.28 token/s、TTFT 22.38 秒；同批 Q4 GPU 无投机 decode 为 50.27 token/s，按同轮比值汇总达到 BF16 CPU 的 11.76 倍。全部原始文件哈希、分类/逐轮/逐题结果已留档；没有重跑或覆盖此前批次。
