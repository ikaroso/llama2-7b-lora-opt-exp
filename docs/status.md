# 项目进度

## 优化完成（2026-09-12）

已完成统一 512/256 的初筛及五轮确认，详见 [完整实验报告](optimization-20260912.md)。原模型与本地配置保留。五轮真实中文问题结果（每轮三条，非空文本流式 TTFT）：基准 prefill 2037.51 token/s、decode 45.53 token/s、TTFT 398.67 ms；batch512 为 2168.42 / 45.27 / 376.72；TinyLlama draft3 为 1703.67 / 66.63 / 453.32。

draft3 提高生成吞吐但增加首字延迟。n-gram 及其组合已实测，部分输出与基准不一致，暂不宣称无损默认优化。batch 的收益较小，bench prefill 波动较大；重复输出、忽略 EOS、三问题样本规模及温度漂移限制均已记录。MTP 和专门草稿头方法缺少适配前置条件，没有虚构实测。PPL/正式准确率仍未完成，本轮未用性能结果代替质量验收。

新增 config.cuda-latency.json、config.cuda-draft.json 两个可选配置及测量/汇总脚本，bench 下保存可提交的逐轮结果和哈希索引。优化前 Git 标签为 baseline-before-optimization-20260912。

## 当前实测进度（2026-09-09）

已存在底座、LoRA、合并后 BF16 checkpoint（config 标注 bfloat16），以及 BF16/Q4_K_M GGUF。CUDA 构建已可执行，bench 记录的源码版本为 b31b71f3a。此次按已有产物和日志核验，没有重新执行下载、合并或全文件哈希检查；镜像与官方权重逐字节一致性仍未证明。

| 产物 | 文件字节数 |
| --- | ---: |
| artifacts/llama2-7b-lora-bf16.gguf | 13478105216 |
| artifacts/llama2-7b-lora-q4_k_m.gguf | 4081004672 |

Q4 文件约缩小 70%。下表为 llama-bench 的 5 次平均吞吐（token/s），固定 p512、n128、t8、b128、ub128。

| 配置 | pp512 | tg128 | 日志前缀（results/ 下） |
| --- | ---: | ---: | --- |
| BF16，ngl=0，FA off | 97.262 | 4.754 | bf16-bench |
| Q4_K_M，ngl=0，FA off | 222.289 | 13.440 | q4-bench |
| Q4_K_M，ngl=99，FA on | 2421.095 | 55.086 | q4-cuda-bench-13745-2315 |

三组 bench 的 meta.json 均记录 exit_code=0。BF16/Q4 的 ngl=0 生成吞吐比约 2.83；Q4 CUDA+FA 相对 Q4 ngl=0 约 4.10，后者包含后端和 FA 变化，不算纯量化收益。ngl=0 记录仍允许默认 KV/操作卸载，不将它宣称为严格禁用所有 GPU 运算的实验。

冒烟日志 bf16-smoke、q4-smoke、q4-cuda-smoke-13745-2315 均有 exit_code=0。中文可读，但二分查找解释不够准确，128 token 上限造成截断，尚不构成质量验收。旧 q4-cuda-bench 没有完成元数据，不纳入结果。

WikiText-2 raw 测试文本已存在（1290590 字节），语料哈希记录为 results/ppl-corpus-sha256-13745-2315.txt。尚未发现 BF16/Q4 PPL 完成结果或投机解码结果。下一步完成 PPL，再执行 [投机解码方案](speculative-plan.md)。runbook 已统一 CMD，协作约定见根目录 AGENTS.md。

以下均为早期历史快照，其中“尚未执行”不代表当前状态。

## ModelScope 下载源更新

官方 HF 基座申请被拒，下载入口已切换到 modelscope/Llama-2-7b-ms，保留 iamshnoo/alpaca-2-7b-chinese LoRA。modelscope.lock.json 固定文件提交、大小和哈希；合并前重新校验基座哈希。实测排除 added_tokens.json 后 tokenizer 词表为 32000，配置为 LlamaForCausalLM、4096 hidden、32 layers。只下载了小型配置和 tokenizer 进行验证，完整权重下载、真实合并和推理尚未执行。官方接口隐藏权重哈希，未证明镜像与官方逐字节一致。后续命令见 runbook.md 第 2 节。

## 最新更新：专用环境与执行手册

已创建 Conda `llama2-7b`，位于 `E:\Software\Anaconda\envs\llama2-7b`。已验证 Python 3.11、torch 2.11.0+cpu、transformers 4.57.6、peft 0.18.0；pip check 通过。随机小型 Llama 的 CPU BF16 前向、非零 LoRA 合并、safetensors 保存与重新加载检查通过。实际转换器 --help 已运行。

项目源码路径已改为 `llama.cpp`，锁定 `b31b71f3a076bfc4278daad442203a9c51c6e676`；构建目标包含 server、completion、quantize、bench、perplexity。完整命令见 [运行手册](runbook.md)。实际 7B 下载/合并/转换/推理未执行。下面及决策记录中的旧环境状态作为历史快照保留。

专用 Conda 环境已安装 CUDA Toolkit 12.9.2（nvcc 12.9.86）及 Ninja 1.13.2。`nvcc --version`、pip check 通过；`scripts/build_cuda.cmd configure` 已使用 VS2022 MSVC 19.44 完成 CUDA 编译器 ABI 检查和 CMake 配置，目标架构为 RTX 4060 的 89。完整 llama.cpp 构建尚未执行。构建脚本在 Conda 激活后显式加载 VS2022，产物目录为 `llama.cpp/build-cuda/bin`。Conda 包精确版本记录在 `conda-win64.lock.txt`（不包含 pip 包，pip 版本另见 `environment.lock.txt`）。

以下为首次建项快照。2026-09-09 后续已确定模型与 LoRA，并完成 PowerShell 7 用户级 Python 修复；最新环境、选型和待验收状态见 [决策记录](decision-2026-09-09.md)。

2026-09-09，本地 Windows / Python 3.11.10 / Git 2.48.1 / CMake 4.1.0。

- 检测到 RTX 4060 系列 GPU，8188 MiB 总显存；检查时已使用约 2726 MiB。
- 未在 PATH 检测到 nvcc；主机 RAM 查询被操作系统拒绝，尚未确认。
- 终端 git ls-remote 无法连接 GitHub 443；未拉取或构建 llama.cpp，尚未生成源码锁。
- 尚未提供基座、LoRA、训练 tokenizer、草稿模型和质量评测集。
- 已实现配置、转换/量化/启动脚本、原始基准记录、设计及运行说明。
- 没有真实推理成绩，也没有量化产物。后续需安装依赖并执行 README 步骤。
