# 完整执行指令：BF16、Q4、吞吐与 PPL

实验结论统一见 [推理优化总报告](final-report.md)；本文件负责执行步骤，不代表其中所有质量评估步骤已完成。

原始 BF16 基线及同运行条件 Q4 的五轮性能补测命令见 [BF16 基线补测](bf16-baseline-20260913.md)。

2026-09-12 起，正式优化实验统一输入 512 / 输出 256，使用真实来源文本并至少五轮确认。下面旧冒烟步骤用于验证可运行，不作为正式优化成绩；正式测量入口及结果见 [512/256 优化实验](optimization-20260912.md)。

适用于本机 Windows，源码固定为 `b31b71f3a076bfc4278daad442203a9c51c6e676`。从项目根目录执行；每条命令成功后再继续。模型下载与真实测试由下面的命令触发，目前不代表已经执行成功。

## 0. 进入专用环境

全文统一使用 **CMD（命令提示符）**，打开 cmd.exe 后逐行执行。构建脚本会自动加载 VS 2022 的 MSVC。

```cmd
cd /d E:\CodeProjects\boruiLi\llama2-7b
call E:\Software\Anaconda\condabin\conda.bat activate llama2-7b
chcp 65001 >nul
set "PATH=%SystemRoot%\System32;%PATH%"
set "BIN=%CD%\llama.cpp\build-cuda\bin"
set "BF16=%CD%\artifacts\llama2-7b-lora-bf16.gguf"
set "Q4=%CD%\artifacts\llama2-7b-lora-q4_k_m.gguf"
set "CORPUS=%CD%\data\wikitext-2-raw\wiki.test.raw"
set "LLAMA_ARG_FIT=off"
set "RUN=%RANDOM%-%RANDOM%"
python --version
python -m pip check
python -c "import torch,transformers,peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

预期 Python 3.11、Torch 2.11 CPU、Transformers 4.57.6、PEFT 0.18.0。CPU torch 的 CUDA False 正常，llama.cpp GPU 后端不依赖它。

重建环境时才执行：

```cmd
conda env create -f environment.yml
conda activate llama2-7b
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-merge.txt
python -m pip freeze > environment.lock.txt
```

## 1. 编译 llama.cpp

专用 Conda 环境已安装 CUDA Toolkit 12.9 和 Ninja。激活后运行 `nvcc --version`。下面的脚本会先激活环境，再明确初始化本机 VS 2022，以 Ninja 编译；无需系统级 CUDA 安装或 VS CUDA 集成。先检查主机内存和磁盘，建议 RAM 32GB、空闲磁盘 60GB 起。

```cmd
git -C llama.cpp rev-parse HEAD
nvcc --version
cmd /c scripts\build_cuda.cmd
```

仅检查配置而不完整编译时使用 `cmd /c scripts\build_cuda.cmd configure`。脚本固定本机 RTX 4060 的 CUDA 架构 89，产物目录 build-cuda；与旧的 Visual Studio build 目录分开。Conda 附带的 VS 2017 探测可能输出提示，脚本随后显式使用 VS 2022。

```cmd
set "BIN=%CD%\llama.cpp\build-cuda\bin"
set "LLAMA_ARG_FIT=off"
"%BIN%\llama-completion.exe" --help
"%BIN%\llama-bench.exe" --help
"%BIN%\llama-perplexity.exe" --help
```

保持该终端打开；新开窗口时先执行第 0 节恢复环境和变量。当前版本使用 llama-completion 做非交互文本生成。项目 bootstrap 也已指向 llama.cpp 并包含上述目标；此处显式构建便于看清参数。

关闭自动 fit 可避免程序为了适应显存偷偷调整上下文或卸载设置。冒烟/PPL 的 CPU 基线需要完全禁用 GPU 运算时，可统一添加 `--device none --no-kv-offload`；bench 对应使用 `-nkvo 1` 并核对实际后端。

## 2. 下载基座和 LoRA

基座改从 [ModelScope 组织仓库](https://modelscope.cn/models/modelscope/Llama-2-7b-ms) 下载，LoRA 仍从 Hugging Face 的公开仓库下载。遵守仓库附带的 Llama 2 许可；不再需要 Hugging Face 官方基座的访问申请。

```cmd
python scripts/download_models.py
```

脚本按 modelscope.lock.json 中每个文件的固定提交及 SHA256 下载约 13.48 GB safetensors 权重和必要配置，避免重复 bin 权重。排除魔搭额外的 added_tokens.json 和 tokenizer.json，以 tokenizer.model 加载的词表已验证为 32000。LoRA 固定为 iamshnoo/alpaca-2-7b-chinese 的 `48951161ce47ed33d8bde26d4302d7bca8ef21e2`。下载后写入 models/sources.json；重试会跳过已校验完整的文件，未完成的单个文件会重新下载。已有其他来源的 sources.json 或多余基座文件时会报错，请将旧文件移到备份目录后重试，不要混用。

已核对架构及 tokenizer；官方 HF 接口隐藏权重哈希，尚未证明镜像权重与官方逐字节相同，也尚未完成真实 7B 合并验证。

## 3. 合并 BF16，重新加载，再转 GGUF

```cmd
if not exist config.local.json copy config.example.json config.local.json
python scripts/pipeline.py merge
python -c "from transformers import AutoModelForCausalLM; import torch; m=AutoModelForCausalLM.from_pretrained('models/merged-bf16',torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True); print(type(m).__name__,m.dtype,m.config.vocab_size)"
python scripts/pipeline.py convert
set "BF16=%CD%\artifacts\llama2-7b-lora-bf16.gguf"
set "Q4=%CD%\artifacts\llama2-7b-lora-q4_k_m.gguf"
```

重新加载进程结束后会释放内存，再进行下一步。合并输出已存在时脚本会拒绝覆盖；中断后改用新的输出目录并核对配置。合并前会重新核对基座文件与 Git 中锁定的 SHA256，读取约 13.48 GB 文件需要一些时间。

## 4. BF16 冒烟测试

8GB GPU 放不下完整 BF16。先用 CPU（ngl=0）建立可复现基线；输出测试与吞吐、PPL 分别运行，避免多个模型进程抢资源。

```cmd
python scripts/run_logged.py results/bf16-smoke-%RUN% -- "%BIN%\llama-completion.exe" -m "%BF16%" -f bench/smoke.txt -n 128 -c 2048 -t 8 -ngl 0 -b 128 -ub 128 -fa off --temp 0 --seed 42 --no-conversation --no-display-prompt
type results\bf16-smoke-%RUN%.stdout.txt
type results\bf16-smoke-%RUN%.stderr.txt
```

通过标准：退出码 0、输出非空、没有乱码/NaN/崩溃，人工确认中文回答基本切题。示例 prompt 是 Alpaca 风格冒烟输入，不能代替正式中文质量评测。

## 5. BF16 llama-bench

```cmd
python scripts/run_logged.py results/bf16-bench-%RUN% -- "%BIN%\llama-bench.exe" -m "%BF16%" -p 512 -n 128 -r 5 -t 8 -ngl 0 -b 128 -ub 128 -fa off -o json
type results\bf16-bench-%RUN%.stdout.txt
```

pp512 表示 prompt processing 吞吐，tg128 表示 token generation 吞吐，单位 token/s。bench 使用合成输入，不代表业务端到端延迟或 TTFT；也不测试投机解码。保留默认预热，重复 5 次记录均值和标准差。

## 6. 本地 Q4_K_M 量化

```cmd
python scripts/pipeline.py quantize
dir "%BF16%" "%Q4%"
certutil -hashfile "%BF16%" SHA256
certutil -hashfile "%Q4%" SHA256
```

底层相当于 `llama-quantize.exe BF16.gguf Q4.gguf Q4_K_M`。CPU 可执行，不需要 GPU 容纳 BF16。

## 7. Q4 冒烟测试和 llama-bench

先保持与 BF16 相同的 CPU 配置，比较量化本身产生的差异：

```cmd
python scripts/run_logged.py results/q4-smoke-%RUN% -- "%BIN%\llama-completion.exe" -m "%Q4%" -f bench/smoke.txt -n 128 -c 2048 -t 8 -ngl 0 -b 128 -ub 128 -fa off --temp 0 --seed 42 --no-conversation --no-display-prompt
type results\q4-smoke-%RUN%.stdout.txt
python scripts/run_logged.py results/q4-bench-%RUN% -- "%BIN%\llama-bench.exe" -m "%Q4%" -p 512 -n 128 -r 5 -t 8 -ngl 0 -b 128 -ub 128 -fa off -o json
type results\q4-bench-%RUN%.stdout.txt
```

分别计算 `Q4 avg_ts / BF16 avg_ts`，pp 和 tg 分开比较；固定采样仍可能因量化改变输出，不要求逐字一致。

CUDA 编译完成后再测实际 GPU 部署，先释放显存（建议空闲约 6GiB）：

```cmd
nvidia-smi
python scripts/run_logged.py results/q4-cuda-smoke-%RUN% -- "%BIN%\llama-completion.exe" -m "%Q4%" -f bench/smoke.txt -n 128 -c 2048 -t 8 -ngl 99 -b 128 -ub 128 -fa on --temp 0 --seed 42 --no-conversation --no-display-prompt
python scripts/run_logged.py results/q4-cuda-bench-%RUN% -- "%BIN%\llama-bench.exe" -m "%Q4%" -p 512 -n 128 -r 5 -t 8 -ngl 99 -b 128 -ub 128 -fa on -o json
type results\q4-cuda-smoke-%RUN%.stderr.txt
```

检查 CUDA 加载、实际卸载层数和 KV 位置。OOM 时先释放其他进程显存，再降低上下文或卸载层数，并以新结果名记录参数。此结果包含 GPU/Flash Attention 收益，不能称为纯量化收益。

## 8. 固定 PPL 测试文本

使用该源码自带下载脚本对应的 WikiText-2 raw 测试集，以下是 Windows 等价步骤：

```cmd
if not exist data mkdir data
if not exist results mkdir results
hf download ggml-org/ci wikitext-2-raw-v1.zip --repo-type dataset --local-dir data
python -m zipfile -e data/wikitext-2-raw-v1.zip data
set "CORPUS=%CD%\data\wikitext-2-raw\wiki.test.raw"
certutil -hashfile "%CORPUS%" SHA256 > results\ppl-corpus-sha256-%RUN%.txt
```

固定同一个文件、上下文、KV 类型与后端；不要套聊天模板，也不要对两边使用不同的文本截断。此英文语料用于量化退化检查，不能证明中文指令质量。

## 9. BF16 与 Q4 的 PPL

先跑 4 chunks 验证流程，再跑全量；CPU 计算可能耗时较长。FP16 KV 显式固定。

```cmd
python scripts/run_logged.py results/bf16-ppl-check-%RUN% -- "%BIN%\llama-perplexity.exe" -m "%BF16%" -f "%CORPUS%" -c 2048 -b 128 -ub 128 -t 8 -ngl 0 -fa off -ctk f16 -ctv f16 --chunks 4
python scripts/run_logged.py results/q4-ppl-check-%RUN% -- "%BIN%\llama-perplexity.exe" -m "%Q4%" -f "%CORPUS%" -c 2048 -b 128 -ub 128 -t 8 -ngl 0 -fa off -ctk f16 -ctv f16 --chunks 4
python scripts/compare_ppl.py results/bf16-ppl-check-%RUN% results/q4-ppl-check-%RUN%

python scripts/run_logged.py results/bf16-ppl-full-%RUN% -- "%BIN%\llama-perplexity.exe" -m "%BF16%" -f "%CORPUS%" -c 2048 -b 128 -ub 128 -t 8 -ngl 0 -fa off -ctk f16 -ctv f16 --chunks -1
python scripts/run_logged.py results/q4-ppl-full-%RUN% -- "%BIN%\llama-perplexity.exe" -m "%Q4%" -f "%CORPUS%" -c 2048 -b 128 -ub 128 -t 8 -ngl 0 -fa off -ctk f16 -ctv f16 --chunks -1
python scripts/compare_ppl.py results/bf16-ppl-full-%RUN% results/q4-ppl-full-%RUN%
```

比较同一个已合并模型的 BF16 与 Q4。PPL 越低越好；delta=Q4-BF16，relative=(Q4/BF16-1)*100%。报告两个最终 PPL、变化和日志里的不确定度，不将少量 chunks 的结果当成全量结果。

## 10. 结果留档

run_logged 为每次执行保存 stdout、stderr、命令与退出码，结果名存在时拒绝覆盖。每轮开始执行 `set "RUN=%RANDOM%-%RANDOM%"` 生成新后缀；同一轮 BF16/Q4 与 PPL 对比沿用同一个 RUN。单独重跑某项时生成新 RUN，并记下需要对比的旧文件名。日志执行期间不会实时显示，可用 `type results\具体文件名.stderr.txt` 查看已有内容。

保留 environment.lock.txt、llama.lock.json、models/sources.json、模型哈希、语料哈希、命令日志、驱动版本。models/data/results 被 Git 忽略，另行备份实际产物。投机解码通过 llama-server 和 benchmark.py 测试，具体见 [投机解码方案](speculative-plan.md)；llama-bench 的数字不能代表投机加速。
