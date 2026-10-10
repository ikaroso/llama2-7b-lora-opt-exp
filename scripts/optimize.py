"""Reproducible 512/256 local screening. Raw results are never overwritten."""
# 阅读顺序：main() 组织整场实验 → post() 发 HTTP 请求 → stream() 读一次生成结果。
# server 模式：读配置 → 启动 llama-server → 准备真实输入 → 预热 → 正式请求 → 保存记录。
# bench 模式：直接运行 llama-bench，用合成输入测吞吐，不测真实问题的 TTFT。
# 本脚本保存原始证据；Spec-Bench 的五轮汇总和表格由 report_specbench.py 生成。
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

# 路径相对于本脚本所在项目定位，不依赖终端当前目录。
ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'llama.cpp/build-cuda/bin'
ENV = os.environ.copy()
# 给子进程补上 Conda 动态库路径；关闭自动 fit，避免服务自动改动实验配置。
ENV['PATH'] = r'E:\Software\Anaconda\envs\llama2-7b\Library\bin;' + ENV.get('PATH', '')
ENV['LLAMA_ARG_FIT'] = 'off'
Q4 = ROOT / 'artifacts/llama2-7b-lora-q4_k_m.gguf'
BF16 = ROOT / 'artifacts/llama2-7b-lora-bf16.gguf'


def write(path, value):
    # 保存易读的 JSON；ensure_ascii=False 让中文保持原文。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def post(port, route, body):
    # 例如 route='completion'：向 http://127.0.0.1:8093/completion 发请求。
    # body 转为 JSON 字节；Request 提供 data 时默认使用 POST 方法。
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{route}',
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=600)


def stream(port, tokens):
    # 一次完整生成请求。tokens 是已构造好的 512 个 token ID，不是 512 个字符。
    # 禁用跨请求提示缓存；忽略 EOS 是为了固定生成 256 token（不代表自然停止回答）。
    # stream=True 要求服务端边生成边返回，才能在客户端测量首段文本到达时间。
    body = dict(prompt=tokens, n_predict=256, temperature=0, seed=42,
        cache_prompt=False, stream=True, ignore_eos=True, return_tokens=True)
    start = time.perf_counter()  # 客户端计时起点；模型已加载，加载耗时不在这里。
    first = None
    events = []
    final = None
    with post(port, 'completion', body) as response:
        for line in response:
            # SSE 流中的有效消息形如 data: {"content": "...", "stop": false}。
            # 跳过空行等非数据行；一个消息可能含多个 token，不能按消息数统计 token。
            if not line.startswith(b'data: '):
                continue
            data = json.loads(line[6:])
            elapsed = time.perf_counter() - start
            if first is None and data.get('content'):
                # TTFT：从发请求到首个非空文本到达，包含本地 HTTP/流式传输开销。
                first = elapsed
            events.append(dict(seconds=elapsed, data=data))
            if data.get('stop'):
                # 最后一条消息 stop=true，含 llama.cpp 的 timings 和最终生成计数。
                final = data
    if final is None or first is None:
        raise RuntimeError('No completed text stream')
    # 这些是 llama.cpp 自己测的引擎数据，不是 Python 用总耗时推算出来的：
    # prompt_n / prompt_ms：输入 token 数 / prefill 毫秒数；
    # predicted_n / predicted_ms：生成 token 数 / decode 毫秒数；
    # draft_n / draft_n_accepted：提出 / 接受的草稿 token 数（投机时可用）。
    # 汇总脚本按总 token / 总时间计算吞吐；本版本 decode 计数用 predicted_n - 1。
    timings = final.get('timings', {})
    # Fail closed on the experimental token budget; token arrays avoid implicit BOS.
    if timings.get('prompt_n') != 512 or final.get('tokens_predicted') != 256:
        raise RuntimeError(f'Unexpected token counts: {final}')
    # Python 只额外计 TTFT 和请求总耗时（秒）。保留全部 SSE，供输出一致性复核。
    return dict(ttft_seconds=first, wall_seconds=time.perf_counter()-start,
        timings=timings, final=final, events=events)


def main():
    # 1. 读取命令行参数。--configs 指向配置列表；每项启动一个独立模型进程。
    # Spec-Bench 的五轮写在配置 label 中（如 reference-r1 ... reference-r5），
    # 所以该协议使用 --repeat 1；不是整个实验只测一次。
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['bench', 'server'])
    p.add_argument('--out', required=True)
    p.add_argument('--configs', required=True)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--dataset', choices=['wikitext', 'cmrc', 'specbench'], default='wikitext')
    p.add_argument('--prompt-reference', help='Existing prompts JSON; fail before warmup if workload differs')
    a = p.parse_args()
    if a.dataset == 'specbench' and (a.mode != 'server' or a.repeat != 1):
        p.error('Spec-Bench uses server with --repeat 1 and five independent labels per method')
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)  # 目录已存在就报错，防止覆盖历史结果。
    configs = json.loads((ROOT / a.configs).read_text(encoding='utf-8'))
    # 保存协议、源码哈希与 llama.cpp 版本，便于追溯当时用的脚本和参数。
    write(out / 'protocol.json', dict(mode=a.mode, repeat=a.repeat, configs=configs,
        python=sys.version, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        llama_lock=json.loads((ROOT/'llama.lock.json').read_text()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_tokens=512, output_tokens=256, ignore_eos=True,
        ttft='client time to first nonempty text SSE, excludes model loading; one warmup',
        dataset=a.dataset,
        prompt_reference=a.prompt_reference,
        input_builder_sha256=(hashlib.sha256((ROOT/'scripts/specbench_inputs.py').read_bytes()).hexdigest() if a.dataset=='specbench' else None),
        source=({'cmrc':'https://github.com/ymcui/cmrc2018/blob/master/data/cmrc2018_dev.json',
                 'specbench':'https://github.com/hemingkx/Spec-Bench/tree/fd2c1cd7d2201ef71db4c5f4e455008f017967bf',
                 'wikitext':'https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip'}[a.dataset])))
    specbench_bundle = None
    for c in configs:
        # 2. 拼接真正执行的 llama.cpp 命令。c.get(key, default) 表示缺省时用默认值。
        label = c['label']
        print('START', label, flush=True)
        # -m 模型；-t CPU 线程；-ngl GPU 卸载层数；-fa Flash Attention 开关。
        # -b/-ub 是逻辑批次/物理微批次的 token 上限，不是请求并发数。
        base = ['-m', str(BF16 if c.get('bf16') else Q4), '-t', str(c.get('threads', 8)),
            '-ngl', str(c.get('ngl', 99)), '-b', str(c.get('batch', 128)),
            '-ub', str(c.get('ubatch', 128)), '-fa', c.get('fa', 'on')]
        # extra 添加投机参数、草稿模型路径，或 BF16 CPU 的禁止卸载开关。
        base += c.get('extra', [])
        command = [str(BIN / ('llama-bench.exe' if a.mode == 'bench' else 'llama-server.exe'))] + base
        if a.mode == 'bench':
            # bench 直接测 pp512/tg256，JSON 输出与 stderr 分开保存；等退出后再测下一组。
            command += ['-p', '512', '-n', '256', '-r', str(a.repeat), '-o', 'json']
            with (out / (label+'.stdout.json')).open('wb') as stdout, (out / (label+'.stderr.txt')).open('wb') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr)
            write(out / (label+'.meta.json'), dict(command=command, exit_code=result.returncode))
            print('DONE', label, result.returncode, flush=True)
            continue
        # 3. server 模式：监听本机；-np 1 只有一个槽位，本脚本逐个发送请求。
        # --metrics 开放汇总监控接口；逐请求 timings 则来自 /completion 响应。
        port = 8093
        command += ['--host', '127.0.0.1', '--port', str(port), '-c', '2048', '-np', '1', '--metrics', '--no-context-shift']
        with (out / (label+'.server.txt')).open('wb') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            # 另起轻量监控进程，每秒采样整张 GPU 的显存、温度、功耗和频率。
            # 这是整卡状态，包含桌面等其他程序，不是模型进程独占显存。
            gpu_log = (out / (label+'.gpu.csv')).open('wb')
            monitor = subprocess.Popen([r'C:\Windows\System32\nvidia-smi.exe',
                '--query-gpu=timestamp,memory.used,temperature.gpu,power.draw,clocks.sm',
                '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu_log, stderr=subprocess.STDOUT)
            write(out / (label+'.meta.json'), dict(command=command, pid=proc.pid))
            try:
                # 加载模型可能较慢，先轮询 /health；就绪后才准备输入和计时。
                for _ in range(180):
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited {proc.returncode}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                            break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError('Server health timeout')
                corpus_path = ROOT / {'cmrc':'data/cmrc2018_dev.json', 'specbench':'data/spec-bench/question.jsonl',
                                      'wikitext':'data/wikitext-2-raw/wiki.test.raw'}[a.dataset]
                raw = corpus_path.read_bytes()
                def tokenize(text, special=False):
                    # 使用当前目标模型的 tokenizer，而不是另一套分词器估算长度。
                    with post(port,'tokenize',dict(content=text,add_special=special)) as response:
                        return json.load(response)['tokens']
                if a.dataset=='specbench':
                    # 4. Spec-Bench：核验固定数据文件，选摘要/RAG 各六题。
                    # build_inputs 保留指令并裁上下文到 512 token，另选一题预热。
                    # 首个配置构造一次，后续复用相同 token 数组；本项目目标模型词表相同。
                    from specbench_inputs import build_inputs
                    lock = json.loads((ROOT/'bench/specbench-source.json').read_text())
                    if hashlib.sha256(raw).hexdigest() != lock['question_sha256']:
                        raise ValueError('Spec-Bench source checksum mismatch')
                    if specbench_bundle is None:
                        def detokenize(tokens):
                            with post(port,'detokenize',dict(tokens=tokens)) as response:
                                return json.load(response)['content']
                        specbench_bundle = build_inputs(raw, tokenize, detokenize)
                        write(out/'specbench-selection.json', specbench_bundle)
                        write(out/'specbench-source.json', lock)
                    prompts=specbench_bundle['tokens']
                    offsets=specbench_bundle['offsets']
                    warmup=specbench_bundle['warmup_tokens']
                    warmup_offset=specbench_bundle['warmup_offset']
                elif a.dataset=='cmrc':
                    # 旧 CMRC 协议：问题和模板完整保留，用剩余 token 预算截文章。
                    # 多取一题作为独立预热，不能把预热混入正式统计。
                    prompts=[]
                    offsets=[]
                    suffix=tokenize('\n\n### Response:\n')
                    for record in json.loads(raw):
                        qa=record['qas'][0]
                        prefix=tokenize('### Instruction:\n'+qa['query_text']+'\n\n### Input:\n',True)
                        context=tokenize(record['context_text'])
                        budget=512-len(prefix)-len(suffix)
                        if budget<=0 or len(context)<budget:
                            continue
                        prompts.append(prefix+context[:budget]+suffix)
                        offsets.append(dict(query_id=qa['query_id'],context_tokens=budget))
                        if len(prompts)==3*a.repeat+1:
                            break
                    if len(prompts)!=3*a.repeat+1:
                        raise ValueError('Not enough qualifying real prompts')
                    warmup=prompts.pop()
                    warmup_offset=offsets.pop()
                else:
                    # 旧 WikiText 初筛：从真实语料的固定位置截 512 token。
                    all_tokens=tokenize(raw.decode('utf-8'))
                    offsets = [4096*i for i in range(3*a.repeat)]
                    prompts = [all_tokens[offset:offset+512] for offset in offsets]
                    warmup=all_tokens[65536:65536+512]
                    warmup_offset=65536
                if a.prompt_reference:
                    # 跨批次补测时，与历史输入逐项核对；不一致就不进入正式测量。
                    reference = json.loads((ROOT / a.prompt_reference).read_text(encoding='utf-8'))
                    actual = dict(corpus_sha256=hashlib.sha256(raw).hexdigest(), offsets=offsets,
                                  tokens=prompts, warmup_offset=warmup_offset)
                    if any(actual[key] != reference[key] for key in actual):
                        raise ValueError('Workload differs from --prompt-reference')
                write(out / (label+'.prompts.json'), dict(corpus_sha256=hashlib.sha256(raw).hexdigest(), offsets=offsets, tokens=prompts, warmup_offset=warmup_offset))
                # 5. 独立预热只让服务进入运行状态，返回值不写入正式样本。
                stream(port, warmup)  # distinct excluded warmup
                per_round = 12 if a.dataset=='specbench' else 3
                with (out / (label+'.samples.jsonl')).open('x', encoding='utf-8') as dest:
                    # 6. 正式请求逐个运行。JSONL 每行是一条完整请求记录；每条完成即落盘。
                    for rep in range(a.repeat):
                        for i in range(per_round):
                            tokens = prompts[per_round*rep+i]
                            result = stream(port, tokens)
                            dest.write(json.dumps(dict(repeat=rep, prompt=per_round*rep+i, **result), ensure_ascii=False)+'\n')
                            dest.flush()
                            print('SAMPLE', label, rep, i, round(result['ttft_seconds'],4), result['timings'], flush=True)
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics') as response:
                    # 附加保存服务汇总监控数据；吞吐计算仍使用每条响应中的 timings。
                    (out / (label+'.metrics.txt')).write_bytes(response.read())
                write(out / (label+'.done.json'), dict(samples=per_round*a.repeat))
            except Exception as exc:
                # 保留失败原因后继续下一个配置。终端 DONE 只表示结束处理，
                # 成功要看 .done.json；.error.json 会被报告脚本判为失败，不能忽略。
                write(out / (label+'.error.json'), dict(error=str(exc)))
                print('ERROR', label, str(exc)[:300], flush=True)
            finally:
                # 7. 无论成功失败都清理监控和模型进程，避免与下一配置争抢资源。
                monitor.terminate()
                monitor.wait(timeout=10)
                gpu_log.close()
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        print('DONE', label, flush=True)


if __name__ == '__main__':
    main()
