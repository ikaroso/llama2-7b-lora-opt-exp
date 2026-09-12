"""Reproducible 512/256 local screening. Raw results are never overwritten."""
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

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'llama.cpp/build-cuda/bin'
ENV = os.environ.copy()
ENV['PATH'] = r'E:\Software\Anaconda\envs\llama2-7b\Library\bin;' + ENV.get('PATH', '')
ENV['LLAMA_ARG_FIT'] = 'off'
Q4 = ROOT / 'artifacts/llama2-7b-lora-q4_k_m.gguf'
BF16 = ROOT / 'artifacts/llama2-7b-lora-bf16.gguf'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def post(port, route, body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{route}',
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=600)


def stream(port, tokens):
    body = dict(prompt=tokens, n_predict=256, temperature=0, seed=42,
        cache_prompt=False, stream=True, ignore_eos=True, return_tokens=True)
    start = time.perf_counter()
    first = None
    events = []
    final = None
    with post(port, 'completion', body) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = json.loads(line[6:])
            elapsed = time.perf_counter() - start
            if first is None and data.get('content'):
                first = elapsed
            events.append(dict(seconds=elapsed, data=data))
            if data.get('stop'):
                final = data
    if final is None or first is None:
        raise RuntimeError('No completed text stream')
    timings = final.get('timings', {})
    # Fail closed on the experimental token budget; token arrays avoid implicit BOS.
    if timings.get('prompt_n') != 512 or final.get('tokens_predicted') != 256:
        raise RuntimeError(f'Unexpected token counts: {final}')
    return dict(ttft_seconds=first, wall_seconds=time.perf_counter()-start,
        timings=timings, final=final, events=events)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['bench', 'server'])
    p.add_argument('--out', required=True)
    p.add_argument('--configs', required=True)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--dataset', choices=['wikitext', 'cmrc'], default='wikitext')
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    configs = json.loads((ROOT / a.configs).read_text(encoding='utf-8'))
    write(out / 'protocol.json', dict(mode=a.mode, repeat=a.repeat, configs=configs,
        python=sys.version, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        llama_lock=json.loads((ROOT/'llama.lock.json').read_text()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_tokens=512, output_tokens=256, ignore_eos=True,
        ttft='client time to first nonempty text SSE, excludes model loading; one warmup',
        dataset=a.dataset,
        source=('https://github.com/ymcui/cmrc2018/blob/master/data/cmrc2018_dev.json' if a.dataset=='cmrc' else 'https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip')))
    for c in configs:
        label = c['label']
        print('START', label, flush=True)
        base = ['-m', str(BF16 if c.get('bf16') else Q4), '-t', str(c.get('threads', 8)),
            '-ngl', str(c.get('ngl', 99)), '-b', str(c.get('batch', 128)),
            '-ub', str(c.get('ubatch', 128)), '-fa', c.get('fa', 'on')]
        base += c.get('extra', [])
        command = [str(BIN / ('llama-bench.exe' if a.mode == 'bench' else 'llama-server.exe'))] + base
        if a.mode == 'bench':
            command += ['-p', '512', '-n', '256', '-r', str(a.repeat), '-o', 'json']
            with (out / (label+'.stdout.json')).open('wb') as stdout, (out / (label+'.stderr.txt')).open('wb') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr)
            write(out / (label+'.meta.json'), dict(command=command, exit_code=result.returncode))
            print('DONE', label, result.returncode, flush=True)
            continue
        port = 8093
        command += ['--host', '127.0.0.1', '--port', str(port), '-c', '2048', '-np', '1', '--metrics', '--no-context-shift']
        with (out / (label+'.server.txt')).open('wb') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            gpu_log = (out / (label+'.gpu.csv')).open('wb')
            monitor = subprocess.Popen([r'C:\Windows\System32\nvidia-smi.exe',
                '--query-gpu=timestamp,memory.used,temperature.gpu,power.draw,clocks.sm',
                '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu_log, stderr=subprocess.STDOUT)
            write(out / (label+'.meta.json'), dict(command=command, pid=proc.pid))
            try:
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
                corpus_path = ROOT / ('data/cmrc2018_dev.json' if a.dataset=='cmrc' else 'data/wikitext-2-raw/wiki.test.raw')
                raw = corpus_path.read_bytes()
                def tokenize(text, special=False):
                    with post(port,'tokenize',dict(content=text,add_special=special)) as response:
                        return json.load(response)['tokens']
                if a.dataset=='cmrc':
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
                    all_tokens=tokenize(raw.decode('utf-8'))
                    offsets = [4096*i for i in range(3*a.repeat)]
                    prompts = [all_tokens[offset:offset+512] for offset in offsets]
                    warmup=all_tokens[65536:65536+512]
                    warmup_offset=65536
                write(out / (label+'.prompts.json'), dict(corpus_sha256=hashlib.sha256(raw).hexdigest(), offsets=offsets, tokens=prompts, warmup_offset=warmup_offset))
                stream(port, warmup)  # distinct excluded warmup
                with (out / (label+'.samples.jsonl')).open('x', encoding='utf-8') as dest:
                    for rep in range(a.repeat):
                        for i in range(3):
                            tokens = prompts[3*rep+i]
                            result = stream(port, tokens)
                            dest.write(json.dumps(dict(repeat=rep, prompt=3*rep+i, **result), ensure_ascii=False)+'\n')
                            dest.flush()
                            print('SAMPLE', label, rep, i, round(result['ttft_seconds'],4), result['timings'], flush=True)
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics') as response:
                    (out / (label+'.metrics.txt')).write_bytes(response.read())
                write(out / (label+'.done.json'), dict(samples=3*a.repeat))
            except Exception as exc:
                write(out / (label+'.error.json'), dict(error=str(exc)))
                print('ERROR', label, str(exc)[:300], flush=True)
            finally:
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
