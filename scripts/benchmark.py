"""Sequential native /completion benchmark; preserve server timings and outputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def request(url, body):
    req = urllib.request.Request(url + '/completion', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    return data, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--label', required=True)
    p.add_argument('--prompts', default='bench/prompts.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='config.local.json')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--tokens', type=int, default=128)
    a = p.parse_args()
    if a.repeat < 1 or a.tokens < 1:
        p.error('repeat and tokens must be positive')
    raw = Path(a.prompts).read_bytes()
    prompts = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not prompts:
        p.error('Empty prompt set')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    def body(prompt):
        return {'prompt': prompt, 'n_predict': a.tokens, 'temperature': 0,
                'seed': 42, 'cache_prompt': False, 'stream': False}
    with out.open('x', encoding='utf-8') as f:
        metadata = {'type': 'metadata', 'label': a.label,
                    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'prompt_sha256': hashlib.sha256(raw).hexdigest(),
                    'config': json.loads(Path(a.config).read_text(encoding='utf-8-sig')),
                    'lock': json.loads(Path('llama.lock.json').read_text()) if Path('llama.lock.json').exists() else None,
                    'request_tokens': a.tokens, 'repeat': a.repeat}
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        request(a.url, body(prompts[0]['prompt']))  # warmup excluded
        for repeat in range(a.repeat):
            for prompt in prompts:
                result, elapsed = request(a.url, body(prompt['prompt']))
                row = {'type': 'sample', 'label': a.label, 'id': prompt['id'],
                       'repeat': repeat, 'wall_seconds': elapsed, 'response': result}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(prompt['id'], round(elapsed, 3), result.get('timings', {}))


if __name__ == '__main__':
    main()
