"""Download the pinned public corpus if missing and write fixed comparison configs."""
import hashlib
import json
from pathlib import Path
import urllib.request
from specbench_inputs import REVISION, SOURCE

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04'


def main():
    target=ROOT/'data/spec-bench/question.jsonl'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(SOURCE,timeout=60) as response:
            raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=EXPECTED:
            raise ValueError('Downloaded data hash mismatch')
        target.write_bytes(raw)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED
    lock=dict(repository='https://github.com/hemingkx/Spec-Bench',revision=REVISION,
              question_url=SOURCE,question_sha256=EXPECTED,per_task=6,
              categories=['summarization','rag'],selection_seed=42)
    (ROOT/'bench/specbench-source.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    methods=[dict(label='reference'),dict(label='ngram',extra=['--spec-type','ngram-simple',
             '--spec-ngram-simple-size-n','12','--spec-ngram-simple-size-m','48','--spec-ngram-simple-min-hits','1']),
             dict(label='draft3',extra=['--spec-type','draft-simple','-md',
                  'models/draft-tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf','-ngld','99','--spec-draft-n-max','3'])]
    configs=[]
    for n in range(1,6):
        shift=(n-1)%3
        configs.extend(dict(c,label=c['label']+f'-r{n}') for c in methods[shift:]+methods[:shift])
    (ROOT/'bench/specbench-confirm.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    print('Pinned dataset verified; 12 inputs per configuration, 3 methods, 5 rounds')


if __name__=='__main__': main()
