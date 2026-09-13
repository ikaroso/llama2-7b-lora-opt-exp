"""Validate every repeat and report original-precision baseline comparisons."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    return dict(values=values, mean=statistics.mean(values), stdev=statistics.stdev(values),
                min=min(values), max=max(values))


def main():
    result = dict(server={}, bench={}, ratios={}, raw_sha256={})
    for mode in ['server', 'bench']:
        folder = ROOT / f'results/bf16-confirm-{mode}-20260913'
        assert not list(folder.glob('*.error.json')), 'Failed runs must be resolved and disclosed'
        for path in folder.iterdir():
            if path.is_file():
                result['raw_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for label in ['bf16-cpu', 'q4-cpu', 'q4-gpu']:
            rounds = []
            for n in range(1, 6):
                prefix = folder / f'{label}-r{n}'
                if mode == 'server':
                    assert json.loads(Path(str(prefix)+'.done.json').read_text())['samples'] == 3
                    rows = [json.loads(s) for s in Path(str(prefix)+'.samples.jsonl').read_text(encoding='utf-8').splitlines()]
                    assert len(rows) == 3
                    assert all(r['timings']['prompt_n']==512 and r['final']['tokens_predicted']==256 and r['timings']['cache_n']==0 for r in rows)
                    rounds.append(dict(round=n, prefill_tps=1536/(sum(r['timings']['prompt_ms'] for r in rows)/1000),
                        decode_tps=sum(r['timings']['predicted_n']-1 for r in rows)/(sum(r['timings']['predicted_ms'] for r in rows)/1000),
                        ttft_ms=statistics.mean(r['ttft_seconds']*1000 for r in rows),
                        wall_seconds=statistics.mean(r['wall_seconds'] for r in rows)))
                else:
                    assert json.loads(Path(str(prefix)+'.meta.json').read_text())['exit_code']==0
                    rows=json.loads(Path(str(prefix)+'.stdout.json').read_text())
                    assert {(r['n_prompt'],r['n_gen']) for r in rows}=={(512,0),(0,256)}
                    rounds.append(dict(round=n, **{('prefill_tps' if r['n_prompt'] else 'decode_tps'):r['avg_ts'] for r in rows}))
            result[mode][label]=dict(rounds=rounds, stats={k:stats([r[k] for r in rounds]) for k in rounds[0] if k!='round'})
        result['ratios'][mode]={}
        for target,base in [('q4-cpu','bf16-cpu'),('q4-gpu','bf16-cpu'),('q4-gpu','q4-cpu')]:
            result['ratios'][mode][target+'/'+base]={k:stats([a[k]/b[k] for a,b in zip(result[mode][target]['rounds'],result[mode][base]['rounds'])]) for k in result[mode][base]['stats']}
    (ROOT/'bench/bf16-results-20260913.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['\n## 五轮完成结果\n','服务端真实问题，各配置每轮三条，以下为五轮均值 ± 样本标准差。\n',
           '| 配置 | Prefill token/s | Decode token/s | TTFT ms |','|---|---:|---:|---:|']
    for label,data in result['server'].items():
        lines.append('| '+label+' | '+' | '.join(f"{data['stats'][k]['mean']:.2f} ± {data['stats'][k]['stdev']:.2f}" for k in ['prefill_tps','decode_tps','ttft_ms'])+' |')
    for mode in ['server','bench']:
        lines += [f'\n### {mode} 逐轮\n', '| 配置 | 轮次 | Prefill token/s | Decode token/s |'+(' TTFT ms |' if mode=='server' else ''),
                  '|---|---:|---:|---:|'+('---:|' if mode=='server' else '')]
        for label,data in result[mode].items():
            for r in data['rounds']:
                lines.append(f"| {label} | {r['round']} | {r['prefill_tps']:.2f} | {r['decode_tps']:.2f} |"+(f" {r['ttft_ms']:.2f} |" if mode=='server' else ''))
    report=ROOT/'docs/bf16-baseline-20260913.md'
    text=report.read_text(encoding='utf-8').split('\n## 五轮完成结果')[0]
    text=text.replace('测量进行中，完成后补充全部五轮、均值、标准差及配对收益。','已完成三个配置的各五轮确认：45 条正式 server 请求，以及 15 个 bench 进程。')
    report.write_text(text+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result['ratios'],indent=2))


if __name__=='__main__':
    main()
