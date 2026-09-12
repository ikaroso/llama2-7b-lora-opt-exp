"""Aggregate confirmation by independent round, not by cherry-picked samples."""
import json
from pathlib import Path
import statistics
import re
import argparse
from summarize_optimization import describe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    summary=json.loads(Path(a.summary).read_text(encoding='utf-8'))
    groups={}
    for record in summary['server']:
        m=re.search(r'([^\\/]+)-r(\d+)\.samples',record['file'])
        if not m:
            continue
        label,round_id=m.groups()
        # Equal input/output budgets; aggregate decode through total decoding time,
        # not arithmetic mean of t/s (which overweights fast/repetitive samples).
        samples=record['samples']
        pp=512*len(samples)/(sum(s['timings']['prompt_ms'] for s in samples)/1000)
        tg=sum(s['timings']['predicted_n']-1 for s in samples)/(sum(s['timings']['predicted_ms'] for s in samples)/1000)
        drafted=sum(s['timings'].get('draft_n',0) for s in samples)
        accepted=sum(s['timings'].get('draft_n_accepted',0) for s in samples)
        groups.setdefault(label,[]).append(dict(round=int(round_id),prefill_tps=pp,decode_tps=tg,
            ttft_ms=statistics.mean(s['ttft_ms'] for s in samples),wall_seconds=statistics.mean(s['wall_seconds'] for s in samples),
            drafted=drafted,accepted=accepted,acceptance=(accepted/drafted if drafted else None),
            matching=sum(s['matches_reference'] is True for s in samples),
            matching_batch=sum(s['matches_batch512'] is True for s in samples),
            samples=len(samples)))
    result={}
    for label,rounds in groups.items():
        rounds.sort(key=lambda r:r['round'])
        if len(rounds)!=5 or any(r['samples']!=3 for r in rounds):
            raise ValueError(f'Incomplete confirmation: {label}')
        result[label]=dict(rounds=rounds,stats={key:describe([r[key] for r in rounds]) for key in ['prefill_tps','decode_tps','ttft_ms','wall_seconds']})
    baseline={r['round']:r for r in result['reference']['rounds']}
    for label,data in result.items():
        data['paired_ratios']={key:describe([r[key]/baseline[r['round']][key] for r in data['rounds']]) for key in ['prefill_tps','decode_tps','ttft_ms','wall_seconds']}
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    for label,data in result.items():
        print(label, {k:(round(v['mean'],3),round(v['stdev'],3)) for k,v in data['stats'].items()})


if __name__=='__main__': main()
