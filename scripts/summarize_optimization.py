"""Summarize raw optimization logs without selecting away failures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import re


def describe(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
        min=values[0], max=values[-1], stdev=statistics.stdev(values) if len(values)>1 else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    summary = dict(bench=[], server=[], failures=[], raw_sha256={})
    for folder in a.folders:
        root = Path(folder)
        references = {}
        batch_references = {}
        for refpath in root.glob('reference*.samples.jsonl'):
            if refpath.name!='reference.samples.jsonl' and not re.fullmatch(r'reference-r\d+\.samples\.jsonl',refpath.name):
                continue
            match=re.search(r'-r(\d+)\.',refpath.name)
            round_id=match.group(1) if match else ''
            for line in refpath.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                references[(round_id,row['prompt'])] = [t for e in row['events'] if not e['data'].get('stop') for t in e['data'].get('tokens', [])]
        for refpath in root.glob('batch512*.samples.jsonl'):
            m=re.search(r'-r(\d+)\.',refpath.name)
            rid=m.group(1) if m else ''
            for line in refpath.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                batch_references[(rid,row['prompt'])]=[t for e in row['events'] if not e['data'].get('stop') for t in e['data'].get('tokens',[])]
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            summary['raw_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name.endswith('.error.json'):
                summary['failures'].append(dict(file=str(path), **json.loads(path.read_text())))
            if path.name.endswith('.stdout.json'):
                try:
                    rows=json.loads(path.read_text())
                    summary['bench'].append(dict(file=str(path), rows=rows))
                except ValueError:
                    summary['failures'].append(dict(file=str(path), error='Incomplete or invalid bench JSON'))
            if path.name.endswith('.samples.jsonl'):
                rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
                if not rows:
                    continue
                matches=[]
                sample_summary=[]
                match=re.search(r'-r(\d+)\.',path.name)
                round_id=match.group(1) if match else ''
                for row in rows:
                    tokens=[t for e in row['events'] if not e['data'].get('stop') for t in e['data'].get('tokens', [])]
                    key=(round_id,row['prompt'])
                    match=tokens==references[key] if key in references else None
                    first_diff=(next((i for i,(x,y) in enumerate(zip(tokens,references[key])) if x!=y),
                        min(len(tokens),len(references[key])) if len(tokens)!=len(references[key]) else None) if key in references else None)
                    matches.append(match)
                    sample_summary.append(dict(prompt=row['prompt'], ttft_ms=row['ttft_seconds']*1000,
                        wall_seconds=row['wall_seconds'], timings=row['timings'], output_tokens=len(tokens),
                        output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(), matches_reference=match,
                        first_difference_token=first_diff,
                        matches_batch512=(tokens==batch_references[key] if key in batch_references else None)))
                summary['server'].append(dict(file=str(path), samples=sample_summary,
                    ttft_ms=describe([r['ttft_seconds']*1000 for r in rows]),
                    prefill_tps=describe([r['timings']['prompt_per_second'] for r in rows]),
                    decode_tps=describe([r['timings']['predicted_per_second'] for r in rows]),
                    wall_seconds=describe([r['wall_seconds'] for r in rows]), matches_reference=matches))
    Path(a.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in summary['bench']:
        print(row['file'], [(r['n_prompt'],r['n_gen'],round(r['avg_ts'],2)) for r in row['rows']])
    for row in summary['server']:
        print(row['file'], 'TTFT', round(row['ttft_ms']['median'],2), 'PP',round(row['prefill_tps']['median'],2),
            'TG',round(row['decode_tps']['median'],2),'matching',row['matches_reference'])
    print('Failures:',len(summary['failures']))


if __name__ == '__main__':
    main()
