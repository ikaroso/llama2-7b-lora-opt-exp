"""Prepare five interleaved rounds; keep the existing experiment immutable."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for mode in ['bench', 'server']:
    rows = []
    for round_id in range(1, 6):
        pair = [dict(label='bf16-cpu', bf16=True), dict(label='q4-cpu')]
        if round_id % 2 == 0:
            pair.reverse()
        for config in pair:
            config.update(ngl=0, fa='off', extra=['--no-kv-offload', '--no-op-offload'])
            if mode == 'bench':
                config['extra'] = ['--no-kv-offload', '1', '--no-op-offload', '1']
        pair.append(dict(label='q4-gpu'))
        for config in pair:
            config['label'] += f'-r{round_id}'
            rows.append(config)
    (root / f'bench/bf16-confirm-{mode}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
