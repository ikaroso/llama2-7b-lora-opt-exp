"""Compare final PPL from two llama-perplexity log prefixes."""
import argparse
import re
from pathlib import Path


def read_ppl(prefix):
    text = '\n'.join(Path(prefix + suffix).read_text(encoding='utf-8', errors='replace')
                     for suffix in ('.stdout.txt', '.stderr.txt'))
    values = re.findall(r'Final estimate: PPL = ([0-9.]+)', text)
    if len(values) != 1:
        raise ValueError(f'{prefix}: expected one final PPL; check completion logs')
    return float(values[0])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bf16_prefix')
    p.add_argument('q4_prefix')
    a = p.parse_args()
    base, quant = read_ppl(a.bf16_prefix), read_ppl(a.q4_prefix)
    print(f'BF16={base:.4f}, Q4={quant:.4f}, delta={quant-base:+.4f}, relative={(quant/base-1)*100:+.2f}%')
