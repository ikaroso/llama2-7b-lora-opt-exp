"""Local llama.cpp workflow. Paths resolve relative to the project, not shell cwd."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'llama.cpp'
LOCK = ROOT / 'llama.lock.json'


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def run(argv, dry=False, cwd=ROOT):
    argv = list(map(str, argv))
    print(shlex.join(argv), flush=True)
    if not dry:
        subprocess.run(argv, cwd=cwd, check=True)


def binary(name):
    suffix = '.exe' if os.name == 'nt' else ''
    candidates = [SRC / 'build-cuda/bin' / (name + suffix), SRC / 'build/bin/Release' / (name + suffix), SRC / 'build/bin' / (name + suffix)]
    return next((p for p in candidates if p.exists()), candidates[0 if os.name == 'nt' else 2])


def server_command(c, precision, spec):
    spec = 'draft-simple' if spec == 'draft' else spec
    cmd = [binary('llama-server'), '-m', path(c[precision + '_gguf']),
           '--host', '127.0.0.1', '--port', c['port'], '-c', c['context'],
           '-t', c['threads'], '-ngl', c['gpu_layers'], '-np', 1,
           '-fa', c['flash_attention']]
    if spec != 'none':
        cmd += ['--spec-type', spec, '--spec-draft-n-max', c['draft_tokens']]
    if spec == 'draft-simple':
        if not c['draft_gguf']:
            raise ValueError('draft_gguf must identify a compatible, locally verified draft GGUF.')
        cmd += ['-md', path(c['draft_gguf']), '-ngld', c['draft_gpu_layers']]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['bootstrap', 'merge', 'convert', 'quantize', 'serve'])
    p.add_argument('--config', default='config.local.json')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--precision', choices=['bf16', 'q4'], default='q4')
    p.add_argument('--spec', choices=['none', 'ngram-simple', 'ngram-map-k', 'draft-simple', 'draft'], default='none')
    a = p.parse_args()
    c = json.loads(path(a.config).read_text(encoding='utf-8-sig'))
    dry = a.dry_run
    if a.stage == 'bootstrap':
        ref = json.loads(LOCK.read_text())['commit'] if LOCK.exists() else c['llama_ref']
        if not SRC.exists():
            run(['git', 'clone', 'https://github.com/ggml-org/llama.cpp.git', SRC], dry)
        if SRC.exists() and not dry:
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=SRC, text=True)
            if dirty.strip():
                raise ValueError('llama.cpp checkout has local changes; refusing checkout/build.')
        run(['git', '-C', SRC, 'checkout', '--detach', ref], dry)
        if not dry:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SRC, text=True).strip()
            LOCK.write_text(json.dumps({'repository': 'https://github.com/ggml-org/llama.cpp', 'commit': commit}, indent=2) + '\n')
        if os.name == 'nt' and c['backend'] == 'cuda':
            run(['cmd.exe', '/c', ROOT / 'scripts/build_cuda.cmd'], dry)
            return
        generator = ['-G', 'Visual Studio 17 2022', '-A', 'x64'] if os.name == 'nt' else []
        run(['cmake', '-S', SRC, '-B', SRC / 'build', *generator, '-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_CUDA=' + ('ON' if c['backend'] == 'cuda' else 'OFF')], dry)
        run(['cmake', '--build', SRC / 'build', '--config', 'Release', '-j', '4',
             '--target', 'llama-server', 'llama-quantize', 'llama-bench', 'llama-completion', 'llama-perplexity'], dry)
    elif a.stage == 'merge':
        run([sys.executable, ROOT / 'scripts/merge_lora.py', '--base', path(c['base_model']),
             '--adapter', path(c['adapter']), '--tokenizer', path(c['tokenizer']),
             '--output', path(c['merged_model'])], dry)
    elif a.stage in ('convert', 'quantize'):
        output = path(c['bf16_gguf' if a.stage == 'convert' else 'q4_gguf'])
        if not dry:
            if output.exists():
                raise ValueError(f'Refusing to overwrite {output}')
            output.parent.mkdir(parents=True, exist_ok=True)
        cmd = ([sys.executable, SRC / 'convert_hf_to_gguf.py', path(c['merged_model']),
                '--outfile', output, '--outtype', 'bf16'] if a.stage == 'convert' else
               [binary('llama-quantize'), path(c['bf16_gguf']), output, 'Q4_K_M'])
        run(cmd, dry)
    else:
        cmd = server_command(c, a.precision, a.spec)
        if not dry:
            for item in [path(c[a.precision + '_gguf'])] + ([path(c['draft_gguf'])] if a.spec in ('draft', 'draft-simple') else []):
                if not item.is_file():
                    raise FileNotFoundError(item)
            help_text = subprocess.check_output([str(binary('llama-server')), '--help'], text=True, errors='replace')
            for flag in ('--spec-type', '--spec-draft-n-max') if a.spec != 'none' else ():
                if flag not in help_text:
                    raise ValueError(f'Pinned llama.cpp lacks {flag}; check version before running.')
        run(cmd, dry)


if __name__ == '__main__':
    main()
