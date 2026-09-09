"""Merge a local PEFT adapter into its exact local Llama 2 base, on CPU."""
import argparse
import json
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ('base', 'adapter', 'tokenizer', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        p.error('Output already exists; use a new directory to avoid partial overwrite.')
    adapter_config = json.loads((Path(args.adapter) / 'adapter_config.json').read_text())
    if adapter_config.get('base_model_name_or_path') != 'meta-llama/Llama-2-7b-hf':
        p.error('This workflow requires an adapter explicitly targeting official Llama-2-7b-hf Base.')
    manifest_path = Path(args.base).parent / 'sources.json'
    if not manifest_path.is_file():
        p.error('Run scripts/download_models.py first to record selected base provenance.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['base']['repo'] != 'modelscope/Llama-2-7b-ms' or not manifest['base'].get('sha256'):
        p.error('Expected completed ModelScope Base download manifest.')
    lock = json.loads((Path(__file__).resolve().parents[1] / 'modelscope.lock.json').read_text(encoding='utf-8-sig'))
    allowed = {entry['Name'] for entry in lock['files']}
    if any(f.is_file() and f.name not in allowed for f in Path(args.base).iterdir()):
        p.error('Unexpected files in base directory; use only the locked download files.')
    for entry in lock['files']:
        with (Path(args.base) / entry['Name']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['Sha256']:
                p.error('Base checksum mismatch: ' + entry['Name'])
    if Path(args.tokenizer).resolve() != Path(args.base).resolve():
        p.error('Use the verified base tokenizer directory.')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map='cpu',
        local_files_only=True, trust_remote_code=False,
    )
    if model.config.model_type != 'llama' or model.config.hidden_size != 4096 or model.config.num_hidden_layers != 32:
        raise ValueError('Expected Llama 2 7B architecture (4096 hidden, 32 layers). Verify checkpoint provenance separately.')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError('Tokenizer vocabulary differs from base embeddings; handle resized training vocab explicitly.')
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(out)
    print(f'Saved merged BF16 checkpoint to {out}')


if __name__ == '__main__':
    main()
