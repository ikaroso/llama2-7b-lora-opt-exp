"""Merge a local PEFT adapter into its exact local Llama 2 base, on CPU."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ('base', 'adapter', 'tokenizer', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        p.error('Output already exists; use a new directory to avoid partial overwrite.')
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
