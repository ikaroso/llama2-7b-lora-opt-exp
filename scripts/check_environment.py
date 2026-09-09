"""Offline dependency/PEFT merge smoke test with random tiny weights, not the real 7B."""
import tempfile
import torch
import transformers
import peft
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model


def main():
    print('versions:', torch.__version__, transformers.__version__, peft.__version__)
    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=4)).to(torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
                                          task_type='CAUSAL_LM'))
    for name, tensor in model.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(tensor, std=0.01)
    model.eval()
    tokens = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        before = model(tokens).logits.float()
        merged = model.merge_and_unload(safe_merge=True).to(torch.bfloat16)
        after = merged(tokens).logits.float()
    torch.testing.assert_close(before, after, atol=0.01, rtol=0.05)
    assert not any('lora_' in key for key in merged.state_dict())
    with tempfile.TemporaryDirectory() as folder:
        merged.save_pretrained(folder, safe_serialization=True)
        restored = LlamaForCausalLM.from_pretrained(folder, torch_dtype=torch.bfloat16, local_files_only=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(after, restored(tokens).logits.float(), atol=0, rtol=0)
    print('PASS: CPU BF16 forward, nonzero LoRA merge, safetensors save and reload')


if __name__ == '__main__':
    main()
