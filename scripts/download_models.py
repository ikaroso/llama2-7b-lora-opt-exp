"""Download the selected base/adapter with pinned provenance; authenticate with hf first."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/sources.json'
BASE = 'meta-llama/Llama-2-7b-hf'
ADAPTER = 'iamshnoo/alpaca-2-7b-chinese'
REV = '48951161ce47ed33d8bde26d4302d7bca8ef21e2'


def main():
    MANIFEST.parent.mkdir(exist_ok=True)
    if MANIFEST.exists():
        metadata = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if metadata['base']['repo'] != BASE or metadata['adapter']['revision'] != REV:
            raise ValueError('Existing model provenance differs from selected models')
    else:
        metadata = {'base': {'repo': BASE, 'revision': HfApi().model_info(BASE).sha},
                    'adapter': {'repo': ADAPTER, 'revision': REV}}
        MANIFEST.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for role, folder, patterns in [
        ('base', 'base', ['*.json', '*.safetensors', 'tokenizer.model']),
        ('adapter', 'lora', ['adapter_config.json', 'adapter_model.bin', 'README.md']),
    ]:
        entry = metadata[role]
        snapshot_download(entry['repo'], revision=entry['revision'],
                          local_dir=ROOT / 'models' / folder, allow_patterns=patterns)
        files = {}
        for f in sorted((ROOT / 'models' / folder).iterdir()):
            if f.is_file():
                with f.open('rb') as stream:
                    files[f.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        entry['sha256'] = files
        MANIFEST.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Downloads complete; provenance:', MANIFEST)


if __name__ == '__main__':
    main()
