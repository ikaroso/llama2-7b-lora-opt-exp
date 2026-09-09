"""Download pinned ModelScope Base and Hugging Face adapter with verified hashes."""
import hashlib
import json
from pathlib import Path
import requests
from urllib.parse import urlencode
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/sources.json'
BASE = 'meta-llama/Llama-2-7b-hf'
ADAPTER = 'iamshnoo/alpaca-2-7b-chinese'
REV = '48951161ce47ed33d8bde26d4302d7bca8ef21e2'


def main():
    MANIFEST.parent.mkdir(exist_ok=True)
    if MANIFEST.exists():
        metadata = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if metadata['base']['repo'] != 'modelscope/Llama-2-7b-ms' or metadata['adapter']['revision'] != REV:
            raise ValueError('Existing model provenance differs from selected models')
    else:
        metadata = {'base': {'repo': 'modelscope/Llama-2-7b-ms', 'provider': 'modelscope', 'upstream': BASE},
                    'adapter': {'repo': ADAPTER, 'revision': REV}}
        MANIFEST.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lock = json.loads((ROOT / 'modelscope.lock.json').read_text(encoding='utf-8-sig'))
    base_dir = ROOT / 'models/base'
    base_dir.mkdir(exist_ok=True)
    allowed = {f['Name'] for f in lock['files']}
    unexpected = [f.name for f in base_dir.iterdir() if f.is_file() and f.name not in allowed and not f.name.endswith('.partial')]
    if unexpected:
        raise ValueError(f'Unexpected base files; move them aside before downloading: {unexpected}')
    hashes = {}
    for item in lock['files']:
        target = base_dir / item['Name']
        def valid(path):
            if not path.exists() or path.stat().st_size != item['Size']:
                return False
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest() == item['Sha256']
        if not valid(target):
            url = 'https://modelscope.cn/api/v1/models/' + lock['repo'] + '/repo?' + urlencode({'Revision': item['Revision'], 'FilePath': item['Name']})
            partial = target.with_name(target.name + '.partial')
            print('Downloading', item['Name'], flush=True)
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
            if not valid(partial):
                raise ValueError(f'Hash/size mismatch: {partial}')
            partial.replace(target)
        hashes[item['Name']] = item['Sha256']
    metadata['base']['sha256'] = hashes
    metadata['base']['files'] = lock['files']
    MANIFEST.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for role, folder, patterns in [
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
