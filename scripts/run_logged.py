"""Run a command, separating stdout/stderr and preserving its exact arguments and exit code."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        p.error('Missing command')
    prefix = Path(a.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(prefix) + suffix) for suffix in ('.stdout.txt', '.stderr.txt', '.meta.json')]
    if any(f.exists() for f in paths):
        p.error('Result prefix already exists; choose a new run name')
    metadata = {'command': cmd, 'cwd': str(Path.cwd()),
                'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with paths[0].open('wb') as out, paths[1].open('wb') as err:
        result = subprocess.run(cmd, stdout=out, stderr=err)
    metadata['exit_code'] = result.returncode
    paths[2].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Exit {result.returncode}; logs: {prefix}.*')
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
