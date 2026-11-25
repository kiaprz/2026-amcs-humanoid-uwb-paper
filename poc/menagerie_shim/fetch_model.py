"""Download missing pinned Menagerie G1 files and check all SHA-256 hashes.

The model is bundled in g1_model/; this script restores or re-verifies it.
Usage (from the repository root):
    python poc/menagerie_shim/fetch_model.py poc/model_manifest.json g1_model
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def main(result_json, target):
    manifest = json.loads(Path(result_json).read_text())
    # Accept poc/model_manifest.json or a stored result.json that embeds it.
    files = manifest.get('files') or manifest['provenance']['model_manifest']['files']
    root = Path(target)
    bad = 0
    for f in files:
        path = root/f['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f['source_url'], path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != f['sha256']:
            bad += 1
            print('hash mismatch:', f['path'])
    print(len(files), 'files,', bad, 'mismatches')
    return bad


if __name__ == '__main__':
    sys.exit(1 if main(*sys.argv[1:3]) else 0)
