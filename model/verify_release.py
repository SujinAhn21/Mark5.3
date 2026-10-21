"""Verify the deployed source and fixed input materials without starting training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, normalize_newlines=False):
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify(root=ROOT, source_only=False):
    root = Path(root)
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    failures = []
    groups = ["source"] if source_only else ["source", "materials"]
    checked = 0
    for group in groups:
        for relative, expected in manifest[group].items():
            path = root / relative
            if not path.is_file() or digest(path, group == "source") != expected:
                failures.append(relative)
            checked += 1
    if failures:
        raise RuntimeError("Release files are missing or differ from the verified release: " + ", ".join(failures))
    print(f"Verified {checked} files; policy={manifest['classification_policy']}")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_only", action="store_true", help="Skip non-Git audio index and weight files")
    args = parser.parse_args()
    verify(source_only=args.source_only)
