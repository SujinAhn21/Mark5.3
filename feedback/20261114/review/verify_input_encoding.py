"""Read-only replay of the 430-file source-to-Drive-input encoding audit."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    report = json.loads(Path(__file__).with_name("input_encoding_audit.json").read_text())
    expected = {r["filename"]: r for r in report["files"]}
    for row in expected.values():
        source = args.codes_root / row["source_relative_path"]
        assert sha(source) == row["source_sha256"], source
        wave, rate = sf.read(source, dtype="float32")
        assert rate == 16000 and wave.ndim == 1, source
        if len(wave) < 48000:
            wave = np.pad(wave, (0, 48000 - len(wave)))
        encoded = io.BytesIO()
        sf.write(encoded, wave, rate, format="WAV", subtype="PCM_16")
        assert hashlib.sha256(encoded.getvalue()).hexdigest() == row["drive_sha256"], source
    dataset = args.codes_root / "Mark5.3/dataset_test.csv"
    assert sha(dataset) == report["test_csv_sha256"]
    with dataset.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(path=r.get("path"), label=r.get("label"), audio_sha256=expected[Path(r["path"]).name]["drive_sha256"])
            for r in csv.DictReader(handle)
        ]
    payload = dict(csv_sha256=sha(dataset), rows=rows)
    identity = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert identity == report["reconstructed_dataset_identity"]
    for row in report["kampa_policies_verified"]:
        policy = args.codes_root / "Mark5.3" / row["path"]
        assert sha(policy) == row["sha256"], policy
        assert json.loads(policy.read_text())["dataset_identity"] == identity
    print(f"Verified {len(expected)}/{len(expected)} PCM_16 input hashes and {len(report['kampa_policies_verified'])} KAMPA test policies.")
    print("Historical KAMPA dataset identity:", identity)


if __name__ == "__main__":
    main()
