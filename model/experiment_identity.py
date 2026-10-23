"""Stable input identities for training reuse and evaluation artifacts."""

import csv
import hashlib
import json
import os
from functools import lru_cache

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_audio(path):
    normalized = str(path).replace("\\", "/")
    if os.path.isfile(normalized):
        return normalized
    marker = "/NewProject/codes/experiments/Mark5.3/"
    if marker in normalized:
        candidate = os.path.join(PROJECT_ROOT, *normalized.split(marker, 1)[1].split("/"))
        if os.path.isfile(candidate):
            return candidate
    return None


@lru_cache(maxsize=32)
def dataset_identity(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            audio = _resolve_audio(row.get("path", ""))
            rows.append({
                "path": row.get("path"),
                "label": row.get("label"),
                "audio_sha256": sha256_file(audio) if audio else None,
            })
    payload = {
        "csv_sha256": sha256_file(csv_path),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=1)
def teacher_release_identity():
    files = []
    for version in [f"mark4.{i}" for i in range(1, 9)]:
        for kind in ("encoder", "classifier"):
            name = f"best_teacher_{kind}_{version}.pth"
            path = os.path.join(RESOURCES_DIR, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[ERROR] teacher 파일 없음: {path}")
            ckpt = torch.load(path, map_location="cpu")
            provenance = ckpt.get("provenance") if isinstance(ckpt, dict) else None
            files.append({"file": name, "sha256": sha256_file(path),
                          "run_id": (provenance or {}).get("run_id")})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def current_experiment_identity(run_tag, extra_files=None, teacher_free=False):
    """[2026-10-22 / 검증 결합기] teacher_free=True(--pseudo_csv 회차)는 학습 중에 교사를 쓰지 않으므로 resources/ 교사 파일을
    신원에 넣지 않는다. 가짜 정답 CSV 자체가 extra_files 로 들어가고, 그 CSV 를 만든 교사는 plots/combiner/*.json 의 점수 캐시 해시에 남는다."""
    data = {}
    for name in ("dataset_labeled.csv", "dataset_val.csv", "dataset_unlabeled.csv"):
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(path):
            data[name] = dataset_identity(path)
    for path in extra_files or ():
        if path and not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        if path and os.path.isfile(path):
            # [2026-10-22] CSV 가 아닌 재료(교사 특징 캐시 .pt)는 파일 해시로 신원을 잡는다.
            data[os.path.basename(path)] = dataset_identity(path) if path.endswith(".csv") else sha256_file(path)
    code = {}
    # [2026-10-22 / Prodigy] 옵티마이저 코드(utils/prodigy_optim.py · third_party/prodigy)도 학습 결과를 바꾸므로 신원에 넣는다.
    for rel in ("model/train_mark5.py", "vild/vild_config.py", "vild/teacher_fusion.py",
                "utils/prodigy_optim.py", "third_party/prodigy/prodigyopt/prodigy.py"):
        path = os.path.join(PROJECT_ROOT, rel)
        code[rel] = sha256_file(path)
    payload = {
        "run_tag": run_tag,
        "teacher_release_sha256": None if teacher_free else teacher_release_identity(),
        "training_data": data,
        "code": code,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


def checkpoint_experiment_id(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return (checkpoint.get("config_metadata") or {}).get("experiment_id")
    except (OSError, ValueError, TypeError, AttributeError):
        return None

