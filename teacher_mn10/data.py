# -*- coding: utf-8 -*-
"""mn10 교사 데이터 — dataset_index_mark4.X.csv 를 읽어 3초 파형 클립과 이진 라벨(0 = 자기 클래스, 1 = others)을 낸다.

[신설 2026-10-22 / mn10 교사 B+]

- 파형 읽기는 비교모델 common/data.py 의 load_clip 과 같다(16 kHz, 정확히 48,000 샘플, 모노 평균).
- 인덱스의 경로는 코랩 드라이브 경로다. 그 파일이 없으면 'MarkX.Y/<나머지>' 꼬리를 이 저장소의 형제 폴더에서 찾는다
  (드라이브 experiments/Mark4.k 와 USB Mark4_series/Mark4.k 모두 Mark4.1~4.8 이 나란히 있다).
- 합친 val(pooled val): 자기 val 430클립 + 다른 교사 7개의 val 양성(다른 타겟 소리)을 음성으로 더한다.
  "남의 타겟 소리에 손을 드는가" 를 학습 중에 재려는 자료다. 다른 교사의 val others 는 넣지 않는다(실제 클래스가
  인덱스에 없어서 자기 클래스 소리가 섞여 있을 수 있다).
"""
import csv
import hashlib
import json
import os
import re

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
CLIP_SAMPLES = 48000
OTHERS = "others"
EXPERT_VERSIONS = [f"mark4.{i}" for i in range(1, 9)]

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)
SIBLINGS_ROOT = os.path.dirname(REPO_ROOT)


def repo_dir(mark_version):
    return os.path.join(SIBLINGS_ROOT, "Mark" + mark_version[4:])


def index_path(mark_version):
    for c in (os.path.join(repo_dir(mark_version), f"dataset_index_{mark_version}.csv"),
              os.path.join(repo_dir(mark_version), "model", f"dataset_index_{mark_version}.csv")):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"[ERROR] dataset_index CSV 가 없습니다: {mark_version}")


def split_of(path):
    m = re.search(r"/data/(train|val|test)/", path.replace("\\", "/"))
    return m.group(1) if m else None


def resolve_audio(path):
    p = path.replace("\\", "/")
    if os.path.isfile(p):
        return p
    m = re.search(r"/(Mark4\.\d)/(.+)$", p)
    if m:
        local = os.path.join(SIBLINGS_ROOT, m.group(1), *m.group(2).split("/"))
        if os.path.isfile(local):
            return local
    raise FileNotFoundError(f"[ERROR] 오디오 파일이 없습니다: {path}")


def read_rows(mark_version):
    with open(index_path(mark_version), newline="", encoding="utf-8-sig") as f:
        return [(r["path"], r["label"]) for r in csv.DictReader(f)]


def target_class(mark_version):
    labels = sorted({label for _, label in read_rows(mark_version) if label != OTHERS})
    if len(labels) != 1:
        raise RuntimeError(f"[ERROR] {mark_version} 인덱스의 타겟 라벨이 하나가 아닙니다: {labels}")
    return labels[0]


def load_clip(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        t, c = wav.shape
        wav = wav.mean(axis=1) if t >= c else wav.mean(axis=0)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    if len(wav) >= CLIP_SAMPLES:
        wav = wav[:CLIP_SAMPLES]
    else:
        wav = np.concatenate([wav, np.zeros(CLIP_SAMPLES - len(wav), dtype=np.float32)])
    return np.ascontiguousarray(wav, dtype=np.float32)


class ClipSet(Dataset):
    """items: [(path, binary_label, source_class), ...] — 파형을 전부 메모리에 올린다(2,430클립 약 470MB)."""

    def __init__(self, items, name=""):
        if not items:
            raise RuntimeError(f"[ERROR] 빈 데이터셋: {name}")
        self.paths = [p for p, _, _ in items]
        self.labels = torch.tensor([y for _, y, _ in items], dtype=torch.long)
        self.sources = [s for _, _, s in items]
        self.waves = torch.from_numpy(np.stack([load_clip(resolve_audio(p)) for p in self.paths]))
        n_pos = int((self.labels == 0).sum())
        print(f"[DATA] {name}: {len(self.paths)}클립 (자기 클래스 {n_pos} · others {len(self.paths) - n_pos})", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.waves[i], self.labels[i], i


def own_split_items(mark_version, split, max_clips=0):
    tgt = target_class(mark_version)
    rows = [(p, l) for p, l in read_rows(mark_version) if split_of(p) == split]
    items = [(p, 0 if l == tgt else 1, l) for p, l in sorted(rows)]
    if max_clips:
        pos = [it for it in items if it[1] == 0][: max_clips // 2]
        neg = [it for it in items if it[1] == 1][: max_clips - len(pos)]
        items = pos + neg
    return items


def pooled_val_items(mark_version, max_clips=0):
    """자기 val 전부 + 다른 교사 7개의 val 양성(라벨 1 = others, source_class = 그 교사의 클래스)."""
    items = own_split_items(mark_version, "val", max_clips)
    for other in EXPERT_VERSIONS:
        if other == mark_version:
            continue
        o_tgt = target_class(other)
        o_rows = sorted((p, l) for p, l in read_rows(other) if split_of(p) == "val" and l == o_tgt)
        if max_clips:
            o_rows = o_rows[: max(1, max_clips // 8)]
        items += [(p, 1, o_tgt) for p, _ in o_rows]
    return items


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_identity(mark_version, split_items):
    """인덱스 CSV 해시 + split 별 (경로, 라벨, 음원 SHA-256) 목록의 해시. 옛 교사 provenance 와 같은 정의."""
    entries = []
    for split, items in split_items.items():
        for path, y, src in sorted(items):
            entries.append({"split": split, "path": path.replace("\\", "/"), "label": src,
                            "sha256": sha256_file(resolve_audio(path))})
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"dataset_index_sha256": sha256_file(index_path(mark_version)),
            "audio_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "split_counts": {k: len(v) for k, v in split_items.items()}}
