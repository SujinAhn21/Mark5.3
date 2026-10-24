#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""최신 판 확인 — 교사 → 점수 캐시 → (특징 캐시 · 결합기 가짜 정답 CSV) → 학생 사이에서 옛 파일이 섞이지 않게 해시로 대조한다.

[신설 2026-10-22 / 검증 결합기] 수진님 지시: "예전 파일을 쓰지 않는지 항상 최신껄 쓸 수 있도록".
파일 이름에는 교사 판이 드러나지 않는다. 교사를 다시 학습해도 같은 이름의 옛 CSV 가 조용히 읽히면 학습이 헛일이 되므로,
산출물마다 입력 파일의 SHA-256 을 json 에 남기고(make_teacher_scores.py · make_combiner_labels.py) 읽는 쪽이 지금 파일과 대조한다.
하나라도 어긋나면 StaleFileError 로 멈춘다.

    python model/label_freshness.py                         # 프로젝트 루트의 결합기 CSV·점수 캐시를 전부 대조(노트북의 학습 전 점검 셀)
    python model/label_freshness.py --csv pseudo_labels_cmb_l200_s1.csv
"""
import argparse
import csv
import functools
import glob
import hashlib
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
_CMB_RE = re.compile(r"^pseudo_labels_(cmb|cmbone|fix)_l(\d+|all)_s(\d+)\.csv$")
# [2026-10-25] 데이터 목록·오디오는 늘 이 저장소 기준이다(시험이 PROJECT_ROOT 를 임시 폴더로 바꿔도 그대로).
DATA_ROOT = PROJECT_ROOT
SPLIT_CSV = {"labeled": "dataset_labeled.csv", "unlabeled": "dataset_unlabeled.csv",
             "val": "dataset_val.csv", "test": "dataset_test.csv"}
SOURCE_ALIAS = {"data_labeled": "data_source_labeled", "data_unlabeled": "data_source_unlabeled",
                "data_val": "data_source_val", "data_test": "data_source_test"}


class StaleFileError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _abs(path, root):
    return path if os.path.isabs(path) else os.path.join(root, path)


def localize(path, root=None):
    """CSV 의 코랩 경로 → 지금 환경의 파일. 코랩에서는 그대로 있다. 로컬(USB)에서는 형제 저장소 Mark5.2 의
    data_* 또는 data_source_* 폴더에서 찾는다."""
    p = str(path).replace("\\", "/")
    if os.path.isfile(p):
        return p
    m = re.search(r"/(Mark5\.2|Mark5\.3|Mark5_EXP2?)/(data_[a-z]+)/(.+)$", p)
    if m:
        base = os.path.join(os.path.dirname(root or DATA_ROOT), m.group(1))
        for folder in (m.group(2), SOURCE_ALIAS.get(m.group(2), m.group(2))):
            c = os.path.join(base, folder, *m.group(3).split("/"))
            if os.path.isfile(c):
                return c
    raise FileNotFoundError(f"[ERROR] 오디오 파일이 없습니다: {path}")


@functools.lru_cache(maxsize=8)
def audio_manifest(splits, limit=0, root=None):
    """[2026-10-25] 점수 캐시를 만든 Mark5.3 오디오의 신원 — split 마다 지금 데이터 목록의 (split, path, label, 오디오 SHA-256)를
    모아 해시한다. 파일 이름은 그대로 두고 내용만 바꾼 클립(10-25 talking 아이 말소리 129클립 교체)이 옛 점수 캐시로 통과하지 않게 한다.
    splits 는 튜플(make_teacher_scores 가 점수를 낸 순서), limit 은 split 마다 앞에서 N개(0 = 전부). 한 프로세스에서 한 번만 계산한다."""
    root = root or DATA_ROOT
    entries = []
    for split in splits:
        with open(os.path.join(root, SPLIT_CSV[split]), newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if limit:
            rows = rows[:limit]
        entries += [[split, r["path"], r.get("label", ""), sha256_file(localize(r["path"], root))] for r in rows]
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _teacher_file_name(entry):
    """점수 캐시 json 의 teacher_files 항목 → resources/ 파일 이름. mn10 은 file 키, 조각 교사(B0)는 class·kind 로 만든다."""
    if entry.get("file"):
        return entry["file"]
    from train_mark5 import specialist_config_for   # 조각 교사: 클래스 → mark4.X
    from vild_config import AudioViLDConfig
    spec = specialist_config_for(AudioViLDConfig(mark_version="mark5.2"))
    return spec[entry["class"]][f"{entry['kind']}_path"]


def check_scores(scores_csv, root=PROJECT_ROOT, resources_dir=None):
    """점수 캐시 CSV 가 그 json 과 짝이고, json 에 적힌 교사 파일이 지금 resources/ 의 파일과 같으며, [2026-10-25] 그 캐시를 만든
    Mark5.3 데이터 목록·오디오가 지금과 같은지. 돌려주는 값: json(meta)."""
    scores_csv = _abs(scores_csv, root)
    meta_path = os.path.splitext(scores_csv)[0] + ".json"
    if not os.path.isfile(scores_csv) or not os.path.isfile(meta_path):
        raise StaleFileError(f"[STALE] 점수 캐시 또는 그 json 이 없습니다: {scores_csv}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if sha256_file(scores_csv) != meta.get("scores_sha256"):
        raise StaleFileError(f"[STALE] 점수 캐시 CSV 가 json 에 적힌 판과 다릅니다(다른 실행의 CSV 가 덮였을 수 있음): {scores_csv}")
    resources_dir = resources_dir or os.path.join(PROJECT_ROOT, "resources")   # 교사는 늘 이 저장소의 resources/
    for entry in meta.get("teacher_files", []):
        path = os.path.join(resources_dir, _teacher_file_name(entry))
        if not os.path.isfile(path):
            raise StaleFileError(f"[STALE] 점수 캐시를 만든 교사 파일이 resources/ 에 없습니다: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise StaleFileError(f"[STALE] resources/ 의 교사가 점수 캐시를 만든 교사와 다릅니다(교사를 다시 학습·반입한 뒤 "
                                 f"make_teacher_scores.py 를 다시 돌리지 않음): {path}")
    want = meta.get("mark53_audio_sha256")       # [2026-10-25] 교사가 같아도 Mark5.3 데이터 목록·오디오가 바뀌면 옛 캐시다
    if want is None:
        raise StaleFileError(f"[STALE] 점수 캐시 json 에 Mark5.3 오디오 해시가 없습니다(2026-10-25 이전 판 make_teacher_scores.py "
                             f"산출물): {scores_csv}")
    if audio_manifest(tuple(meta.get("split_counts") or {}), int(meta.get("limit") or 0)) != want:
        raise StaleFileError(f"[STALE] 지금 Mark5.3 데이터 목록·오디오가 점수 캐시를 만든 판과 다릅니다(오디오를 바꾼 뒤 "
                             f"make_teacher_scores.py 를 다시 돌리지 않음): {scores_csv}")
    return meta


def check_features(feats_path, scores_csv, root=PROJECT_ROOT):
    """특징 캐시(.pt)가 지금 점수 캐시와 같은 실행에서 나왔는지."""
    import torch
    feats_path, scores_csv = _abs(feats_path, root), _abs(scores_csv, root)
    if not os.path.isfile(feats_path):
        raise StaleFileError(f"[STALE] 특징 캐시가 없습니다: {feats_path}")
    blob = torch.load(feats_path, map_location="cpu", weights_only=False)
    if blob.get("scores_sha256") != sha256_file(scores_csv):
        raise StaleFileError(f"[STALE] 특징 캐시가 지금 점수 캐시와 다른 실행에서 나왔습니다: {feats_path}")
    return blob


def check_pseudo_csv(csv_path, root=PROJECT_ROOT, resources_dir=None):
    """결합기·고정 규칙 가짜 정답 CSV(pseudo_labels_{cmb|cmbone|fix}_l*_s*.csv)가 결합기 json 에 적힌 판이고, 그 json 이 가리키는
    점수 캐시가 지금 판이며, 그 점수 캐시를 만든 교사가 지금 resources/ 의 교사인지. 이 규칙 밖 이름(self·gen 등)은 None."""
    csv_path = _abs(csv_path, root)
    m = _CMB_RE.match(os.path.basename(csv_path))
    if not m:
        return None
    kind, lname, seed = m.groups()
    meta_path = os.path.join(os.path.dirname(csv_path), "plots", "combiner", f"combiner_l{lname}_s{seed}.json")
    if not os.path.isfile(csv_path) or not os.path.isfile(meta_path):
        raise StaleFileError(f"[STALE] 가짜 정답 CSV 또는 결합기 json 이 없습니다: {csv_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    want = (meta.get("outputs") or {}).get(os.path.basename(csv_path))
    if want is None:
        raise StaleFileError(f"[STALE] 결합기 json 에 이 CSV 의 해시가 없습니다(2026-10-22 이전 판 make_combiner_labels.py 산출물): {csv_path}")
    if sha256_file(csv_path) != want:
        raise StaleFileError(f"[STALE] 가짜 정답 CSV 가 결합기 json 에 적힌 판과 다릅니다: {csv_path}")
    scores_csv = os.path.join(os.path.dirname(csv_path), meta["scores_csv"])
    if not os.path.isfile(scores_csv) or sha256_file(scores_csv) != meta["scores_sha256"]:
        raise StaleFileError(f"[STALE] 가짜 정답을 만든 점수 캐시가 지금 판이 아닙니다(교사 점수를 다시 만든 뒤 결합기를 다시 돌리지 않음): "
                             f"{meta['scores_csv']}")
    check_scores(scores_csv, root=os.path.dirname(csv_path), resources_dir=resources_dir)
    return meta


def main():
    ap = argparse.ArgumentParser(description="가짜 정답 CSV·점수 캐시·특징 캐시가 지금 교사의 최신 판인지 해시로 대조한다")
    ap.add_argument("--csv", nargs="*", default=None, help="안 주면 프로젝트 루트의 pseudo_labels_{cmb,cmbone,fix}_*.csv 전부")
    ap.add_argument("--scores", nargs="*", default=None, help="안 주면 plots/teacher_scores_*.csv 전부(_limit 시험판 제외)")
    args = ap.parse_args()
    csvs = args.csv if args.csv is not None else sorted(p for p in glob.glob(os.path.join(PROJECT_ROOT, "pseudo_labels_*.csv"))
                                                        if _CMB_RE.match(os.path.basename(p)))
    scores = args.scores if args.scores is not None else sorted(p for p in glob.glob(os.path.join(PROJECT_ROOT, "plots", "teacher_scores_*.csv"))
                                                                if "_limit" not in p)
    for s in scores:
        meta = check_scores(s)
        print(f"[OK] 점수 캐시 {os.path.basename(s)} · 교사 release {meta['teacher_release_sha256'][:12]}")
        feats = os.path.join(os.path.dirname(_abs(s, PROJECT_ROOT)), os.path.basename(s).replace("teacher_scores_", "teacher_feats_").replace(".csv", ".pt"))
        if os.path.isfile(feats):
            check_features(feats, s)
            print(f"[OK] 특징 캐시 {os.path.basename(feats)}")
    for c in csvs:
        meta = check_pseudo_csv(c)
        if meta is not None:
            print(f"[OK] {os.path.basename(c)} · 점수 캐시 {meta['scores_sha256'][:12]}")
    print(f"[완료] 가짜 정답 CSV {len(csvs)}개 · 점수 캐시 {len(scores)}개 모두 최신 판")


if __name__ == "__main__":
    main()
