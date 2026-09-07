#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""esc50_slicer.py — ESC-50 을 우리 클립 규격(16 kHz · 모노 · 3.0초 · float32 wav)으로 잘라 둔다.

[신설 2026-09-07 / Mark5_EXP2] 왜 필요한가.
  2026-09-07 실측으로 mark5.0 의 others 316개 중 3분의 2가 타겟과 같은 종류의 소리였다
  (말소리 91 · AIHub 가 스스로 '층간소음'으로 분류한 문 여닫기·런닝머신·골프·악기·고양이 등 117).
  그래서 others 를 "층간소음이 아닌 먼 소리"로 다시 정의하기로 했고(수진님 결정), 그 풀을 채울 재료와
  "학습에서 한 번도 안 본 소리"(open-set 평가)를 ESC-50(2,000클립 · 50종 · 5초 · 44.1 kHz)에서 가져온다.

역할(role) — 50종을 세 묶음으로 나눈다. 이 배정은 실험 설계의 일부라 여기 상수로 못 박는다.
  far_train   (21종)  others 학습 풀. 사람이 들어도 8종 어디에도 안 들어가고 층간소음 민원 항목도 아닌 소리.
  far_heldout (10종)  open-set 먼 층. 학습·검증에 절대 쓰지 않고 평가에만 쓴다. 정답은 others.
  near_analysis(19종) 타겟과 겹치거나 층간소음 항목인 소리(개·고양이·발소음·문·세탁기·청소기·변기·물·아기 울음·박수·엔진…).
                      학습 금지. 경계 소리 분석에만 쓴다.
  ESC-50 자체 라이선스: CC BY-NC 3.0 (esc10 표시 클립은 CC BY). 학술용이라 문제없고 논문에 인용한다(Piczak, ACM MM 2015).

창 고르기.
  ESC-50 클립은 5초인데 앞뒤가 무음으로 채워진 것이 많다(표본 200개 중 38개가 절반 넘게 무음).
  0.0~2.0초 시작점을 0.1초 간격으로 훑어 RMS 가 가장 큰 3초 창을 고르고, 16 kHz 로 리샘플(torchaudio, mark5_slicer 와 같은 경로)한 뒤
  validate_dataset.check_waveform(수집 단계 공용 품질 게이트: 길이·RMS·peak·클리핑)을 통과한 것만 저장한다.

저장.
  <out_root>/<role>/<category>/esc50_<category>_<원본stem>.wav
  <out_root>/esc50_manifest.csv   provenance 와 같은 열 이름(+role·esc10·license). 아직 data_provenance.xlsx 에는 넣지 않는다 —
                                  어느 클립을 어느 split 에 배정할지는 다음 단계(데이터 재구성 스크립트)가 정하면서 넣는다.
  out_root 기본값은 ~/workspace/esc50_clips (WSL 홈, aihub_raw 등 원본과 같은 층). 매니페스트 사본은 codes/esc50_clips/ 에 둔다.
  전문가 8개·Mark5_EXP2 가 같은 풀을 나눠 쓴다. 드라이브(AImodels/codes/experiments/esc50_clips)에는 수진님이 올린다.

사용:
  python preprocessing/esc50_slicer.py --dry_run
  python preprocessing/esc50_slicer.py --limit 5          # 동작 확인
  python preprocessing/esc50_slicer.py                    # 2,000개 전부(로컬 몇 분)
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CODES_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform  # noqa: E402

TARGET_SR = 16000
SEG_SEC = 3.0
WINDOW_STARTS = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 1)   # 5초 원본에서 3초 창 시작점 후보
SOURCE = "esc50"
LICENSE = {True: "CC BY 3.0 (ESC-10 subset)", False: "CC BY-NC 3.0"}

ROLES = {
    "far_train": [
        "sea_waves", "crackling_fire", "chirping_birds", "insects", "frog", "hen", "rooster", "cow", "pig",
        "crow", "sheep", "breathing", "coughing", "brushing_teeth", "snoring", "drinking_sipping",
        "mouse_click", "can_opening", "clock_tick", "car_horn", "airplane",
    ],
    "far_heldout": [
        "rain", "wind", "crickets", "church_bells", "clock_alarm", "keyboard_typing", "sneezing",
        "siren", "train", "helicopter",
    ],
    "near_analysis": [
        "dog", "cat", "footsteps", "door_wood_knock", "door_wood_creaks", "washing_machine",
        "vacuum_cleaner", "toilet_flush", "pouring_water", "water_drops", "thunderstorm", "crying_baby",
        "laughing", "clapping", "engine", "chainsaw", "hand_saw", "fireworks", "glass_breaking",
    ],
}
ROLE_OF = {c: r for r, cs in ROLES.items() for c in cs}
# 매니페스트의 target_class 칸. others 학습 풀만 'others', 나머지는 평가 전용임을 이름으로 못 박는다.
TARGET_OF_ROLE = {"far_train": "others", "far_heldout": "openset_far", "near_analysis": "openset_near"}

def resample_16k(seg, sr):
    """mark5_slicer.slice_resample 과 같은 경로(torchaudio)로 16 kHz 로 맞춘다."""
    if sr == TARGET_SR:
        return seg.astype(np.float32)
    import torch
    import torchaudio.functional as AF
    t = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).unsqueeze(0)
    return AF.resample(t, sr, TARGET_SR).squeeze(0).numpy().astype(np.float32)


def pick_window(x, sr):
    """RMS 가 가장 큰 3초 창의 (start_sec, rms) 를 돌려준다."""
    n = int(SEG_SEC * sr)
    best_s, best_rms = 0.0, -1.0
    for s in WINDOW_STARTS:
        a = int(s * sr)
        seg = x[a:a + n]
        if len(seg) < n:
            break
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms > best_rms:
            best_s, best_rms = float(s), rms
    return best_s, best_rms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esc50_root", type=str, default=os.path.expanduser("~/workspace/esc50_raw/ESC-50-master"))
    # 기본 출력은 WSL 홈(ext4). USB(FAT32/9p)에 384MB 를 작은 파일로 쓰면 매우 느리고, 오디오는 어차피 git 에 안 올린다.
    # 매니페스트 CSV 만 codes/esc50_clips/ 에 사본을 둔다(provenance 와 같은 층). 드라이브 업로드는 수진님이 한다.
    ap.add_argument("--out_root", type=str, default=os.path.expanduser("~/workspace/esc50_clips"))
    ap.add_argument("--manifest_copy_dir", type=str, default=os.path.join(CODES_ROOT, "esc50_clips"),
                    help="매니페스트 CSV 사본을 둘 폴더(빈 문자열이면 안 만듦)")
    ap.add_argument("--roles", nargs="+", default=list(ROLES), choices=list(ROLES))
    ap.add_argument("--limit", type=int, default=0, help="동작 확인용: 앞에서 N개만")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 wav 도 다시 만든다")
    args = ap.parse_args()

    meta_path = os.path.join(args.esc50_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.esc50_root, "audio")
    if not os.path.exists(meta_path):
        raise SystemExit(f"[ERROR] ESC-50 메타가 없습니다: {meta_path}")
    meta = pd.read_csv(meta_path)
    missing = sorted(set(meta["category"]) - set(ROLE_OF))
    extra = sorted(set(ROLE_OF) - set(meta["category"]))
    if missing or extra:
        raise SystemExit(f"[ERROR] 역할 배정과 ESC-50 카테고리가 안 맞습니다. 배정 누락 {missing} · 없는 이름 {extra}")
    meta["role"] = meta["category"].map(ROLE_OF)
    meta = meta[meta["role"].isin(args.roles)].sort_values(["role", "category", "filename"]).reset_index(drop=True)
    if args.limit:
        meta = meta.head(args.limit)

    print(f"[INFO] ESC-50 {len(meta)}개 · 역할별: " + ", ".join(f"{r} {int((meta['role'] == r).sum())}" for r in args.roles))
    print(f"[INFO] 출력: {args.out_root}")
    if args.dry_run:
        for r in args.roles:
            print(f"  {r:<14} {len(ROLES[r]):2d}종: {', '.join(ROLES[r])}")
        return

    rows, rejected, skipped = [], [], 0
    t0 = time.time()
    for i, m in meta.iterrows():
        src = os.path.join(audio_dir, m["filename"])
        stem = os.path.splitext(m["filename"])[0]
        out_dir = os.path.join(args.out_root, m["role"], m["category"])
        name = f"esc50_{m['category']}_{stem}.wav"
        out = os.path.join(out_dir, name)
        try:
            data, sr = sf.read(src, dtype="float32", always_2d=True)
        except Exception as e:                                   # 파일 하나가 깨져도 전체 매니페스트는 남긴다(리뷰 지적 13)
            rejected.append((m["filename"], m["category"], f"읽기실패 {type(e).__name__}"))
            continue
        x = data.mean(axis=1)
        start, _ = pick_window(x, sr)
        a = int(start * sr)
        seg = resample_16k(x[a: a + int(SEG_SEC * sr)], sr)      # pick_window 와 같은 인덱스 식(리뷰 지적 14)
        seg = seg[:int(SEG_SEC * TARGET_SR)]                      # 리샘플 반올림으로 1샘플 남는 것을 정확히 48,000샘플로
        reason = check_waveform(seg, TARGET_SR)
        if reason is not None:
            rejected.append((m["filename"], m["category"], reason))
            continue
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            seg, _sr = sf.read(out, dtype="float32")              # 측정값·해시를 디스크 파일 기준으로 통일(리뷰 지적 10)
        else:
            os.makedirs(out_dir, exist_ok=True)
            sf.write(out, seg, TARGET_SR, subtype="FLOAT")
        with open(out, "rb") as _f:
            sha = hashlib.sha256(_f.read()).hexdigest()
        ax = np.abs(seg)
        rows.append({
            "local_filename": name, "original_labels": m["category"],
            "target_class": TARGET_OF_ROLE[m["role"]], "assigned_split": "",
            "mark_version": "", "sha256": sha,
            "size_bytes": os.path.getsize(out), "download_date": date.today().isoformat(),
            "source_type": "original", "removed_20260715": "active", "source": SOURCE,
            "aihub_src_file": m["filename"], "aihub_category": m["category"], "aihub_volume": f"fold{int(m['fold'])}",
            "seg_start_sec": round(start, 3), "seg_end_sec": round(start + SEG_SEC, 3), "duration_sec": SEG_SEC,
            "rms": round(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))), 6), "peak": round(float(ax.max()), 6),
            "role": m["role"], "esc50_target": int(m["target"]), "esc10": bool(m["esc10"]),
            "license": LICENSE[bool(m["esc10"])], "rel_path": os.path.relpath(out, args.out_root).replace("\\", "/"),
        })
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(meta)}  {time.time() - t0:.0f}s")

    os.makedirs(args.out_root, exist_ok=True)
    man = os.path.join(args.out_root, "esc50_manifest.csv")
    partial = set(args.roles) != set(ROLES)
    df_new = pd.DataFrame(rows)
    rej_new = pd.DataFrame(rejected, columns=["filename", "category", "reason"])
    rej_path = os.path.join(args.out_root, "esc50_rejected.csv")
    if args.limit:
        man = os.path.join(args.out_root, "esc50_manifest_smoke.csv")
        df_new.to_csv(man, index=False, encoding="utf-8-sig")
    else:
        # 역할 일부만 다시 돌렸으면 기존 매니페스트에서 그 역할 행만 갈아 끼운다(통째로 덮지 않음 — 리뷰 지적 5).
        if partial and os.path.exists(man):
            old = pd.read_csv(man, encoding="utf-8-sig")
            df_new = pd.concat([old[~old["role"].isin(args.roles)], df_new], ignore_index=True)
            df_new = df_new.sort_values(["role", "aihub_category", "local_filename"]).reset_index(drop=True)
        df_new.to_csv(man, index=False, encoding="utf-8-sig")
        if partial and os.path.exists(rej_path):
            old_r = pd.read_csv(rej_path, encoding="utf-8-sig")
            old_r = old_r[~old_r["category"].map(ROLE_OF).isin(args.roles)]
            rej_new = pd.concat([old_r, rej_new], ignore_index=True)
        if len(rej_new):
            rej_new.to_csv(rej_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 저장 {len(rows)}개 (이미 있어 건너뜀 {skipped}) · 품질 미달 {len(rejected)}개 · {time.time() - t0:.0f}s")
    if rejected:
        by = pd.DataFrame(rejected, columns=["f", "category", "reason"])
        print("  품질 미달 사유:", by["reason"].str.replace(r"[0-9.%s()<]+", "", regex=True).value_counts().to_dict())
        print("  품질 미달 상위 카테고리:", by["category"].value_counts().head(8).to_dict())
    if len(df_new):
        print("  매니페스트 역할별 수:", df_new["role"].value_counts().to_dict())
    print(f"[INFO] 매니페스트: {man}")
    if args.manifest_copy_dir and not args.limit:
        os.makedirs(args.manifest_copy_dir, exist_ok=True)
        import shutil
        shutil.copy2(man, os.path.join(args.manifest_copy_dir, os.path.basename(man)))
        if os.path.exists(rej_path):
            shutil.copy2(rej_path, os.path.join(args.manifest_copy_dir, os.path.basename(rej_path)))
        print(f"[INFO] 매니페스트 사본: {args.manifest_copy_dir}")


if __name__ == "__main__":
    main()
