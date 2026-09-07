#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""천장(upper bound)용 라벨 CSV 생성 — unlabeled 2,000개에 가려 두었던 진짜 라벨을 되돌려 준다.

[신설 2026-09-06 / Mark5_EXP] 근거: Cances, Labbé, Pellegrini (EURASIP J. Audio Speech Music Process., 2022)
의 반지도 평가 프로토콜 — "라벨 일부만 쓴 지도학습(바닥) / 반지도(SSL) / 라벨 전부를 쓴 지도학습(천장)".
우리는 2026-09-05 까지 "unlabeled 에 사람 라벨이 없다"고 보고 천장을 뺐는데, journal/codes/data_provenance.xlsx
에 unlabeled 2,000행 전부의 target_class 가 기록돼 있었다(9클래스 222개씩). 이 스크립트가 그 라벨을 꺼내
consensus_labels.csv 와 같은 스키마로 만든다. 비교모델(--cond ceil)과 Mark5_EXP(--ceiling_csv)가 이 파일을 읽는다.

입력
    data_provenance.xlsx      기본 위치 = 프로젝트 루트의 부모(codes/). --provenance 로 바꿀 수 있다.
                              mark_version == --mark_version, assigned_split == "unlabeled" 인 행만 쓴다.
                              target_class 는 "unlabeled:construction" 꼴이라 앞의 "unlabeled:" 를 뗀다.
    dataset_unlabeled.csv     path 열 하나. 이 순서 그대로 출력한다(경로 문자열도 그대로 — 코랩 드라이브 경로).

출력 (--out 기본 = 프로젝트 루트/ceiling_labels.csv, 통계는 plots/ceiling_labels_stats.json)
    path,label,confidence,source,generated_at
        confidence 는 진짜 라벨이므로 전부 1.0 (비교모델 load_consensus_rows 의 임계값을 통과시키기 위한 자리)
        source 는 "provenance"

검사 (하나라도 걸리면 파일을 쓰지 않고 멈춘다)
    · dataset_unlabeled.csv 의 클립 전부가 provenance 에 있어야 한다(빠지면 '천장'이 아니다)
    · 라벨은 9-class(others 포함) 안이어야 한다
    · provenance 쪽에 같은 파일명이 두 번 있으면(라벨이 다르면) 멈춘다

부가
    consensus_labels.csv 가 같은 폴더에 있으면 진짜 라벨과 대조해 라벨 정확도·others 오염을 함께 찍는다
    (2026-09-06 실측: 채택 1,917 중 타겟 8종 정확도 0.9462, others 222개 중 150개 오염 = 7.8%).

예)  python model/make_ceiling_labels.py
     python model/make_ceiling_labels.py --provenance /path/to/data_provenance.xlsx --out ceiling_labels.csv
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# vild_config 를 import 하면 torch·sentence_transformers 가 딸려 온다(로컬에서 1분 이상).
# 라벨 이름만 필요하므로 여기 적어 두고 vild_config.py:45-49 와 같은지 실행 때 대조한다.
CLASS_NAMES = ["heavy_impact", "dragging", "construction", "machine_noise",
               "media_talking", "water_toilet", "water_shower", "dog_bark", "others"]


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_class_names_against_config():
    """vild_config.py 의 classes 정의를 텍스트로 읽어 위 상수와 같은지 본다(import 없이)."""
    cfg = os.path.join(PROJECT_ROOT, "vild", "vild_config.py")
    try:
        txt = open(cfg, encoding="utf-8").read()
    except OSError:
        return
    missing = [c for c in CLASS_NAMES if f'"{c}"' not in txt and f"'{c}'" not in txt]
    if missing:
        raise SystemExit(f"[ERROR] vild_config.py 에 없는 클래스 이름이 있습니다: {missing} — CLASS_NAMES 를 맞추십시오.")


def read_provenance(xlsx_path, mark_version):
    """provenance 에서 (파일명 -> 진짜 라벨). openpyxl 로 읽는다."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("[ERROR] openpyxl 이 없습니다. codes/.venv 로 실행하거나 pip install openpyxl 을 하십시오.")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    need = ["local_filename", "target_class", "assigned_split", "mark_version"]
    idx = {}
    for k in need:
        if k not in header:
            raise SystemExit(f"[ERROR] provenance 에 '{k}' 열이 없습니다. 열: {header[:12]} ...")
        idx[k] = header.index(k)
    src_i = header.index("source") if "source" in header else None

    label_of, source_of, dup = {}, {}, []
    n_rows = 0
    for r in rows:
        if r is None or len(r) <= max(idx.values()):
            continue
        if str(r[idx["mark_version"]] or "").strip() != mark_version:
            continue
        if str(r[idx["assigned_split"]] or "").strip() != "unlabeled":
            continue
        n_rows += 1
        fname = str(r[idx["local_filename"]] or "").strip()
        tc = str(r[idx["target_class"]] or "").strip()
        label = tc.split(":", 1)[1] if ":" in tc else tc
        if fname in label_of and label_of[fname] != label:
            dup.append((fname, label_of[fname], label))
        label_of[fname] = label
        if src_i is not None:
            source_of[fname] = str(r[src_i] or "").strip()
    if dup:
        raise SystemExit(f"[ERROR] provenance 에 같은 파일명이 다른 라벨로 두 번 있습니다(예: {dup[0]}). {len(dup)}건")
    return label_of, source_of, n_rows


def main():
    ap = argparse.ArgumentParser(description="unlabeled 2,000개에 진짜 라벨(provenance)을 붙인 천장용 CSV 를 만든다")
    ap.add_argument("--mark_version", type=str, default="mark5.0")
    ap.add_argument("--provenance", type=str, default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--unlabeled_csv", type=str, default=os.path.join(BASE_DIR, "dataset_unlabeled.csv"))
    ap.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "ceiling_labels.csv"))
    ap.add_argument("--consensus_csv", type=str, default=os.path.join(PROJECT_ROOT, "consensus_labels.csv"),
                    help="있으면 진짜 라벨과 대조해 정확도·others 오염을 찍는다(없어도 됨)")
    args = ap.parse_args()
    _check_class_names_against_config()

    prov = os.path.abspath(args.provenance)
    if not os.path.isfile(prov):
        raise SystemExit(f"[ERROR] provenance 파일이 없습니다: {prov}")
    if not os.path.isfile(args.unlabeled_csv):
        raise SystemExit(f"[ERROR] unlabeled 인덱스가 없습니다: {args.unlabeled_csv}")

    label_of, source_of, n_prov = read_provenance(prov, args.mark_version)
    print(f"[INFO] provenance: {prov}")
    print(f"[INFO]   {args.mark_version} · unlabeled 행 {n_prov}개 · 파일명 {len(label_of)}개")

    with open(args.unlabeled_csv, newline="", encoding="utf-8-sig") as f:
        unl = list(csv.DictReader(f))
    if not unl or "path" not in unl[0]:
        raise SystemExit(f"[ERROR] unlabeled 인덱스 형식이 다릅니다(path 열 필요): {args.unlabeled_csv}")

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    out_rows, missing, bad = [], [], []
    for r in unl:
        fname = os.path.basename(str(r["path"]).replace("\\", "/"))
        label = label_of.get(fname)
        if label is None:
            missing.append(fname)
            continue
        if label not in CLASS_NAMES:
            bad.append((fname, label))
            continue
        out_rows.append({"path": r["path"], "label": label, "confidence": "1.0",
                         "source": "provenance", "generated_at": generated_at})
    if missing:
        raise SystemExit(f"[ERROR] provenance 에 없는 unlabeled 클립 {len(missing)}개(예: {missing[:3]}). "
                         "전부 있어야 천장이 됩니다. 파일을 쓰지 않습니다.")
    if bad:
        raise SystemExit(f"[ERROR] 9-class 밖 라벨 {len(bad)}개(예: {bad[:3]}). 파일을 쓰지 않습니다.")
    if len(out_rows) != len(unl):
        raise SystemExit(f"[ERROR] 출력 {len(out_rows)}행 != 인덱스 {len(unl)}행")

    dist = collections.Counter(r["label"] for r in out_rows)
    src_dist = collections.Counter(source_of.get(os.path.basename(r["path"]), "?") for r in out_rows)

    # ---- (부가) 합의 라벨과 대조 ----
    compare = None
    cpath = args.consensus_csv
    if cpath and os.path.isfile(cpath):
        with open(cpath, newline="", encoding="utf-8-sig") as f:
            cons = {os.path.basename(r["path"]): r["label"] for r in csv.DictReader(f)}
        truth = {os.path.basename(r["path"]): r["label"] for r in out_rows}
        tgt = [k for k in truth if truth[k] != "others"]
        oth = [k for k in truth if truth[k] == "others"]
        kept_tgt = [k for k in tgt if k in cons]
        correct = sum(1 for k in kept_tgt if cons[k] == truth[k])
        contaminated = sum(1 for k in oth if k in cons)
        confusion = collections.Counter((truth[k], cons[k]) for k in cons if k in truth)
        compare = {
            "consensus_csv": os.path.relpath(cpath, PROJECT_ROOT), "n_consensus_kept": len(cons),
            "n_target_clips": len(tgt), "n_target_kept": len(kept_tgt), "n_target_correct": correct,
            "target_label_accuracy": round(correct / max(1, len(kept_tgt)), 4),
            "n_others_clips": len(oth), "n_others_kept_as_target": contaminated,
            "others_contamination_ratio_of_kept": round(contaminated / max(1, len(cons)), 4),
            "confusion_true_to_consensus": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items()) if a != b},
        }
        print(f"[INFO] 합의 라벨 대조({os.path.basename(cpath)}): 채택 {len(cons)} · 타겟 정확도 {compare['target_label_accuracy']} "
              f"· others {len(oth)}개 중 {contaminated}개가 타겟 라벨을 받음({compare['others_contamination_ratio_of_kept'] * 100:.1f}%)")

    stats = {
        "generated_at": generated_at, "mark_version": args.mark_version,
        "definition": "upper bound (Cances et al. 2022): withheld true labels restored to all unlabeled clips; "
                      "used as supervised training data, no teacher",
        "provenance": {"path": os.path.relpath(prov, PROJECT_ROOT), "bytes": os.path.getsize(prov),
                       "sha1": _sha1(prov), "rows_matched": n_prov},
        "unlabeled_csv": os.path.relpath(args.unlabeled_csv, PROJECT_ROOT),
        "n_rows": len(out_rows), "class_distribution": {c: int(dist.get(c, 0)) for c in CLASS_NAMES},
        "source_distribution": dict(src_dist),
        "consensus_comparison": compare,
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    stats_path = os.path.join(PROJECT_ROOT, "plots", "ceiling_labels_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "confidence", "source", "generated_at"])
        w.writeheader()
        w.writerows(out_rows)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f" 천장용 라벨: {len(out_rows)}개 (unlabeled 인덱스 {len(unl)}개 전부)")
    print(f" 클래스 분포: {stats['class_distribution']}")
    print(f" 출처 분포  : {stats['source_distribution']}")
    print(f" 저장: {out_path}\n       {stats_path}")
    print(" 비교모델 세 폴더에 복사하십시오:  cp ceiling_labels.csv ../Baseline_*/")
    print("=" * 70)


if __name__ == "__main__":
    main()
