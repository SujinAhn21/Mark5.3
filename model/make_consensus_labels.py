#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""합의 라벨 오프라인 생성 — unlabeled 2,000개 클립에 teacher 8명의 합의를 한 번 매겨 CSV 로 고정한다.

[신설 2026-09-05 / 개정 2026-09-05 저녁, code-reviewer 지적 반영] 비교모델 확장 실험(B 블록)용.
근거 문서: Downloads/Mark5_비교실험_조건표_20260905.pdf 4절.

Mark5 는 학습 중 매 epoch, 증강된 세그먼트를 teacher 8명이 보고 합의 라벨을 다시 만든다
(train_mark5.py:746-752). 비교모델(PANNs·EfficientAT·CED-Tiny)의 학습 루프에 그것을 그대로
넣으려면 그쪽 코드를 통째로 고쳐야 하므로, 대신 **증강 없는 원본**에 한 번만 매겨 CSV 로
고정하고 비교모델은 그 CSV 를 labeled 2,000개와 합쳐 지도학습하듯 돈다.

세그먼트 단위 규칙은 Mark5 와 같다
  - teacher 8개 · 합의 조립(담당 칸 = margin, others 칸 = -max(margin), teacher_fusion.py:150-200)
  - 세그먼트 확신도 = 1 - uncertainty = 전문가 8명의 양성 확률 최댓값 (teacher_fusion.py:211)
  - 세그먼트 채택 = 확신도 >= 0.5 (train_mark5.py:614, 748). 이 규칙에서는 others 가 argmax 이면
    확신도가 반드시 0.5 미만이라 채택 세그먼트의 argmax 는 항상 타겟 8종이다(해석적으로 증명됨:
    others 칸 = -M, 확신도 = sigmoid(M) 이므로 others 가 이기려면 M < 0 → 확신도 < 0.5).
클립 단위 규칙 (이 스크립트가 새로 정한 것 — 비교모델은 3초 클립 전체를 한 번에 받는다)
  - 클립 채택 = 세그먼트 5개 중 채택 세그먼트가 --min_segments(기본 3, 과반) 개 이상
  - 클립 라벨 = 채택 세그먼트들의 argmax **다수결** (동점이면 확신도 합이 큰 쪽). 채택 세그먼트의
    argmax 는 타겟 8종뿐이므로 others 는 구조상 나오지 않는다(나오면 [ERROR] 로 멈춤).
  - 클립 확신도 = 채택 세그먼트 확신도의 평균 (>= 0.5 가 보장됨)
  - vote_agreement = 다수결 표 / 채택 세그먼트 수 (1.0 이면 채택 세그먼트가 전부 같은 라벨)
  ※ 처음 판(같은 날 오전)은 eval.py 의 aggregate_segment_probs(가중치 = softmax 최댓값² × saliency)로
    확률을 합치고 그 argmax 를 라벨, 같은 가중치의 확신도 평균을 채택 기준으로 썼다. code-reviewer 가
    (1) 라벨과 채택이 서로 다른 양 위에 있어 others 가 채택될 수 있고, (2) 그 가중치는 student 확률용이라
    teacher 융합 로짓에서는 "아무도 아님"이 확실한 세그먼트를 오히려 더 무겁게 친다는 것을 반례로
    보였다. 그래서 세그먼트 채택 규칙(Mark5 와 동일) 위에 다수결을 얹는 위 규칙으로 바꿨다.
Mark5 와 다른 것 (논문에 그대로 밝힌다)
  - 증강 view 마다 재계산하지 않는다(원본 1회) · 확신도 가중 없음(hard label) · feature KD 없음
  - Mark5 는 세그먼트 단위로 선별·학습하고 여기서는 클립 단위(위 규칙)로 선별한다.
  - 분포 정렬은 끈다(2026-08-23 부터 꺼져 있고, EMA 상태가 순서에 따라 달라져 오프라인 생성에 맞지 않음).
  - 채택률은 Mark5 학습 로그의 세그먼트 채택률(증강 view, 약 0.92)과 직접 비교할 수 없다
    (단위와 조건이 다르다). stats 에 세그먼트·클립 채택률을 따로 적는다.

결정성: fusion 에 학습 파라미터가 없고(optimizer 에는 student 만 들어감, train_mark5.py:633) 파서·teacher
에 난수가 없으며 증강도 없으므로 결과는 seed 와 무관하다. cudnn deterministic 을 켜 GPU 끝자리도 고정한다.
CSV 하나를 비교모델 15회(3모델 × 5 seed)가 공유한다.

산출물 (--out 기본 = 프로젝트 루트/consensus_labels.csv, 보조 파일은 plots/. --limit 을 주면 이름에 _limitN 이 붙어
기본 산출물을 덮어쓰지 않는다)
    consensus_labels.csv               채택 클립: path, label, confidence, n_segments, n_segments_kept, vote_agreement, generated_at
    plots/consensus_labels_all.csv     2,000개 전부: keep · note(kept/few_segments/file_missing/parse_fail) · 클립 통계 · 클래스별 평균 확률
    plots/consensus_segments.csv       세그먼트 10,000행 원자료(확신도·argmax·확률 9개) — 다른 규칙·임계값을 CSV 계산만으로 비교하기 위함
    plots/consensus_labels_stats.json  채택 수·클래스 분포·확신도 분포·teacher 체크포인트 신원·집계 규칙

예)  python model/make_consensus_labels.py --mark_version mark5.0
     python model/make_consensus_labels.py --mark_version mark5.0 --limit 3      # 동작 확인용(산출물 이름에 _limit3)
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"),
          os.path.join(PROJECT_ROOT, "shared_vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.append(p)

from train_mark5 import EnsembleTeacher, SPECIALIST_CONFIG, _resolve_csv_path, _resolve_resource_path  # noqa: E402
from vild_config import AudioViLDConfig                       # noqa: E402
from vild_parser_teacher import AudioParser                   # noqa: E402
from seed_utils import set_seed                               # noqa: E402

_SOURCE_ALIAS = {"data_labeled": "data_source_labeled", "data_unlabeled": "data_source_unlabeled",
                 "data_val": "data_source_val", "data_test": "data_source_test"}


def _localize(path):
    """CSV 의 코랩 드라이브 경로를 지금 환경의 파일로. 코랩에서는 경로가 그대로 있어 그대로 쓰고,
    로컬에서는 'Mark5_EXP/' 뒤의 꼬리를 프로젝트 루트에서 찾는다(전처리 전 폴더 data_source_* 도 시도).
    비교모델 폴더의 common/paths.py 와 같은 규칙. 못 찾으면 None."""
    p = str(path).replace("\\", "/")
    if os.path.isfile(p):
        return p
    m = re.search(r"Mark5_EXP[\\/](.+)$", p)
    parts = [x for x in (m.group(1) if m else os.path.basename(p)).split("/") if x]
    cands = [os.path.join(PROJECT_ROOT, *parts)]
    if parts and parts[0] in _SOURCE_ALIAS:
        cands.append(os.path.join(PROJECT_ROOT, _SOURCE_ALIAS[parts[0]], *parts[1:]))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _file_identity(path):
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": os.path.relpath(path, PROJECT_ROOT), "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), "sha1": h.hexdigest()}


def _with_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def main():
    ap = argparse.ArgumentParser(description="unlabeled 클립에 teacher 8명의 합의 라벨을 오프라인으로 매긴다")
    ap.add_argument("--mark_version", type=str, default="mark5.0")
    ap.add_argument("--min_conf", type=float, default=None, help="세그먼트 채택 임계값(기본: config.pseudo_label_min_confidence=0.5)")
    ap.add_argument("--min_segments", type=int, default=3, help="클립 채택에 필요한 채택 세그먼트 수(5개 중, 기본 3=과반)")
    ap.add_argument("--clips_per_batch", type=int, default=8, help="teacher 한 번 호출에 묶는 클립 수(세그먼트 5개씩)")
    ap.add_argument("--limit", type=int, default=0, help="동작 확인용: 앞에서 N 개만. 산출물 이름에 _limitN 이 붙는다")
    ap.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "consensus_labels.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    set_seed(args.seed, deterministic=bool(getattr(config, "cudnn_deterministic", True)))
    device = config.device
    min_conf = args.min_conf if args.min_conf is not None else float(getattr(config, "pseudo_label_min_confidence", 0.5))
    others_rule = getattr(config, "fusion_others_rule", "min")
    score_mode = getattr(config, "fusion_score_mode", "margin")
    if getattr(config, "use_distribution_alignment", False):
        print("[WARN] config.use_distribution_alignment=True 이지만 오프라인 생성에서는 끕니다(EMA 상태가 순서에 의존).")
    class_names = config.get_classes_for_evaluation()
    label_map = config.get_target_label_map()
    others_idx = class_names.index("others")
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] 합의 라벨 생성: mark_version={args.mark_version}, device={device}, 세그먼트 채택 임계값={min_conf}, "
          f"클립 채택 = 채택 세그먼트 >= {args.min_segments}/5, 라벨 = 채택 세그먼트 다수결, 분포 정렬=off, 증강=없음")
    if score_mode == "margin":
        print("[INFO] teacher fusion: 담당 칸 = margin(target-others), others 칸 = -max(margin)  (others_rule 은 이 모드에서 미사용)")
    else:
        print(f"[INFO] teacher fusion: 담당 칸 = raw logit, others 칸 조립 규칙 = {others_rule}")

    # ---- 산출물 경로 (동작 확인 실행이 진짜 산출물을 덮어쓰지 않게) ----
    out_path = os.path.abspath(args.out)
    default_out = os.path.abspath(os.path.join(PROJECT_ROOT, "consensus_labels.csv"))
    aux_dir = os.path.join(PROJECT_ROOT, "plots") if out_path == default_out else os.path.dirname(out_path)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = _with_suffix(out_path, suffix)
    all_path = os.path.join(aux_dir, f"consensus_labels_all{suffix}.csv")
    seg_path = os.path.join(aux_dir, f"consensus_segments{suffix}.csv")
    stats_path = os.path.join(aux_dir, f"consensus_labels_stats{suffix}.json")

    # ---- teacher 8개 (train_mark5.py 와 같은 목록·같은 설정) + 신원 기록 ----
    teacher_files = []
    for name, spec in SPECIALIST_CONFIG.items():
        for key in ("encoder_path", "classifier_path"):
            teacher_files.append({"specialist": name, **_file_identity(_resolve_resource_path(spec[key]))})
    parser = AudioParser(config, segment_mode=True)
    teacher = EnsembleTeacher(SPECIALIST_CONFIG, device, others_rule=others_rule, score_mode=score_mode,
                              use_distribution_alignment=False)

    csv_path = _resolve_csv_path("dataset_unlabeled.csv")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
        print(f"[WARN] --limit {args.limit}: 동작 확인용 축소 실행입니다. 산출물 이름에 '{suffix}' 가 붙습니다.")
    print(f"[INFO] unlabeled 인덱스: {csv_path} ({len(rows)}개)")

    # ---- 클립을 묶어 teacher 에 통과 ----
    all_rows, kept_rows, seg_rows = [], [], []
    n_missing, n_parse_fail = 0, 0
    seg_total, seg_kept_total = 0, 0
    t0 = time.time()
    pending = []                                             # (row, records)

    def flush(pending):
        nonlocal seg_total, seg_kept_total
        if not pending:
            return
        batch = torch.cat([torch.stack([r["tensor"] for r in recs], dim=0) for _, recs in pending], dim=0).to(device)
        with torch.no_grad():
            out = teacher(batch, label_map)
        probs_all = torch.softmax(out.logits, dim=1).cpu().numpy()                  # [sumK, 9]
        conf_all = (1.0 - out.uncertainty).detach().cpu().numpy().astype(np.float64)
        pos = 0
        for row, recs in pending:
            K = len(recs)
            probs, conf = probs_all[pos:pos + K], conf_all[pos:pos + K]
            pos += K
            path = row["path"]
            seg_argmax = probs.argmax(axis=1)
            kept = conf >= min_conf
            n_kept = int(kept.sum())
            seg_total += K
            seg_kept_total += n_kept
            for i, r in enumerate(recs):
                seg_rows.append({"path": path, "segment_index": r["segment_index"], "start_frame": r["start_frame"],
                                 "saliency": round(float(r["saliency"]), 6), "confidence": round(float(conf[i]), 6),
                                 "kept": int(kept[i]), "argmax": class_names[int(seg_argmax[i])],
                                 **{f"prob_{c}": round(float(probs[i, j]), 6) for j, c in enumerate(class_names)}})
            mean_prob = probs.mean(axis=0)                     # 참고용(라벨 결정에는 쓰지 않음)
            rec = {"path": path, "keep": 0, "label": "", "confidence": float("nan"), "n_segments": K,
                   "n_segments_kept": n_kept, "vote_agreement": float("nan"),
                   "seg_conf_mean": round(float(conf.mean()), 6), "seg_conf_max": round(float(conf.max()), 6),
                   "note": "few_segments"}
            rec.update({f"meanprob_{c}": round(float(mean_prob[j]), 6) for j, c in enumerate(class_names)})
            if n_kept >= args.min_segments:
                votes = collections.Counter(int(a) for a in seg_argmax[kept])
                conf_sum = collections.defaultdict(float)
                for a, c in zip(seg_argmax[kept], conf[kept]):
                    conf_sum[int(a)] += float(c)
                label_idx = max(votes, key=lambda k: (votes[k], conf_sum[k]))
                if label_idx == others_idx:
                    raise SystemExit(f"[ERROR] 채택 세그먼트의 argmax 가 others 입니다({path}) — fusion 규칙이 바뀌었는지 확인하십시오.")
                rec.update({"keep": 1, "label": class_names[label_idx],
                            "confidence": round(float(conf[kept].mean()), 6),
                            "vote_agreement": round(votes[label_idx] / n_kept, 4), "note": "kept"})
                kept_rows.append({"path": path, "label": rec["label"], "confidence": rec["confidence"],
                                  "n_segments": K, "n_segments_kept": n_kept,
                                  "vote_agreement": rec["vote_agreement"], "generated_at": generated_at})
            all_rows.append(rec)

    for i, row in enumerate(rows):
        path = row["path"]
        local = _localize(path)
        if local is None:
            n_missing += 1
            all_rows.append({"path": path, "keep": 0, "note": "file_missing"})
            continue
        records = parser.load_and_segment_with_metadata(local)
        if not records:
            n_parse_fail += 1
            all_rows.append({"path": path, "keep": 0, "note": "parse_fail"})
            continue
        pending.append((row, records))
        if len(pending) >= args.clips_per_batch:
            flush(pending)
            pending = []
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"[INFO] {i + 1}/{len(rows)}  채택 {len(kept_rows)}  ({time.time() - t0:.0f}s)", flush=True)
    flush(pending)

    # ---- 실패 방어: 절반 이상 못 읽었거나 채택이 0이면 산출물을 쓰지 않고 멈춘다 ----
    n_bad = n_missing + n_parse_fail
    if n_bad and n_bad * 2 >= len(rows):      # 절반 이상(정수 나눗셈 함정 피함)
        raise SystemExit(f"[ERROR] {len(rows)}개 중 {n_bad}개를 못 읽었습니다(파일 없음 {n_missing}, 파싱 실패 {n_parse_fail}). "
                         "드라이브 마운트·데이터 폴더를 확인하십시오. 산출물을 쓰지 않습니다.")
    if not kept_rows:
        raise SystemExit("[ERROR] 채택된 클립이 0개입니다. 임계값·teacher 를 확인하십시오. 산출물을 쓰지 않습니다.")

    # ---- 통계 ----
    kept_dist = collections.Counter(r["label"] for r in kept_rows)
    few = [r for r in all_rows if r.get("note") == "few_segments"]
    few_dist = collections.Counter(int(np.argmax([r[f"meanprob_{c}"] for c in class_names])) for r in few)
    confs = np.array([r["confidence"] for r in all_rows if r.get("note") in ("kept", "few_segments") and not np.isnan(r["confidence"])])
    seg_conf = np.array([r["confidence"] for r in seg_rows], dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    seg_hist, _ = np.histogram(seg_conf, bins=edges) if len(seg_conf) else (np.zeros(10, dtype=int), None)
    n_read = len(rows) - n_bad
    stats = {
        "generated_at": generated_at, "mark_version": args.mark_version,
        "rule": {"segment_min_conf": min_conf, "clip_min_segments": args.min_segments,
                 "clip_label": "majority vote of argmax over kept segments (tie -> larger confidence sum)",
                 "clip_confidence": "mean confidence of kept segments",
                 "score_mode": score_mode, "others_rule": others_rule if score_mode != "margin" else "n/a (margin mode)",
                 "distribution_alignment": False, "augmentation": None,
                 "segment_selection_mode": config.segment_selection_mode, "max_segments": getattr(config, "max_segments", 5)},
        "n_unlabeled": len(rows), "n_file_missing": n_missing, "n_parse_fail": n_parse_fail, "n_read": n_read,
        "n_kept": len(kept_rows), "keep_ratio_clip": round(len(kept_rows) / max(1, n_read), 4),
        "keep_ratio_segment": round(seg_kept_total / max(1, seg_total), 4),
        "keep_ratio_note": "Mark5 학습 로그의 채택률(증강 view, 세그먼트 단위, 약 0.92)과 직접 비교 불가 — 여기는 원본·클립 단위",
        "kept_class_distribution": {c: int(kept_dist.get(c, 0)) for c in class_names},
        "few_segments_meanprob_argmax_distribution": {c: int(few_dist.get(i, 0)) for i, c in enumerate(class_names)},
        "n_segments_kept_histogram": {str(k): int(v) for k, v in sorted(collections.Counter(r["n_segments_kept"] for r in all_rows if "n_segments_kept" in r).items())},
        "vote_agreement_mean_kept": round(float(np.mean([r["vote_agreement"] for r in kept_rows])), 4),
        "confidence_mean_kept": round(float(np.mean([r["confidence"] for r in kept_rows])), 4),
        "segment_confidence_histogram_0.1bins": {f"{edges[k]:.1f}-{edges[k + 1]:.1f}": int(seg_hist[k]) for k in range(10)},
        "teacher_checkpoints": teacher_files,
        "limit": args.limit, "clips_per_batch": args.clips_per_batch,
        "elapsed_sec_excluding_teacher_load": round(time.time() - t0, 1),
    }

    # ---- 저장 ----
    os.makedirs(aux_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "confidence", "n_segments", "n_segments_kept", "vote_agreement", "generated_at"])
        w.writeheader()
        w.writerows(kept_rows)
    all_fields = ["path", "keep", "note", "label", "confidence", "n_segments", "n_segments_kept", "vote_agreement",
                  "seg_conf_mean", "seg_conf_max"] + [f"meanprob_{c}" for c in class_names]
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in all_fields})
    with open(seg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()))
        w.writeheader()
        w.writerows(seg_rows)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f" 합의 라벨: {len(kept_rows)}/{n_read} 클립 채택 (클립 채택률 {stats['keep_ratio_clip']}, 세그먼트 채택률 {stats['keep_ratio_segment']}), "
          f"파일 없음 {n_missing}, 파싱 실패 {n_parse_fail}")
    print(f" 채택 클래스 분포: {stats['kept_class_distribution']}")
    print(f" 채택 세그먼트 수 분포(클립당): {stats['n_segments_kept_histogram']}   다수결 일치율 평균 {stats['vote_agreement_mean_kept']}")
    print(f" 세그먼트 확신도 분포: {stats['segment_confidence_histogram_0.1bins']}")
    print(f" 저장: {out_path}\n       {all_path}\n       {seg_path}\n       {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
