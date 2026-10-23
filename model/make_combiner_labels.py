#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""검증 결합기 가짜 정답 — 교사 점수 캐시로 9클래스 결합기를 회차의 정답 라벨만으로 학습하고, 라벨 없는 클립마다
한 라벨 · 두 후보 라벨 · 침묵을 정한다.

[신설 2026-10-22 / 검증 결합기] 설계 문서 '검증 결합기와 세 단계 전달'.

    python model/make_combiner_labels.py --scores plots/teacher_scores_mn10.csv --labeled_ns 200 500 1000 0 --seeds 1 2 3 4 5

결합기
    입력  교사 8명의 클립 점수 p 를 로짓 log(p/(1-p)) 로(p 는 1e-4 ~ 1-1e-4 로 자른다) → 8개 값
    모델  9클래스 다항 로지스틱 회귀, L2(C=1.0) — 가중치 72 + 절편 9 = 81개
    학습  그 회차의 학생이 쓰는 정답 라벨만: train_mark5.select_labeled_subset(같은 seed·n) — 0 은 전량(2,000)
세 단계 (라벨 없는 클립마다, 결합기 확률 q)
    한 라벨     max q ≥ τ1(0.9)                         → label = argmax
    두 후보     max q < τ1 이고 1등+2등 ≥ τ2(0.95)        → label = 1등, label2 = 2등
    침묵        그 밖                                     → 파일에 넣지 않는다
산출물 (프로젝트 루트, 실행기의 PSEUDO_CSV_FMT 규칙)
    pseudo_labels_cmb_l<n|all>_s<seed>.csv     Mark5.3 용: path,label,confidence,label2,mode(one|two)
    pseudo_labels_cmbone_l<n|all>_s<seed>.csv  비교모델 용: 한 라벨 행만 path,label,confidence(표준 가짜 정답 방식)
    pseudo_labels_fix_l<n|all>_s<seed>.csv     아블레이션 '고정 규칙'(결합기 없음): 1등 ≥ 0.5 · 1·2등 차이 ≥ 0.4 → 한 라벨,
                                              전원 < 0.2 → others (10-22 까지의 Mark5.3 규칙). 정답 라벨을 쓰지 않으므로 n·seed 와 무관하나
                                              실행기 이름 규칙에 맞춰 회차마다 같은 내용으로 쓴다.
    plots/combiner/combiner_l<n|all>_s<seed>.json  결합기 계수 · τ · 단계별·클래스별 개수 · 쓴 정답 라벨 목록 해시 · 점수 캐시 해시
                                                  · 위 CSV 3개의 해시(outputs) — 학습·비교모델이 model/label_freshness.py 로 대조한다
시작 전에 점수 캐시가 지금 resources/ 교사의 최신 판인지 대조한다(label_freshness.check_scores, 어긋나면 멈춤).
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"),
          os.path.join(PROJECT_ROOT, "shared_vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.append(p)

TAU1, TAU2 = 0.9, 0.95
FIX_MIN_CONF, FIX_MIN_GAP, FIX_OTHERS_TAU = 0.5, 0.4, 0.2
CLIP_EPS = 1e-4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_scores(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    experts = [k[2:] for k in rows[0] if k.startswith("p_")]
    by_split = collections.defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    return experts, by_split


def to_features(rows, experts):
    p = np.array([[float(r["p_" + e]) for e in experts] for r in rows], dtype=np.float64)
    p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
    return p, np.log(p / (1 - p))


def fit_combiner(x, y, classes):
    from sklearn.linear_model import LogisticRegression
    missing = sorted(set(range(len(classes))) - set(y.tolist()))
    if missing:
        raise RuntimeError(f"[ERROR] 결합기 학습 자료에 없는 클래스가 있습니다: {[classes[i] for i in missing]}")
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(x, y)
    if list(clf.classes_) != list(range(len(classes))):
        raise RuntimeError("[ERROR] 결합기 클래스 순서가 어긋났습니다.")
    return clf


def three_steps(q, tau1=TAU1, tau2=TAU2):
    order = np.argsort(-q, axis=1, kind="stable")
    q1 = q[np.arange(len(q)), order[:, 0]]
    q2 = q[np.arange(len(q)), order[:, 1]]
    one = q1 >= tau1
    two = (~one) & (q1 + q2 >= tau2)
    return order, q1, q2, one, two


def fixed_rule(p, experts, classes):
    """10-22 까지의 Mark5.3 규칙(make_consensus_labels.py --clip_rule mean --others_from_reject 0.2 와 같은 판정)."""
    order = np.argsort(-p, axis=1, kind="stable")
    top = p[np.arange(len(p)), order[:, 0]]
    gap = top - p[np.arange(len(p)), order[:, 1]]
    keep = (top >= FIX_MIN_CONF) & (gap >= FIX_MIN_GAP)
    rej = (~keep) & (top < FIX_OTHERS_TAU)
    out = []
    for i in range(len(p)):
        if keep[i]:
            out.append((classes.index(experts[order[i, 0]]), float(top[i])))
        elif rej[i]:
            out.append((classes.index("others"), float(1.0 - top[i])))
        else:
            out.append(None)
    return out


def write_csv(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="검증 결합기로 회차별 가짜 정답(한 라벨·두 후보·침묵)을 만든다")
    ap.add_argument("--scores", required=True, help="make_teacher_scores.py 산출 CSV")
    ap.add_argument("--labeled_ns", type=int, nargs="+", default=[200, 500, 1000, 0], help="0 = 정답 라벨 전량(2,000)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--tau1", type=float, default=TAU1)
    ap.add_argument("--tau2", type=float, default=TAU2)
    ap.add_argument("--out_root", default=PROJECT_ROOT, help="산출물 기준 폴더(기본 프로젝트 루트). 시험할 때만 바꾼다")
    args = ap.parse_args()

    from train_mark5 import select_labeled_subset
    from vild_config import AudioViLDConfig
    classes = list(AudioViLDConfig(mark_version="mark5.2").get_classes_for_evaluation())
    scores_path = args.scores if os.path.isabs(args.scores) else os.path.join(PROJECT_ROOT, args.scores)
    from label_freshness import check_scores
    check_scores(scores_path)          # [2026-10-22] 옛 교사의 점수 캐시면 여기서 멈춘다
    experts, by_split = load_scores(scores_path)
    if not by_split.get("labeled") or not by_split.get("unlabeled"):
        raise SystemExit("[ERROR] 점수 캐시에 labeled·unlabeled split 이 모두 있어야 합니다.")
    scores_sha = sha256_file(scores_path)
    lab_rows = by_split["labeled"]
    unl_rows = by_split["unlabeled"]
    p_u, x_u = to_features(unl_rows, experts)
    fix = fixed_rule(p_u, experts, classes)
    out_root = os.path.abspath(args.out_root)
    out_dir = os.path.join(out_root, "plots", "combiner")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lab_by_path = {r["path"]: r for r in lab_rows}

    for n in args.labeled_ns:
        for seed in args.seeds:
            lname = "all" if n == 0 else str(n)
            if n == 0:
                chosen = [{"path": r["path"], "label": r["label"]} for r in lab_rows]
            else:
                chosen, _ = select_labeled_subset([{"path": r["path"], "label": r["label"]} for r in lab_rows],
                                                  n, seed, classes)
            rows = [lab_by_path[r["path"]] for r in chosen]
            _, x_l = to_features(rows, experts)
            y_l = np.array([classes.index(r["label"]) for r in rows])
            clf = fit_combiner(x_l, y_l, classes)
            q = clf.predict_proba(x_u)
            order, q1, q2, one, two = three_steps(q, args.tau1, args.tau2)

            full, single, fixed = [], [], []
            for i, r in enumerate(unl_rows):
                if one[i]:
                    rec = {"path": r["path"], "label": classes[order[i, 0]], "confidence": round(float(q1[i]), 6),
                           "label2": "", "mode": "one"}
                    full.append(rec)
                    single.append({k: rec[k] for k in ("path", "label", "confidence")})
                elif two[i]:
                    full.append({"path": r["path"], "label": classes[order[i, 0]], "label2": classes[order[i, 1]],
                                 "confidence": round(float(q1[i] + q2[i]), 6), "mode": "two"})
                if fix[i] is not None:
                    fixed.append({"path": r["path"], "label": classes[fix[i][0]], "confidence": round(fix[i][1], 6)})
            names = {k: f"pseudo_labels_{k}_l{lname}_s{seed}.csv" for k in ("cmb", "cmbone", "fix")}
            write_csv(os.path.join(out_root, names["cmb"]), ["path", "label", "confidence", "label2", "mode"], full)
            write_csv(os.path.join(out_root, names["cmbone"]), ["path", "label", "confidence"], single)
            write_csv(os.path.join(out_root, names["fix"]), ["path", "label", "confidence"], fixed)
            meta = {
                "labeled_n": n or len(lab_rows), "seed": seed, "tau1": args.tau1, "tau2": args.tau2,
                "experts": experts, "classes": classes, "C": 1.0, "clip_eps": CLIP_EPS,
                "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                "labeled_paths_sha256": hashlib.sha256("\n".join(sorted(r["path"] for r in rows)).encode()).hexdigest(),
                "scores_csv": os.path.relpath(scores_path, out_root), "scores_sha256": scores_sha,
                "outputs": {nm: sha256_file(os.path.join(out_root, nm)) for nm in names.values()},
                "counts": {"unlabeled": len(unl_rows), "one": int(one.sum()), "two": int(two.sum()),
                           "silent": int(len(unl_rows) - one.sum() - two.sum()), "fixed_rule": len(fixed)},
                "one_by_class": dict(collections.Counter(classes[order[i, 0]] for i in range(len(unl_rows)) if one[i])),
                "generated_at": stamp,
            }
            with open(os.path.join(out_dir, f"combiner_l{lname}_s{seed}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[결합기] 정답 {meta['labeled_n']:>4} · seed {seed}: 한 라벨 {meta['counts']['one']} · 두 후보 {meta['counts']['two']} · "
                  f"침묵 {meta['counts']['silent']} (고정 규칙 {len(fixed)})", flush=True)
    print(f"[완료] {len(args.labeled_ns) * len(args.seeds)}회차 · 점수 캐시 {os.path.basename(scores_path)} ({scores_sha[:12]})")


if __name__ == "__main__":
    main()
