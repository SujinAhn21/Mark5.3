#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""교사 채택 관문 — mn10 교사(B+)를 기준 교사(조각 교사)와 다중분류 지표로 비교해 채택 여부를 판정한다.

[신설 2026-10-23] 설계 문서 '채택 관문'. 교사를 학습·반입하고 점수 캐시를 만든 뒤, Mark5.3 을 학습하기 전에 돌린다.

    python model/teacher_gate_report.py --candidate plots/teacher_scores_mn10.csv --reference plots/teacher_scores_segment.csv

비교 자료: 정답이 원래 있는 Mark5.3 val 430 과 정답 라벨 2,000(점수 캐시의 val · labeled split). 두 캐시는 같은 클립·같은 정답이어야 한다.
지표 (10-22 다중분류 측정 mc_summary.py 와 같은 규칙)
    9클래스 판정   교사 8명 점수 중 1등이 0.5 이상이면 그 클래스, 모두 0.5 미만이면 others. Acc · Macro P · R · F1(9클래스 평균)
    손들기 행렬    교사 k 가 정답 j 클립에 0.5 이상을 낸 비율. '남의 클래스에 손드는 비율의 평균' = k ≠ j 인 칸(8 × 8, others 포함) 평균
    가짜 정답      정답 라벨 2,000 에서 ① 고정 규칙(make_combiner_labels.fixed_rule) ② 검증 결합기 5겹 교차검증(4겹으로 학습 →
                   나머지 1겹에 τ1·τ2 세 단계 적용, 겹 나눔 시드 0). 채택 수 · 맞은 수 · 정밀도. 두 후보는 정답이 1·2등 안에 든 수.
채택 조건 (설계 문서, 네 개 모두 만족하면 채택 · 하나도 못 넘으면 기준 교사(B0) · 일부만이면 결과를 보고한 뒤 함께 정한다)
    C1 val 430 Macro F1 이 더 높음 · C2 정답 2,000 Macro F1 이 더 높음
    C3 결합기 교차검증 정밀도가 같거나 높으면서 맞은 수가 더 많음 · C4 남의 클래스 손들기 평균(val + 정답 2,000)이 더 낮음
⚠ 기준 교사: 공정한 비교는 같은 새 데이터로 학습한 B0(--recipe run3)다. 3차 교사(10-23 이전 데이터)는 교사 train 이 Mark5.3 정답 라벨과
    같은 녹음을 82~227개 공유했으므로 정답 2,000 지표가 실제보다 좋게 나온다. 설계 문서의 3차 기준값(10-22 측정)은 참고로만 함께 적는다.
산출물: plots/teacher_gate_<후보 종류>_vs_<기준 종류>.json (지표 전부 · 조건별 판정 · 두 캐시의 SHA-256 · 최신 판 확인 여부)
"""
import argparse
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

from make_combiner_labels import (TAU1, TAU2, load_scores, to_features, fit_combiner, three_steps, fixed_rule,  # noqa: E402
                                  sha256_file)

RAISE_TAU = 0.5
CV_FOLDS, CV_SEED = 5, 0
# 설계 문서(10-22 측정)의 3차 교사 값 — 참고로만 적는다(옛 데이터, 위 ⚠).
RUN3_REFERENCE = {"val": [0.8744, 0.8749, 0.8743, 0.8719], "labeled": [0.9170, 0.9171, 0.9170, 0.9138],
                  "fixed_rule_labeled": {"correct": 1400, "adopted": 1453}}


def four(y, pred, labels):
    """Acc · Macro P · R · F1 (labels 전체 평균, 10-22 mc_summary.four 와 같은 식)."""
    y, pred = np.asarray(y), np.asarray(pred)
    ps, rs, fs = [], [], []
    for c in labels:
        tp = int(np.sum((pred == c) & (y == c))); fp = int(np.sum((pred == c) & (y != c))); fn = int(np.sum((pred != c) & (y == c)))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        ps.append(pr); rs.append(rc); fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"Accuracy": float(np.mean(y == pred)), "Macro Precision": float(np.mean(ps)),
            "Macro Recall": float(np.mean(rs)), "Macro F1": float(np.mean(fs))}


def nine_class(p, y, experts, classes):
    pred = np.where(p.max(1) >= RAISE_TAU, np.array(experts, dtype=object)[p.argmax(1)], "others")
    return four(y, pred, classes)


def raise_matrix(p, y, experts, classes):
    """{교사: {정답 클래스: 손든 비율}} 과 남의 클래스 칸 평균."""
    y = np.asarray(y)
    mat = {e: {c: (float(np.mean(p[y == c, k] >= RAISE_TAU)) if np.any(y == c) else None) for c in classes}
           for k, e in enumerate(experts)}
    off = [v for e in experts for c, v in mat[e].items() if c != e and v is not None]
    return mat, float(np.mean(off))


def fixed_counts(p, y, experts, classes):
    out = fixed_rule(p, experts, classes)
    adopted = [(i, lab) for i, o in enumerate(out) if o is not None for lab in [classes[o[0]]]]
    correct = sum(1 for i, lab in adopted if y[i] == lab)
    return {"adopted": len(adopted), "correct": correct, "precision": (correct / len(adopted)) if adopted else None}


def combiner_cv(x, y, classes, tau1, tau2):
    from sklearn.model_selection import StratifiedKFold
    yi = np.array([classes.index(v) for v in y])
    one_a = one_c = two_a = two_c = 0
    for tr, te in StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED).split(x, yi):
        clf = fit_combiner(x[tr], yi[tr], classes)
        order, _, _, one, two = three_steps(clf.predict_proba(x[te]), tau1, tau2)
        one_a += int(one.sum()); one_c += int(np.sum(one & (order[:, 0] == yi[te])))
        hit2 = (order[:, 0] == yi[te]) | (order[:, 1] == yi[te])
        two_a += int(two.sum()); two_c += int(np.sum(two & hit2))
    return {"one": {"adopted": one_a, "correct": one_c, "precision": (one_c / one_a) if one_a else None},
            "two": {"adopted": two_a, "true_in_top2": two_c}, "folds": CV_FOLDS, "fold_seed": CV_SEED}


def measure(path, classes, tau1, tau2, fresh):
    meta = None
    if fresh:
        from label_freshness import check_scores
        meta = check_scores(path)
    experts, by = load_scores(path)
    if experts != [c for c in classes if c != "others"]:
        raise SystemExit(f"[ERROR] {path}: 교사 순서가 9클래스 순서와 다릅니다 {experts}")
    res, sets = {"file": os.path.relpath(path, PROJECT_ROOT), "sha256": sha256_file(path), "freshness_checked": fresh,
                 "teacher_release_sha256": (meta or {}).get("teacher_release_sha256")}, {}
    for split in ("val", "labeled"):
        rows = by.get(split) or []
        if not rows:
            raise SystemExit(f"[ERROR] {path}: {split} split 이 없습니다.")
        p, x = to_features(rows, experts)
        y = [r["label"] for r in rows]
        sets[split] = ({r["path"]: r["label"] for r in rows}, p, y)
        mat, off = raise_matrix(p, y, experts, classes)
        res[split] = {"n": len(rows), "nine_class": nine_class(p, y, experts, classes), "raise": mat, "raise_off_mean": off}
    pl, xl = to_features(by["labeled"], experts)
    yl = [r["label"] for r in by["labeled"]]
    res["labeled"]["fixed_rule"] = fixed_counts(pl, yl, experts, classes)
    res["labeled"]["combiner_cv"] = combiner_cv(xl, yl, classes, tau1, tau2)
    pv, yv = sets["val"][1], sets["val"][2]
    p_all, y_all = np.vstack([pv, pl]), list(yv) + list(yl)
    res["raise_off_mean_val_labeled"] = raise_matrix(p_all, y_all, experts, classes)[1]
    return res, {k: v[0] for k, v in sets.items()}


def kind_of(path):
    b = os.path.splitext(os.path.basename(path))[0]
    return b[len("teacher_scores_"):] if b.startswith("teacher_scores_") else b


def main():
    ap = argparse.ArgumentParser(description="mn10 교사 채택 관문(설계 문서 네 조건)")
    ap.add_argument("--candidate", required=True, help="후보 교사 점수 캐시(보통 plots/teacher_scores_mn10.csv)")
    ap.add_argument("--reference", required=True, help="기준 교사 점수 캐시(B0 권장, 위 ⚠)")
    ap.add_argument("--tau1", type=float, default=TAU1)
    ap.add_argument("--tau2", type=float, default=TAU2)
    ap.add_argument("--allow_stale_reference", action="store_true",
                    help="기준 캐시의 교사가 지금 resources/ 에 없어도 비교한다(예: 3차 교사 캐시를 B0 반입 뒤에). 후보는 늘 확인")
    ap.add_argument("--no_freshness_check", action="store_true", help="시험용: 두 캐시 모두 최신 판 확인을 건너뛴다(json 에 기록)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vild_config import AudioViLDConfig
    classes = list(AudioViLDConfig(mark_version="mark5.2").get_classes_for_evaluation())
    cand_path, ref_path = (p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p) for p in (args.candidate, args.reference))
    cand, cand_sets = measure(cand_path, classes, args.tau1, args.tau2, fresh=not args.no_freshness_check)
    ref, ref_sets = measure(ref_path, classes, args.tau1, args.tau2,
                            fresh=not (args.no_freshness_check or args.allow_stale_reference))
    for split in ("val", "labeled"):
        if cand_sets[split] != ref_sets[split]:
            raise SystemExit(f"[ERROR] 두 캐시의 {split} 클립·정답이 다릅니다 — 같은 Mark5.3 데이터로 만든 캐시끼리만 비교합니다.")

    cc, rc = cand["labeled"]["combiner_cv"]["one"], ref["labeled"]["combiner_cv"]["one"]
    cond = {
        "C1_val_macro_f1": cand["val"]["nine_class"]["Macro F1"] > ref["val"]["nine_class"]["Macro F1"],
        "C2_labeled_macro_f1": cand["labeled"]["nine_class"]["Macro F1"] > ref["labeled"]["nine_class"]["Macro F1"],
        "C3_combiner_cv": (cc["precision"] or 0.0) >= (rc["precision"] or 0.0) and cc["correct"] > rc["correct"],
        "C4_raise_off_mean": cand["raise_off_mean_val_labeled"] < ref["raise_off_mean_val_labeled"],
    }
    n_ok = sum(cond.values())
    verdict = "채택(B+)" if n_ok == 4 else ("기준 교사(B0)" if n_ok == 0 else "일부만 만족 — 결과를 보고하고 함께 정함")

    fmt = lambda m: " · ".join(f"{m[k]:.4f}" for k in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"))
    ck, rk = kind_of(cand_path), kind_of(ref_path)
    print(f"[관문] 후보 {ck} ({cand['sha256'][:12]}) vs 기준 {rk} ({ref['sha256'][:12]})"
          + ("  ※ 최신 판 확인 생략" if args.no_freshness_check else ("  ※ 기준 캐시 최신 판 확인 생략" if args.allow_stale_reference else "")))
    for split, name in (("val", "Mark5.3 val"), ("labeled", "정답 라벨")):
        print(f"  9클래스 Acc · P · R · F1  {name} {cand[split]['n']:>4}:  후보 {fmt(cand[split]['nine_class'])}  |  기준 {fmt(ref[split]['nine_class'])}"
              f"  |  3차(10-22 기록) {' · '.join(f'{v:.4f}' for v in RUN3_REFERENCE[split])}")
    for nm, r in (("후보", cand), ("기준", ref)):
        fr, cv = r["labeled"]["fixed_rule"], r["labeled"]["combiner_cv"]
        print(f"  가짜 정답(정답 라벨) {nm}: 고정 규칙 맞음 {fr['correct']} / 채택 {fr['adopted']} ({(fr['precision'] or 0):.4f}) · "
              f"결합기 {CV_FOLDS}겹 한 라벨 맞음 {cv['one']['correct']} / 채택 {cv['one']['adopted']} ({(cv['one']['precision'] or 0):.4f}) · "
              f"두 후보 {cv['two']['adopted']}(정답이 1·2등 안 {cv['two']['true_in_top2']})")
    print(f"  (3차 10-22 기록: 고정 규칙 {RUN3_REFERENCE['fixed_rule_labeled']['correct']} / {RUN3_REFERENCE['fixed_rule_labeled']['adopted']})")
    print(f"  남의 클래스 손들기 평균(val + 정답 라벨): 후보 {cand['raise_off_mean_val_labeled']:.4f} · 기준 {ref['raise_off_mean_val_labeled']:.4f}")
    for k, v in cond.items():
        print(f"  {k}: {'만족' if v else '불만족'}")
    print(f"[판정] {n_ok}/4 → {verdict}")

    out = args.out or os.path.join(PROJECT_ROOT, "plots", f"teacher_gate_{ck}_vs_{rk}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"candidate": cand, "reference": ref, "conditions": cond, "n_satisfied": n_ok, "verdict": verdict,
                   "tau1": args.tau1, "tau2": args.tau2, "raise_tau": RAISE_TAU, "run3_reference_2026_10_22": RUN3_REFERENCE,
                   "classes": classes, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
