#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""others 판정 규칙(운영점)을 회차마다 val 에서 다시 고르고, 재학습·평가 재실행 없이 결과 파일을 다시 쓴다.

[신설 2026-09-07 / 리뷰 반영 2판] 왜 필요한가.
  vild_config.others_direct_threshold=0.09 는 2026-08-23 에 최종 반지도 모델(u2000) 하나의 val 로 고른 값인데
  바닥(u0)·천장(ceil)·데이터효율(u500~u1500) 30회 전부에 고정돼 쓰였다. 진짜 others 라벨로 배운 지도학습 student 는
  Prob_others 가 정상적으로 올라와서 0.09 로는 타겟 클립까지 others 로 쏟아졌다(천장 5회 others 정밀도 0.27~0.44).
  운영점은 모델(조건)마다 val 에서 고르는 것이 비교의 전제라 30회 전부를 그 규칙으로 통일한다.

규칙과 후보 (vild/postprocess_utils.select_others_threshold — eval.py 가 쓰는 것과 같은 함수).
  후보 = "argmax"(9클래스 argmax, others 강제 없음 — 비교모델의 기본 규칙) + direct(Prob_others >= thr 이면 others,
  아니면 others 를 빼고 argmax) 임계값 격자 0.02~0.95/0.01. val Macro F1 최대를 고르고, 동률 격자점은 가운데,
  argmax 가 동률이면 argmax. 비교모델 3개도 같은 후보·같은 절차로 고른다(Baseline_*/reselect_others_threshold.py).

다시 쓰는 것 (회차 = run_tag, val·test 둘 다).
  performance_summary_<tag>[_val].csv   요약 블록 끝에 "Others Threshold"·"Others Rule" 행 추가(집계는 METRICS 만 읽어 무해)
  prediction_details_<tag>[_val].csv    Predicted Label · Forced To Others · Abstained 갱신(Prob_* 는 그대로)
  confusion_matrix_<tag>[_val].png · roc_curve_<tag>[_val].png   다시 그림. ROC AUC 와 ROC 곡선은 규칙 적용 **전**
                                        확률(Prob_*)로 잰다(운영점과 무관해야 하는 지표 — 리뷰 지적 7).
  calibration_details_<tag>[_val].csv   전체 경로가 예측 CSV 에 없어 다시 못 만든다 → 백업 폴더로 옮긴다(어긋난 채 남기지 않음).
  others_threshold_<tag>.json           고른 규칙·임계값·val F1. 체크포인트 지문은 CSV 사후계산이라 없음(null).
  others_threshold_selected_<mv>.csv    회차별 표. 기존 표가 있으면 run_tag 기준으로 병합(--tags 부분 실행에도 안 사라짐).
  원본은 plots/backup_fixed_thr_<기존값>/ 에 첫 실행 때만 보관(집계 glob 은 하위 폴더를 안 본다).

검증.
  --check [--check_dir DIR]  다시 계산해 파일과 같은지만 본다(쓰기 없음). 각 파일의 "Others Rule"/"Others Threshold" 행을
      읽어 그 규칙으로 재현하고, 행이 없으면 direct/기존 config 값(0.09)으로 본다. 비교 대상 폴더를 바꾸려면 --check_dir.
      ROC AUC 는 옛 파일이 보정 후 확률로 잰 값이라 다를 수 있어 정보로만 찍고 판정에서 뺀다.
  2026-09-07 실측: 옛 0.09 원본(백업) 30회 × (test·val) 전부 예측 라벨 불일치 0, AUC 외 지표 차이 1e-9 이하.

사용:
  python model/reselect_others_threshold.py --mark_version mark5.0 --check --check_dir plots/backup_fixed_thr_0.09
  python model/reselect_others_threshold.py --mark_version mark5.0 --dry_run
  python model/reselect_others_threshold.py --mark_version mark5.0
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild")):
    if p not in sys.path:
        sys.path.append(p)
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from postprocess_utils import apply_abstention, apply_others_calibration, select_others_threshold  # noqa: E402
from vild_config import AudioViLDConfig  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
PRED_PREFIX = "prediction_details_"
SUMMARY_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                   "ROC AUC (Macro, OvR)", "Others FPR"]
COMPARE_METRICS = [m for m in SUMMARY_METRICS if not m.startswith("ROC AUC")]


def find_tags(mark_version):
    tags = []
    for p in sorted(glob.glob(os.path.join(PLOT_DIR, f"{PRED_PREFIX}{mark_version}*_val.csv"))):
        tag = os.path.basename(p)[len(PRED_PREFIX):-len("_val.csv")]
        if "_smoke" in tag:
            continue
        if os.path.exists(os.path.join(PLOT_DIR, f"{PRED_PREFIX}{tag}.csv")):
            tags.append(tag)
        else:
            print(f"[WARN] {tag}: val 은 있는데 test 예측 CSV 가 없어 건너뜁니다.")
    return tags


def load_pred(tag, split, plot_dir=PLOT_DIR):
    path = os.path.join(plot_dir, f"{PRED_PREFIX}{tag}{'' if split == 'test' else '_' + split}.csv")
    df = pd.read_csv(path, encoding="utf-8")
    class_names = [c[len("Prob_"):] for c in df.columns if c.startswith("Prob_")]
    probs = df[[f"Prob_{c}" for c in class_names]].to_numpy(dtype=np.float32)   # eval.py 와 같은 float32 산술
    unknown = sorted(set(df["True Label"]) - set(class_names))
    if unknown:
        raise SystemExit(f"[ERROR] {path}: 클래스 밖 라벨 {unknown}")
    labels = np.array([class_names.index(l) for l in df["True Label"]], dtype=int)
    return path, df, class_names, probs, labels


def decide(probs, class_names, config):
    preds, forced, abst = [], [], []
    for row in probs:
        cal, pred, meta = apply_others_calibration(row.copy(), class_names, config)
        cal, pred, a = apply_abstention(cal, class_names, config, pred)
        preds.append(int(pred))
        forced.append(bool(meta["forced_to_others"]))
        abst.append(bool(a))
    return np.array(preds), forced, abst


def compute_metrics(labels, preds, raw_probs, class_names):
    C = len(class_names)
    idx = list(range(C))
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=idx, zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(labels, preds, average="macro", labels=idx, zero_division=0)
    try:
        auc_m = roc_auc_score(labels, raw_probs, multi_class="ovr", average="macro", labels=idx)
    except ValueError:
        auc_m = float("nan")
    cm = confusion_matrix(labels, preds, labels=idx)
    oi = class_names.index("others")
    s = int(cm[oi].sum())
    fpr = float((s - cm[oi][oi]) / s) if s > 0 else float("nan")
    return {"Accuracy": acc, "Macro Precision": mp, "Macro Recall": mr, "Macro F1": mf1,
            "ROC AUC (Macro, OvR)": auc_m, "Others FPR": fpr, "per_class": (p, r, f1), "cm": cm}


def set_rule(config, rule, thr):
    config.others_decision_mode = rule
    if rule == "direct":
        config.others_direct_threshold = float(thr)


def write_summary(path, out_tag, m, class_names, rule, thr):
    p, r, f1 = m["per_class"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Performance Summary for {out_tag}\n\n")
        pd.DataFrame({
            "Metric": SUMMARY_METRICS + ["Others Threshold", "Others Rule"],
            "Score": [m[k] for k in SUMMARY_METRICS] + [float(thr) if rule == "direct" else float("nan"), rule],
        }).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({"Class": class_names, "Precision": p, "Recall": r, "F1-Score": f1}).to_csv(f, index=False)


def write_confusion_png(path, out_tag, cm, class_names):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {out_tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_roc_png(path, out_tag, labels, raw_probs, class_names):
    plt.figure(figsize=(7, 6))
    labels = np.asarray(labels)
    for i, cname in enumerate(class_names):
        fpr_c, tpr_c, _ = roc_curve((labels == i).astype(int), raw_probs[:, i])
        plt.plot(fpr_c, tpr_c, label=f"{cname} AUC={auc(fpr_c, tpr_c):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({out_tag})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def read_existing_summary(path):
    """(지표 dict, 클래스별 dict, rule, thr). rule/thr 행이 없으면 (None, None)."""
    out, per_class, rule, thr = {}, {}, None, None
    in_class = False
    with open(path, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f):
            if not parts or parts[0].startswith("#") or not parts[0].strip():
                continue
            if parts[0] == "Class":
                in_class = True
                continue
            if parts[0] == "Metric":
                continue
            if in_class:
                per_class[parts[0]] = tuple(float(x) for x in parts[1:4])
            elif parts[0] == "Others Rule":
                rule = parts[1].strip()
            elif parts[0] == "Others Threshold":
                try:
                    thr = float(parts[1])
                except ValueError:
                    thr = None
            else:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    if rule is None and thr is not None:
        rule = "direct"
    return out, per_class, rule, thr


def backup(paths, backup_dir, move=False):
    os.makedirs(backup_dir, exist_ok=True)
    for p in paths:
        if os.path.exists(p):
            dst = os.path.join(backup_dir, os.path.basename(p))
            if not os.path.exists(dst):
                (shutil.move if move else shutil.copy2)(p, dst)
            elif move:
                os.remove(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mark_version", type=str, default="mark5.0")
    ap.add_argument("--tags", nargs="*", default=None, help="일부 run_tag 만")
    ap.add_argument("--fixed", type=str, default=None, help="val 선택 대신 이 규칙을 전부에 적용: 실수(direct) 또는 argmax")
    ap.add_argument("--check", action="store_true", help="파일에 적힌 규칙으로 다시 계산해 같은지만 확인(쓰기 없음)")
    ap.add_argument("--check_dir", type=str, default=None, help="--check 의 비교 대상 폴더(기본 plots/)")
    ap.add_argument("--dry_run", action="store_true", help="고를 규칙과 지표만 출력(쓰기 없음)")
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    if getattr(config, "others_decision_mode", "direct") not in ("direct", "argmax"):
        raise SystemExit("[ERROR] others_decision_mode 가 direct/argmax 가 아닙니다. 이 스크립트는 그 규칙 전용입니다.")
    old_thr = float(getattr(config, "others_direct_threshold", 0.14))
    tags = args.tags or find_tags(args.mark_version)
    if not tags:
        raise SystemExit(f"[ERROR] {PLOT_DIR} 에 {PRED_PREFIX}{args.mark_version}*_val.csv 가 없습니다.")
    mode = "check" if args.check else ("dry_run" if args.dry_run else "write")
    check_dir = os.path.join(PROJECT_ROOT, args.check_dir) if args.check_dir and not os.path.isabs(args.check_dir) else (args.check_dir or PLOT_DIR)
    print(f"[INFO] 회차 {len(tags)}개 · config 기본 direct {old_thr:g} · 모드 {mode}" + (f" · 비교 폴더 {check_dir}" if args.check else ""))

    backup_dir = os.path.join(PLOT_DIR, f"backup_fixed_thr_{old_thr:g}")
    rows, n_fail = [], 0
    for tag in tags:
        if args.check:
            ok = True
            for split in ("val", "test"):
                out_tag = tag if split == "test" else f"{tag}_{split}"
                spath = os.path.join(check_dir, f"performance_summary_{out_tag}.csv")
                ppath = os.path.join(check_dir, f"{PRED_PREFIX}{out_tag}.csv")
                if not (os.path.exists(spath) and os.path.exists(ppath)):
                    print(f"  SKIP  {out_tag:<34} 비교 파일 없음")
                    continue
                old, old_pc, rule, thr = read_existing_summary(spath)
                if rule is None:
                    rule, thr = "direct", old_thr
                _, df, class_names, probs, labels = load_pred(tag, split, check_dir)
                set_rule(config, rule, thr)
                preds, _, _ = decide(probs, class_names, config)
                m = compute_metrics(labels, preds, probs, class_names)
                diffs = {k: abs(old.get(k, np.nan) - m[k]) for k in COMPARE_METRICS
                         if not (np.isnan(old.get(k, np.nan)) and np.isnan(m[k]))}
                pc_diff = max(abs(old_pc[c][j] - m["per_class"][j][i]) for i, c in enumerate(class_names) for j in range(3))
                lab_mism = int((np.array([class_names[i] for i in preds]) != df["Predicted Label"].to_numpy()).sum())
                worst = max(diffs.values()) if diffs else 0.0
                auc_diff = abs(old.get("ROC AUC (Macro, OvR)", np.nan) - m["ROC AUC (Macro, OvR)"])
                good = worst < 1e-9 and pc_diff < 1e-9 and lab_mism == 0
                ok &= good
                print(f"  {'PASS' if good else 'FAIL'}  {out_tag:<34} 규칙 {rule}{'' if thr is None else f' {thr:g}'} · "
                      f"지표(AUC 제외) 최대차 {worst:.2e} · 클래스별 {pc_diff:.2e} · 예측 불일치 {lab_mism} · AUC 차(정보) {auc_diff:.4f}")
            n_fail += 0 if ok else 1
            continue

        vpath, vdf, class_names, vprobs, vlabels = load_pred(tag, "val")
        tpath, tdf, cn2, tprobs, tlabels = load_pred(tag, "test")
        if cn2 != class_names:
            raise SystemExit(f"[ERROR] {tag}: val/test 클래스 열 순서가 다릅니다.")
        if args.fixed is not None:
            if args.fixed.strip().lower() == "argmax":
                rule, thr, vf1 = "argmax", None, None
            else:
                rule, thr, vf1 = "direct", float(args.fixed), None
            selected_on = "fixed"
        else:
            rule, thr, vf1, _curve = select_others_threshold(vprobs, vlabels, class_names, config)
            selected_on = "val"
        set_rule(config, rule, thr)
        out = {}
        for split, (path, df, probs, labels) in (("val", (vpath, vdf, vprobs, vlabels)),
                                                ("test", (tpath, tdf, tprobs, tlabels))):
            preds, forced, abst = decide(probs, class_names, config)
            out[split] = (path, df, preds, forced, abst, compute_metrics(labels, preds, probs, class_names), probs, labels)

        mt, mv = out["test"][5], out["val"][5]
        shown = f"direct {thr:.2f}" if rule == "direct" else "argmax"
        rows.append({"run_tag": tag, "rule": rule, "threshold": thr, "previous_threshold": old_thr, "selected_on": selected_on,
                     "val_macro_f1_at_threshold": vf1,
                     **{f"val {k}": mv[k] for k in SUMMARY_METRICS}, **{f"test {k}": mt[k] for k in SUMMARY_METRICS}})
        print(f"  {tag:<28} {old_thr:g} -> {shown:<12} | test Acc {mt['Accuracy']:.4f} P {mt['Macro Precision']:.4f} "
              f"R {mt['Macro Recall']:.4f} F1 {mt['Macro F1']:.4f} AUC {mt['ROC AUC (Macro, OvR)']:.4f} othersFPR {mt['Others FPR']:.4f}")
        if args.dry_run:
            continue

        for split, (path, df, preds, forced, abst, m, probs, labels) in out.items():
            out_tag = tag if split == "test" else f"{tag}_{split}"
            spath = os.path.join(PLOT_DIR, f"performance_summary_{out_tag}.csv")
            cpath = os.path.join(PLOT_DIR, f"confusion_matrix_{out_tag}.png")
            rpath = os.path.join(PLOT_DIR, f"roc_curve_{out_tag}.png")
            calpath = os.path.join(PLOT_DIR, f"calibration_details_{out_tag}.csv")
            backup([spath, path, cpath, rpath], backup_dir)
            backup([calpath], backup_dir, move=True)      # 다시 못 만드는 파일은 옮겨서 어긋난 채 남기지 않는다
            df = df.copy()
            df["Predicted Label"] = [class_names[i] for i in preds]
            df["Forced To Others"] = forced
            df["Abstained"] = abst
            df.to_csv(path, index=False, encoding="utf-8")
            write_summary(spath, out_tag, m, class_names, rule, thr)
            write_confusion_png(cpath, out_tag, m["cm"], class_names)
            write_roc_png(rpath, out_tag, labels, probs, class_names)
        g = getattr(config, "others_threshold_grid", (0.02, 0.95, 0.01))
        with open(os.path.join(PLOT_DIR, f"others_threshold_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "val_select" if selected_on == "val" else "fixed", "rule": rule, "threshold": thr,
                       "selected_on": selected_on, "criterion": "macro_f1",
                       "grid": [float(g[0]), float(g[1]), float(g[2])], "candidates": "argmax + direct grid",
                       "val_macro_f1_at_threshold": vf1, "n_val_clips": int(len(vlabels)),
                       "previous_fixed_threshold": old_thr, "run_tag": tag, "checkpoint": None,
                       "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "note": "재학습·평가 재실행 없이 prediction_details CSV 로 재계산(체크포인트 지문 없음)"},
                      f, ensure_ascii=False, indent=2)

    if args.check:
        print(f"\n[CHECK] 회차 {len(tags)}개 중 불일치 {n_fail}개 -> {'전부 재현됨' if n_fail == 0 else '재현 실패 있음'}")
        sys.exit(0 if n_fail == 0 else 1)
    if rows and not args.dry_run:
        table = os.path.join(PLOT_DIR, f"others_threshold_selected_{args.mark_version}.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(table):
            old = pd.read_csv(table, encoding="utf-8-sig")
            new = pd.concat([old[~old["run_tag"].isin(new["run_tag"])], new], ignore_index=True)
            new = new.sort_values("run_tag").reset_index(drop=True)
        new.to_csv(table, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 회차 {len(rows)}개 다시 씀 · 선택표 {table}({len(new)}행) · 원본 백업 {backup_dir}")
        print("[INFO] 이어서: python model/aggregate_results.py --mark_version %s --split test / --split val" % args.mark_version)


if __name__ == "__main__":
    main()
