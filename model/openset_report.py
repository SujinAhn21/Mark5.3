# -*- coding: utf-8 -*-
"""open-set 표 — 네 모델의 두 숫자를 한 표에 낸다 (체크리스트 10절 "open-set (학습 0회)").

[신설 2026-09-17 / Mark5_EXP2]

    ① 학습에 없던 소리를 others 로 보낸 비율   — data_eval/far_heldout (ESC-50 먼 소리 10종, 381) 과
                                                  data_eval/boundary_B (경계 소리, 1,272) 를 각각 낸다.
    ② 정규 test 에서 타겟 8종을 others 로 잘못 보낸 비율 — 타겟 클립 가운데 others 로 간 것의 비율.
    (지금까지 있던 0.28 대 0.05 는 ② 만 잰 것이고 ① 은 아무도 잰 적이 없다 — 체크리스트 10절.)

①·② 는 서로 당기는 값이라(others 로 많이 보내면 ① 은 오르고 ② 도 오른다) **항상 같이** 적는다. 그 옆에 정규 test 의
Accuracy · Macro Precision · Macro Recall · Macro F1 네 지표도 같이 둔다(지표는 늘 4개 세트).

어디서 읽나 — 모두 prediction_details CSV 의 "True Label"·"Predicted Label" 두 열만 쓴다. 학습·추론을 다시 하지 않는다.
    Mark5.2(이 저장소)  plots/prediction_details_<tag>_eval_far_heldout.csv · _eval_boundary_B.csv · <tag>.csv(test)
                          ← model/eval.py --split eval_far_heldout / eval_boundary_B / (test)  (운영점 = val 에서 고른 others 규칙)
    비교모델 4개         ../Baseline_*/plots/prediction_details_<tag>_eval_far_heldout.csv 등 (EfficientAT·CED·PANNs·CLAP)
                          ← 각 폴더 eval.py --split eval_far_heldout ... (운영점 = softmax argmax, others 도 한 클래스)
  "같은 운영점 규칙" 이란 각 모델이 정규 test 에서 쓰는 규칙 그대로를 평가 세트에도 쓴다는 뜻이다. 세트를 보고 임계값을 새로
  고르지 않는다(그러면 ① 만 좋아지게 맞출 수 있다). Mark5.2 의 규칙은 test performance_summary 의 Others Rule·Others
  Threshold 행에서 읽어 표의 rule 열에 적는다.

같은 태그에서 seed(_s<k>) 만 다른 회차는 한 묶음(group)으로 평균 ± 표본표준편차(ddof=1) 를 낸다. 회차마다 값은
openset_raw.csv 에 그대로 남긴다.

카테고리별 분해 — 평가 세트 인덱스(dataset_eval_<세트>.csv, category 열)가 이 저장소 루트·model/ 또는 옆 Mark5.2 에 있으면
파일 이름으로 붙여 "어느 소리가 타겟으로 새는가" 를 openset_by_category_<세트>.csv 로 낸다(묶음 × 카테고리, seed 평균).

산출물 (plots/):
    openset_raw.csv                    회차별 (dir, tag, group, seed, n_far, far_to_others, n_bnd, bnd_to_others, n_target, target_to_others,
                                        others_recall_test, Accuracy, Macro Precision, Macro Recall, Macro F1, rule)
    openset_table.csv                  묶음별 평균 ± SD (n = seed 수)
    openset_by_category_far_heldout.csv · openset_by_category_boundary_B.csv
    (화면) 마크다운 표 — 논문 표에 그대로 옮긴다.

사용:  python model/openset_report.py                       # 이 저장소 + 옆의 Baseline_* 3개에서 있는 것 전부
       python model/openset_report.py --ours_pattern "mark5.2_PL_SA_u2000_s*" --baseline_cond sup
       python model/openset_report.py --no_baselines
"""
import argparse
import csv
import fnmatch
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from aggregate_results import _read_summary                                   # noqa: E402  요약 CSV 앞 블록 파서(쉼표 포함 지표명 안전)

TARGET_CLASSES = ["heavy_impact", "dragging", "construction", "machine_noise",
                  "talking", "water_toilet", "water_shower", "dog_bark"]       # Mark5.2 타겟 8종(others 제외). 비교모델도 같은 이름·순서
EVAL_SETS = ("far_heldout", "boundary_B")
BASELINE_DIRS = ("Baseline_EfficientAT_mn10", "Baseline_CED_Tiny", "Baseline_PANNs_CNN6", "Baseline_CLAP")   # [2026-09-17] CLAP 네 번째
_SEED_RE = re.compile(r"^(?P<group>.+)_s(?P<seed>\d+)$")


def _rate(df, mask, pred_others):
    """mask 인 행 가운데 others 로 예측한 비율. 행이 없으면 nan."""
    n = int(mask.sum())
    return n, (float((pred_others & mask).sum()) / n if n else float("nan"))


def _read_pred(path):
    df = pd.read_csv(path, encoding="utf-8")
    need = {"Filename", "True Label", "Predicted Label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[ERROR] {path}: 열 {sorted(need - set(df.columns))} 이 없습니다.")
    return df


def _find_eval_index(set_name, model_root):
    """dataset_eval_<세트>.csv 를 모델 폴더 루트·model/ → 이 저장소 루트·model/ → 옆 Mark5.2 루트 순으로 찾는다. 없으면 None."""
    fn = f"dataset_eval_{set_name}.csv"
    cands = [os.path.join(model_root, fn), os.path.join(model_root, "model", fn),
             os.path.join(PROJECT_ROOT, fn), os.path.join(PROJECT_ROOT, "model", fn),
             os.path.join(PROJECT_ROOT, "..", "Mark5.2", fn)]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _category_map(index_csv):
    """파일 이름(basename) -> category."""
    out = {}
    with open(index_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[os.path.basename(r["path"])] = r.get("category", "")
    return out


def _discover(model_root, plot_subdir="plots", pattern=None):
    """prediction_details_<tag>_eval_far_heldout.csv 가 있는 tag 목록(둘 중 하나라도 있으면 포함)."""
    pdir = os.path.join(model_root, plot_subdir)
    tags = set()
    for s in EVAL_SETS:
        for p in glob.glob(os.path.join(pdir, f"prediction_details_*_eval_{s}.csv")):
            tag = os.path.basename(p)[len("prediction_details_"):-len(f"_eval_{s}.csv")]
            if "_limit" in tag:                                            # 동작 확인용 산출물은 표에 안 넣는다
                continue
            if pattern and not fnmatch.fnmatch(tag, pattern):
                continue
            tags.add(tag)
    return sorted(tags)


def evaluate_tag(model_root, tag, label):
    """한 회차의 ①·②와 test 4지표. 없는 파일은 nan 으로 두고 어떤 파일이 없었는지 note 에 적는다."""
    pdir = os.path.join(model_root, "plots")
    m = _SEED_RE.match(tag)
    row = {"model": label, "dir": os.path.basename(model_root), "tag": tag,
           "group": m.group("group") if m else tag, "seed": int(m.group("seed")) if m else None}
    notes, per_cat = [], {}
    for s in EVAL_SETS:
        p = os.path.join(pdir, f"prediction_details_{tag}_eval_{s}.csv")
        key = "far" if s == "far_heldout" else "bnd"
        if not os.path.isfile(p):
            row[f"n_{key}"], row[f"{key}_to_others"] = 0, float("nan"); notes.append(f"{s} 없음"); continue
        df = _read_pred(p)
        pred_others = df["Predicted Label"].astype(str).eq("others")
        n, r = _rate(df, pd.Series(True, index=df.index), pred_others)
        row[f"n_{key}"], row[f"{key}_to_others"] = n, r
        not_others = df["True Label"].astype(str).ne("others").sum()
        if not_others:
            notes.append(f"{s}: 정답이 others 아닌 행 {int(not_others)}개(인덱스 확인)")
        idx = _find_eval_index(s, model_root)
        if idx:
            cmap = _category_map(idx)
            cats = df["Filename"].astype(str).map(lambda fn: cmap.get(os.path.basename(fn), "?"))
            per_cat[s] = pd.DataFrame({"category": cats, "to_others": pred_others.astype(float)}) \
                .groupby("category")["to_others"].agg(["count", "mean"]).reset_index()
    # ② 정규 test
    p = os.path.join(pdir, f"prediction_details_{tag}.csv")
    if os.path.isfile(p):
        df = _read_pred(p)
        pred_others = df["Predicted Label"].astype(str).eq("others")
        is_target = df["True Label"].astype(str).isin(TARGET_CLASSES)
        is_others = df["True Label"].astype(str).eq("others")
        row["n_target"], row["target_to_others"] = _rate(df, is_target, pred_others)
        row["n_others_test"], row["others_recall_test"] = _rate(df, is_others, pred_others)
    else:
        row.update({"n_target": 0, "target_to_others": float("nan"), "n_others_test": 0, "others_recall_test": float("nan")})
        notes.append("test 없음")
    summ = os.path.join(pdir, f"performance_summary_{tag}.csv")
    metrics = _read_summary(summ) if os.path.isfile(summ) else {}
    for k in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"):
        row[k] = metrics.get(k, float("nan"))
    # Mark5.2 의 eval.py 는 val 에서 고른 규칙을 요약 CSV 의 Others Rule·Others Threshold 행에 적는다(문자열이라 직접 읽는다).
    # 그 행이 없으면 비교모델(softmax argmax, others 도 한 클래스)이다.
    rule = _read_rule(summ) if os.path.isfile(summ) else None
    row["rule"] = rule or "argmax(softmax)"
    row["note"] = "; ".join(notes)
    return row, per_cat


def _read_rule(summary_csv):
    rule, thr = None, None
    with open(summary_csv, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f):
            if len(parts) >= 2 and parts[0].strip() == "Others Rule":
                rule = parts[1].strip()
            elif len(parts) >= 2 and parts[0].strip() == "Others Threshold":
                thr = parts[1].strip()
    if rule is None:
        return None
    if rule == "direct" and thr not in (None, ""):
        try:
            return f"direct(thr {float(thr):g})"
        except ValueError:
            return f"direct(thr {thr})"
    return rule                                                      # argmax 등


def _mean_sd(v):
    v = np.asarray([x for x in v if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def build_table(raw):
    cols = ["far_to_others", "bnd_to_others", "target_to_others", "others_recall_test",
            "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    rows = []
    for (model, group), g in raw.groupby(["model", "group"], sort=False):
        r = {"model": model, "group": group, "n": int(g["seed"].notna().sum() or len(g)),
             "n_far": int(g["n_far"].max()), "n_bnd": int(g["n_bnd"].max()), "n_target": int(g["n_target"].max()),
             "rule": " / ".join(sorted(set(g["rule"].astype(str))))}
        for c in cols:
            m, s = _mean_sd(g[c].tolist())
            r[f"{c}_mean"], r[f"{c}_sd"] = m, s
        rows.append(r)
    return pd.DataFrame(rows)


def _fmt(m, s, n):
    if m != m:
        return "—"
    return f"{m:.4f}" if n <= 1 else f"{m:.4f} ± {s:.4f}"


def print_markdown(table):
    print("\n| 모델 | 묶음(tag) | n | ① far_heldout→others | ① boundary_B→others | ② test 타겟→others | test others 재현율 | Accuracy | Macro P | Macro R | Macro F1 | 운영점 규칙 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        n = r["n"]
        cells = [r["model"], r["group"], str(n),
                 _fmt(r["far_to_others_mean"], r["far_to_others_sd"], n) + (f" (N={r['n_far']})" if r["n_far"] else ""),
                 _fmt(r["bnd_to_others_mean"], r["bnd_to_others_sd"], n) + (f" (N={r['n_bnd']})" if r["n_bnd"] else ""),
                 _fmt(r["target_to_others_mean"], r["target_to_others_sd"], n) + (f" (N={r['n_target']})" if r["n_target"] else ""),
                 _fmt(r["others_recall_test_mean"], r["others_recall_test_sd"], n),
                 _fmt(r["Accuracy_mean"], r["Accuracy_sd"], n), _fmt(r["Macro Precision_mean"], r["Macro Precision_sd"], n),
                 _fmt(r["Macro Recall_mean"], r["Macro Recall_sd"], n), _fmt(r["Macro F1_mean"], r["Macro F1_sd"], n), r["rule"]]
        print("| " + " | ".join(cells) + " |")
    print("\n  ① = 정답이 전부 others 인 평가 세트에서 others 로 판정된 비율(클수록 좋음, = 1 − Others FPR).")
    print("  ② = 정규 test 의 타겟 8종 클립 가운데 others 로 판정된 비율(작을수록 좋음). ①·② 는 서로 당기므로 항상 같이 본다.")
    print("  값은 seed 평균 ± 표본표준편차(ddof=1), n = seed 수. 운영점은 test 와 같은 규칙(세트를 보고 다시 고르지 않음).")


def main():
    ap = argparse.ArgumentParser(description="open-set 표 (① 평가 세트→others 비율, ② test 타겟→others 비율)")
    ap.add_argument("--ours_pattern", type=str, default=None, help="이 저장소 태그 glob (예: mark5.2_PL_SA_u2000_s*). 기본: 평가 세트 산출물이 있는 전부")
    ap.add_argument("--ours_label", type=str, default="Mark5.2 (ours)")
    ap.add_argument("--baseline_dirs", type=str, nargs="*", default=None, help="비교모델 폴더들. 기본: 옆의 Baseline_* 3개 중 있는 것")
    ap.add_argument("--baseline_cond", type=str, default=None, help="비교모델 조건만 고를 때 (sup | pl | ceil). 기본: 전부")
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--out_dir", type=str, default=PLOT_DIR)
    args = ap.parse_args()

    jobs = [(PROJECT_ROOT, args.ours_label, args.ours_pattern)]
    if not args.no_baselines:
        dirs = args.baseline_dirs
        if dirs is None:
            dirs = [os.path.join(PROJECT_ROOT, "..", d) for d in BASELINE_DIRS]
        for d in dirs:
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                print(f"[WARN] 비교모델 폴더 없음(건너뜀): {d}")
                continue
            pat = f"*_{args.baseline_cond}_lr*" if args.baseline_cond else None
            jobs.append((d, os.path.basename(d).replace("Baseline_", ""), pat))

    raw_rows, per_cat_all = [], {s: [] for s in EVAL_SETS}
    for root, label, pat in jobs:
        tags = _discover(root, pattern=pat)
        if not tags:
            print(f"[WARN] {label}: 평가 세트 산출물(prediction_details_*_eval_*.csv)이 없습니다 — eval.py --split eval_far_heldout / eval_boundary_B 를 먼저.")
            continue
        for tag in tags:
            row, per_cat = evaluate_tag(root, tag, label)
            raw_rows.append(row)
            for s, dfc in per_cat.items():
                dfc = dfc.copy(); dfc.insert(0, "group", row["group"]); dfc.insert(0, "model", label); dfc["seed"] = row["seed"]
                per_cat_all[s].append(dfc)
            print(f"[INFO] {label:22s} {tag:45s} ① far {row['far_to_others']:.4f} (N={row['n_far']}) · bnd {row['bnd_to_others']:.4f} (N={row['n_bnd']})"
                  f" · ② target→others {row['target_to_others']:.4f} (N={row['n_target']})" + (f"  [{row['note']}]" if row["note"] else ""))
    if not raw_rows:
        raise SystemExit("[ERROR] 집계할 회차가 하나도 없습니다.")

    os.makedirs(args.out_dir, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(args.out_dir, "openset_raw.csv"), index=False, encoding="utf-8")
    table = build_table(raw)
    table.to_csv(os.path.join(args.out_dir, "openset_table.csv"), index=False, encoding="utf-8")
    for s, parts in per_cat_all.items():
        if not parts:
            continue
        cat = pd.concat(parts, ignore_index=True)
        agg = cat.groupby(["model", "group", "category"], sort=False).agg(
            n_clips=("count", "max"), to_others_mean=("mean", "mean"),
            to_others_sd=("mean", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), n_seeds=("seed", "nunique")).reset_index()
        agg.to_csv(os.path.join(args.out_dir, f"openset_by_category_{s}.csv"), index=False, encoding="utf-8")
    print_markdown(table)
    print(f"\n[INFO] 저장: {args.out_dir}/openset_raw.csv · openset_table.csv"
          + "".join(f" · openset_by_category_{s}.csv" for s, p in per_cat_all.items() if p))


if __name__ == "__main__":
    main()
