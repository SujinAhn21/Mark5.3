#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""[신설 2026-10-26] 논문 결과물 4개 — test 4지표(Accuracy · Macro Precision · Macro Recall · Macro F1), seed 1~5 평균 ± 표준편차.

    python model/make_four_results.py --baseline_root /content/drive/MyDrive/NewProject/codes/experiments

① 데이터 효율 곡선   정답 200·500·1,000·전량 — Mark5.3(결합기 가짜 정답, 라벨 없음 2,000) · EfficientAT · CED-Tiny · PANNs(정답 라벨만)
② 라벨 없는 클립 수  정답 200 고정 — 라벨 없음 0·500·1,000·2,000 (Mark5.3)
③ 비교표             ①과 같은 회차를 정답 수준별 표로
④ 지식 융합 이득 표  정답 200 — 평범한 구조(가짜 정답 없음) · 전문가 의사라벨만(고정 규칙, 결합기 없음) · 제안모델(전문가 + 결합기)

비교모델은 지식 융합 구조가 아니므로 우리 가짜 정답을 받은 조건(plc)은 쓰지 않는다(2026-10-26 수진님 확정).
회차 이름은 실행기(run_experiments._run_tag · preset cmb 공통 플래그)와 비교모델 이름 규칙(common/naming.run_tag)을 그대로 따른다.
산출물: plots/four_results_{1,2,3,4}_test.csv · plots/four_results_test.md · plots/four_results_{1,2}_test.png
"""
import argparse
import csv
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)

METRICS = ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1")
SEEDS = (1, 2, 3, 4, 5)
LABELED = (200, 500, 1000, None)            # None = 정답 전량(2,000)
UNLABELED = (0, 500, 1000, 2000)
BASELINES = (("Baseline_EfficientAT_mn10", "efficientat_mn10", "EfficientAT"),
             ("Baseline_CED_Tiny", "ced_tiny", "CED-Tiny"),
             ("Baseline_PANNs_CNN6", "panns_cnn6", "PANNs"))
MODEL = "Mark5.3"


def _lab_txt(l):
    return "2000" if l is None else str(l)


def read_summary(path):
    """performance_summary_*.csv 의 'Metric,Score' 줄에서 4지표. 파일이 없으면 None."""
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for m in METRICS:
                if line.startswith(m + ","):
                    out[m] = float(line.split(",", 1)[1])
    return out if len(out) == len(METRICS) else None


def mark53_tag(seed, labeled_n, unlabeled_n, ps):
    import run_experiments as RE
    kw = RE._parse_runner_extra(RE.CMB_BASE_EXTRA)[1]
    return RE._run_tag("mark5.2", seed, unlabeled_n, labeled_n=labeled_n, ps=ps, ablation_kw=kw)


def mark53_runs(root, labeled_n, unlabeled_n, ps):
    """Mark5.3 한 조건의 seed 1~5 test 결과 [(seed, {지표: 값})] 와 빠진 회차 이름."""
    got, missing = [], []
    for s in SEEDS:
        tag = mark53_tag(s, labeled_n, unlabeled_n, ps)
        r = read_summary(os.path.join(root, "plots", f"performance_summary_{tag}_plain.csv"))
        (got.append((s, r)) if r else missing.append(tag))
    return got, missing


def baseline_runs(baseline_root, repo, mtag, labeled_n):
    """비교모델 한 정답 수준의 seed 1~5 test 결과(정답 라벨만, run_tag <model>_sup_adam[_l<n>]_s<seed>)."""
    got, missing = [], []
    for s in SEEDS:
        tag = f"{mtag}_sup_adam" + ("" if labeled_n is None else f"_l{labeled_n}") + f"_s{s}"
        r = read_summary(os.path.join(baseline_root, repo, "plots", f"performance_summary_{tag}.csv"))
        (got.append((s, r)) if r else missing.append(f"{repo}/{tag}"))
    return got, missing


def summarize(runs):
    """{지표: (평균, 표준편차, n)}. n=1 이면 표준편차 0."""
    out = {}
    for m in METRICS:
        v = [r[m] for _, r in runs]
        out[m] = (statistics.mean(v) if v else float("nan"), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
    return out


def _cell(st, m):
    mean, sd, n = st[m]
    return "—" if n == 0 else f"{mean:.4f} ± {sd:.4f}" + ("" if n == len(SEEDS) else f" (n={n})")


def _md(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def _write_csv(path, rows):
    fields = ["result", "row", "x", "n"] + [f"{m} mean" for m in METRICS] + [f"{m} sd" for m in METRICS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for res, row, x, st in rows:
            n = min(st[m][2] for m in METRICS)
            w.writerow([res, row, x, n] + [f"{st[m][0]:.6f}" for m in METRICS] + [f"{st[m][1]:.6f}" for m in METRICS])


def _plot(path, xs, lines, xlabel):
    """lines = [(이름, [stat 또는 None ...])]. 4지표를 2×2 로. 한글 글꼴이 없는 환경을 생각해 그림 글자는 영어."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, m in zip(axes.flat, METRICS):
        for name, stats in lines:
            pts = [(x, st[m][0], st[m][1]) for x, st in zip(xs, stats) if st is not None and st[m][2] > 0]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                            marker="o", markersize=4, capsize=3, linewidth=1.2, label=name)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="논문 결과물 4개(test 4지표, seed 1~5 평균 ± 표준편차)")
    ap.add_argument("--baseline_root", required=True, help="비교모델 저장소 3개가 들어 있는 폴더(코랩: …/NewProject/codes/experiments)")
    ap.add_argument("--root", default=PROJECT_ROOT, help="Mark5.3 폴더(기본: 이 저장소). 시험할 때만 바꾼다")
    ap.add_argument("--no_plot", action="store_true", help="그림을 만들지 않는다")
    args = ap.parse_args()
    plots = os.path.join(args.root, "plots")
    os.makedirs(plots, exist_ok=True)
    missing = []

    # ① · ③ 정답 수준별: Mark5.3(결합기, 라벨 없음 2,000) + 비교모델 3개
    eff = {MODEL: []}
    for l in LABELED:
        runs, miss = mark53_runs(args.root, l, 2000, "cmb")
        eff[MODEL].append(summarize(runs))
        missing += miss
    for repo, mtag, name in BASELINES:
        eff[name] = []
        for l in LABELED:
            runs, miss = baseline_runs(args.baseline_root, repo, mtag, l)
            eff[name].append(summarize(runs))
            missing += miss

    # ② 정답 200 고정, 라벨 없음 0·500·1,000·2,000 (0 = 가짜 정답 없음, 2,000 = ①의 정답 200 회차)
    unl = []
    for u in UNLABELED:
        runs, miss = mark53_runs(args.root, 200, u, None if u == 0 else "cmb")
        unl.append(summarize(runs))
        missing += [t for t in miss if t not in missing]

    # ④ 정답 200: 평범한 구조(= ②의 0) · 전문가 의사라벨만(고정 규칙) · 제안모델(= ①의 정답 200)
    fix_runs, miss = mark53_runs(args.root, 200, 2000, "fix")
    missing += miss
    ka = [("평범한 구조 (가짜 정답 없음)", unl[0]),
          ("전문가 의사라벨만 (고정 규칙, 결합기 없음)", summarize(fix_runs)),
          ("제안모델 (전문가 + 결합기)", eff[MODEL][0])]

    models = [MODEL] + [b[2] for b in BASELINES]
    csv_rows1 = [("1_label_efficiency", name, _lab_txt(l), st) for name in models for l, st in zip(LABELED, eff[name])]
    csv_rows2 = [("2_unlabeled_curve", MODEL, str(u), st) for u, st in zip(UNLABELED, unl)]
    csv_rows3 = [("3_comparison", name, _lab_txt(l), eff[name][i]) for i, l in enumerate(LABELED) for name in models]
    csv_rows4 = [("4_ka_gain", row, "200", st) for row, st in ka]
    for i, rows in enumerate((csv_rows1, csv_rows2, csv_rows3, csv_rows4), start=1):
        _write_csv(os.path.join(plots, f"four_results_{i}_test.csv"), rows)

    md = ["# 논문 결과물 4개 — test, seed 1~5 평균 ± 표준편차", "",
          "비교모델은 정답 라벨만으로 학습한 결과다(지식 융합 구조가 아니므로 우리 가짜 정답을 쓰지 않음).", ""]
    md += _md("① 데이터 효율 — 정답 라벨 수별 (Accuracy · Macro P · Macro R · Macro F1)",
              ["정답 라벨", "모델"] + list(METRICS),
              [[_lab_txt(l), name] + [_cell(eff[name][i], m) for m in METRICS]
               for i, l in enumerate(LABELED) for name in models])
    md += _md("② 라벨 없는 클립 수 — 정답 200 고정 (Mark5.3)", ["라벨 없는 클립"] + list(METRICS),
              [[str(u)] + [_cell(st, m) for m in METRICS] for u, st in zip(UNLABELED, unl)])
    md += _md("③ 우리 모델 대 비교모델 — 정답 라벨 수별 비교표", ["정답 라벨"] + [f"{name} Acc / P / R / F1" for name in models],
              [[_lab_txt(l)] + [" / ".join("—" if eff[name][i][m][2] == 0 else f"{eff[name][i][m][0]:.4f}" for m in METRICS)
                                for name in models] for i, l in enumerate(LABELED)])
    md += _md("④ 지식 융합 구조의 이득 — 정답 200", ["구성"] + list(METRICS),
              [[row] + [_cell(st, m) for m in METRICS] for row, st in ka])
    if missing:
        md += [f"### 아직 없는 회차 {len(missing)}개", ""] + [f"- {t}" for t in missing] + [""]
    md_path = os.path.join(plots, "four_results_test.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    if not args.no_plot:
        _plot(os.path.join(plots, "four_results_1_test.png"), [200, 500, 1000, 2000],
              [(name, eff[name]) for name in models], "Labeled clips")
        _plot(os.path.join(plots, "four_results_2_test.png"), list(UNLABELED),
              [(MODEL + " (labeled 200)", unl)], "Unlabeled clips")
    print("\n".join(md))
    print(f"[OK] {md_path} · four_results_{{1,2,3,4}}_test.csv" + ("" if args.no_plot else " · four_results_{1,2}_test.png"))
    if missing:
        print(f"[주의] 아직 없는 회차 {len(missing)}개 — 표의 (n=…) 칸은 그 수만큼만 평균했다")


if __name__ == "__main__":
    main()
