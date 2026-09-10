# make_paper_tables.py
# [신설 2026-09-05 / Mark5_EXP]
#
# 반복실험 집계 결과를 **논문에 그대로 붙일 수 있는 표**로 뽑는다.
# 마크다운(.md)과 LaTeX(.tex) 두 벌을 같은 숫자로 만든다.
#
# --------------------------------------------------------------------------
# 왜 따로 만드나
# --------------------------------------------------------------------------
# aggregate_results.py 가 내는 paper_table_*.csv 는 지표마다 평균±SD·95% CI·min~max·
# 중앙값[Q1,Q3] 네 칸을 다 담아 열이 24개다. 원자료로는 그게 맞지만 논문 표로는 못 쓴다.
# 저널 표의 관례는 **평균 ± 표준편차만 싣고**, 신뢰구간·사분위수는 심사자가 물을 때
# 본문이나 부록에서 꺼내는 것이다. 그래서 여기서 열을 줄여 세 표로 나눈다.
#
#   표 1  제안 모델 성능 (u2000, seed 5회)      <- 비교모델이 생기면 행을 추가한다
#         [2026-09-06] 바닥(u0)·천장(ceiling, 진짜 라벨) 행이 있으면 같이 싣는다 — Cances et al. (2022) 3기준선
#   표 2  라벨 없는 데이터 양에 따른 성능(데이터 효율) — 천장 행은 구분선 아래 따로(축이 다르다: 개수 vs 라벨 출처)
#   표 2-b 메운 비율 (aggregate_gap_closed_*.csv 가 있을 때)
#   표 3  효율 (파라미터·크기·연산량·CPU 지연)
#   그리고 본문에 문장으로 넣을 검정 결과(대응표본 t · Cohen's d · Holm 보정 p)
#
# 지표는 Accuracy·Macro Precision·Macro Recall·Macro F1·ROC AUC 다섯이다.
# P·R 을 빼지 않는 이유는 오류 방향(놓침 vs 오탐)이 이 연구의 핵심 주장이기 때문이다
# (u0 은 P 0.853 / R 0.728 로 갈리고, 라벨 없는 데이터는 R 만 크게 올린다).
#
# --------------------------------------------------------------------------
# 쓰는 법
# --------------------------------------------------------------------------
#   python model/make_paper_tables.py --mark_version mark5.0 --split test
#   -> plots/paper_tables_<mark_version>_<split>.md / .tex
#
# 비교모델을 넣을 때는 --extra_csv 로 같은 열 이름의 CSV 를 주면 표 1 에 행이 붙는다.
# (열: model,Accuracy,Macro Precision,Macro Recall,Macro F1,ROC AUC (Macro, OvR)
#      값은 "0.8512 ± 0.0071" 처럼 문자열로 적어도 되고 숫자만 적어도 된다.)

import argparse
import csv
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

# (원자료 열 이름, 표에 쓸 짧은 이름, LaTeX 표기)
METRICS = [
    ("Accuracy",             "Accuracy",  "Accuracy"),
    ("Macro Precision",      "Macro P",   "Macro P"),
    ("Macro Recall",         "Macro R",   "Macro R"),
    ("Macro F1",             "Macro F1",  "Macro F1"),
    ("ROC AUC (Macro, OvR)", "ROC AUC",   "ROC AUC"),
]

# [2026-09-10] 키 이름·열 제목 정정.
#   실제 입력 클립은 3초인데 "5s_clip" 으로 저장돼 있었다. 세그먼트 5개(1.01초 창을
#   0.5초 간격으로 겹침)가 약 3초를 덮는 것을 "5초"로 잘못 이름 붙인 것이고 숫자는 맞다.
#   새 키 이름에는 길이를 넣지 않는다(클립 길이가 바뀌어도 이름이 안 틀리게).
#   각 항목의 첫 원소가 새 이름이고 그 뒤가 옛 이름이다. 옛 이름으로 저장된 기존 산출물
#   (aggregate_latency CSV)도 그대로 읽히게 하려는 것이다. 이름을 못 찾으면 그 행이
#   조용히 사라지는 구조라 대체 이름을 반드시 남겨 둔다.
LATENCY_ROWS = [
    (("total_parameters",),                                     "Parameters",                  "", "int"),
    (("model_size_mb",),                                        "Model size (MB)",             "", "f3"),
    (("gflops_per_clip_approx", "gflops_per_5s_clip_approx"),   "GFLOPs / 3-s clip",           "", "f3"),
    (("cpu_latency_ms_per_clip", "cpu_latency_ms_per_5s_clip"), "CPU latency / 3-s clip (ms)", "", "f2"),
]


def fmt_ms(values, nd=4):
    """평균 ± 표본표준편차. 회차가 하나뿐이면 SD 를 붙이지 않는다.

    [feedback_verify_test_conditions_first] n=1 에 ddof=1 을 쓰면 NaN 이 되는데,
    그것을 '± nan' 으로 찍으면 표가 깨진다. 그래서 개수를 먼저 본다.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def fmt_ms_tex(values, nd=4):
    s = fmt_ms(values, nd)
    return s.replace("±", r"$\pm$")


def load_raw(mark_version, split):
    p = os.path.join(PLOT_DIR, f"aggregate_raw_{mark_version}_{split}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"[ERROR] 원자료가 없습니다: {p}\n"
                         f"        먼저 aggregate_results.py 를 돌리십시오.")
    return pd.read_csv(p, encoding="utf-8-sig")


def load_latency(mark_version):
    p = os.path.join(PLOT_DIR, f"aggregate_latency_{mark_version}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, encoding="utf-8-sig", index_col=0)


def load_tests(mark_version, split):
    p = os.path.join(PLOT_DIR, f"aggregate_tests_{mark_version}_{split}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def load_gap(mark_version, split):
    """[2026-09-06] aggregate_results.py 가 낸 메운 비율 표. 없으면 None."""
    p = os.path.join(PLOT_DIR, f"aggregate_gap_closed_{mark_version}_{split}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def split_conditions(raw_all):
    """[2026-09-06] 원자료를 반지도(ssl)와 천장(ceil)으로 나눈다. cond 열이 없는 옛 파일은 전부 ssl."""
    if "cond" not in raw_all.columns:
        return raw_all, raw_all.iloc[0:0]
    return (raw_all[raw_all["cond"] == "ssl"].reset_index(drop=True),
            raw_all[raw_all["cond"] == "ceil"].reset_index(drop=True))


def proposed_rows(raw, extra_csv, raw_ceil=None):
    """표 1 의 행 목록. [(모델 이름, {지표: 표시 문자열}), ...]

    [2026-09-06] Cances et al. (2022) 3기준선 순서로 싣는다:
        Proposed, supervised only (u0)  = 바닥
        Proposed (ours)                 = 반지도(제안)
        Proposed, upper bound           = 천장(진짜 라벨) — raw_ceil 이 있을 때만
    그 뒤에 --extra_csv 의 비교모델 행이 붙는다.
    """
    top = int(raw["unlabeled"].max())
    g = raw[raw["unlabeled"] == top]
    rows = []
    lo = int(raw["unlabeled"].min())
    if lo != top:
        g0 = raw[raw["unlabeled"] == lo]
        rows.append((f"Proposed, supervised only (u={lo})", {c: g0[c].values for c, _, _ in METRICS}, len(g0)))
    rows.append(("Proposed (ours)", {c: g[c].values for c, _, _ in METRICS}, len(g)))
    if raw_ceil is not None and len(raw_ceil):
        rows.append(("Proposed, upper bound (true labels)", {c: raw_ceil[c].values for c, _, _ in METRICS}, len(raw_ceil)))
    if extra_csv:
        if not os.path.exists(extra_csv):
            raise SystemExit(f"[ERROR] --extra_csv 파일이 없습니다: {extra_csv}")
        ex = pd.read_csv(extra_csv, encoding="utf-8-sig")
        # [주의] 지표 이름 'ROC AUC (Macro, OvR)' 에는 쉼표가 들어 있다. CSV 에서 이 열 이름을
        # 따옴표로 감싸지 않으면 두 열로 쪼개져 값이 조용히 빠진다(표에 '-' 만 남는다).
        # 조용한 실패를 막으려고 빠진 열을 반드시 알린다.
        missing = [c for c, _, _ in METRICS if c not in ex.columns]
        if missing:
            print("[WARN] --extra_csv 에 다음 열이 없어 그 칸은 '-' 로 둡니다: "
                  + ", ".join(missing))
            if any("," in m for m in missing):
                print("       지표 이름에 쉼표가 있는 열은 CSV 에서 따옴표로 감싸야 합니다. "
                      'ㅡ 예: "ROC AUC (Macro, OvR)"')
        if "model" not in ex.columns:
            print("[WARN] --extra_csv 에 'model' 열이 없어 이름을 '?' 로 둡니다.")
        for _, r in ex.iterrows():
            name = str(r.get("model", "?"))
            vals = {c: (r[c] if c in ex.columns else None) for c, _, _ in METRICS}
            rows.append((name, vals, None))
    return rows, top


def cell(v, tex=False, nd=4):
    """배열이면 평균±SD 로, 이미 문자열이면 그대로(비교모델 수기 입력 대비)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, str):
        return v.replace("±", r"$\pm$") if tex else v
    if np.ndim(v) == 0:
        return f"{float(v):.{nd}f}"
    return fmt_ms_tex(v, nd) if tex else fmt_ms(v, nd)


def build_markdown(raw, lat, tests, rows, top, split, n_seeds, raw_ceil=None, gap=None):
    L = []
    L.append(f"# 논문용 표 ({split} split)")
    L.append("")
    L.append("> aggregate_results.py 의 원자료에서 뽑았다. 값은 평균 ± 표본표준편차(ddof=1).")
    L.append("> 신뢰구간·사분위수는 표에 넣지 않는다 — 필요하면 `aggregate_summary_*.csv` 에 있다.")
    L.append("")

    L.append("## 표 1. 제안 모델 성능")
    L.append("")
    L.append("| Model | " + " | ".join(s for _, s, _ in METRICS) + " |")
    L.append("|" + "---|" * (len(METRICS) + 1))
    for name, vals, _n in rows:
        L.append(f"| {name} | " + " | ".join(cell(vals[c]) for c, _, _ in METRICS) + " |")
    L.append("")
    L.append(f"캡션: unlabeled = {top}, mean ± SD over {n_seeds} seeds.")
    L.append("비교모델을 넣을 때는 이 표에 행을 추가한다.")
    if raw_ceil is not None and len(raw_ceil):
        L.append("'upper bound' 행은 unlabeled 전부에 진짜 라벨을 되돌려 준 지도학습이며 제안 모델이 아니다"
                 "(Cances et al. 2022 의 Supervised 100% 기준선).")
    L.append("")

    L.append("## 표 2. 라벨 없는 데이터 양에 따른 성능")
    L.append("")
    L.append("| Unlabeled | " + " | ".join(s for _, s, _ in METRICS) + " |")
    L.append("|" + "---|" * (len(METRICS) + 1))
    for u in sorted(raw["unlabeled"].unique()):
        g = raw[raw["unlabeled"] == u]
        L.append(f"| {int(u)} | " + " | ".join(fmt_ms(g[c].values) for c, _, _ in METRICS) + " |")
    if raw_ceil is not None and len(raw_ceil):
        L.append(f"| {top} (true labels, upper bound) | "
                 + " | ".join(fmt_ms(raw_ceil[c].values) for c, _, _ in METRICS) + " |")
    L.append("")
    L.append(f"캡션: mean ± SD over {n_seeds} seeds. unlabeled = 0 은 전문가 합의를 전혀 쓰지 않은 순수 지도학습이다.")
    if raw_ceil is not None and len(raw_ceil):
        L.append("마지막 행(upper bound)은 축이 다르다 — 위 행들은 unlabeled 를 **몇 개** 쓰느냐, "
                 "마지막 행은 그 unlabeled 의 **라벨 출처**(합의 라벨 대신 진짜 라벨)를 바꾼 것이다. 표에서 구분선을 긋는다.")
    L.append("")

    if gap is not None and len(gap):
        L.append("## 표 2-b. 메운 비율 — (SSL − 바닥) ÷ (천장 − 바닥)")
        L.append("")
        cols = [c for c in gap.columns if c != "지표"]
        L.append("| 지표 | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for _, r in gap.iterrows():
            short = next((s_ for c, s_, _ in METRICS if c == r["지표"]), r["지표"])
            vals = []
            for c in cols:
                v = r[c]
                vals.append("-" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v)))
            L.append(f"| {short} | " + " | ".join(vals) + " |")
        L.append("")
        L.append("캡션: 1.0 이면 라벨이 완벽했을 때만큼 올랐다는 뜻. 'seed 짝' 값은 같은 seed 의 세 조건으로 seed 마다 계산한 것.")
        L.append("")

    if lat is not None:
        L.append("## 표 3. 효율")
        L.append("")
        L.append("| Item | Value |")
        L.append("|---|---|")
        for keys, label, _u, kind in LATENCY_ROWS:
            key = next((k for k in keys if k in lat.index), None)
            if key is None:
                continue
            r = lat.loc[key]
            m, sd = float(r["mean"]), float(r["std"])
            if kind == "int":
                s = f"{int(round(m)):,}"
            elif kind == "f3":
                s = f"{m:.3f}"
            else:
                s = f"{m:.2f}"
            if sd and not np.isnan(sd) and sd > 0:
                s += f" ± {sd:.2f}"
            L.append(f"| {label} | {s} |")
        L.append("")
        L.append("캡션: 파라미터·크기·연산량은 아키텍처에서 정해지는 값이라 반복해도 변하지 않는다.")
        L.append("CPU 지연만 5회 측정한 평균 ± SD 다.")
        L.append("")

    if tests is not None:
        L.append("## 본문에 문장으로 넣을 검정 결과")
        L.append("")
        L.append("| 비교 | 지표 | 평균차 | Cohen's d | p (paired) | p (Holm) |")
        L.append("|---|---|---|---|---|---|")
        dcol = "Cohen's d"
        keep = [f"u0 vs u{top}", f"u{top - 500} vs u{top}"] if top >= 500 else [f"u0 vs u{top}"]
        if "비교" in tests.columns and (tests["비교"] == f"u{top} vs ceil").any():
            keep += [f"u{top} vs ceil", "u0 vs ceil"]      # [2026-09-06] 천장 검정
        for cmp_name in keep:
            for c, s, _ in METRICS:
                sub = tests[(tests["비교"] == cmp_name) & (tests["지표"] == c)]
                if not len(sub):
                    continue
                r = sub.iloc[0]
                diff, d = float(r["평균차"]), float(r[dcol])
                p, ph = float(r["p (paired)"]), float(r["p (paired, Holm)"])
                L.append(f"| {cmp_name} | {s} | {diff:.4f} | {d:.2f} | {p:.6f} | {ph:.4f} |")
        L.append("")
        L.append("전체 30행은 `aggregate_tests_*.csv` 에 있다. 논문 본문에는 위 쌍만 문장으로 쓴다.")
        L.append("")

    return "\n".join(L) + "\n"


def build_latex(raw, lat, tests, rows, top, split, n_seeds, raw_ceil=None, gap=None):
    L = []
    L.append("% 논문용 표 — make_paper_tables.py 가 생성. booktabs 패키지가 필요하다.")
    L.append("% \\usepackage{booktabs}")
    L.append("")

    ncol = len(METRICS)
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{Performance on the {split} set "
             f"(unlabeled = {top}, mean $\\pm$ SD over {n_seeds} seeds).}}")
    L.append("\\label{tab:performance}")
    L.append("\\begin{tabular}{l" + "c" * ncol + "}")
    L.append("\\toprule")
    L.append("Model & " + " & ".join(t for _, _, t in METRICS) + " \\\\")
    L.append("\\midrule")
    for name, vals, _n in rows:
        L.append(f"{name} & " + " & ".join(cell(vals[c], tex=True) for c, _, _ in METRICS) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")

    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{Effect of the amount of unlabeled data on the {split} set "
             f"(mean $\\pm$ SD over {n_seeds} seeds). "
             f"The first row uses no specialist consensus at all."
             + (" The last row restores the withheld true labels of all unlabeled clips "
                "(supervised upper bound, not the proposed model)." if (raw_ceil is not None and len(raw_ceil)) else "")
             + "}}")
    L.append("\\label{tab:data-efficiency}")
    L.append("\\begin{tabular}{r" + "c" * ncol + "}")
    L.append("\\toprule")
    L.append("Unlabeled & " + " & ".join(t for _, _, t in METRICS) + " \\\\")
    L.append("\\midrule")
    for u in sorted(raw["unlabeled"].unique()):
        g = raw[raw["unlabeled"] == u]
        L.append(f"{int(u)} & " + " & ".join(fmt_ms_tex(g[c].values) for c, _, _ in METRICS) + " \\\\")
    if raw_ceil is not None and len(raw_ceil):
        L.append("\\midrule")
        L.append(f"{top} (true labels) & " + " & ".join(fmt_ms_tex(raw_ceil[c].values) for c, _, _ in METRICS) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    if gap is not None and len(gap):
        L.append("% [2026-09-06] Fraction of the gap closed = (SSL - floor) / (upper bound - floor).")
        for _, r in gap.iterrows():
            short = next((t for c, _, t in METRICS if c == r["지표"]), r["지표"])
            mean_key = [c for c in gap.columns if c.startswith("메운 비율(평균")]
            seed_key = [c for c in gap.columns if c.startswith("메운 비율(seed 짝 평균")]
            sd_key = [c for c in gap.columns if c.startswith("메운 비율(seed 짝 SD")]
            v = r[mean_key[0]] if mean_key else float("nan")
            ps = r[seed_key[0]] if seed_key else float("nan")
            sd = r[sd_key[0]] if sd_key else float("nan")
            L.append(f"% {short}: gap closed {v:.3f} (per-seed {ps:.3f} $\\pm$ {sd:.3f})")
        L.append("")

    if lat is not None:
        L.append("\\begin{table}[t]")
        L.append("\\centering")
        L.append("\\caption{Efficiency of the proposed student model. "
                 "Parameter count, model size and FLOPs are architectural constants; "
                 "CPU latency is the mean $\\pm$ SD of five measurements.}")
        L.append("\\label{tab:efficiency}")
        L.append("\\begin{tabular}{lc}")
        L.append("\\toprule")
        L.append("Item & Value \\\\")
        L.append("\\midrule")
        for keys, label, _u, kind in LATENCY_ROWS:
            key = next((k for k in keys if k in lat.index), None)
            if key is None:
                continue
            r = lat.loc[key]
            m, sd = float(r["mean"]), float(r["std"])
            if kind == "int":
                s = f"{int(round(m)):,}"
            elif kind == "f3":
                s = f"{m:.3f}"
            else:
                s = f"{m:.2f}"
            if sd and not np.isnan(sd) and sd > 0:
                s += f" $\\pm$ {sd:.2f}"
            L.append(f"{label} & {s} \\\\")
        L.append("\\bottomrule")
        L.append("\\end{tabular}")
        L.append("\\end{table}")
        L.append("")

    if tests is not None:
        L.append("% 검정 결과 — 표로 싣지 말고 본문 문장에 괄호로 넣는 것을 권한다.")
        keep = [f"u0 vs u{top}"]
        if "비교" in tests.columns and (tests["비교"] == f"u{top} vs ceil").any():
            keep += [f"u{top} vs ceil", "u0 vs ceil"]      # [2026-09-06 리뷰 반영] md 와 같은 쌍을 tex 에도
        for cmp_name in keep:
            for c, s, _ in METRICS:
                sub = tests[(tests["비교"] == cmp_name) & (tests["지표"] == c)]
                if not len(sub):
                    continue
                r = sub.iloc[0]
                dcol = "Cohen's d"
                L.append(f"% {cmp_name} / {s}: diff {r['평균차']:.4f}, "
                         f"d = {r[dcol]:.2f}, paired t p = {r['p (paired)']:.6f}, "
                         f"Holm p = {r['p (paired, Holm)']:.4f}")
        L.append("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="반복실험 집계를 논문용 표(마크다운·LaTeX)로 뽑는다.")
    ap.add_argument("--mark_version", type=str, default="mark5.0")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--extra_csv", type=str, default="",
                    help="비교모델 결과 CSV(열: model + 지표 5종). 표 1 에 행으로 붙는다.")
    args = ap.parse_args()

    raw_all = load_raw(args.mark_version, args.split)
    raw, raw_ceil = split_conditions(raw_all)            # [2026-09-06] 천장 행 분리
    if raw.empty:
        raise SystemExit("[ERROR] 반지도 조건(cond=ssl) 원자료가 없습니다.")
    lat = load_latency(args.mark_version)
    tests = load_tests(args.mark_version, args.split)
    gap = load_gap(args.mark_version, args.split)

    rows, top = proposed_rows(raw, args.extra_csv, raw_ceil)
    n_seeds = int(raw[raw["unlabeled"] == top].shape[0])

    md = build_markdown(raw, lat, tests, rows, top, args.split, n_seeds, raw_ceil=raw_ceil, gap=gap)
    tex = build_latex(raw, lat, tests, rows, top, args.split, n_seeds, raw_ceil=raw_ceil, gap=gap)

    os.makedirs(PLOT_DIR, exist_ok=True)
    md_path = os.path.join(PLOT_DIR, f"paper_tables_{args.mark_version}_{args.split}.md")
    tex_path = os.path.join(PLOT_DIR, f"paper_tables_{args.mark_version}_{args.split}.tex")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[INFO] 마크다운 저장: {md_path}")
    print(f"[INFO] LaTeX 저장   : {tex_path}")
    print(f"[INFO] 표 1 행 {len(rows)}개 · 표 2 조건 {raw['unlabeled'].nunique()}개"
          + (f" + 천장 {len(raw_ceil)}회" if len(raw_ceil) else "") + f" · seeds {n_seeds}")


if __name__ == "__main__":
    main()
