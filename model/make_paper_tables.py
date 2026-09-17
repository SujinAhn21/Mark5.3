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
# [확장 2026-09-17 / 체크리스트 11절 "표 전부에 Accuracy·Macro P·Macro R·Macro F1 네 열"] 새 표 다섯 개를 같은 파일에 붙인다.
#   표 4  라벨 효율(본표)     labeled 전량·1,000·500·200 x {supervised(u0), ours(u<max>), upper(true labels)}
#                             + --baseline_raw 의 비교모델 {sup, pl, ceil} 행 (aggregate_baseline.py 의 aggregate_raw_<model>_<split>.csv)
#   표 5  데이터 효율 곡선     labeled 200 고정, unlabeled 0·500·1,000·2,000 (+ 비교모델 pl u500·u1000)
#   표 6  pseudo-label 출처    labeled 200 + unlabeled 3,800 에서 consensus / self / gen
#   표 7  open-set             model/openset_report.py 의 plots/openset_table.csv (①·②·test others 재현율·4지표·운영점 규칙)
#   표 8  제로샷               plots/zeroshot_<run_tag>.json(seed 평균) + zeroshot_clap_*.json (CLAP 참조선)
#   검정  aggregate_tests_label_eff_* · aggregate_tests_plsource_* · aggregate_tests_vs_baseline_* 의 쌍도 본문 검정 절에 붙인다.
#   원자료(aggregate_raw)에 labeled·spill·pseudo_source 열이 생겼으므로 표 1·2 는 "본 조건"(labeled 전량, spill 아님,
#   pseudo_source consensus)만 골라 쓴다 — 전에는 전부 u2000 으로 섞였다.
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


def main_slice(raw_all):
    """[2026-09-17] 표 1·2 용 "본 조건" 행만: labeled 전량(최댓값), spill 아님, pseudo_source consensus(또는 없음)."""
    m = pd.Series(True, index=raw_all.index)
    if "labeled" in raw_all.columns and raw_all["labeled"].notna().any():
        full = raw_all["labeled"].max()
        m &= raw_all["labeled"].fillna(full) == full
    if "spill" in raw_all.columns:
        m &= ~raw_all["spill"].astype(str).str.lower().eq("true")
    if "pseudo_source" in raw_all.columns:
        m &= raw_all["pseudo_source"].fillna("consensus").astype(str).eq("consensus")
    return raw_all[m].reset_index(drop=True)


def _load_csv(name):
    p = os.path.join(PLOT_DIR, name)
    return pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) else None


def load_baseline_raw(paths):
    """비교모델 원자료(aggregate_raw_<model>_<split>.csv) 여러 개를 하나로. 열: model, cond, seed, labeled[, unlabeled], 지표."""
    frames = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[WARN] --baseline_raw 파일이 없습니다(건너뜀): {p}"); continue
        b = pd.read_csv(p, encoding="utf-8-sig")
        if "labeled" not in b.columns:
            b["labeled"] = np.nan
        if "unlabeled" not in b.columns:
            b["unlabeled"] = np.nan
        frames.append(b)
    return pd.concat(frames, ignore_index=True) if frames else None


_COND_ORDER = {"supervised": 0, "ours": 1, "upper": 2}
_BASE_COND = {"sup": "supervised (labeled only)", "pl": "+ consensus pseudo-labels", "ceil": "upper (true labels)"}
_DISPLAY = {"efficientat_mn10": "EfficientAT mn10", "ced_tiny": "CED-Tiny", "panns_cnn6": "PANNs CNN6"}


def _disp(model):
    return _DISPLAY.get(str(model), str(model))


def _ours_u(cond):
    """'ours (u2000)' -> 2000, 그 밖(supervised·upper)은 NaN."""
    import re
    m = re.match(r"^ours \(u(\d+)\)$", str(cond))
    return float(m.group(1)) if m else np.nan


def label_eff_rows(le, base, full_labeled):
    """표 4 행: [(labeled, 모델, 조건, {지표: 문자열}, n)]. le = paper_table_label_eff csv, base = 비교모델 원자료(없으면 None)."""
    rows = []
    if le is None or le.empty:
        return rows
    levels = sorted(le["labeled"].unique(), reverse=True)
    for lab in levels:
        g = le[le["labeled"] == lab].copy()
        # 본표에는 supervised · ours(그 수준에서 unlabeled 최대) · upper 만. ours(u500)·ours(u1000) 같은 중간점은 표 5(곡선) 몫.
        u_of = g["condition"].map(_ours_u)
        umax = u_of.max() if u_of.notna().any() else None
        g = g[(u_of.isna()) | (u_of == umax)].copy()
        g["_o"] = g["condition"].map(lambda c: _COND_ORDER.get(str(c).split(" ")[0], 9))
        for _, r in g.sort_values("_o").iterrows():
            vals = {c: (f"{r[f'{c} (mean)']:.4f} ± {r[f'{c} (std)']:.4f}" if f"{c} (mean)" in le.columns and not np.isnan(r[f"{c} (mean)"])
                        and not np.isnan(r.get(f"{c} (std)", np.nan)) else
                        (f"{r[f'{c} (mean)']:.4f}" if f"{c} (mean)" in le.columns and not np.isnan(r[f"{c} (mean)"]) else "-"))
                    for c, _, _ in METRICS}
            rows.append((int(lab), "Proposed (Mark5.2)", str(r["condition"]), vals, int(r["N"])))
        if base is not None:
            bb = base[(base["labeled"].fillna(full_labeled).astype(int) == int(lab)) & (base["unlabeled"].isna())]
            for model in sorted(bb["model"].unique(), key=lambda m: list(_DISPLAY).index(m) if m in _DISPLAY else 9):
                for cond in ("sup", "pl", "ceil"):
                    gg = bb[(bb["model"] == model) & (bb["cond"] == cond)]
                    if gg.empty:
                        continue
                    rows.append((int(lab), _disp(model), _BASE_COND[cond], {c: gg[c].values if c in gg.columns else None for c, _, _ in METRICS}, len(gg)))
    return rows


def curve_rows(le, base, lab=200):
    """표 5 행: labeled=lab 고정, unlabeled 개수별. [(모델, unlabeled 표기, {지표: 문자열|배열}, n)]
    Mark5.2 행은 paper_table_label_eff 의 labeled=lab 행(supervised = u0, ours (uN) = N)에서 온다
    (aggregate_raw_<mv>_<split>.csv 는 본 조건 15회만 담고, 라벨 효율 회차는 label_eff 표에만 있다)."""
    rows = []
    if le is not None and not le.empty:
        g = le[le["labeled"] == lab].copy()
        g["_u"] = g["condition"].map(lambda c: 0.0 if str(c) == "supervised" else _ours_u(c))
        g = g[g["_u"].notna()].sort_values("_u")
        for _, r in g.iterrows():
            vals = {c: (f"{r[f'{c} (mean)']:.4f} ± {r[f'{c} (std)']:.4f}" if f"{c} (mean)" in le.columns and not np.isnan(r[f"{c} (mean)"])
                        and not np.isnan(r.get(f"{c} (std)", np.nan)) else
                        (f"{r[f'{c} (mean)']:.4f}" if f"{c} (mean)" in le.columns and not np.isnan(r[f"{c} (mean)"]) else "-"))
                    for c, _, _ in METRICS}
            rows.append(("Proposed (Mark5.2)", str(int(r["_u"])), vals, int(r["N"])))
    if base is not None and base["labeled"].notna().any():
        bb = base[base["labeled"].astype(float) == float(lab)]
        for model in sorted(bb["model"].unique(), key=lambda m: list(_DISPLAY).index(m) if m in _DISPLAY else 9):
            g0 = bb[(bb["model"] == model) & (bb["cond"] == "sup") & bb["unlabeled"].isna()]
            if len(g0):
                rows.append((_disp(model), "0", {c: g0[c].values for c, _, _ in METRICS}, len(g0)))
            pl = bb[(bb["model"] == model) & (bb["cond"] == "pl")]
            for u in sorted(pl["unlabeled"].dropna().unique()):
                g = pl[pl["unlabeled"] == u]
                rows.append((_disp(model), str(int(u)), {c: g[c].values for c, _, _ in METRICS}, len(g)))
            gf = pl[pl["unlabeled"].isna()]
            if len(gf):
                rows.append((_disp(model), "all consensus", {c: gf[c].values for c, _, _ in METRICS}, len(gf)))
    return rows


def plsource_rows(ps):
    rows = []
    if ps is None or ps.empty:
        return rows
    names = {"consensus": "8 binary experts, consensus (ours)", "self": "self-training (student trained on 200)",
             "gen": "single generalist (EfficientAT mn10 fine-tuned on 200)"}
    for _, r in ps.iterrows():
        vals = {c: (f"{r[f'{c} (mean)']:.4f} ± {r[f'{c} (std)']:.4f}" if f"{c} (mean)" in ps.columns and not np.isnan(r[f"{c} (mean)"]) else "-")
                for c, _, _ in METRICS}
        rows.append((int(r["labeled"]), names.get(str(r["pseudo_source"]), str(r["pseudo_source"])), vals, int(r["N"])))
    return rows


OPENSET_COLS = [("far_to_others", "far_heldout→others ①"), ("bnd_to_others", "boundary_B→others ①"),
                ("target_to_others", "test target→others ②"), ("others_recall_test", "test others recall"),
                ("Accuracy", "Accuracy"), ("Macro Precision", "Macro P"), ("Macro Recall", "Macro R"), ("Macro F1", "Macro F1")]


def openset_rows(os_tab):
    rows = []
    if os_tab is None or os_tab.empty:
        return rows
    for _, r in os_tab.iterrows():
        n = int(r["n"])
        vals = {}
        for k, _ in OPENSET_COLS:
            m, sd = r.get(f"{k}_mean", np.nan), r.get(f"{k}_sd", np.nan)
            vals[k] = "-" if (m is None or np.isnan(m)) else (f"{m:.4f}" if (n <= 1 or np.isnan(sd)) else f"{m:.4f} ± {sd:.4f}")
        rows.append((str(r["model"]), str(r["group"]), n, vals, str(r.get("rule", ""))))
    return rows


def load_zeroshot(mark_version):
    """plots/zeroshot_<run_tag>.json 들을 (모델, 묶음) 로 모은다. 동작 확인용(_limit)은 뺀다. 반환: [(이름, [json...])]"""
    import glob, json, re
    ours, clap = {}, {}
    for p in sorted(glob.glob(os.path.join(PLOT_DIR, "zeroshot_*.json"))):
        name = os.path.basename(p)[len("zeroshot_"):-len(".json")]
        if "_limit" in name or name.startswith("confusion_") or name.startswith("predictions_"):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if name.startswith("clap_"):
            clap.setdefault(name, []).append(d)
        elif name.startswith(mark_version):
            m = re.match(r"^(?P<g>.+)_s\d+$", name)
            ours.setdefault(m.group("g") if m else name, []).append(d)
    return [(f"Proposed (Mark5.2) {g}", v) for g, v in ours.items()] + [(f"LAION-CLAP {g[len('clap_'):]}", v) for g, v in clap.items()]


ZS_COLS = [("zeroshot6", "accuracy", "zero-shot 6 acc"), ("zeroshot6", "macro_recall_new6", "new-6 macro R"),
           ("zeroshot6", "macro_precision_new6_15way", "new-6 macro P (15-way)"), ("zeroshot6", "macro_recall_new6_15way", "new-6 macro R (15-way)"),
           ("zeroshot6", "macro_f1_new6_15way", "new-6 macro F1 (15-way)"),
           ("test9_with_15_anchors", "accuracy", "test-9 Accuracy"), ("test9_with_15_anchors", "macro_precision", "test-9 Macro P"),
           ("test9_with_15_anchors", "macro_recall", "test-9 Macro R"), ("test9_with_15_anchors", "macro_f1", "test-9 Macro F1")]


def zeroshot_rows(groups, lat=None):
    """표 8 행: [(이름, n, {열: 문자열}, {새 클래스: recall 문자열}, 파라미터 문자열)]"""
    rows = []
    for name, ds in groups:
        n = len(ds)
        vals = {}
        for sec, key, label in ZS_COLS:
            v = [d.get(sec, {}).get(key, np.nan) for d in ds]
            vals[label] = fmt_ms(v)
        per = {}
        classes = list(ds[0].get("new_classes", []))
        for c in classes:
            v = [d.get("zeroshot6", {}).get("per_class", {}).get(c, {}).get("recall", np.nan) for d in ds]
            per[c] = fmt_ms(v, nd=3)
        params = ds[0].get("model", {}).get("parameters")
        if not params and name.startswith("Proposed") and lat is not None and "total_parameters" in lat.index:
            params = float(lat.loc["total_parameters"]["mean"])
        rows.append((name, n, vals, per, (f"{int(round(params)):,}" if params else "")))
    return rows


def _md_table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return L


def _tex_escape(x):
    return str(x).replace("_", r"\_").replace("%", r"\%").replace("→", r"$\rightarrow$").replace("±", r"$\pm$").replace("①", "(1)").replace("②", "(2)")


def _tex_table(caption, label, header, rows, colspec=None):
    colspec = colspec or ("l" * 1 + "c" * (len(header) - 1))
    L = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
         "\\begin{tabular}{" + colspec + "}", "\\toprule", " & ".join(_tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "midrule":
            L.append("\\midrule"); continue
        L.append(" & ".join(_tex_escape(x) for x in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def _tests_rows(tests, short_map):
    """검정 CSV -> (비교, 지표, 평균차, d, p, Holm p) 행. 지표 5종 전부."""
    out = []
    if tests is None or tests.empty:
        return out
    for cmp_name in tests["비교"].drop_duplicates():
        for c, s_, _ in METRICS:
            sub = tests[(tests["비교"] == cmp_name) & (tests["지표"] == c)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            out.append((cmp_name, s_, f"{float(r['평균차']):.4f}", f"{float(r['Cohen' + chr(39) + 's d']):.2f}",
                        f"{float(r['p (paired)']):.6f}", f"{float(r['p (paired, Holm)']):.4f}"))
    return out


def build_extra(mark_version, split, raw_all, base, lat):
    """[2026-09-17 / 11절] 표 4~8 + 추가 검정을 마크다운·LaTeX 두 벌로. 자료가 없는 표는 그 사실을 적고 건너뛴다."""
    md, tex = [], []
    full_labeled = int(raw_all["labeled"].max()) if "labeled" in raw_all.columns and raw_all["labeled"].notna().any() else 0
    metric_hdr = [s_ for _, s_, _ in METRICS]

    # 표 4 라벨 효율
    le = _load_csv(f"paper_table_label_eff_{mark_version}_{split}.csv")
    rows4 = label_eff_rows(le, base, full_labeled)
    md += ["## 표 4. 라벨 효율(본표) — labeled 개수별, 세 기준선(supervised / ours / upper) + 비교모델", ""]
    if rows4:
        md += _md_table(["Labeled", "Model", "Condition"] + metric_hdr + ["n"],
                        [(lab, mdl, cond, *[cell(vals[c]) for c, _, _ in METRICS], n) for lab, mdl, cond, vals, n in rows4])
        md += ["", "캡션: mean ± SD over seeds. supervised = labeled 만(u0), ours = labeled + unlabeled 합의 pseudo-label, upper = unlabeled 에 진짜 라벨.",
               "비교모델 행은 aggregate_baseline.py 원자료(--baseline_raw)에서 같은 labeled 수준·같은 역할끼리 붙였다.", ""]
        tex_rows, last = [], None
        for lab, mdl, cond, vals, n in rows4:
            if last is not None and lab != last:
                tex_rows.append("midrule")
            tex_rows.append((lab, mdl, cond, *[cell(vals[c], tex=True) for c, _, _ in METRICS]))
            last = lab
        tex += _tex_table(f"Label efficiency on the {split} set (mean $\\pm$ SD over seeds).", "tab:label_eff",
                          ["Labeled", "Model", "Condition"] + metric_hdr, tex_rows, "rll" + "c" * len(METRICS))
    else:
        md += ["(자료 없음 — aggregate_results.py 가 만든 paper_table_label_eff_*.csv 가 없다)", ""]

    # 표 5 데이터 효율 곡선(labeled 200)
    rows5 = curve_rows(le, base, lab=200)
    md += ["## 표 5. 데이터 효율 곡선 — labeled 200 고정, unlabeled 개수별", ""]
    if rows5:
        md += _md_table(["Model", "Unlabeled"] + metric_hdr + ["n"],
                        [(mdl, u, *[cell(vals[c]) for c, _, _ in METRICS], n) for mdl, u, vals, n in rows5])
        md += ["", "캡션: mean ± SD over seeds. 비교모델의 unlabeled 는 합의 라벨(consensus_labels.csv) 서브샘플 개수, 'all consensus' 는 전부.", ""]
        tex += _tex_table("Data efficiency with 200 labeled clips (mean $\\pm$ SD over seeds).", "tab:curve_l200",
                          ["Model", "Unlabeled"] + metric_hdr,
                          [(mdl, u, *[cell(vals[c], tex=True) for c, _, _ in METRICS]) for mdl, u, vals, n in rows5], "lr" + "c" * len(METRICS))
    else:
        md += ["(자료 없음 — labeled 200 회차가 원자료에 없다)", ""]

    # 표 6 pseudo-label 출처
    ps = _load_csv(f"paper_table_plsource_{mark_version}_{split}.csv")
    rows6 = plsource_rows(ps)
    md += ["## 표 6. pseudo-label 출처 비교 — labeled 200 + unlabeled 3,800, 라벨을 붙이는 주체만 바꿈", ""]
    if rows6:
        md += _md_table(["Labeled", "Pseudo-label source"] + metric_hdr + ["n"],
                        [(lab, src, *[cell(vals[c]) for c, _, _ in METRICS], n) for lab, src, vals, n in rows6])
        md += ["", "캡션: mean ± SD over seeds. 세 조건 모두 같은 3,800 풀(dataset_unlabeled 2,000 + 남은 labeled 1,800)에 라벨을 붙였다.", ""]
        tex += _tex_table("Pseudo-label source comparison (200 labeled + 3,800 unlabeled; mean $\\pm$ SD over seeds).", "tab:plsource",
                          ["Labeled", "Pseudo-label source"] + metric_hdr,
                          [(lab, src, *[cell(vals[c], tex=True) for c, _, _ in METRICS]) for lab, src, vals, n in rows6], "rl" + "c" * len(METRICS))
    else:
        md += ["(자료 없음 — paper_table_plsource_*.csv 가 없다)", ""]

    # 표 7 open-set
    os_tab = _load_csv("openset_table.csv")
    rows7 = openset_rows(os_tab)
    md += ["## 표 7. open-set — ① 학습에 없던 소리를 others 로 보낸 비율, ② test 타겟을 others 로 잘못 보낸 비율", ""]
    if rows7:
        hdr = ["Model", "Run group", "n"] + [lbl for _, lbl in OPENSET_COLS] + ["Operating rule"]
        md += _md_table(hdr, [(m_, g_, n_, *[v_[k] for k, _ in OPENSET_COLS], r_) for m_, g_, n_, v_, r_ in rows7])
        md += ["", "캡션: ① 은 클수록 좋고(정답이 전부 others 인 세트, = 1 − Others FPR), ② 는 작을수록 좋다. 둘은 서로 당기므로 항상 같이 본다.",
               "운영점은 각 모델이 test 에서 쓰는 규칙 그대로(세트를 보고 다시 고르지 않음). 값은 seed 평균 ± SD.", ""]
        tex += _tex_table("Open-set behaviour: (1) fraction of unseen sounds sent to others; (2) fraction of target clips wrongly sent to others.",
                          "tab:openset", ["Model", "n"] + [lbl for _, lbl in OPENSET_COLS],
                          [(m_, n_, *[v_[k] for k, _ in OPENSET_COLS]) for m_, g_, n_, v_, r_ in rows7], "lr" + "c" * len(OPENSET_COLS))
    else:
        md += ["(자료 없음 — model/openset_report.py 의 plots/openset_table.csv 가 없다)", ""]

    # 표 8 제로샷
    rows8 = zeroshot_rows(load_zeroshot(mark_version), lat)
    md += ["## 표 8. 제로샷 — 학습에 없던 6종(door·treadmill·golf_putting·violin·piano·cat)을 문장 앵커만 추가해 판정", ""]
    if rows8:
        zs_hdr = [lbl for _, _, lbl in ZS_COLS]
        md += _md_table(["Model", "Params", "n"] + zs_hdr, [(nm, pr or "-", n_, *[v_[l] for l in zs_hdr]) for nm, n_, v_, per, pr in rows8])
        classes = list(rows8[0][3].keys())
        if classes:
            md += ["", "새 클래스별 재현율:", ""]
            md += _md_table(["Model"] + classes, [(nm, *[per.get(c, "-") for c in classes]) for nm, n_, v_, per, pr in rows8])
        md += ["", "캡션: 15 앵커(기존 9 + 새 6)로 판정. test-9 는 정규 test 430 클립을 같은 15 앵커로 판정한 9종 지표(새 앵커를 붙인 뒤에도 기존 성능이 유지되는지).",
               "CLAP 행은 LAION-CLAP 체크포인트에 같은 문장을 넣은 참조선(학습 0회). Params 는 모델 전체 파라미터 수(CLAP json 의 값).", ""]
        tex += _tex_table("Zero-shot extension to six unseen classes (15 text anchors; mean $\\pm$ SD over seeds).", "tab:zeroshot",
                          ["Model", "n"] + zs_hdr, [(nm, n_, *[v_[l] for l in zs_hdr]) for nm, n_, v_, per, pr in rows8], "lr" + "c" * len(zs_hdr))
        if classes:
            tex += _tex_table("Zero-shot recall per unseen class.", "tab:zeroshot_per_class", ["Model"] + classes,
                              [(nm, *[per.get(c, "-") for c in classes]) for nm, n_, v_, per, pr in rows8], "l" + "c" * len(classes))
    else:
        md += ["(자료 없음 — model/zeroshot_eval.py / clap_zeroshot.py 의 plots/zeroshot_*.json 이 없다)", ""]

    # 추가 검정
    extra_tests = [("라벨 효율 (수준별 u0 vs u<max>, u<max> vs ceil)", _load_csv(f"aggregate_tests_label_eff_{mark_version}_{split}.csv")),
                   ("pseudo-label 출처 (consensus 대비)", _load_csv(f"aggregate_tests_plsource_{mark_version}_{split}.csv")),
                   ("비교모델 대비 (seed 짝지음)", _load_csv(f"aggregate_tests_vs_baseline_{mark_version}_{split}.csv"))]
    any_t = False
    for title, t in extra_tests:
        rows_t = _tests_rows(t, None)
        if not rows_t:
            continue
        if not any_t:
            md += ["## 본문 검정(추가) — 표 4·6 과 비교모델", ""]; any_t = True
        md += [f"### {title}", ""] + _md_table(["비교", "지표", "평균차", "Cohen's d", "p (paired)", "p (Holm)"], rows_t) + [""]
        tex.append(f"% ---- 검정: {title}")
        tex += [f"% {a} / {b}: diff {c_}, d = {d_}, paired t p = {e_}, Holm p = {f_}" for a, b, c_, d_, e_, f_ in rows_t] + [""]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


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
    ap.add_argument("--mark_version", type=str, default="mark5.2")   # [변경 2026-09-17] 기본 mark5.2
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--extra_csv", type=str, default="",
                    help="비교모델 결과 CSV(열: model + 지표 5종). 표 1 에 행으로 붙는다.")
    ap.add_argument("--baseline_raw", type=str, nargs="*", default=[],
                    help="[2026-09-17 / 11절] 비교모델 원자료 aggregate_raw_<model>_<split>.csv 들 — 표 4·5 에 비교모델 행이 붙는다")
    args = ap.parse_args()

    raw_all = load_raw(args.mark_version, args.split)
    raw_main = main_slice(raw_all)                        # [2026-09-17] 표 1·2 는 본 조건(labeled 전량·spill 아님·consensus)만
    raw, raw_ceil = split_conditions(raw_main)            # [2026-09-06] 천장 행 분리
    if raw.empty:
        raise SystemExit("[ERROR] 반지도 조건(cond=ssl) 원자료가 없습니다.")
    lat = load_latency(args.mark_version)
    tests = load_tests(args.mark_version, args.split)
    gap = load_gap(args.mark_version, args.split)

    rows, top = proposed_rows(raw, args.extra_csv, raw_ceil)
    n_seeds = int(raw[raw["unlabeled"] == top].shape[0])

    md = build_markdown(raw, lat, tests, rows, top, args.split, n_seeds, raw_ceil=raw_ceil, gap=gap)
    tex = build_latex(raw, lat, tests, rows, top, args.split, n_seeds, raw_ceil=raw_ceil, gap=gap)
    base = load_baseline_raw(args.baseline_raw) if args.baseline_raw else None
    md_x, tex_x = build_extra(args.mark_version, args.split, raw_all, base, lat)   # [2026-09-17 / 11절] 표 4~8 + 추가 검정
    md += "\n" + md_x
    tex += "\n" + tex_x

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
