# aggregate_results.py
# [신설 2026-08-30 / Mark5_EXP]
# [확장 2026-09-03] 95% 신뢰구간 · min/max · 중앙값 · 사분위수 · 유의성 검정 · 효과크기 추가.
# [수정 2026-09-03] 코드 검수 반영 — H3·M1·M2·M3·M4·M5 및 낮음 5건.
#
# run_experiments.py 가 만든 25회분 결과를 모아 논문에 넣을 표와 그림을 만든다.
#
#   표 1 (성능)      : 조건별 Accuracy · Macro P · Macro R · Macro F1 · ROC AUC 의
#                      평균 ± 표준편차 · 95% 신뢰구간 · min/max · 중앙값 · 사분위수
#   표 1-b (검정)    : 조건 쌍 사이의 대응표본 t-검정 · Welch t-검정 · Mann-Whitney U ·
#                      Cohen's d · Holm 보정 p
#   그림 1 (데이터효율곡선) : 가로축 unlabeled 개수, 세로축 성능, 오차막대 = 표준편차
#   표 2 보조        : 속도(profile_efficiency.py) 반복 측정 집계
#
# --------------------------------------------------------------------------
# 통계량을 어떻게 내는가 — 다섯 가지를 구분한다
# --------------------------------------------------------------------------
# (1) 표본표준편차(ddof=1)
#     우리가 가진 5회는 "돌릴 수 있는 모든 학습"이 아니라 그중 뽑은 5개 표본이다.
#     모집단 표준편차(ddof=0)를 쓰면 흩어짐을 과소평가한다.
#     논문에서 "N=5, mean ± SD" 로 적을 때의 관례이기도 하다.
#
# (2) 95% 신뢰구간 — 표준편차와 뜻이 다르다
#     표준편차는 "5회가 서로 얼마나 흩어져 있나"이고,
#     신뢰구간은 "그 평균값 자체를 얼마나 믿을 수 있나"이다.
#
#     ⚠ n=5 는 30 보다 훨씬 작으므로 z(1.96) 가 아니라 t 분포를 써야 한다.
#       자유도 4 에서 t = 2.776 이고, 구간은  평균 ± 2.776 x SD / sqrt(5) 다.
#       z 를 쓰면 구간이 1.42배 좁게 나와 틀린 값이 된다(실측 확인).
#
# (3) min · max · 중앙값 · 사분위수(Q1 · Q3 · IQR)
#     5회 중 하나가 크게 무너진 경우가 있는지 보여준다.
#     흩어짐이 대칭이 아니면 평균·표준편차 대신 중앙값·사분위수를 쓰라는 것이 권고이므로
#     둘 다 뽑아 두고, 실제 분포를 본 뒤 논문에 어느 쪽을 적을지 고른다.
#
# (4) 유의성 검정 — 대응표본과 독립표본을 둘 다 낸다  [M4, 2026-09-03]
#     train_mark5.py 는 unlabeled 뽑기에 전역 난수기가 아니라 random.Random(seed) 를
#     따로 쓴다. 그래서 "같은 seed = 같은 초기 가중치, 달라지는 것은 unlabeled 양 하나뿐"이
#     되도록 조건 사이가 seed 로 **짝지어져(blocking)** 있다.
#       - 대응표본 t-검정 : 그 짝짓기를 살린 주 결과. seed 자체의 편차가 오차에서 빠져 검정력이 높다.
#       - Welch t-검정    : 짝짓기를 무시한 보수적 하한. 틀린 것이 아니라 검정력이 낮을 뿐이다.
#       - Mann-Whitney U  : 정규성을 가정하지 않는 비모수. n=5 대 5 의 양측 최소 p 가 0.0079 라
#                           그 아래로 못 내려간다.
#     ※ 대응표본 비모수(Wilcoxon signed-rank)는 일부러 쓰지 않는다. n=5 에서 양측 최소 p 가
#       0.0625 라 어떤 경우에도 0.05 를 못 넘는다.
#     ※ 짝짓기가 완전하지는 않다 — 초기 가중치는 같지만 unlabeled 개수가 달라지면 배치 구성과
#       셔플 순서가 달라진다. 그래서 대응표본을 주 결과로 하되 Welch 를 함께 싣는다.
#
# (5) 다중비교 보정 (Holm)  [M5, 2026-09-03]
#     쌍 5개 x 지표 6개 = 30번을 검정 종류마다 돌리므로 우연히 유의해지는 것이 나온다.
#     Holm-Bonferroni 로 보정한 p 를 같이 낸다. 논문에는 보정 전후를 같이 적는다.
#
# --------------------------------------------------------------------------
# 사용법
# --------------------------------------------------------------------------
#   python model/aggregate_results.py --mark_version mark5.0
#   python model/aggregate_results.py --mark_version mark5.0 --split val
#   python model/aggregate_results.py --mark_version mark5.0 --latency_only
#   python model/aggregate_results.py --mark_version mark5.0 --no_tests   # 검정 생략
#   python model/aggregate_results.py --mark_version mark5.0 --allow_mixed_config
#
# --------------------------------------------------------------------------
# [신설 2026-09-06] 천장(upper bound) 조건 — _ceil_s<k> 태그
# --------------------------------------------------------------------------
# run_experiments.py --ceiling 이 만든 performance_summary_<mv>_ceil_s<k>.csv 를 따로 모아
#   · 조건별 요약·논문표에 "ceiling" 행을 덧붙이고(기존 u0~u2000 행은 그대로),
#   · u0 vs ceil · u<max> vs ceil 대응표본 검정을 같은 검정표 CSV 에 이어 쓰고(별도 Holm family),
#   · 메운 비율 gap_closed = (u<max> − u0) ÷ (ceil − u0) 를 지표별로 내어
#     aggregate_gap_closed_<mv>_<split>.csv 에 저장한다(평균 기준과 seed 짝 기준 둘 다),
#   · 데이터효율곡선에 상한을 지표별 색의 가는 점선으로 그린다.
# 원자료 CSV(aggregate_raw_*)에는 cond 열(ssl/ceil)이 생긴다. make_paper_tables.py 가 그 열을 본다.

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as _sps
    HAS_SCIPY = True
except ImportError:                                # scipy 가 없어도 나머지는 돌아가야 한다
    HAS_SCIPY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

# performance_summary CSV 의 요약 블록에 있는 행 이름. eval.py 와 문자열이 같아야 한다.
METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
           "ROC AUC (Macro, OvR)", "Others FPR"]

# 콘솔에 검정 결과를 뿌릴 때 쓰는 대표 지표(전체는 CSV 에 저장된다).
KEY_METRICS = ["Accuracy", "Macro F1"]

# 저장 CSV 인코딩. 열 이름에 한글과 ± 가 들어가는데 utf-8 로 쓰면 윈도우 엑셀에서 깨진다. [낮음]
CSV_ENCODING = "utf-8-sig"


def _read_summary(path):
    """performance_summary CSV 의 앞쪽 요약 블록만 dict 로 읽는다.

    파일 구조가 [주석 1줄][빈 줄][요약 표][빈 줄][# Class-wise Metrics][클래스별 표] 라
    통째로 read_csv 하면 두 표가 섞인다. 'Class' 가 나오는 줄에서 끊는다.

    ⚠ [수정 2026-09-03] 반드시 csv.reader 로 읽는다. line.split(",") 을 쓰면
      지표 이름 안에 쉼표가 있는 행을 통째로 놓친다. eval.py 는 pandas 로 쓰기 때문에
      "ROC AUC (Macro, OvR)" 가 따옴표에 감싸여 저장되는데, split(",") 은 그것을
      ['"ROC AUC (Macro', ' OvR)"', '0.9871'] 로 쪼개고 float(' OvR)"') 가 실패해
      그 행이 버려졌다. 논문 표 1 에 들어갈 지표가 조용히 빠지는 버그였다.
    """
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f):
            if not parts:
                continue
            key = parts[0].strip()
            if key.startswith("#") or not key:
                continue
            if key == "Class":                   # 클래스별 표 시작 -> 요약 끝
                break
            if key == "Metric":
                continue
            if len(parts) >= 2:
                try:
                    out[key] = float(parts[1])
                except ValueError:
                    pass
    return out


def is_complete(path):
    """[M1, 2026-09-03] 요약 CSV 가 온전한지 본다.

    _read_summary 는 한 줄씩 float 로 바꿔보고 실패한 줄만 버리므로, 파일이 중간에
    잘려 있어도 조용히 통과한다. 실제로 120바이트에서 자른 파일에서 마지막 자리가
    잘린 0.9871412218852 가 유효한 값으로 통과하는 것을 확인했다.
    METRICS 6개가 다 들어왔는지로 완결성을 판정한다.
    """
    try:
        d = _read_summary(path)
    except Exception:
        return False, []
    missing = [m for m in METRICS if m not in d]
    return (len(missing) == 0), missing


def _parse_tag(tag):
    """run_tag 를 (설정, unlabeled 개수, seed) 로 가른다.

    ⚠ [H3 수정 2026-09-03] 예전에는 re.search 로 _u·_s 만 찾아서, 설정이 다른 실험이라도
      _u<n>_s<k> 만 붙어 있으면 같은 조건에 합쳐졌다. mark5.0_PL_SA_u2000_s* 5개에
      _DA05 3개와 _DKD 3개가 섞이자 경고 없이 N=11, 평균 0.874 -> 0.7655 가 됐다.
      앞부분(설정 태그)까지 잡아내 나중에 혼입을 검사한다.
    """
    m = re.match(r"^(?P<base>.+?)_u(?P<n>\d+)_s(?P<s>\d+)$", tag)
    if m:
        return m.group("base"), int(m.group("n")), int(m.group("s")), "ssl"
    # [2026-09-06] 천장 조건: _u<n> 자리에 _ceil 이 들어간다(vild_config.run_tag).
    m = re.match(r"^(?P<base>.+?)_ceil_s(?P<s>\d+)$", tag)
    if m:
        return m.group("base"), None, int(m.group("s")), "ceil"
    return None, None, None, None


def collect(mark_version, split, allow_mixed=False):
    """조건별로 결과를 모은다. split='test' 면 태그 없는 파일, 'val' 이면 _val 파일."""
    suffix = "" if split == "test" else f"_{split}"
    patterns = [os.path.join(PLOT_DIR, f"performance_summary_{mark_version}*_u*_s*{suffix}.csv"),
                os.path.join(PLOT_DIR, f"performance_summary_{mark_version}*_ceil_s*{suffix}.csv")]  # [2026-09-06] 천장

    rows, broken = [], []
    seen = set()
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        if path in seen:
            continue
        seen.add(path)
        name = os.path.basename(path)
        tag = name[len("performance_summary_"):-len(".csv")]
        if suffix and tag.endswith(suffix):
            tag = tag[: -len(suffix)]
        elif not suffix and re.search(r"_(val)$", tag):
            continue                              # test 를 모을 때 _val 파일은 제외
        base, n, s, cond = _parse_tag(tag)
        if cond is None:
            continue

        ok, missing = is_complete(path)           # [M1] 잘린 파일 검출
        if not ok:
            broken.append((name, missing))
            continue

        rec = {"config": base, "cond": cond, "unlabeled": n, "seed": s, "run_tag": tag}
        rec.update(_read_summary(path))
        rows.append(rec)

    if broken:
        print("\n[WARN] 요약 CSV 가 온전하지 않아 제외한 파일:")
        for name, missing in broken:
            print(f"  {name}  (빠진 지표: {', '.join(missing) if missing else '읽기 실패'})")
        print("[WARN] 학습이 끝났는데 평가만 실패한 것일 수 있습니다. eval 을 다시 돌리십시오.")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 천장 행의 unlabeled 칸: 정의상 unlabeled 전량(=u 최댓값)에 라벨을 붙인 것이므로 그 값을 적는다.
    # (조건 구분은 cond 열이 맡는다. 같은 unlabeled 값이라도 cond 가 다르면 다른 조건이다.)
    if (df["cond"] == "ceil").any():
        ssl_max = df.loc[df["cond"] == "ssl", "unlabeled"].max()
        df.loc[df["cond"] == "ceil", "unlabeled"] = int(ssl_max) if pd.notna(ssl_max) else 2000
    df["unlabeled"] = df["unlabeled"].astype(int)
    df = df.sort_values(["cond", "unlabeled", "seed"], ascending=[False, True, True]).reset_index(drop=True)

    # [H3] 설정이 다른 결과가 섞였는지 검사
    configs = sorted(df["config"].unique())
    if len(configs) > 1:
        print("\n[ERROR] 설정이 다른 실험 결과가 섞여 있습니다:")
        for c in configs:
            print(f"  {c}  ({(df['config'] == c).sum()}건)")
        if not allow_mixed:
            print("[ERROR] 이대로 집계하면 조건별 평균이 오염됩니다.")
            print("        섞을 의도였다면 --allow_mixed_config 를 주십시오.")
            sys.exit(1)
        print("[WARN] --allow_mixed_config 로 그대로 진행합니다.")

    return df


# ==========================================================================
# 통계량
# ==========================================================================

def t_crit(n, conf=0.95):
    """자유도 n-1 의 양측 t 임계값. scipy 가 없으면 표에서 찾아 쓴다.

    n=5 -> df=4 -> 2.776 이다. z(1.96) 를 쓰면 구간이 좁게 나와 틀린다.
    """
    if n < 2:
        return float("nan")
    if HAS_SCIPY:
        return float(_sps.t.ppf(1 - (1 - conf) / 2.0, df=n - 1))
    # scipy 가 없을 때를 위한 95% 양측 t 표(df 1~20). 우리 n=5 는 df=4.
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
             14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
             20: 2.086}
    df_ = n - 1
    if df_ in table:
        return table[df_]
    # [낮음 수정] df>20 에서 1.96 을 조용히 돌려주면 구간이 좁아진다. 표 끝값을 쓰고 알린다.
    print(f"[WARN] scipy 없이 df={df_} 의 t 값을 못 구해 df=20 값(2.086)을 씁니다. "
          f"구간이 실제보다 약간 좁을 수 있습니다.")
    return table[20]


def ci95(values):
    """평균의 95% 신뢰구간 (하한, 상한, 반폭). t 분포 기준."""
    x = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    sd = x.std(ddof=1)
    half = t_crit(n) * sd / np.sqrt(n)
    return m - half, m + half, half


# 값이 "전부 같다"를 판정할 때 쓰는 허용오차.  [재검수 반영 2026-09-03]
# `std == 0` 이나 `ptp == 0` 은 부동소수 뺄셈의 1 ulp 차이에 뚫린다. 실측:
#   k/430 눈금에서 같은 정수 차이(30)인 다섯 쌍 -> diff 의 ptp = 1.1e-16 (== 0 이 False)
#   그 상태로 ttest_rel 을 부르면 t = 3.1e15 · p = 6e-62 가 나와 논문 표에 실린다.
# 지표는 전부 [0, 1] 이고 실제 최소 눈금이 1/430 ≈ 2.3e-3 이므로, 1e-9 아래 흔들림은
# 계산 오차이지 실제 차이가 아니다.
_SAME_ATOL = 1e-9


def _all_same(v):
    """배열의 값이 (허용오차 안에서) 전부 같은가."""
    v = np.asarray(v, dtype=float)
    return len(v) > 0 and float(np.ptp(v)) <= _SAME_ATOL


def cohens_d(a, b):
    """두 집단의 효과크기. 합동표준편차(pooled SD)로 나눈 평균차."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled <= _SAME_ATOL:                       # 0 뿐 아니라 ulp 수준 분산도 "정의 안 됨"
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def d_label(d):
    """Cohen 의 통상 기준으로 크기를 말로 붙인다."""
    if pd.isna(d):
        return "-"
    ad = abs(d)
    if ad < 0.2:
        return "미미"
    if ad < 0.5:
        return "작음"
    if ad < 0.8:
        return "중간"
    return "큼"


def holm(pvals):
    """[M5] Holm-Bonferroni 보정. 입력 순서 그대로 보정된 p 를 돌려준다.

    작은 p 부터 (m-순위) 를 곱하고, 앞의 값보다 작아지지 않게 누적 최댓값을 취한 뒤 1 로 자른다.
    Bonferroni 보다 덜 보수적이면서 family-wise error 를 같은 수준으로 통제한다.
    """
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        out[i] = min(running, 1.0)
    return out


def summarize(df):
    """조건별 회수 · 평균 · 표본표준편차 · 95% 신뢰구간 · min · max · 중앙값 · 사분위수."""
    present = [m for m in METRICS if m in df.columns]
    rows = {}
    for n, g in df.groupby("unlabeled"):
        rec = {"N": len(g)}
        for m in present:
            v = g[m].dropna().values
            if len(v) == 0:
                continue
            lo, hi, half = ci95(v)
            # [M2] 지표마다 실제로 쓰인 표본 수를 따로 남긴다.
            # 그룹 크기(N)와 다를 수 있는데, 예전에는 N 만 표에 찍혀서
            # "N=5" 옆에 3회로 계산한 평균이 붙는 일이 생길 수 있었다.
            rec[f"{m} (n)"] = len(v)
            rec[f"{m} (mean)"] = v.mean()
            rec[f"{m} (std)"] = v.std(ddof=1) if len(v) > 1 else np.nan
            rec[f"{m} (ci_low)"] = lo
            rec[f"{m} (ci_high)"] = hi
            rec[f"{m} (ci_half)"] = half
            rec[f"{m} (min)"] = v.min()
            rec[f"{m} (max)"] = v.max()
            rec[f"{m} (median)"] = float(np.median(v))
            # 사분위수. linear 보간(numpy 기본). n=5 면 Q1 은 2번째, Q3 는 4번째 값이 된다.
            rec[f"{m} (q1)"] = float(np.percentile(v, 25))
            rec[f"{m} (q3)"] = float(np.percentile(v, 75))
            rec[f"{m} (iqr)"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        rows[n] = rec
    out = pd.DataFrame(rows).T
    out.index.name = "unlabeled"
    if "N" in out.columns:
        out["N"] = out["N"].astype(int)
    return out.sort_index()


def fmt_table(summary):
    """논문에 그대로 옮길 수 있는 문자열 표.

    지표마다 네 칸을 만든다.
      "<지표>"               -> "0.8744 ± 0.0045 (n=5)"
      "<지표> 95% CI"        -> "[0.8688, 0.8800]"          (t 분포 기준)
      "<지표> min~max"       -> "0.8698~0.8814"
      "<지표> 중앙값[Q1,Q3]"  -> "0.8744 [0.8710, 0.8780]"   (분포가 비대칭일 때 쓴다)

    [M2] n 을 값 옆에 같이 적는다. 조건마다, 지표마다 표본 수가 다를 수 있다.
    [M3] n=1 이면 표준편차가 0 인 것이 아니라 정의되지 않는다. "± 0.0000" 으로 적으면
         "흔들림이 전혀 없다"는 틀린 주장이 되므로 SD 없이 적는다.
    """
    rows = []
    for n, r in summary.iterrows():
        # [2026-09-06] 천장 행은 인덱스가 "ceiling" 문자열이다.
        row = {"unlabeled": (n if isinstance(n, str) else int(n)), "N": int(r["N"])}
        for m in METRICS:
            mk = f"{m} (mean)"
            if mk not in r or pd.isna(r[mk]):
                continue
            n_m = r.get(f"{m} (n)", np.nan)
            n_m = int(n_m) if not pd.isna(n_m) else 0
            sd = r.get(f"{m} (std)", np.nan)
            if pd.isna(sd):
                row[m] = f"{r[mk]:.4f} (n={n_m})"          # [M3] n=1 등
            else:
                row[m] = f"{r[mk]:.4f} ± {sd:.4f} (n={n_m})"
            lo, hi = r.get(f"{m} (ci_low)", np.nan), r.get(f"{m} (ci_high)", np.nan)
            row[f"{m} 95% CI"] = "-" if pd.isna(lo) else f"[{lo:.4f}, {hi:.4f}]"
            mn, mx = r.get(f"{m} (min)", np.nan), r.get(f"{m} (max)", np.nan)
            row[f"{m} min~max"] = "-" if pd.isna(mn) else f"{mn:.4f}~{mx:.4f}"
            md = r.get(f"{m} (median)", np.nan)
            q1, q3 = r.get(f"{m} (q1)", np.nan), r.get(f"{m} (q3)", np.nan)
            row[f"{m} 중앙값[Q1,Q3]"] = ("-" if pd.isna(md)
                                        else f"{md:.4f} [{q1:.4f}, {q3:.4f}]")
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_tests(df):
    """조건 쌍 사이의 차이를 검정한다.

    비교하는 쌍
      - 인접 쌍 (0 vs 500, 500 vs 1000, ...)  : "조금씩 늘려도 오르는가"
      - 양 끝 쌍 (0 vs 최대)                   : "합의가 얼마나 기여했는가" — 논문 핵심

    [M4] 대응표본과 독립표본을 둘 다 낸다. 조건 사이가 seed 로 짝지어져 있으므로
         대응표본이 주 결과이고 Welch 는 보수적 하한이다(파일 머리말 (4) 참조).
    [M5] 검정 종류마다 Holm 보정 p 를 함께 낸다.
    """
    ns = sorted(df["unlabeled"].unique())
    if len(ns) < 2:
        return pd.DataFrame()

    pairs = [(ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
    if (ns[0], ns[-1]) not in pairs:
        pairs.append((ns[0], ns[-1]))

    rows = []
    for lo_n, hi_n in pairs:
        rows += _compare_groups(df[df["unlabeled"] == lo_n], df[df["unlabeled"] == hi_n],
                                f"u{lo_n} vs u{hi_n}")
    return _finish_tests(rows)


def _compare_groups(a, b, label):
    """두 조건(a=작은쪽/기준, b=큰쪽)의 지표별 비교 행을 만든다. pairwise_tests 와 ceiling_tests 가 같이 쓴다.
    [2026-09-06] pairwise_tests 의 안쪽 루프를 그대로 떼어냈다(계산은 바뀌지 않음)."""
    present = [m for m in METRICS if m in a.columns and m in b.columns]
    rows = []
    if True:
        for m in present:
            x = a[m].dropna().values                       # 작은 조건
            y = b[m].dropna().values                       # 큰 조건
            if len(x) < 2 or len(y) < 2:
                continue
            d = cohens_d(y, x)                             # 큰 - 작은 (양수면 늘어서 좋아짐)
            rec = {
                "비교": label,
                "지표": m,
                "평균(작은쪽)": round(float(x.mean()), 4),
                "평균(큰쪽)": round(float(y.mean()), 4),
                "평균차": round(float(y.mean() - x.mean()), 4),
                "Cohen's d": round(d, 3) if not pd.isna(d) else np.nan,
                "효과크기": d_label(d),
                "n(작은쪽)": len(x), "n(큰쪽)": len(y),     # [낮음] n1/n2 라벨 혼동 제거
            }

            if HAS_SCIPY:
                # ---- 대응표본 (주 결과) : seed 로 짝을 맞춘다 ----
                merged = pd.merge(a[["seed", m]], b[["seed", m]],
                                  on="seed", suffixes=("_lo", "_hi")).dropna()
                if len(merged) >= 2:
                    lo_v = merged[f"{m}_lo"].values
                    hi_v = merged[f"{m}_hi"].values
                    diff = hi_v - lo_v
                    # [재검수 반영] `std == 0` 이 아니라 허용오차 안에서 "전부 같은가"로 본다.
                    # 상수 배열이라도 값에 따라 분산이 정확히 0 이 아니다. 실측:
                    #   [0.8]*5  -> std(ddof=1) = 0.0            (== 0  True)
                    #   [0.91]*5 -> std(ddof=1) = 1.241e-16      (== 0  False)  <- 가드를 통과
                    # diff 는 뺄셈 결과라 더 위험하다(_SAME_ATOL 주석의 실측 참조).
                    if _all_same(diff):
                        rec["paired t"] = np.nan
                        rec["p (paired)"] = np.nan
                        rec["비고"] = "대응표본 차이가 전부 같아 t 정의 안 됨"
                    else:
                        t_stat, t_p = _sps.ttest_rel(hi_v, lo_v)
                        rec["paired t"] = round(float(t_stat), 3)
                        rec["p (paired)"] = float(t_p)
                    rec["n(짝)"] = len(merged)
                else:
                    rec["paired t"] = np.nan
                    rec["p (paired)"] = np.nan
                    rec["n(짝)"] = len(merged)

                # ---- 독립표본 Welch (보수적 하한) ----
                # [재검수 반영] 예전 가드는 `std==0 and std==0 and 평균이 같음` 이었다.
                # 그래서 **두 조건이 각각 상수인데 평균이 다르면 일부러 통과**했고,
                # Welch 가 t = ±inf · p = 0.0 을 돌려줬다. Holm 보정도 0.0 이 되어
                # 논문 표에 "완벽하게 유의"로 실린다. 실은 분산 추정이 무너진 것뿐이다.
                # test 는 430 클립, Others FPR 의 분모는 47 이라 5회가 같은 값에 몰릴 수 있다.
                x_const = _all_same(x)
                y_const = _all_same(y)
                if x_const and y_const:
                    rec["Welch t"] = np.nan
                    rec["p (Welch)"] = np.nan
                    rec["비고"] = (rec.get("비고", "") +
                                  " / 두 조건 모두 값이 하나뿐이라 Welch t 정의 안 됨").strip(" /")
                else:
                    w_stat, w_p = _sps.ttest_ind(y, x, equal_var=False)
                    rec["Welch t"] = round(float(w_stat), 3)
                    rec["p (Welch)"] = float(w_p)

                # ---- 비모수 ----
                if _all_same(np.concatenate([x, y])):
                    rec["Mann-Whitney U"] = np.nan
                    rec["p (Mann-Whitney)"] = np.nan
                    rec["비고"] = (rec.get("비고", "") + " / 값이 전부 같아 MWU 불가").strip(" /")
                else:
                    u_stat, u_p = _sps.mannwhitneyu(y, x, alternative="two-sided")
                    rec["Mann-Whitney U"] = float(u_stat)
                    rec["p (Mann-Whitney)"] = float(u_p)
            rows.append(rec)
    return rows


def _finish_tests(rows):
    """비교 행 목록 -> DataFrame + 검정 종류별 Holm 보정 + 열 순서. 호출한 쪽의 행들이 하나의 family 다."""
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "비고" in out.columns:
        out["비고"] = out["비고"].fillna("")          # 화면·CSV 에서 정상 행이 NaN 으로 찍히지 않게

    # [M5] 검정 종류마다 따로 Holm 보정. 한 종류의 p 들이 하나의 family 다.
    for src, dst in [("p (paired)", "p (paired, Holm)"),
                     ("p (Welch)", "p (Welch, Holm)"),
                     ("p (Mann-Whitney)", "p (MWU, Holm)")]:
        if src in out.columns:
            out[dst] = holm(out[src].values)

    cols = ["비교", "지표", "평균(작은쪽)", "평균(큰쪽)", "평균차",
            "Cohen's d", "효과크기",
            "paired t", "p (paired)", "p (paired, Holm)",
            "Welch t", "p (Welch)", "p (Welch, Holm)",
            "Mann-Whitney U", "p (Mann-Whitney)", "p (MWU, Holm)",
            "n(작은쪽)", "n(큰쪽)", "n(짝)", "비고"]     # [낮음] 비고를 목록에 포함
    return out[[c for c in cols if c in out.columns]]


def ceiling_tests(df_ssl, df_ceil):
    """[2026-09-06] 천장 대 바닥(u0)·천장 대 제안(u<max>) 검정. pairwise_tests 와 별도 family 로 Holm 보정."""
    if df_ssl.empty or df_ceil.empty:
        return pd.DataFrame()
    ns = sorted(df_ssl["unlabeled"].unique())
    rows = []
    for n in ([ns[0], ns[-1]] if len(ns) > 1 else [ns[0]]):
        rows += _compare_groups(df_ssl[df_ssl["unlabeled"] == n], df_ceil, f"u{n} vs ceil")
    return _finish_tests(rows)


def gap_closed(df_ssl, df_ceil):
    """[2026-09-06] 메운 비율 = (u<max> − u0) ÷ (ceil − u0). Cances et al. (2022) 식 읽기.

    두 가지로 낸다.
      · 평균 기준  : 조건별 평균값으로 한 번 계산(논문 본문에 쓰는 값)
      · seed 짝 기준: 같은 seed 의 u0·u<max>·ceil 세 값으로 seed 마다 계산해 평균 ± SD(흔들림을 보여줌)
    분모(ceil − u0)가 허용오차 안에서 0 이하면 정의되지 않으므로 NaN 으로 둔다.
    """
    if df_ssl.empty or df_ceil.empty:
        return pd.DataFrame()
    ns = sorted(df_ssl["unlabeled"].unique())
    if len(ns) < 2:
        return pd.DataFrame()
    lo_n, hi_n = ns[0], ns[-1]
    a = df_ssl[df_ssl["unlabeled"] == lo_n]
    b = df_ssl[df_ssl["unlabeled"] == hi_n]
    c = df_ceil
    rows = []
    for m in [x for x in METRICS if x in df_ssl.columns and x in df_ceil.columns]:
        fa, fb, fc = a[m].mean(), b[m].mean(), c[m].mean()
        denom = fc - fa
        ratio = (fb - fa) / denom if denom > _SAME_ATOL else np.nan
        merged = a[["seed", m]].merge(b[["seed", m]], on="seed", suffixes=("_floor", "_ssl")) \
                               .merge(c[["seed", m]].rename(columns={m: f"{m}_ceil"}), on="seed").dropna()
        per_seed = []
        for _, r in merged.iterrows():
            d = r[f"{m}_ceil"] - r[f"{m}_floor"]
            per_seed.append((r[f"{m}_ssl"] - r[f"{m}_floor"]) / d if d > _SAME_ATOL else np.nan)
        ps = np.asarray(per_seed, dtype=float)
        ps = ps[~np.isnan(ps)]
        rows.append({
            "지표": m, "바닥 u%d" % lo_n: round(float(fa), 4), "제안 u%d" % hi_n: round(float(fb), 4),
            "천장 ceil": round(float(fc), 4), "천장−바닥": round(float(denom), 4),
            "메운 비율(평균 기준)": round(float(ratio), 4) if not np.isnan(ratio) else np.nan,
            "메운 비율(seed 짝 평균)": round(float(ps.mean()), 4) if len(ps) else np.nan,
            "메운 비율(seed 짝 SD)": round(float(ps.std(ddof=1)), 4) if len(ps) > 1 else np.nan,
            "n(짝)": int(len(ps)),
        })
    return pd.DataFrame(rows)


def plot_curve(summary, split, out_png, ceiling=None):
    """데이터효율곡선. 오차막대는 표본표준편차.

    [2026-09-06] ceiling(천장 요약, summarize() 결과 1행)을 주면 지표별 상한을 같은 색의 가는 점선으로
    가로로 긋고 ± SD 를 옅은 띠로 칠한다. 범례에는 "Upper bound (true labels)" 한 줄만 더 넣는다
    (지표 4개 × 점선 4개를 다 넣으면 범례가 그림 높이의 절반을 먹는다 — 7.2×4.5 in 기준 계산).
    그림 안 글자는 전부 영문이다(코랩·서버에 한글 폰트가 없어 □ 로 깨진다).

    [재검수 반영] SD 가 없는 조건(n=1)의 오차막대를 0 으로 그리지 않는다.
    예전에는 fillna(0) 이라 회차가 하나뿐인 조건이 "흔들림이 전혀 없다"로 보였다.
    표에서는 M3 로 이미 없앤 오해인데 그림에만 남아 있었다. NaN 으로 두면
    그 점의 막대만 안 그려지고 점 자체는 그대로 찍힌다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = summary.index.values.astype(float)

    # [2026-09-05] 논문에 그대로 넣을 수 있게 다듬었다.
    #   (1) 지표 4종 — Accuracy·Macro F1 만 그리던 것을 P·R 까지 넓혔다.
    #       수진님 지시: "Accuracy 와 Macro F1 이 나오는 곳에는 Precision·Recall 도 세트로."
    #       교수님이 P·R 을 요구하신 이유가 오류 방향(놓침 vs 오탐)이라, 곡선에서도 두 값이
    #       갈리는 구간이 보여야 한다(u0 은 P 0.853 / R 0.728 로 크게 갈린다).
    #   (2) x 오프셋 — 클래스가 균형이면 macro recall 이 accuracy 와 수학적으로 같아져
    #       두 선이 완전히 포개져 하나가 안 보인다. 점을 조금씩 어긋놓아 네 선을 다 보이게 한다.
    #       오프셋은 x 범위에 비례시켜 조건 개수가 바뀌어도 깨지지 않게 했다.
    #   (3) 300 dpi, 얇은 선·흰 마커, 위·오른쪽 테두리 제거, 연한 격자 — 인쇄 대비.
    SPEC = [("Accuracy",        "o", "-",  "#1f4e79", 1.6),
            ("Macro Precision", "^", "--", "#c55a11", 1.3),
            ("Macro Recall",    "v", ":",  "#2e7d32", 1.3),
            ("Macro F1",        "s", "-.", "#7b2d8e", 1.3)]

    span = float(xs.max() - xs.min()) if len(xs) > 1 else 0.0
    dodge = span * 0.011          # 2000 범위 기준 약 22

    plt.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    no_sd = []
    for i, (metric, marker, ls, color, lw) in enumerate(SPEC):
        mk, sk = f"{metric} (mean)", f"{metric} (std)"
        if mk not in summary.columns:
            continue
        ys = summary[mk].values.astype(float)
        if sk in summary.columns:
            es = summary[sk].values.astype(float)
            miss = summary.index.values[np.isnan(es)]
            if len(miss):
                no_sd.append(f"{metric}: u" + ", u".join(str(int(v)) for v in miss))
        else:
            es = np.full_like(ys, np.nan)
        ax.errorbar(xs + (i - (len(SPEC) - 1) / 2.0) * dodge, ys, yerr=es,
                    marker=marker, linestyle=ls, color=color, label=metric,
                    markersize=4.6, linewidth=lw, markeredgewidth=0.9,
                    markerfacecolor="white", capsize=2.5, capthick=0.9, elinewidth=0.9)

    if no_sd:
        print("[WARN] 회차가 1회뿐이라 오차막대를 그리지 못한 조건이 있습니다 — "
              + " · ".join(no_sd))

    # ---- [2026-09-06] 천장(상한) 점선 ----
    y_extra = []
    if ceiling is not None and len(ceiling):
        from matplotlib.lines import Line2D
        crow = ceiling.iloc[0]
        for metric, marker, ls, color, lw in SPEC:
            mk, sk = f"{metric} (mean)", f"{metric} (std)"
            if mk not in ceiling.columns or pd.isna(crow[mk]):
                continue
            y = float(crow[mk])
            ax.axhline(y, color=color, linestyle=(0, (4, 2.2)), linewidth=0.9, alpha=0.8, zorder=1)
            y_extra.append(y)
            if sk in ceiling.columns and not pd.isna(crow[sk]) and crow[sk] > 0:
                sd = float(crow[sk])
                ax.axhspan(y - sd, y + sd, color=color, alpha=0.05, linewidth=0, zorder=0)
                y_extra += [y - sd, y + sd]
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="#555555", linestyle=(0, (4, 2.2)), linewidth=0.9))
        labels.append("Upper bound (true labels)")
    else:
        handles, labels = ax.get_legend_handles_labels()

    # y 범위: 점·오차막대·상한 띠를 전부 담는다(axhline/axhspan 은 자동 범위에 안 잡힐 수 있다).
    y_vals = []
    for metric, *_ in SPEC:
        mk, sk = f"{metric} (mean)", f"{metric} (std)"
        if mk in summary.columns:
            ys_ = summary[mk].values.astype(float)
            es_ = summary[sk].values.astype(float) if sk in summary.columns else np.zeros_like(ys_)
            es_ = np.nan_to_num(es_)
            y_vals += list(ys_ - es_) + list(ys_ + es_)
    y_vals += y_extra
    y_vals = [v for v in y_vals if not np.isnan(v)]
    if y_vals:
        lo_y, hi_y = min(y_vals), max(y_vals)
        pad = max(0.01, (hi_y - lo_y) * 0.08)
        ax.set_ylim(lo_y - pad, min(1.0 + pad * 0.5, hi_y + pad))

    # 회차 수가 조건마다 같을 때만 제목에 n 을 적는다(다르면 오해를 부른다).
    n_txt = ""
    if "N" in summary.columns:
        ns = summary["N"].dropna().unique()
        if len(ns) == 1:
            n_txt = f", n = {int(ns[0])}"

    ax.set_xlabel("Number of unlabeled samples")
    ax.set_ylabel("Score")
    ax.set_title(f"Data efficiency ({split}, mean ± SD{n_txt})", pad=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([int(v) if float(v).is_integer() else v for v in xs])
    if span:
        ax.set_xlim(xs.min() - span * 0.07, xs.max() + span * 0.07)
    ax.grid(True, color="#dddddd", linewidth=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#555555")
    ax.tick_params(width=0.8, color="#555555", length=3.5)
    ax.legend(handles, labels, frameon=True, framealpha=0.92, edgecolor="#dddddd", loc="lower right",
              borderpad=0.5, handlelength=2.4, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 곡선 저장: {out_png}")


def collect_latency(mark_version):
    """profile_efficiency.py 의 반복 측정 결과를 모은다.

    이 스크립트는 아키텍처를 재는 것이라 학습·seed 와 무관하고 파일명에 run_tag 가 없다.
    그래서 반복 측정분은 efficiency_profile_<mark_version>_r<k>.json 으로 저장한 것을 읽는다.
    """
    paths = sorted(glob.glob(os.path.join(PLOT_DIR, f"efficiency_profile_{mark_version}*.json")))
    recs = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"[WARN] {os.path.basename(p)} 읽기 실패: {e}")
            continue
        flat = {"file": os.path.basename(p)}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                flat[k] = v
        recs.append(flat)
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description="Mark5_EXP 반복실험 결과 집계")
    ap.add_argument("--mark_version", type=str, default="mark5.0")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--latency_only", action="store_true")
    ap.add_argument("--no_tests", action="store_true", help="유의성 검정을 건너뛴다")
    ap.add_argument("--allow_mixed_config", action="store_true",
                    help="설정이 다른 실험 결과가 섞여 있어도 진행한다(기본은 중단)")
    args = ap.parse_args()

    if not args.latency_only:
        df_all = collect(args.mark_version, args.split, allow_mixed=args.allow_mixed_config)
        if df_all.empty:
            print(f"[ERROR] {args.split} 결과를 찾지 못했습니다.")
            print(f"        찾은 경로: {PLOT_DIR}/performance_summary_{args.mark_version}*_u*_s*.csv")
            print("        run_experiments.py 를 먼저 돌리십시오.")
            sys.exit(1)
        # [2026-09-06] 천장 행은 따로 떼어 둔다. 아래 기존 흐름(요약·검정·곡선)은 반지도 조건(ssl)만 본다.
        df = df_all[df_all["cond"] == "ssl"].reset_index(drop=True)
        df_ceil = df_all[df_all["cond"] == "ceil"].reset_index(drop=True)
        if df.empty:
            print(f"[ERROR] 반지도 조건(_u<n>_s<k>) 결과가 없습니다. 천장 결과만 {len(df_ceil)}건 있습니다.")
            print("        먼저 25회(run_experiments.py) 결과가 있어야 천장과 대조할 수 있습니다.")
            sys.exit(1)

        # [재검수 반영] --allow_mixed_config 로 진행할 때 머리말이 설정 하나만 찍으면
        # 섞여 있다는 사실이 화면에서 사라진다. 있는 대로 다 적는다.
        cfgs = sorted(df_all["config"].unique())
        cfg_txt = cfgs[0] if len(cfgs) == 1 else f"{len(cfgs)}종 혼합 — " + ", ".join(cfgs)
        print("=" * 78)
        print(f" 개별 실행 결과 ({args.split}) — 반지도 {len(df)}건"
              + (f" + 천장 {len(df_ceil)}건" if len(df_ceil) else "") + f" · 설정 {cfg_txt}")
        print("=" * 78)
        show = ["cond", "unlabeled", "seed"] + [m for m in METRICS if m in df_all.columns]
        print(df_all[show].to_string(index=False))

        summary = summarize(df)
        table = fmt_table(summary)
        ceil_summary, ceil_table = None, None
        if len(df_ceil):
            ceil_summary = summarize(df_ceil)
            ceil_summary.index = ["ceiling"] * len(ceil_summary)
            ceil_summary.index.name = "unlabeled"
            ceil_table = fmt_table(ceil_summary)

        print("\n" + "=" * 78)
        print(f" 조건별 요약 ({args.split}) — 평균 ± 표본SD · 95% 신뢰구간 · min~max · 중앙값[Q1,Q3]")
        print("=" * 78)
        # [M2] t 값은 조건·지표마다 표본 수가 다를 수 있으므로 하나로 뭉뚱그려 적지 않는다.
        ns_used = sorted({int(v) for c in summary.columns if c.endswith(" (n)")
                          for v in summary[c].dropna().values})
        if ns_used:
            print(" (95% CI 는 t 분포 기준. " +
                  " · ".join(f"n={n} -> t={t_crit(n):.3f}" for n in ns_used if n >= 2) + ")")
        show_table = table if ceil_table is None else pd.concat([table, ceil_table], ignore_index=True)
        for m in METRICS:
            cols = [c for c in [m, f"{m} 95% CI", f"{m} min~max", f"{m} 중앙값[Q1,Q3]"]
                    if c in show_table.columns]
            if not cols:
                continue
            print(f"\n--- {m} ---")
            print(show_table[["unlabeled"] + cols].to_string(index=False))
        if ceil_summary is not None:
            print("\n (마지막 'ceiling' 행 = 천장: unlabeled 전부에 진짜 라벨을 붙인 지도학습. 제안 모델이 아니라 상한 기준선)")

        raw_csv = os.path.join(PLOT_DIR, f"aggregate_raw_{args.mark_version}_{args.split}.csv")
        sum_csv = os.path.join(PLOT_DIR, f"aggregate_summary_{args.mark_version}_{args.split}.csv")
        tab_csv = os.path.join(PLOT_DIR, f"paper_table_{args.mark_version}_{args.split}.csv")
        # 원자료에는 천장 행도 같이 들어간다(cond 열로 구분). 요약·논문표에는 "ceiling" 행이 맨 끝에 붙는다.
        df_all.to_csv(raw_csv, index=False, encoding=CSV_ENCODING)
        (summary if ceil_summary is None else pd.concat([summary, ceil_summary])).to_csv(sum_csv, encoding=CSV_ENCODING)
        show_table.to_csv(tab_csv, index=False, encoding=CSV_ENCODING)
        print(f"\n[INFO] 원자료   : {raw_csv}")
        print(f"[INFO] 집계     : {sum_csv}")
        print(f"[INFO] 논문표   : {tab_csv}")

        # ---- 유의성 검정 ----
        if not args.no_tests:
            if not HAS_SCIPY:
                print("\n[WARN] scipy 가 없어 검정을 건너뜁니다(Cohen's d 만 계산됩니다).")
            tests = pairwise_tests(df)
            if not tests.empty:
                print("\n" + "=" * 78)
                print(f" 조건 쌍 비교 ({args.split})")
                print("=" * 78)
                key = tests[tests["지표"].isin(KEY_METRICS)]
                # [재검수 반영] n(짝) 과 비고를 화면에도 낸다. 주 결과인 p (paired) 가
                # NaN 으로 비는 경우(차이가 전부 같음, seed 가 어긋나 짝이 모자람)에
                # 화면만 보면 왜 비었는지 알 수 없었다. 이유는 비고 열에 들어 있다.
                show_cols = [c for c in ["비교", "지표", "평균차", "Cohen's d", "효과크기",
                                         "p (paired)", "p (paired, Holm)",
                                         "p (Welch)", "p (Mann-Whitney)",
                                         "n(짝)", "비고"]
                             if c in key.columns]
                with pd.option_context("display.width", 220):
                    print(key[show_cols].to_string(index=False))
                if HAS_SCIPY:
                    print("\n 읽는 법")
                    print("  · p (paired)  = 주 결과. 조건 사이가 seed 로 짝지어져 있어 이것을 먼저 본다.")
                    print("  · p (Welch)   = 짝짓기를 무시한 보수적 하한. 틀린 게 아니라 검정력이 낮다.")
                    print("  · Holm        = 다중비교 보정 후. 논문에는 보정 전후를 같이 적는다.")
                    print("  ⚠ n=5 대 n=5 의 Mann-Whitney 양측 최소 p 는 0.0079 입니다.")
                    print("    완벽하게 갈려도 그 아래로 못 내려가므로 p 값만으로 판단하지 마십시오.")
                    print("    Cohen's d 를 같이 보는 이유가 이것입니다.")
                # [2026-09-06] 천장 검정(u0 vs ceil · u<max> vs ceil)을 같은 CSV 에 이어 쓴다. Holm 은 family 별.
                ctests = ceiling_tests(df, df_ceil) if len(df_ceil) else pd.DataFrame()
                if not ctests.empty:
                    print("\n" + "=" * 78)
                    print(f" 천장(상한) 대비 검정 ({args.split}) — 별도 Holm family")
                    print("=" * 78)
                    keyc = ctests[ctests["지표"].isin(KEY_METRICS)]
                    with pd.option_context("display.width", 220):
                        print(keyc[[c for c in show_cols if c in keyc.columns]].to_string(index=False))
                    tests = pd.concat([tests, ctests], ignore_index=True)
                tst_csv = os.path.join(PLOT_DIR,
                                       f"aggregate_tests_{args.mark_version}_{args.split}.csv")
                tests.to_csv(tst_csv, index=False, encoding=CSV_ENCODING)
                print(f"\n[INFO] 검정표   : {tst_csv} (전체 지표 · 보정 p 포함"
                      + (" · 천장 검정 포함" if not ctests.empty else "") + ")")

        # ---- [2026-09-06] 메운 비율 ----
        if len(df_ceil):
            gap = gap_closed(df, df_ceil)
            if not gap.empty:
                print("\n" + "=" * 78)
                print(f" 메운 비율 gap_closed = (u<max> − u0) ÷ (ceil − u0)  ({args.split})")
                print("=" * 78)
                with pd.option_context("display.width", 220):
                    print(gap.to_string(index=False))
                print("  · 1.0 이면 라벨이 완벽했을 때만큼 올랐다는 뜻. 0.5 면 갈 수 있는 거리의 절반.")
                print("  · 'seed 짝' 값은 같은 seed 의 세 조건으로 seed 마다 계산한 것의 평균 ± SD.")
                gap_csv = os.path.join(PLOT_DIR, f"aggregate_gap_closed_{args.mark_version}_{args.split}.csv")
                gap.to_csv(gap_csv, index=False, encoding=CSV_ENCODING)
                print(f"[INFO] 메운 비율 : {gap_csv}")

        plot_curve(summary, args.split,
                   os.path.join(PLOT_DIR, f"data_efficiency_curve_{args.mark_version}_{args.split}.png"),
                   ceiling=ceil_summary)

        # 회수가 조건마다 다르면 표가 들쭉날쭉해진다. 조용히 넘기지 않는다.
        counts = df.groupby("unlabeled").size()
        if len(df_ceil):
            counts = pd.concat([counts, pd.Series({"ceiling": len(df_ceil)})])
        if counts.nunique() != 1:
            print("\n[WARN] 조건마다 실행 횟수가 다릅니다:")
            print(counts.to_string())
            print("[WARN] 논문 표에 넣기 전에 부족한 조건을 채우십시오.")

    # ---- 속도 ----
    lat = collect_latency(args.mark_version)
    if lat.empty:
        print("\n[INFO] 속도 측정 결과(efficiency_profile_*.json)가 없습니다.")
        return

    print("\n" + "=" * 78)
    print(f" 속도·효율 측정 — {len(lat)}회")
    print("=" * 78)
    print(lat.to_string(index=False))

    num = lat.drop(columns=["file"]).select_dtypes("number")
    if len(lat) > 1:
        n_lat = len(lat)
        ci_lo, ci_hi = {}, {}
        for c in num.columns:
            lo, hi, _ = ci95(num[c].values)
            ci_lo[c], ci_hi[c] = lo, hi
        stat = pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1),
                             "ci_low": pd.Series(ci_lo), "ci_high": pd.Series(ci_hi),
                             "min": num.min(), "max": num.max(),
                             "median": num.median(),
                             "q1": num.quantile(0.25), "q3": num.quantile(0.75),
                             "iqr": num.quantile(0.75) - num.quantile(0.25)})
        print(f"\n--- 평균 ± 표본SD · 95% CI (n={n_lat}, t={t_crit(n_lat):.3f}) ---")
        print(stat.to_string())
        out = os.path.join(PLOT_DIR, f"aggregate_latency_{args.mark_version}.csv")
        stat.to_csv(out, encoding=CSV_ENCODING)
        print(f"\n[INFO] 저장: {out}")
    else:
        print("\n[INFO] 1회뿐이라 표준편차를 낼 수 없습니다. 5회 측정하는 것을 권합니다.")


if __name__ == "__main__":
    main()
