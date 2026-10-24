"""--main_pseudo_source(집계·논문 표) 점검 — preset cmb 80회 태그로 만든 가짜 결과로 고르기·짝 검정이 맞는지 본다 (2026-10-23, 10-25 80회).

aggregate_results.collect 만 가짜 표로 바꾸고(체크포인트·출력 정책 대조를 건너뛰려고), 그 뒤의 main()·baseline_paired·
make_paper_tables.main_slice·label_eff_rows 는 실제 코드를 돌린다. 산출물은 임시 폴더(PLOT_DIR 를 바꿈)에만 쓴다.
기대값(손 계산, 본실험 설정 묶음 = 본실험 4 + 곡선 2(u500·u1000) + 지도만 4(정답 200·전량·1,000·500 × u0) + 고정 규칙 2 조합 = 12, seed 5 → 60행)
    cmb       본표(정답 전량) = u2000 cmb 5 + u0 5 = 10행 · 라벨 효율 = (cmb 6 + 지도만 4) × 5 = 50행 · fix 0행
    cmb       표 4 / 라벨 효율 검정: 정답 2,000·1,000·500·200 네 수준 모두 supervised(u0) 행(N=5)과 u0 vs u2000 검정이 있다
    consensus 본표 = u0 5행 · 라벨 효율 = 지도만 4 × 5 = 20행
    짝 검정(정답 200, 비교모델 sup·pl·plc seed 1~5): cmb → 'ours' 짝 = plc 5 · consensus → u0 만 있어 'ours' 짝 없음
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "model"), os.path.join(ROOT, "vild"), os.path.join(ROOT, "shared_vild"), os.path.join(ROOT, "utils")):
    sys.path.insert(0, p)
import aggregate_results as AG          # noqa: E402
import run_experiments as RE            # noqa: E402
import make_paper_tables as MP          # noqa: E402


def require(ok, msg):
    if not ok:
        raise SystemExit(f"[FAIL] {msg}")
    print(f"[OK] {msg}")


def fake_df():
    rows, rng = [], np.random.default_rng(0)
    main_kw = RE._parse_runner_extra(RE.CMB_BASE_EXTRA)[1]
    for (l, n, ps, ex) in RE.CMB_PLAN:
        if ex:                                          # 아블레이션(설정이 다른 묶음)은 --bases 로 빠지므로 여기 넣지 않는다
            continue
        for s in range(1, 6):
            tag = RE._run_tag("mark5.2", s, n, labeled_n=l, ps=ps, ablation_kw=main_kw)
            base, nn, ss, cond, lab, sp, pss = AG._parse_tag_full(tag)
            rec = {"config": base, "cond": cond, "unlabeled": nn, "seed": ss, "run_tag": tag,
                   "labeled": lab if lab is not None else AG._full_labeled_count(), "spill": bool(sp),
                   "pseudo_source": pss if pss else "consensus"}
            rec.update({m: float(0.9 + 0.01 * rng.random()) for m in AG.METRICS})
            rows.append(rec)
    return pd.DataFrame(rows)


def run_main(df, ps, out_dir):
    AG.PLOT_DIR = out_dir
    AG.collect = lambda *a, **k: df.copy()
    sys.argv = ["aggregate_results.py", "--split", "test", "--no_tests"] + (["--main_pseudo_source", ps] if ps else [])
    AG.main()
    raw = pd.read_csv(os.path.join(out_dir, "aggregate_raw_mark5.2_test_plain.csv"), encoding="utf-8-sig")
    raw_all = pd.read_csv(os.path.join(out_dir, "aggregate_raw_all_mark5.2_test_plain.csv"), encoding="utf-8-sig")
    return raw, raw_all


def main():
    df = fake_df()
    full = AG._full_labeled_count()
    require(len(df) == 12 * 5 and set(df["pseudo_source"]) == {"cmb", "fix", "consensus"}, f"가짜 결과 {len(df)}행(본 설정 12조합 × 5)")
    with tempfile.TemporaryDirectory() as d:
        raw, raw_all = run_main(df, "cmb", d)
        require(len(raw) == 10 and set(raw["unlabeled"]) == {0, 2000} and (raw["labeled"] == full).all(), f"cmb 본표 {len(raw)}행(기대 10)")
        require(len(raw_all) == 50 and "fix" not in set(raw_all["pseudo_source"]), f"cmb 라벨 효율 원자료 {len(raw_all)}행(기대 50) · fix 없음")
        # [2026-10-25] 표 4 의 정답 1,000·500 줄에 지도만 행이 없던 결함(70회 판) — 네 수준 모두 supervised N=5 와 u0 vs u2000 검정
        le_tab = pd.read_csv(os.path.join(d, "paper_table_label_eff_mark5.2_test_plain.csv"), encoding="utf-8-sig")
        sup = le_tab[le_tab["condition"] == "supervised"]
        require(sorted(sup["labeled"]) == [200, 500, 1000, full] and (sup["N"] == 5).all(),
                f"라벨 효율 표: supervised 행 수준 {sorted(sup['labeled'])} · N {sorted(sup['N'])}(기대 200·500·1000·{full}, 각 5)")
        _, le_tests = AG.label_efficiency(raw_all)
        require(sorted(set(le_tests["labeled"])) == [200, 500, 1000, full] if not le_tests.empty else False,
                f"라벨 효율 검정(u0 vs u2000) 수준 {sorted(set(le_tests['labeled'])) if not le_tests.empty else []}")
        rows4 = MP.label_eff_rows(le_tab, None, full, "plc")
        sup4 = sorted(r[0] for r in rows4 if r[1].startswith("Proposed") and r[2] == "supervised")
        require(sup4 == [200, 500, 1000, full], f"표 4(make_paper_tables) Proposed supervised 행 {sup4}")
        ms = MP.main_slice(raw_all, "cmb")
        require(len(ms) == 10, f"make_paper_tables.main_slice(cmb) {len(ms)}행(기대 10)")
        # 표 생성기 main 을 같은 임시 폴더에서 끝까지(집계 _plain 산출물을 읽어야 한다 — 10-23 이전에는 이름이 달라 멈췄다)
        MP.PLOT_DIR = d
        sys.argv = ["make_paper_tables.py", "--split", "test", "--main_pseudo_source", "cmb"]
        MP.main()
        md = [f for f in os.listdir(d) if f.startswith("paper_tables_mark5.2_test") and f.endswith(".md")]
        require(len(md) == 1, f"make_paper_tables 산출물 {md}")
        text = open(os.path.join(d, md[0]), encoding="utf-8").read()
        require("자료 없음 — aggregate_results.py 가 만든 paper_table_label_eff" not in text, "표 4(라벨 효율)가 집계 산출물을 읽음")
        # [2026-10-23] 노트북 '2 Mark5.3 본실험 80회' 마지막 셀과 같은 인자: 비교모델 원자료(sup·plc × 정답 전량·1,000·500·200 × seed 1)
        bdir = os.path.join(d, "base"); os.makedirs(bdir)
        braw = []
        for mt in ("efficientat_mn10", "panns_cnn6", "ced_tiny"):
            bb = pd.DataFrame([{"model": mt, "cond": c, "seed": 1, "lr": "prodigy", "labeled": lab, "unlabeled": "",
                                "run_tag": f"{mt}_{c}_pd_s1", **{m: 0.85 for m in AG.METRICS}}
                               for c in ("sup", "plc") for lab in (full, 1000, 500, 200)])
            braw.append(os.path.join(bdir, f"aggregate_raw_{mt}_test.csv")); bb.to_csv(braw[-1], index=False, encoding="utf-8-sig")
        sys.argv = ["make_paper_tables.py", "--split", "test", "--main_pseudo_source", "cmb", "--baseline_cond", "plc", "--baseline_raw", *braw]
        MP.main()
        text = open(os.path.join(d, md[0]), encoding="utf-8").read()
        require(MP._BASE_COND["plc"] in text and MP._BASE_COND["pl"] not in text,
                f"make_paper_tables --baseline_raw(seed 1 비교모델 3개): 표에 '{MP._BASE_COND['plc']}' 행이 붙고 pl 행은 없음")
    with tempfile.TemporaryDirectory() as d:
        raw, raw_all = run_main(df, None, d)
        require(len(raw) == 5 and set(raw["unlabeled"]) == {0}, f"consensus(기본) 본표 {len(raw)}행(기대 5, u0)")
        require(len(raw_all) == 20, f"consensus(기본) 라벨 효율 원자료 {len(raw_all)}행(기대 20)")

    # 비교모델 짝 검정: 정답 200, sup·pl·plc seed 1~5
    le = df[(df["pseudo_source"] != "fix")].copy()
    with tempfile.TemporaryDirectory() as d:
        b = pd.DataFrame([{"model": "efficientat_mn10", "cond": c, "seed": s, "labeled": 200, "unlabeled": np.nan,
                           **{m: 0.9 for m in AG.METRICS}} for c in ("sup", "pl", "plc") for s in range(1, 6)])
        bp = os.path.join(d, "aggregate_raw_efficientat_mn10_test.csv"); b.to_csv(bp, index=False)
        t_cmb = AG.baseline_paired(le, [bp], AG.PL_COUNTERPART["cmb"])
        roles = sorted({str(x).split(": ")[0].split(" ")[-1] for x in t_cmb["비교"]}) if not t_cmb.empty else []
        require(not t_cmb.empty and any("ours" in str(x) for x in t_cmb["비교"]), f"cmb 짝 검정에 ours 행이 있음 {roles}")
        n_ours = int(t_cmb[t_cmb["비교"].astype(str).str.contains("ours")]["n(짝)"].iloc[0])
        require(n_ours == 5, f"cmb 의 ours 짝 수 {n_ours}(기대 5 = plc seed 1~5)")
        le_cons = le[le["pseudo_source"] == "consensus"]
        t_cons = AG.baseline_paired(le_cons, [bp], AG.PL_COUNTERPART["consensus"])
        require(t_cons.empty or not any("ours" in str(x) for x in t_cons["비교"]), "consensus 에서는 ours 짝이 없음(u2000 회차 없음)")
        base = b.assign(model="efficientat_mn10")
        lab_table = pd.DataFrame([{"labeled": 200, "condition": "supervised", "N": 5}])
        rows_plc = MP.label_eff_rows(lab_table, base, full, "plc")
        rows_pl = MP.label_eff_rows(lab_table, base, full, "pl")
        require(any(r[2] == MP._BASE_COND["plc"] for r in rows_plc) and not any(r[2] == MP._BASE_COND["pl"] for r in rows_plc),
                "표 4: pl_cond=plc 면 plc 행만(pl 행 없음)")
        require(any(r[2] == MP._BASE_COND["pl"] for r in rows_pl), "표 4: 기본은 pl 행")
    # openset_report: 이 저장소 결과는 _plain 꼬리, 비교모델은 꼬리 없음 — 각자 제 이름만 찾고 섞이지 않는가
    import openset_report as OR
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "plots"))
        pred = pd.DataFrame({"Filename": ["a.wav", "b.wav"], "True Label": ["others", "others"], "Predicted Label": ["others", "dragging"]})
        for s in OR.EVAL_SETS:
            pred.to_csv(os.path.join(d, "plots", f"prediction_details_oursT_s1_eval_{s}_plain.csv"), index=False)
            pred.to_csv(os.path.join(d, "plots", f"prediction_details_baseT_s1_eval_{s}.csv"), index=False)
        require(OR._discover(d, sfx=OR.OURS_SFX) == ["oursT_s1"] and OR._discover(d) == ["baseT_s1"], "openset_report: _plain / 꼬리 없음 각각 찾음")
        row, _ = OR.evaluate_tag(d, "oursT_s1", "ours", OR.OURS_SFX)
        require(row["n_far"] == 2 and abs(row["far_to_others"] - 0.5) < 1e-9, f"openset_report: _plain 예측 파일을 읽음(far {row['far_to_others']})")
    print("[완료] --main_pseudo_source 점검 통과")


if __name__ == "__main__":
    main()
