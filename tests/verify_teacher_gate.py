"""teacher_gate_report.py 점검 — 손으로 계산한 기대값과 맞는지 합성 점수 캐시로 본다. 성능 실험이 아니다 (2026-10-23).

합성 자료: 9클래스, val 클래스당 10 · 정답 라벨 클래스당 20. 타겟 클립은 자기 교사 0.9 · 나머지 0.05, others 클립은 전원 0.1.
기준 캐시만 heavy_impact 클립 2개(val)·2개(정답 라벨)를 진짜 dragging 클립과 똑같이(dragging 교사 0.9) 만든다 → 구별 불가 오류.
기대값(손 계산)
    기준 val    Acc 88/90=0.977778 · P (8+10/12)/9=0.981481 · R (8+0.8)/9=0.977778 · F1 (7+0.888889+0.909091)/9=0.977553
    기준 정답   Acc 178/180=0.988889 · P (8+20/22)/9=0.989899 · R (8+0.9)/9=0.988889 · F1 (7+0.947368+0.952381)/9=0.988861
    기준 고정 규칙 맞음 178 / 채택 180 · 후보 180 / 180
    손들기 평균(val+정답) 기준 (4/30)/64=0.00208333 · 후보 0
    후보 vs 기준 → C1·C2·C4 만족(C3 은 결합기가 후보를 모두 채택하면 만족) · 같은 캐시끼리 → 네 조건 모두 불만족 → 기준 교사(B0)
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CLASSES = ["heavy_impact", "dragging", "construction", "machine_noise", "talking", "water_toilet", "water_shower", "dog_bark"]


def require(ok, msg):
    if not ok:
        raise SystemExit(f"[FAIL] {msg}")
    print(f"[OK] {msg}")


def make_cache(path, n_val, n_lab, n_err_val, n_err_lab):
    rows = []
    for split, n, n_err in (("val", n_val, n_err_val), ("labeled", n_lab, n_err_lab)):
        for c in CLASSES + ["others"]:
            for i in range(n):
                p = {e: (0.1 if c == "others" else 0.05) for e in CLASSES}
                if c != "others":
                    p[c] = 0.9
                if c == "heavy_impact" and i < n_err:            # 진짜 dragging 클립과 같은 점수
                    p = {e: 0.05 for e in CLASSES}; p["dragging"] = 0.9
                rows.append({"path": f"/x/{split}/{c}_{i:03d}.wav", "split": split, "label": c, **{f"p_{e}": p[e] for e in CLASSES}})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "label"] + [f"p_{e}" for e in CLASSES])
        w.writeheader(); w.writerows(rows)


def run(cand, ref, out):
    r = subprocess.run([PY, os.path.join(ROOT, "model", "teacher_gate_report.py"), "--candidate", cand, "--reference", ref,
                        "--no_freshness_check", "--out", out], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("[FAIL] teacher_gate_report.py 실패")
    return json.load(open(out, encoding="utf-8"))


def close(a, b):
    return abs(a - b) < 5e-6


def main():
    with tempfile.TemporaryDirectory() as d:
        ref, cand = os.path.join(d, "teacher_scores_ref.csv"), os.path.join(d, "teacher_scores_cand.csv")
        make_cache(ref, 10, 20, 2, 2)
        make_cache(cand, 10, 20, 0, 0)
        g = run(cand, ref, os.path.join(d, "gate.json"))
        rv, rl = g["reference"]["val"]["nine_class"], g["reference"]["labeled"]["nine_class"]
        require(all(close(rv[k], v) for k, v in zip(("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"),
                                                    (0.977778, 0.981481, 0.977778, 0.977553))), f"기준 val 4지표 {rv}")
        require(all(close(rl[k], v) for k, v in zip(("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"),
                                                    (0.988889, 0.989899, 0.988889, 0.988861))), f"기준 정답 4지표 {rl}")
        require(all(close(g["candidate"][s]["nine_class"][k], 1.0) for s in ("val", "labeled")
                    for k in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1")), "후보 4지표 전부 1")
        fr, fc = g["reference"]["labeled"]["fixed_rule"], g["candidate"]["labeled"]["fixed_rule"]
        require((fr["correct"], fr["adopted"], fc["correct"], fc["adopted"]) == (178, 180, 180, 180), f"고정 규칙 {fr} · {fc}")
        require(close(g["reference"]["raise_off_mean_val_labeled"], (4 / 30) / 64) and g["candidate"]["raise_off_mean_val_labeled"] == 0.0,
                "손들기 평균")
        require(abs(g["reference"]["val"]["raise"]["dragging"]["heavy_impact"] - 0.2) < 1e-9, "손들기 칸 (dragging 교사, heavy 클립) val 0.2")
        cv = g["candidate"]["labeled"]["combiner_cv"]["one"]
        require(cv["correct"] <= cv["adopted"] <= 180, f"결합기 교차검증 개수 범위 {cv}")
        require(g["conditions"]["C1_val_macro_f1"] and g["conditions"]["C2_labeled_macro_f1"] and g["conditions"]["C4_raise_off_mean"],
                "C1·C2·C4 만족")
        print(f"[INFO] C3 {g['conditions']['C3_combiner_cv']} · 후보 결합기 {cv} · 기준 {g['reference']['labeled']['combiner_cv']['one']} · 판정 {g['verdict']}")
        g2 = run(cand, cand, os.path.join(d, "gate_same.json"))
        require(g2["n_satisfied"] == 0 and g2["verdict"] == "기준 교사(B0)", "같은 캐시끼리 → 네 조건 모두 불만족 · B0")
        g3 = run(cand, ref, os.path.join(d, "gate_again.json"))
        require(g3["candidate"]["labeled"]["combiner_cv"] == g["candidate"]["labeled"]["combiner_cv"], "결합기 교차검증 재현(시드 고정)")
    print("[완료] teacher_gate_report 점검 통과")


if __name__ == "__main__":
    main()
