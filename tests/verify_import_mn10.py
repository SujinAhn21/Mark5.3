"""import_teachers.py --kind mn10 점검 — 가짜 체크포인트 8개로 반입·검사·되돌림을 본다. 진짜 resources/ 는 건드리지 않는다 (2026-10-23).

기대: ① 판정 통과 8개 → resources 에 mn10_teacher_mark4.{1..8}.pt + teacher_manifest_mn10.json, source_manifest materials 에 9개 키 추가,
         조각 교사 manifest(teacher_manifest.json)는 그대로 ② 4.2 판정 불합격 → 멈춤, resources 그대로 ③ --allow_failed_verdict → 반입
      ④ 클래스가 다른 체크포인트 → 멈춤
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TARGETS = ["heavy_impact", "dragging", "construction", "machine_noise", "talking", "water_toilet", "water_shower", "dog_bark"]


def require(ok, msg):
    if not ok:
        raise SystemExit(f"[FAIL] {msg}")
    print(f"[OK] {msg}")


def make_source(root, fail_version=None, wrong_class_version=None):
    for i, t in enumerate(TARGETS, 1):
        mv = f"mark4.{i}"
        d = os.path.join(root, f"Mark4.{i}")
        os.makedirs(os.path.join(d, "model"), exist_ok=True); os.makedirs(os.path.join(d, "plots"), exist_ok=True)
        cls = "dragging" if mv == wrong_class_version and t != "dragging" else t
        prov = {"run_id": f"run-{mv}", "dataset_index_sha256": "a" * 64, "teacher_code_sha256": "b" * 64,
                "target_class": cls, "classes": [cls, "others"]}
        torch.save({"schema_version": 3, "model_kind": "mn10_binary_clip", "mark_version": mv, "provenance": prov,
                    "model_state": {"w": torch.zeros(2)}}, os.path.join(d, "model", f"mn10_teacher_{mv}.pt"))
        json.dump({"provenance": {"run_id": f"run-{mv}"}, "verdict": {"pass": mv != fail_version}},
                  open(os.path.join(d, "plots", f"mn10_teacher_eval_{mv}.json"), "w"))


def run(src, res, man, *extra):
    r = subprocess.run([PY, os.path.join(ROOT, "model", "import_teachers.py"), "--kind", "mn10", "--source_root", src,
                        "--resources_dir", res, "--source_manifest", man, *extra], capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


def main():
    with tempfile.TemporaryDirectory() as d:
        res, man = os.path.join(d, "resources"), os.path.join(d, "source_manifest.json")
        os.makedirs(res)
        seg_manifest = os.path.join(res, "teacher_manifest.json")
        open(seg_manifest, "w").write('{"kind": "segment"}')
        seg_sha = hashlib.sha256(open(seg_manifest, "rb").read()).hexdigest()
        json.dump({"materials": {"resources/teacher_manifest.json": seg_sha}}, open(man, "w"))
        src = os.path.join(d, "Mark4_series"); make_source(src)
        code, out = run(src, res, man)
        require(code == 0, "① 판정 통과 8개 반입 exit 0" + ("" if code == 0 else "\n" + out[-1500:]))
        files = sorted(f for f in os.listdir(res) if f.startswith("mn10_teacher_"))
        require(files == [f"mn10_teacher_mark4.{i}.pt" for i in range(1, 9)], f"① resources 의 mn10 교사 8개 {files}")
        m = json.load(open(os.path.join(res, "teacher_manifest_mn10.json")))
        require(m["kind"] == "mn10" and len(m["files"]) == 8, "① teacher_manifest_mn10.json (kind mn10 · 8파일)")
        require(hashlib.sha256(open(seg_manifest, "rb").read()).hexdigest() == seg_sha, "① 조각 교사 manifest 그대로")
        mat = json.load(open(man))["materials"]
        require(len(mat) == 1 + 9 and "resources/teacher_manifest_mn10.json" in mat, f"① source_manifest materials {len(mat)}개")

        src2 = os.path.join(d, "s2"); make_source(src2, fail_version="mark4.2")
        before = {f: os.path.getmtime(os.path.join(res, f)) for f in os.listdir(res)}
        code, out = run(src2, res, man)
        require(code != 0 and "판정 불합격" in out, "② 4.2 판정 불합격 → 멈춤")
        require({f: os.path.getmtime(os.path.join(res, f)) for f in os.listdir(res)} == before, "② resources 그대로")
        code, out = run(src2, res, man, "--allow_failed_verdict")
        require(code == 0 and "[WARN]" in out, "③ --allow_failed_verdict → 경고 뒤 반입")
        src3 = os.path.join(d, "s3"); make_source(src3, wrong_class_version="mark4.3")
        code, out = run(src3, res, man)
        require(code != 0 and "클래스 불일치" in out, "④ 클래스가 다른 체크포인트 → 멈춤")

        # ⑤ 기본(--kind segment) 경로가 예전과 같은지: 가짜 조각 교사 16파일 → teacher_manifest.json 갱신, mn10 manifest 그대로
        src5 = os.path.join(d, "s5")
        emb = torch.ones(2, 3)
        for i, t in enumerate(TARGETS, 1):
            mv = f"mark4.{i}"; md = os.path.join(src5, f"Mark4.{i}", "model"); os.makedirs(md, exist_ok=True)
            prov = {"run_id": f"seg-{mv}", "classes": [t, "others"], "target_class": t, "clip_aggregation": "mean_probability",
                    "dataset_index_sha256": "a", "audio_manifest_sha256": "b", "text_anchor_sha256": "c"}
            for part in ("encoder", "classifier"):
                torch.save({"schema_version": 2, "provenance": prov, "text_embeddings": emb},
                           os.path.join(md, f"best_teacher_{part}_{mv}.pth"))
        mn10_sha = hashlib.sha256(open(os.path.join(res, "teacher_manifest_mn10.json"), "rb").read()).hexdigest()
        r = subprocess.run([PY, os.path.join(ROOT, "model", "import_teachers.py"), "--source_root", src5, "--resources_dir", res,
                            "--source_manifest", man], capture_output=True, text=True, cwd=ROOT)
        require(r.returncode == 0, "⑤ 기본(조각 교사) 반입 exit 0" + ("" if r.returncode == 0 else "\n" + (r.stdout + r.stderr)[-1500:]))
        segm = json.load(open(seg_manifest))
        require(segm.get("kind") == "segment" and len(segm["files"]) == 16, "⑤ teacher_manifest.json = 조각 교사 16파일")
        require(hashlib.sha256(open(os.path.join(res, "teacher_manifest_mn10.json"), "rb").read()).hexdigest() == mn10_sha,
                "⑤ mn10 manifest 그대로")
    print("[완료] import_teachers --kind mn10 점검 통과")


if __name__ == "__main__":
    main()
