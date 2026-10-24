"""Verify the deployed source and fixed input materials without starting training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, normalize_newlines=False):
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify(root=ROOT, source_only=False):
    root = Path(root)
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    failures = []
    groups = ["source"] if source_only else ["source", "materials"]
    checked = 0
    for group in groups:
        for relative, expected in manifest[group].items():
            path = root / relative
            if not path.is_file() or digest(path, group == "source") != expected:
                failures.append(relative)
            checked += 1
    if failures:
        raise RuntimeError("Release files are missing or differ from the verified release: " + ", ".join(failures))
    print(f"Verified {checked} files; policy={manifest['classification_policy']}")
    return checked


# [2026-10-23] 10-22 Prodigy 커밋 뒤로 source 해시가 갱신되지 않아 이 검사가 HEAD 에서도 멈췄다. 손으로 고치면 또 어긋나므로
# 코드를 고친 뒤(커밋 전) --refresh_source 로 다시 계산한다. materials(데이터 목록·교사·가중치)는 import_teachers.py 가 관리하므로 건드리지 않는다.
NEW_SOURCE_FILES = [
    "model/make_teacher_scores.py", "model/make_combiner_labels.py", "model/label_freshness.py", "model/teacher_gate_report.py",
    "teacher_mn10/__init__.py", "teacher_mn10/augment.py", "teacher_mn10/data.py", "teacher_mn10/evaluate.py",
    "teacher_mn10/model_def.py", "teacher_mn10/train_loop.py", "teacher_mn10/train_teacher.py",
    "tests/verify_two_candidate.py", "tests/verify_teacher_cache.py", "tests/verify_teacher_gate.py",
    "tests/verify_import_mn10.py", "tests/verify_cmb_aggregation.py",
    "notebooks/1 교사 점수·채택 관문.ipynb", "notebooks/2 Mark5.3 본실험 80회.ipynb", "notebooks/3 비교모델 결합기 조건 24회.ipynb",
]
# 일부러 없앤 파일만 여기 적는다(목록에 없는데 사라진 파일은 그대로 오류). 10-23: 한 조건 노트북 → 위 세 노트북으로 바뀜.
# 10-25: 본실험 노트북 70회 → 80회(지도만 정답 1,000·500 추가)로 이름이 바뀜.
RETIRED_SOURCE_FILES = ["notebooks/Mark5.3 제안모델 실험.ipynb", "notebooks/2 Mark5.3 본실험 70회.ipynb"]


def refresh_source(root=ROOT):
    """source 묶음의 해시를 지금 파일로 다시 계산하고 NEW_SOURCE_FILES 를 더하며 RETIRED_SOURCE_FILES 를 뺀다. 없는 파일이 있으면 멈춘다."""
    root = Path(root)
    path = root / "source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    names = sorted((set(manifest["source"]) | set(NEW_SOURCE_FILES)) - set(RETIRED_SOURCE_FILES))
    missing = [n for n in names if not (root / n).is_file()]
    if missing:
        raise RuntimeError("source 에 적힌 파일이 없습니다: " + ", ".join(missing))
    old = manifest["source"]
    manifest["source"] = {n: digest(root / n, True) for n in names}
    changed = [n for n in names if old.get(n) != manifest["source"][n]]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:          # Path.write_text 의 newline 은 3.10 부터
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[refresh] source {len(names)}개 (새로 등록 {len(set(names) - set(old))} · 해시 바뀜 {len(changed)})")
    for n in changed:
        print(f"  {'+' if n not in old else '~'} {n}")
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_only", action="store_true", help="Skip non-Git audio index and weight files")
    parser.add_argument("--refresh_source", action="store_true",
                        help="[2026-10-23] source 묶음 해시를 지금 파일로 다시 쓰고(새 파일 등록 포함) 검사한다. 커밋 전에만")
    args = parser.parse_args()
    if args.refresh_source:
        refresh_source()
    verify(source_only=args.source_only)
