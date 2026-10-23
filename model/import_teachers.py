"""Validate and atomically import one Mark4 teacher release into resources/.

--kind segment (기본): 조각 교사 16파일(best_teacher_{encoder,classifier}_mark4.X.pth) → resources/teacher_manifest.json
--kind mn10 [2026-10-23]: mn10 교사 8파일(model/mn10_teacher_mark4.X.pt) → resources/teacher_manifest_mn10.json
    체크포인트 schema 3 · model_kind · 클래스 순서와, Mark4 의 학습 후 판정(plots/mn10_teacher_eval_mark4.X.json)이 '통과'인지 본다.
    불합격 교사는 --allow_failed_verdict 없이는 반입하지 않는다(설계 문서 '학습 후 판정'). 조각 교사 manifest 는 건드리지 않는다.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
SOURCE_MANIFEST = os.path.join(PROJECT_ROOT, "source_manifest.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pair(encoder_path, classifier_path, version, target):
    encoder = torch.load(encoder_path, map_location="cpu")
    classifier = torch.load(classifier_path, map_location="cpu")
    required = {"run_id", "classes", "target_class", "clip_aggregation",
                "dataset_index_sha256", "audio_manifest_sha256", "text_anchor_sha256"}
    ep = encoder.get("provenance") or {}
    cp = classifier.get("provenance") or {}
    if int(encoder.get("schema_version", 0)) < 2 or int(classifier.get("schema_version", 0)) < 2:
        raise RuntimeError(f"{version}: schema v2 teacher가 아닙니다.")
    missing = sorted(required - set(ep)) + sorted(required - set(cp))
    if missing:
        raise RuntimeError(f"{version}: provenance 누락 {missing}")
    if ep != cp:
        raise RuntimeError(f"{version}: encoder/classifier provenance 불일치")
    if ep["classes"] != [target, "others"] or ep["target_class"] != target:
        raise RuntimeError(f"{version}: 클래스 순서 불일치 {ep['classes']}")
    if ep["clip_aggregation"] != "mean_probability":
        raise RuntimeError(f"{version}: clip aggregation 불일치")
    ee, ce = encoder.get("text_embeddings"), classifier.get("text_embeddings")
    if ee is None or ce is None or not torch.equal(ee, ce):
        raise RuntimeError(f"{version}: text anchor 쌍 불일치")
    return ep


MN10_SCHEMA, MN10_KIND = 3, "mn10_binary_clip"      # teacher_mn10/train_teacher.SCHEMA_VERSION · model_def.MODEL_KIND


def validate_mn10(path, eval_json, version, target, allow_failed_verdict=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prov = ck.get("provenance") or {}
    if ck.get("schema_version") != MN10_SCHEMA or ck.get("model_kind") != MN10_KIND:
        raise RuntimeError(f"{version}: mn10 교사가 아닙니다(schema {ck.get('schema_version')} · {ck.get('model_kind')})")
    if ck.get("mark_version") != version or prov.get("target_class") != target or prov.get("classes") != [target, "others"]:
        raise RuntimeError(f"{version}: 버전·클래스 불일치 {ck.get('mark_version')} {prov.get('classes')}")
    for key in ("run_id", "dataset_index_sha256", "teacher_code_sha256"):
        if not prov.get(key):
            raise RuntimeError(f"{version}: provenance 누락 {key}")
    if not os.path.isfile(eval_json):
        raise FileNotFoundError(f"[ERROR] 학습 후 판정 json 없음: {eval_json}")
    with open(eval_json, encoding="utf-8") as handle:
        ev = json.load(handle)
    if (ev.get("provenance") or {}).get("run_id") != prov["run_id"]:
        raise RuntimeError(f"{version}: 판정 json 이 이 체크포인트의 것이 아닙니다(run_id 다름)")
    if not ev.get("verdict", {}).get("pass"):
        msg = f"{version}: 학습 후 판정 불합격 {ev.get('verdict')}"
        if not allow_failed_verdict:
            raise RuntimeError(msg + " — 원인을 찾거나 --allow_failed_verdict 로 반입")
        print(f"[WARN] {msg} (--allow_failed_verdict 로 반입)")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_root", required=True,
                    help="Mark4.1~Mark4.8 폴더를 포함하는 Mark4_series 경로")
    ap.add_argument("--resources_dir", default=RESOURCES_DIR)
    ap.add_argument("--kind", choices=["segment", "mn10"], default="segment",
                    help="segment = 조각 교사 16파일(기본) · mn10 = mn10 교사 8파일(2026-10-23)")
    ap.add_argument("--allow_failed_verdict", action="store_true", help="mn10: 학습 후 판정 불합격 교사도 반입(경고만)")
    ap.add_argument("--source_manifest", default=SOURCE_MANIFEST, help="고칠 source_manifest.json (시험할 때만 바꾼다)")
    args = ap.parse_args()
    source_manifest_path = args.source_manifest
    manifest_name = "teacher_manifest.json" if args.kind == "segment" else "teacher_manifest_mn10.json"

    targets = ["heavy_impact", "dragging", "construction", "machine_noise",
               "talking", "water_toilet", "water_shower", "dog_bark"]
    os.makedirs(args.resources_dir, exist_ok=True)
    release = []
    with tempfile.TemporaryDirectory(prefix="teacher_release_", dir=args.resources_dir) as stage:
        for index, target in enumerate(targets, 1):
            version = f"mark4.{index}"
            source_model = os.path.join(args.source_root, f"Mark4.{index}", "model")
            names = ([f"best_teacher_encoder_{version}.pth", f"best_teacher_classifier_{version}.pth"] if args.kind == "segment"
                     else [f"mn10_teacher_{version}.pt"])
            staged = []
            for name in names:
                source = os.path.join(source_model, name)
                if not os.path.isfile(source):
                    raise FileNotFoundError(f"[ERROR] 새 teacher 파일 없음: {source}")
                destination = os.path.join(stage, name)
                shutil.copy2(source, destination)
                staged.append(destination)
            if args.kind == "segment":
                provenance = validate_pair(staged[0], staged[1], version, target)
            else:
                provenance = validate_mn10(staged[0], os.path.join(args.source_root, f"Mark4.{index}", "plots",
                                                                   f"mn10_teacher_eval_{version}.json"),
                                           version, target, args.allow_failed_verdict)
            for path in staged:
                release.append({"file": os.path.basename(path), "sha256": sha256_file(path),
                                "bytes": os.path.getsize(path), "run_id": provenance["run_id"],
                                "target_class": target})

        canonical = json.dumps(release, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest = {
            "schema_version": 1, "kind": args.kind,
            "teacher_release_sha256": hashlib.sha256(canonical).hexdigest(),
            "files": release,
        }
        manifest_stage = os.path.join(stage, manifest_name)
        with open(manifest_stage, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

        if not os.path.isfile(source_manifest_path):
            raise FileNotFoundError(f"[ERROR] source manifest 없음: {source_manifest_path}")
        with open(source_manifest_path, "r", encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        materials = source_manifest.get("materials")
        if not isinstance(materials, dict):
            raise RuntimeError("[ERROR] source_manifest.json materials 형식이 올바르지 않습니다.")
        materials[f"resources/{manifest_name}"] = sha256_file(manifest_stage)
        for item in release:
            materials[f"resources/{item['file']}"] = item["sha256"]
        source_manifest_stage = os.path.join(stage, "source_manifest.json")
        with open(source_manifest_stage, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(source_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        backup = tempfile.mkdtemp(prefix="teacher_backup_", dir=args.resources_dir)
        replaced = []
        source_manifest_replaced = False
        try:
            for item in release:
                name = item["file"]
                destination = os.path.join(args.resources_dir, name)
                if os.path.exists(destination):
                    shutil.copy2(destination, os.path.join(backup, name))
                os.replace(os.path.join(stage, name), destination)
                replaced.append(name)
            manifest_destination = os.path.join(args.resources_dir, manifest_name)
            if os.path.exists(manifest_destination):
                shutil.copy2(manifest_destination, os.path.join(backup, manifest_name))
            os.replace(manifest_stage, manifest_destination)
            shutil.copy2(source_manifest_path, os.path.join(backup, "source_manifest.json"))
            os.replace(source_manifest_stage, source_manifest_path)
            source_manifest_replaced = True
        except Exception:
            for name in replaced:
                old = os.path.join(backup, name)
                destination = os.path.join(args.resources_dir, name)
                if os.path.exists(old):
                    os.replace(old, destination)
                elif os.path.exists(destination):
                    os.remove(destination)
            old_manifest = os.path.join(backup, manifest_name)
            manifest_destination = os.path.join(args.resources_dir, manifest_name)
            if os.path.exists(old_manifest):
                os.replace(old_manifest, manifest_destination)
            elif os.path.exists(manifest_destination):
                os.remove(manifest_destination)
            old_source_manifest = os.path.join(backup, "source_manifest.json")
            if source_manifest_replaced and os.path.exists(old_source_manifest):
                os.replace(old_source_manifest, source_manifest_path)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)

    for item in release:
        actual = sha256_file(os.path.join(args.resources_dir, item["file"]))
        if actual != item["sha256"]:
            raise RuntimeError(f"[ERROR] 반입 후 SHA256 불일치: {item['file']}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

