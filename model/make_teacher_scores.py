#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""교사 점수 캐시 — 교사 8개가 Mark5.3 클립마다 낸 '자기 클래스 확률'을 한 번 계산해 CSV 로 고정한다.

[신설 2026-10-22 / 검증 결합기] 설계 문서 '검증 결합기와 세 단계 전달' · 'Mark5.3 학습 변경(교사 출력 캐시)'.

교사는 증강 없는 원본만 보므로(지금의 --teacher_clean_view 와 같은 조건) 점수는 학습 내내 같다. 그래서 학습 중에 교사를
돌리지 않고 이 캐시를 쓴다. 결합기(make_combiner_labels.py)와 채택 관문(teacher_gate_report.py)이 이 파일을 읽는다.

    python model/make_teacher_scores.py --teacher_kind mn10       # resources/mn10_teacher_mark4.{1..8}.pt (B+ 교사)
    python model/make_teacher_scores.py --teacher_kind segment    # resources/best_teacher_*_mark4.{1..8}.pth (B0 조각 교사)
교사는 먼저 resources/ 로 반입한다: model/import_teachers.py --kind mn10 | segment(기본) — [2026-10-23]

교사별 점수
    mn10    : 3초 클립 통째 → softmax 의 자기 클래스 확률
    segment : 조각 5개(salient top-k) 각각의 자기 클래스 확률 → 조각 평균(지금 Mark5.3 의 --teacher_seg_mean 과 같은 집계)

산출물 (--out 기본 plots/teacher_scores_<kind>.csv)
    path, split, label(정답이 있는 split 만), p_<클래스> 8열(heavy_impact … dog_bark 순서)
    같은 이름 .json : 교사 종류 · 파일별 SHA-256 · release 해시 · split 별 클립 수 · 만든 시각
    plots/teacher_feats_mn10.pt (mn10 만) : unlabeled 클립의 융합 특징 [N, 960] — 아블레이션 '옛 교사 경로 복원'의 특징 정렬 목표.
        융합 = 교사별 특징을 단위 벡터로 만든 뒤 자기 클래스 확률로 가중 평균(vild/teacher_fusion.py 와 같은 식).
        scores_sha256 에 같은 실행의 점수 캐시 해시를 넣어 짝을 고정한다(model/label_freshness.check_features).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"),
          os.path.join(PROJECT_ROOT, "shared_vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.append(p)

SPLIT_CSV = {"labeled": "dataset_labeled.csv", "unlabeled": "dataset_unlabeled.csv",
             "val": "dataset_val.csv", "test": "dataset_test.csv"}
SOURCE_ALIAS = {"data_labeled": "data_source_labeled", "data_unlabeled": "data_source_unlabeled",
                "data_val": "data_source_val", "data_test": "data_source_test"}
TEACHER_VERSIONS = [f"mark4.{i}" for i in range(1, 9)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def localize(path):
    """CSV 의 코랩 경로 → 지금 환경의 파일. 코랩에서는 그대로 있다. 로컬(USB)에서는 형제 저장소 Mark5.2 의
    data_* 또는 data_source_* 폴더에서 찾는다."""
    p = str(path).replace("\\", "/")
    if os.path.isfile(p):
        return p
    m = re.search(r"/(Mark5\.2|Mark5\.3|Mark5_EXP2?)/(data_[a-z]+)/(.+)$", p)
    if m:
        root = os.path.join(os.path.dirname(PROJECT_ROOT), m.group(1))
        for folder in (m.group(2), SOURCE_ALIAS.get(m.group(2), m.group(2))):
            c = os.path.join(root, folder, *m.group(3).split("/"))
            if os.path.isfile(c):
                return c
    raise FileNotFoundError(f"[ERROR] 오디오 파일이 없습니다: {path}")


def read_split(split):
    from train_mark5 import _resolve_csv_path
    with open(_resolve_csv_path(SPLIT_CSV[split]), newline="", encoding="utf-8-sig") as f:
        return [{"path": r["path"], "label": r.get("label", "")} for r in csv.DictReader(f)]


def class_order(config):
    from train_mark5 import specialist_config_for
    return list(specialist_config_for(config).keys())          # heavy_impact … dog_bark (mark4.1 … mark4.8 순서)


class Mn10Scorer:
    """resources/mn10_teacher_mark4.X.pt 8개. 체크포인트의 schema·종류·클래스를 확인한 뒤 평가 모드로 쓴다."""

    def __init__(self, classes, teacher_dir, device):
        from teacher_mn10.model_def import MODEL_KIND, build_teacher
        self.device, self.models, self.files = device, [], []
        for cls, mv in zip(classes, TEACHER_VERSIONS):
            path = os.path.join(teacher_dir, f"mn10_teacher_{mv}.pt")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            prov = ck.get("provenance") or {}
            if ck.get("schema_version") != 3 or ck.get("model_kind") != MODEL_KIND:
                raise RuntimeError(f"[ERROR] mn10 교사 체크포인트가 아닙니다(schema 3 · {MODEL_KIND} 필요): {path}")
            if prov.get("target_class") != cls or prov.get("classes") != [cls, "others"]:
                raise RuntimeError(f"[ERROR] {path} 의 클래스가 {cls} 가 아닙니다: {prov.get('classes')}")
            m = build_teacher(load_pretrained=False).to(device)
            m.load_state_dict(ck["model_state"])
            m.eval()
            self.models.append(m)
            self.files.append({"class": cls, "file": os.path.basename(path), "sha256": sha256_file(path),
                               "run_id": prov.get("run_id"), "saved_epoch": ck.get("epoch")})

    @torch.no_grad()
    def __call__(self, paths, with_features=False):
        from teacher_mn10.data import load_clip
        waves = torch.stack([torch.from_numpy(load_clip(localize(p))) for p in paths]).to(self.device)
        probs, fsum, wsum = [], 0.0, 0.0
        for m in self.models:
            logits, feats = m.forward_with_features(waves, augment=False)
            p = torch.softmax(logits, dim=1)[:, 0]
            probs.append(p.cpu())
            if with_features:
                w = p.unsqueeze(1)
                fsum = fsum + torch.nn.functional.normalize(feats, dim=-1) * w
                wsum = wsum + w
        probs = torch.stack(probs, dim=1)                                                     # [B, 8]
        if not with_features:
            return probs
        return probs, (fsum / wsum.clamp_min(1e-6)).cpu()                                     # [B, 960] 융합 특징


class SegmentScorer:
    """resources/best_teacher_*_mark4.X.pth (조각 교사, schema v2). Mark5.3 의 EnsembleTeacher 로 조각별 자기 클래스 확률을 얻어 평균한다."""

    def __init__(self, classes, config, device):
        from train_mark5 import EnsembleTeacher, specialist_config_for
        from vild_parser_teacher import AudioParser
        spec = specialist_config_for(config)
        self.teacher = EnsembleTeacher(spec, device)
        if self.teacher.clip_input:
            raise RuntimeError("[ERROR] resources/ 의 교사가 클립 입력판입니다. 조각 교사(B0)를 반입하십시오.")
        self.parser = AudioParser(config, segment_mode=True)
        self.classes, self.device = classes, device
        self.files = self.teacher.release_identity["files"]

    @torch.no_grad()
    def __call__(self, paths):
        from train_mark5 import specialist_config_for  # noqa: F401  (EnsembleTeacher 와 같은 모듈 상태)
        from vild_config import AudioViLDConfig
        label_map = {c: i for i, c in enumerate(AudioViLDConfig(mark_version="mark5.2").get_classes_for_evaluation())}
        rows = []
        for p in paths:
            recs = self.parser.load_and_segment_with_metadata(localize(p))
            if not recs:
                raise RuntimeError(f"[ERROR] 조각을 만들지 못했습니다: {p}")
            out = self.teacher(torch.stack([r["tensor"] for r in recs]).to(self.device), label_map)
            seg = torch.stack([out.specialist_weights[c].squeeze(1).float() for c in self.classes], dim=1)   # [K, 8]
            rows.append(seg.mean(dim=0))
        return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser(description="교사 8개의 클립 점수를 한 번 계산해 CSV 로 고정한다")
    ap.add_argument("--teacher_kind", required=True, choices=["mn10", "segment"])
    ap.add_argument("--teacher_dir", default=os.path.join(PROJECT_ROOT, "resources"))
    ap.add_argument("--splits", nargs="+", default=["labeled", "unlabeled", "val", "test"], choices=list(SPLIT_CSV))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="동작 시험용: split 마다 앞에서 N개. 산출물 이름에 _limitN")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vild_config import AudioViLDConfig
    config = AudioViLDConfig(mark_version="mark5.2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = class_order(config)
    scorer = (Mn10Scorer(classes, args.teacher_dir, device) if args.teacher_kind == "mn10"
              else SegmentScorer(classes, config, device))
    release = hashlib.sha256(json.dumps(scorer.files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sfx = f"_limit{args.limit}" if args.limit else ""
    out = args.out or os.path.join(PROJECT_ROOT, "plots", f"teacher_scores_{args.teacher_kind}{sfx}.csv")
    print(f"[INFO] 교사 {args.teacher_kind} · release {release[:16]} · device {device} · split {args.splits}", flush=True)

    rows, counts, t0 = [], {}, time.time()
    feat_paths, feat_rows = [], []     # [2026-10-22] mn10 · unlabeled 만
    for split in args.splits:
        items = read_split(split)
        if args.limit:
            items = items[:args.limit]
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            if args.teacher_kind == "mn10" and split == "unlabeled":
                probs, feats = scorer([r["path"] for r in chunk], with_features=True)
                feat_paths.extend(r["path"] for r in chunk); feat_rows.append(feats.half())
            else:
                probs = scorer([r["path"] for r in chunk])
            if not torch.isfinite(probs).all():
                raise FloatingPointError("[ERROR] 교사 점수에 NaN/Inf 가 있습니다.")
            for r, pr in zip(chunk, probs.tolist()):
                rows.append({"path": r["path"], "split": split, "label": r["label"] if split != "unlabeled" else "",
                             **{f"p_{c}": round(v, 7) for c, v in zip(classes, pr)}})
        counts[split] = len(items)
        print(f"[INFO] {split}: {len(items)}클립 ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "label"] + [f"p_{c}" for c in classes])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    meta = {"teacher_kind": args.teacher_kind, "classes": classes, "teacher_files": scorer.files,
            "teacher_release_sha256": release, "split_counts": counts, "limit": args.limit,
            "scores_sha256": sha256_file(out), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if feat_rows:
        fpath = os.path.join(os.path.dirname(os.path.abspath(out)),
                             os.path.basename(out).replace("teacher_scores_", "teacher_feats_").replace(".csv", ".pt"))
        feats = torch.cat(feat_rows)
        if not torch.isfinite(feats.float()).all():
            raise FloatingPointError("[ERROR] 교사 특징에 NaN/Inf 가 있습니다.")
        torch.save({"paths": feat_paths, "features": feats, "dim": int(feats.shape[1]), "scores_sha256": meta["scores_sha256"],
                    "teacher_release_sha256": release, "fusion": "unit-normalized teacher features weighted by own-class probability"},
                   fpath + ".tmp")
        os.replace(fpath + ".tmp", fpath)
        meta["features_file"] = os.path.basename(fpath)
        meta["features_sha256"] = sha256_file(fpath)
        print(f"[완료] 특징 캐시 {fpath} ({feats.shape[0]}×{feats.shape[1]})", flush=True)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[완료] {out} ({len(rows)}행) · release {release[:16]}", flush=True)


if __name__ == "__main__":
    main()
