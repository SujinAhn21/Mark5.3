# -*- coding: utf-8 -*-
"""mn10 이진 교사 학습 진입점.

[신설 2026-10-22 / mn10 교사 B+]

    python -m teacher_mn10.train_teacher --mark_version mark4.1                    # 본 학습 + 학습 후 판정
    python -m teacher_mn10.train_teacher --mark_version mark4.1 --overfit_check 64  # 학습 전: 작은 묶음 과적합 시험
    python -m teacher_mn10.train_teacher --mark_version mark4.1 --max_clips 32 --epochs 1   # 동작 시험(_smoke)

산출물(저장소 루트 기준, _smoke 는 이름에 붙는다)
    model/mn10_teacher_<mark>.pt               저장한 epoch 의 가중치 + provenance(schema 3)
    plots/mn10_teacher_history_<mark>.csv      epoch 별 train/val 손실·정확도·합친 val AUC·학습률(완주 표식)
    plots/mn10_teacher_curve_<mark>.png
    plots/mn10_teacher_eval_<mark>.json        학습 후 판정(evaluate.py)
"""
import argparse
import glob
import hashlib
import os
import platform
import random
import sys
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (ClipSet, REPO_ROOT, dataset_identity, own_split_items, pooled_val_items, target_class)
from .evaluate import evaluate_teacher
from .model_def import MODEL_KIND, build_teacher
from .train_loop import fit

SCHEMA_VERSION = 3


def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def runtime_info(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        info["sm"] = f"{major}{minor}"
    print("[RUNTIME] " + " · ".join(f"{k} {v}" for k, v in info.items()), flush=True)
    return info


def code_sha256():
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))):
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def overfit_check(train_items, device, n_clips, max_steps=300, target_loss=0.05, batch_size=16):
    """작은 묶음 과적합 시험: 증강 없이 n_clips 만 반복 학습해 train 손실이 target_loss 아래로 가는지 본다."""
    pos = [it for it in train_items if it[1] == 0][: n_clips // 2]
    neg = [it for it in train_items if it[1] == 1][: n_clips - len(pos)]
    small = ClipSet(pos + neg, f"과적합 시험 {len(pos) + len(neg)}클립")
    model = build_teacher().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(small, batch_size=batch_size, shuffle=True)
    step, last = 0, float("inf")
    while step < max_steps:
        for waves, y, _ in loader:
            model.train()
            loss = F.cross_entropy(model(waves.to(device), augment=False), y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                with torch.no_grad():
                    model.eval()
                    full = F.cross_entropy(model(small.waves.to(device), augment=False), small.labels.to(device)).item()
                last = full
                print(f"[OVERFIT] step {step}: 묶음 전체 손실 {full:.4f}", flush=True)
                if full < target_loss:
                    print(f"[OVERFIT] 통과 — {step} step 만에 손실 {full:.4f} < {target_loss}", flush=True)
                    return True
            if step >= max_steps:
                break
    print(f"[OVERFIT] 불합격 — {max_steps} step 뒤 손실 {last:.4f} (기준 {target_loss}). 학습 경로를 먼저 점검하십시오.", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", required=True, choices=[f"mark4.{i}" for i in range(1, 9)])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_clips", type=int, default=0, help="동작 시험용: split 마다 이 개수만. 산출물 이름에 _smoke")
    ap.add_argument("--overfit_check", type=int, default=0, help="학습 전 점검: 이 개수 클립으로 과적합 시험만 하고 끝낸다")
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    set_seed(args.seed)
    rt = runtime_info(device)
    if rt.get("sm", "0").startswith(("10", "12")):
        raise SystemExit("[STOP] Blackwell 계열 GPU(sm_100·sm_120) 입니다. L4 런타임으로 다시 실행하십시오(2026-10-17 G4 계산 오류).")
    mv, tgt = args.mark_version, target_class(args.mark_version)
    print(f"[INFO] {mv} mn10 교사 — 자기 클래스 {tgt} 대 others", flush=True)

    train_items = own_split_items(mv, "train", args.max_clips)
    if args.overfit_check:
        ok = overfit_check(train_items, device, args.overfit_check, max_steps=args.overfit_steps)
        sys.exit(0 if ok else 1)

    val_items = own_split_items(mv, "val", args.max_clips)
    pooled_items = pooled_val_items(mv, args.max_clips)
    identity = dataset_identity(mv, {"train": train_items, "val": val_items})
    sets = {"train": ClipSet(train_items, "train"), "val": ClipSet(val_items, "val"), "pooled": ClipSet(pooled_items, "합친 val")}
    loader = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = build_teacher().to(device)

    sfx = "_smoke" if args.max_clips else ""
    ckpt = os.path.join(REPO_ROOT, "model", f"mn10_teacher_{mv}{sfx}.pt")
    hist = os.path.join(REPO_ROOT, "plots", f"mn10_teacher_history_{mv}{sfx}.csv")
    curve = os.path.join(REPO_ROOT, "plots", f"mn10_teacher_curve_{mv}{sfx}.png")
    out_json = os.path.join(REPO_ROOT, "plots", f"mn10_teacher_eval_{mv}{sfx}.json")
    run_id, code_sha = str(uuid.uuid4()), code_sha256()

    def meta(epoch, metrics):
        return {"schema_version": SCHEMA_VERSION, "model_kind": MODEL_KIND, "mark_version": mv,
                "provenance": {**identity, "run_id": run_id, "seed": args.seed, "best_epoch": epoch,
                               "target_class": tgt, "classes": [tgt, "others"], "select_metric": "pooled_val_auc",
                               "clip_aggregation": "clip", "teacher_code_sha256": code_sha,
                               "recipe": {"optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
                                          "batch_clips": args.batch_size, "sched": "ReduceLROnPlateau(0.5,3)",
                                          "patience": args.patience, "epochs_max": args.epochs, "spec_augment": True},
                               "runtime": rt}}

    fit(model, loader, sets["val"], sets["pooled"], device, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience, ckpt_path=ckpt, history_csv_path=hist,
        curve_png_path=curve, tag=f"{mv} mn10 teacher", ckpt_meta_fn=meta)
    result = evaluate_teacher(mv, ckpt, out_json, device, sets=sets)
    if not result["verdict"]["pass"]:
        print("[WARN] 학습 후 판정 불합격 — 이 교사를 채택 관문에 올리기 전에 원인을 찾으십시오.", flush=True)


if __name__ == "__main__":
    main()
