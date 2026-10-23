# -*- coding: utf-8 -*-
"""mn10 교사 학습 루프 — EfficientAT 비교모델의 검증된 Adam 루프(커밋 a4da6a8 의 common/train_loop.py)를 그대로 따른다.

[신설 2026-10-22 / mn10 교사 B+] 설계 문서 '학습이 덜 될 위험을 미리 없애는 조치'.

    Adam(lr 1e-4, weight_decay 1e-4) · ReduceLROnPlateau(mode=min, factor 0.5, patience 3) · 조기 종료 patience 10
    · 최대 80 epoch · 손실 = cross-entropy · SpecAugment 는 학습 배치에만 · 클립 16개 배치
비교모델 루프와 다른 점은 하나다. 학습률 규칙과 조기 종료는 비교모델처럼 자기 val 손실로 판단하고,
**저장할 epoch 는 합친 val 의 '자기 대 나머지' AUC 로 고른다**(다중분류 교사로 쓸 것이므로). AUC 가 같으면 val 손실이 낮은 쪽.
"""
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def predict_positive(model, dataset, device, batch_size=64):
    """평가 모드에서 자기 클래스 확률 [N] 과 CE 손실 평균(표본 평균)."""
    model.eval()
    probs, loss_sum = [], 0.0
    for s in range(0, len(dataset), batch_size):
        waves = dataset.waves[s:s + batch_size].to(device)
        y = dataset.labels[s:s + batch_size].to(device)
        logits = model(waves, augment=False)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs.append(torch.softmax(logits, dim=1)[:, 0].cpu())
    return torch.cat(probs), loss_sum / len(dataset)


def auc_pos(scores, labels):
    """자기 클래스(label 0)를 양성으로 한 ROC AUC. 동점은 0.5 로 센다."""
    pos = scores[labels == 0]
    neg = scores[labels == 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = torch.cat([pos, neg]).argsort()
    ranks = torch.empty(len(order), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    allv = torch.cat([pos, neg])
    for v in allv.unique():                       # 동점 평균 순위
        m = allv == v
        if int(m.sum()) > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum().item()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fit(model, train_loader, val_set, pooled_set, device, *, lr=1e-4, weight_decay=1e-4, epochs=80, patience=10,
        sched_factor=0.5, sched_patience=3, use_spec_augment=True,
        ckpt_path, history_csv_path, curve_png_path=None, tag="", ckpt_meta_fn=None):
    """ckpt_meta_fn(epoch, metrics) -> dict : 체크포인트에 함께 넣을 기록(provenance 등)."""
    for stale in (ckpt_path, history_csv_path):          # 옛 산출물과 섞이지 않게 먼저 지운다
        if stale and os.path.exists(stale):
            os.remove(stale)
            print(f"[INFO] 옛 산출물 삭제: {stale}")
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=sched_factor, patience=sched_patience)
    print(f"[INFO] 학습 절차: Adam lr={lr:g}, weight_decay={weight_decay}, "
          f"ReduceLROnPlateau(factor={sched_factor}, patience={sched_patience}), 조기 종료 patience={patience}(val 손실), "
          f"저장 기준 = 합친 val AUC, SpecAugment={'on' if use_spec_augment else 'off'}, "
          f"배치 {train_loader.batch_size}클립, 최대 {epochs} epoch", flush=True)

    best_val, no_improve, stopped_early = math.inf, 0, False
    best_sel, best_epoch = (-math.inf, math.inf), -1
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, n_batches, correct, total = 0.0, 0, 0, 0
        for waves, y, _ in train_loader:
            waves, y = waves.to(device), y.to(device)
            logits = model(waves, augment=use_spec_augment)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("[ERROR] train 손실이 NaN/Inf 입니다.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        train_loss, train_acc = loss_sum / max(n_batches, 1), correct / max(total, 1)
        val_p, val_loss = predict_positive(model, val_set, device)
        val_acc = float(((val_p < 0.5).long() == val_set.labels).float().mean())
        pooled_p, _ = predict_positive(model, pooled_set, device)
        pooled_auc = auc_pos(pooled_p, pooled_set.labels)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
               "val_acc": val_acc, "pooled_auc": pooled_auc, "lr": cur_lr, "seconds": round(elapsed, 1)}
        history.append(row)
        print(f"[Epoch {epoch + 1}] Train {train_loss:.4f} (acc {train_acc:.4f}) | Val {val_loss:.4f} (acc {val_acc:.4f}) "
              f"| 합친 val AUC {pooled_auc:.4f} | lr {cur_lr:g} | {elapsed:.1f}s", flush=True)

        sel = (pooled_auc, val_loss)
        if sel[0] > best_sel[0] or (sel[0] == best_sel[0] and sel[1] < best_sel[1]):
            best_sel, best_epoch = sel, epoch + 1
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            payload = {"model_state": model.state_dict(), "epoch": epoch + 1, "metrics": dict(row), "tag": tag}
            if ckpt_meta_fn:
                payload.update(ckpt_meta_fn(epoch + 1, dict(row)))
            tmp = ckpt_path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, ckpt_path)
            print(f"[SAVE] 합친 val AUC {pooled_auc:.4f} (val 손실 {val_loss:.4f}) — epoch {epoch + 1} 저장", flush=True)

        prev_lr = cur_lr
        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] != prev_lr:
            print(f"[LR] {prev_lr:g} -> {optimizer.param_groups[0]['lr']:g} (val={val_loss:.6f})", flush=True)
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped_early = True
                print(f"[INFO] 조기 종료(val 손실 {patience} epoch 개선 없음). 저장된 epoch = {best_epoch}", flush=True)
                break

    if best_epoch < 0:
        raise RuntimeError("[ERROR] 저장된 epoch 가 없습니다(AUC 가 NaN 인지 확인).")
    os.makedirs(os.path.dirname(os.path.abspath(history_csv_path)), exist_ok=True)
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:        # 완주 표식: 루프가 끝난 뒤에만 쓴다
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"[INFO] 학습 기록 저장(완주 표식): {history_csv_path}", flush=True)
    if curve_png_path:
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(ep, [h["train_loss"] for h in history], label="Train CE", linewidth=2)
        ax[0].plot(ep, [h["val_loss"] for h in history], label="Val CE", linewidth=2)
        ax[1].plot(ep, [h["pooled_auc"] for h in history], label="Pooled val AUC", linewidth=2)
        for a in ax:
            a.axvline(best_epoch, color="gray", linestyle=":", label=f"saved epoch {best_epoch}")
            a.set_xlabel("Epoch"); a.grid(True); a.legend()
        ax[0].set_ylabel("Loss"); ax[1].set_ylabel("AUC"); fig.suptitle(tag)
        fig.tight_layout(); fig.savefig(curve_png_path); plt.close(fig)
    return {"best_epoch": best_epoch, "best_pooled_auc": best_sel[0], "epochs_run": len(history),
            "stopped_early": stopped_early, "history": history}
