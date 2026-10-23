# -*- coding: utf-8 -*-
"""mn10 교사 학습 후 판정 — 설계 문서 '학습 후 판정(기준을 미리 정함)'.

1. 증강 없는 train 정확도가 같은 자료의 3차 교사보다 낮지 않은가(낮으면 덜 학습된 것으로 본다).
2. 이진 val Accuracy · Macro P · Macro R · Macro F1 (임계값 0.5).
3. 합친 val AUC 와, 다른 타겟 클래스마다 0.5 이상으로 손든 비율(손들기).
4. BatchNorm 통계 재추정 시험: train 원본으로 BN 통계를 다시 잡아도 val 정확도가 0.02 넘게 바뀌지 않는가.
결과는 plots/mn10_teacher_eval_<mark>.json 에 쓴다.
"""
import copy
import json
import os

import torch

from .data import ClipSet, own_split_items, pooled_val_items, target_class
from .model_def import build_teacher
from .train_loop import auc_pos, predict_positive

# 3차 교사(_run3_20260923)를 같은 train 2,000클립(증강 없음)에 채점한 정확도(2026-10-22 실측, 조각 5개 평균 · 임계값 0.5).
REF_RUN3_TRAIN_ACC = {"mark4.1": 0.9870, "mark4.2": 0.9875, "mark4.3": 0.9990, "mark4.4": 0.9960,
                      "mark4.5": 0.9980, "mark4.6": 0.9990, "mark4.7": 1.0000, "mark4.8": 0.9775}
BN_TOLERANCE = 0.02


def four_metrics(labels, preds):
    labels, preds = labels.tolist(), preds.tolist()
    acc = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    ps, rs, fs = [], [], []
    for c in (0, 1):
        tp = sum(a == c and b == c for a, b in zip(labels, preds))
        fp = sum(a != c and b == c for a, b in zip(labels, preds))
        fn = sum(a == c and b != c for a, b in zip(labels, preds))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        ps.append(p); rs.append(r); fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"Accuracy": acc, "Macro Precision": sum(ps) / 2, "Macro Recall": sum(rs) / 2, "Macro F1": sum(fs) / 2}


@torch.no_grad()
def bn_reestimate(model, train_set, device, batch_size=64):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None                     # 누적 평균
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.eval()
    for s in range(0, len(train_set), batch_size):
        m(train_set.waves[s:s + batch_size].to(device), augment=False)
    return m


def evaluate_teacher(mark_version, ckpt_path, out_json, device, max_clips=0, sets=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_teacher(load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if sets is None:
        sets = {"train": ClipSet(own_split_items(mark_version, "train", max_clips), "train"),
                "val": ClipSet(own_split_items(mark_version, "val", max_clips), "val"),
                "pooled": ClipSet(pooled_val_items(mark_version, max_clips), "합친 val")}
    tr_p, tr_loss = predict_positive(model, sets["train"], device)
    tr_metrics = four_metrics(sets["train"].labels, (tr_p < 0.5).long())
    va_p, va_loss = predict_positive(model, sets["val"], device)
    va_metrics = four_metrics(sets["val"].labels, (va_p < 0.5).long())
    po_p, _ = predict_positive(model, sets["pooled"], device)
    claims = {}
    for src in sorted(set(sets["pooled"].sources)):
        idx = [i for i, s in enumerate(sets["pooled"].sources) if s == src]
        claims[src] = float((po_p[idx] >= 0.5).float().mean())
    tgt = target_class(mark_version)
    other_claims = {k: v for k, v in claims.items() if k not in (tgt, "others")}
    bn_model = bn_reestimate(model, sets["train"], device)
    bn_p, _ = predict_positive(bn_model, sets["val"], device)
    bn_acc = four_metrics(sets["val"].labels, (bn_p < 0.5).long())["Accuracy"]

    ref = REF_RUN3_TRAIN_ACC.get(mark_version)
    under_trained = ref is not None and tr_metrics["Accuracy"] < ref
    bn_unstable = abs(bn_acc - va_metrics["Accuracy"]) > BN_TOLERANCE
    result = {
        "mark_version": mark_version, "target_class": tgt, "checkpoint": os.path.basename(ckpt_path),
        "saved_epoch": ckpt.get("epoch"), "provenance": ckpt.get("provenance", {}),
        "train_clean": {"loss": tr_loss, **tr_metrics, "run3_reference_accuracy": ref},
        "val_binary": {"loss": va_loss, "threshold": 0.5, **va_metrics},
        "pooled_val": {"auc": auc_pos(po_p, sets["pooled"].labels), "n": len(sets["pooled"]),
                       "claim_rate_on_other_targets": other_claims,
                       "mean_claim_rate_on_other_targets": (sum(other_claims.values()) / len(other_claims)) if other_claims else None},
        "bn_reestimate": {"val_accuracy_after": bn_acc, "delta": bn_acc - va_metrics["Accuracy"], "tolerance": BN_TOLERANCE},
        "verdict": {"under_trained": under_trained, "bn_unstable": bn_unstable,
                    "pass": (not under_trained) and (not bn_unstable)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    v = result["verdict"]
    print(f"[EVAL] {mark_version}: train(증강 없음) Acc {tr_metrics['Accuracy']:.4f} (3차 {ref}) · "
          f"val Acc/P/R/F1 {va_metrics['Accuracy']:.4f}/{va_metrics['Macro Precision']:.4f}/"
          f"{va_metrics['Macro Recall']:.4f}/{va_metrics['Macro F1']:.4f} · 합친 val AUC {result['pooled_val']['auc']:.4f} · "
          f"BN 재추정 Δ {result['bn_reestimate']['delta']:+.4f} → {'통과' if v['pass'] else '불합격'}", flush=True)
    return result
