#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""제로샷 평가(학습 0회) — Mark5.2 의 텍스트 앵커 헤드에 학습에 없던 6종 문장을 추가하고 두 세트에서 잰다. (체크리스트 10절)

[신설 2026-09-17 / Mark5_EXP2]

기여 ②: "텍스트 앵커 헤드로 문장 하나에 클래스를 추가한다". softmax 헤드(비교모델)는 클래스를 늘리려면 재학습이 필요한데,
우리 헤드는 앵커 벡터를 하나 더 붙이면 된다. 그것이 실제로 되는지를 잰다.

앵커 = 기존 9종(prompt_bank.json) + 새 6종(zeroshot_prompt_bank.json — door·treadmill·golf_putting·violin·piano·cat).
같은 문장 인코더(all-MiniLM-L6-v2)·같은 템플릿 4개·동의어 평균이라 새 앵커는 기존 앵커와 같은 방식으로 만들어진다.

잰 것
  (1) 제로샷 세트 dataset_eval_zeroshot6.csv (631, 정답 = 새 6종) — 15-way 판정.
      새 클래스별 재현율(정답을 그 클래스로 보낸 비율) · 오분류 방향(가장 많이 간 곳) · 6종 평균.
  (2) 정규 test dataset_test.csv (430, 정답 = 9종) — 같은 15 앵커로 판정.
      9종 Accuracy · Macro P · Macro R · Macro F1. 새 앵커 6개가 붙은 뒤에도 기존 성능이 유지되는지
      (plots/performance_summary_<run_tag>.csv 의 9 앵커 결과와 나란히 적는다 — 있으면).
others 규칙은 그 회차의 val 자동 선택 json(plots/others_threshold_<run_tag>.json)을 그대로 쓴다(15-way softmax 위에서). 없으면 config 값.

산출물 (plots/):
  zeroshot_<run_tag>.json                 요약(위 지표 전부 + 앵커·규칙·체크포인트 sha256)
  zeroshot_predictions_<run_tag>_zeroshot6.csv / _test.csv   클립별 정답·판정·15 확률
  zeroshot_confusion_<run_tag>.csv        새 6종 x 판정 15종 혼동표

사용:  python model/zeroshot_eval.py --mark_version mark5.2 --seed 1 --unlabeled_n 2000
       (학습 때 준 인자와 같게 줘야 그 체크포인트를 찾는다. 인자 없으면 태그 없는 본체 체크포인트)
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
SHARED_DIR = os.path.join(PROJECT_ROOT, "shared_vild")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from vild_config import AudioViLDConfig                                                          # noqa: E402
from vild_model import LearnableBackgroundEmbedding, ViLDTextHead, build_audio_encoder          # noqa: E402
from vild_head import DualBranchStudentHead                                                     # noqa: E402
from vild_parser_teacher import AudioParser                                                     # noqa: E402
from postprocess_utils import (aggregate_segment_probs, apply_abstention, apply_class_pair_calibration,  # noqa: E402
                               apply_others_calibration, apply_temporal_smoothing)
from checkpoint_utils import load_checkpoint, resolve_state_dict                                # noqa: E402
from prompt_bank import get_prompt_texts_for_class                                              # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support                    # noqa: E402

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")


def _resolve_csv(name):
    for cand in (os.path.join(PROJECT_ROOT, name), os.path.join(BASE_DIR, name), os.path.join(PROJECT_ROOT, "preprocessing", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"[ERROR] {name} 이 없습니다(Mark5.2/preprocessing/generate_eval_index.py 로 만들어 복사하십시오).")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_student(config, device):
    model_path = os.path.join(BASE_DIR, f"student_model_{config.run_tag}.pth")
    if not os.path.isfile(model_path):
        raise SystemExit(f"[ERROR] 체크포인트가 없습니다: {model_path}")
    ckpt = load_checkpoint(model_path, map_location=device)
    _enc = (ckpt.get("config_metadata") or {}).get("encoder_type")
    if _enc is not None and _enc != getattr(config, "encoder_type", "cnn"):
        raise SystemExit(f"[ERROR] 체크포인트 encoder_type={_enc!r} 과 config {config.encoder_type!r} 이 다릅니다.")
    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(ckpt, "model_state_dict", "encoder_state_dict", "model"))
    branch = DualBranchStudentHead(config.embedding_dim, distill_final_relu=getattr(config, "distill_branch_final_relu", False)).to(device)
    bs = ckpt.get("branch_state_dict") or ckpt.get("head_state_dict") or ckpt.get("head")
    if bs is not None:
        branch.load_state_dict(bs, strict=False)
    text_head = ViLDTextHead(config).to(device)
    bg = None
    if config.use_background_embedding and ckpt.get("background_state_dict") is not None:
        bg = LearnableBackgroundEmbedding(config.embedding_dim).to(device)
        bg.load_state_dict(ckpt["background_state_dict"]); bg.eval()
    encoder.eval(); branch.eval(); text_head.eval()
    return encoder, branch, text_head, bg, model_path, _sha256(model_path)


def build_anchors(config, zeroshot_bank, device):
    """기존 9 앵커(config) + 새 6 앵커(zeroshot 뱅크, 같은 문장 인코더·템플릿)."""
    from sentence_transformers import SentenceTransformer
    base = config.get_class_text_embeddings(for_evaluation=True).to(device)      # [9, D]
    with open(zeroshot_bank, encoding="utf-8") as f:
        bank = json.load(f)
    new_classes = list(bank["class_synonyms"].keys())
    st = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    embs = []
    prompts_used = {}
    for c in new_classes:
        prompts = get_prompt_texts_for_class(c, zeroshot_bank)
        prompts_used[c] = prompts
        embs.append(st.encode(prompts, convert_to_tensor=True).to(device).mean(dim=0))
    new = torch.stack(embs, dim=0)
    return torch.cat([base, new], dim=0), new_classes, prompts_used


def apply_rule_from_json(config):
    thr_json = os.path.join(PLOT_DIR, f"others_threshold_{config.run_tag}.json")
    if os.path.exists(thr_json):
        with open(thr_json, encoding="utf-8") as f:
            saved = json.load(f)
        config.others_decision_mode = saved.get("rule", config.others_decision_mode)
        if config.others_decision_mode == "direct" and saved.get("threshold") is not None:
            config.others_direct_threshold = float(saved["threshold"])
        return f"val 선택 json({config.others_decision_mode}" + (f", thr {config.others_direct_threshold:g})" if config.others_decision_mode == "direct" else ")")
    return f"config 기본({config.others_decision_mode}" + (f", thr {config.others_direct_threshold:g})" if config.others_decision_mode == "direct" else ")")


def predict_set(rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device):
    """클립마다 eval.py 와 같은 계산(세그먼트 5개 → 두 갈래 혼합 → 집계 → pair 보정 → others 규칙)을 15 앵커로."""
    use_bg_override = getattr(config, "use_background_override_at_eval", False)
    student_input = getattr(config, "student_input", "mel")
    w = getattr(config, "distill_branch_eval_weight", 0.5)
    oi = class_names.index("others")
    out = []
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            recs = parser.load_and_segment_with_metadata(r["path"])
            if not recs:
                out.append((r, None, None)); continue
            seg_probs, sal = [], [x["saliency"] for x in recs]
            for x in recs:
                if student_input == "waveform":
                    feat = encoder(x["wave"].unsqueeze(0).to(device))
                else:
                    seg = x["tensor"]; seg = seg.unsqueeze(0) if seg.ndim == 3 else seg
                    feat = encoder(seg.to(device))
                sup, dis = branch(feat)
                sl, dl = text_head(sup, text_emb), text_head(dis, text_emb)
                if bg is not None and use_bg_override:
                    bgn = F.normalize(bg(), dim=0)
                    sl = sl.clone(); sl[:, oi] = torch.maximum(sl[:, oi], (F.normalize(sup, dim=1) @ bgn) / text_head.temperature)
                    dl = dl.clone(); dl[:, oi] = torch.maximum(dl[:, oi], (F.normalize(dis, dim=1) @ bgn) / text_head.temperature)
                seg_probs.append(((1 - w) * torch.softmax(sl, dim=-1) + w * torch.softmax(dl, dim=-1)).squeeze(0).cpu().numpy())
            if config.enable_temporal_smoothing:
                seg_probs = apply_temporal_smoothing(seg_probs, config.temporal_smoothing_alpha)
            agg, _ = aggregate_segment_probs(seg_probs, sal, config)
            cal = apply_class_pair_calibration(agg, class_names, config)
            cal, pred, _ = apply_others_calibration(cal.copy(), class_names, config)
            cal, pred, _ = apply_abstention(cal, class_names, config, pred)
            out.append((r, int(pred), cal))
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}")
    return out


def compute_metrics(zs, te, base_classes, new_classes, class_names):
    """(1) 제로샷 세트·(2) 정규 test 의 지표. zs·te 는 [(row, pred_idx|None, probs|None)] — zeroshot_eval 과 clap_zeroshot 이 같이 쓴다.

    - per_class[c]: n · recall(정답을 c 로 보낸 비율) · top_wrong(가장 많이 간 다른 곳) · sent_to_others
    - zeroshot6: accuracy(6종 전체 클립 기준) · macro_recall_new6 · 15-way 라벨 공간에서 새 6종의 macro P/R/F1
      (예측이 기존 9종으로 새면 재현율이, 기존 클립이 새 종으로 오면 정밀도가 떨어진다 — test 클립도 함께 넣는다)
    - test9: 9종 Accuracy · Macro P · Macro R · Macro F1(15 앵커로 판정, 라벨은 9종) · 새 클래스로 샌 클립 수
    """
    C, B = len(class_names), len(base_classes)
    conf = np.zeros((len(new_classes), C), dtype=int)
    for r, pred, _ in zs:
        if pred is None:
            continue
        conf[new_classes.index(r["label"]), pred] += 1
    per_class = {}
    for i, c in enumerate(new_classes):
        n = int(conf[i].sum()); hit = int(conf[i, B + i]) if n else 0
        wrong = conf[i].copy(); wrong[B + i] = 0
        top = int(np.argmax(wrong)) if wrong.sum() > 0 else None
        per_class[c] = {"n": n, "recall": (hit / n) if n else float("nan"),
                        "top_wrong": class_names[top] if top is not None else None,
                        "top_wrong_share": (float(wrong[top]) / n) if (top is not None and n) else 0.0,
                        "sent_to_others": (int(conf[i, class_names.index("others")]) / n) if n else float("nan")}
    n_zs = sum(v["n"] for v in per_class.values())
    zs_acc = float(sum(v["n"] * v["recall"] for v in per_class.values() if v["n"]) / max(1, n_zs))
    zs_macro_recall = float(np.nanmean([v["recall"] for v in per_class.values()])) if n_zs else float("nan")
    y_true_all = [new_classes.index(r["label"]) + B for r, p, _ in zs if p is not None] + \
                 [base_classes.index(r["label"]) for r, p, _ in te if p is not None and r["label"] in base_classes]
    y_pred_all = [p for r, p, _ in zs if p is not None] + [p for r, p, _ in te if p is not None and r["label"] in base_classes]
    p6, r6, f6, _ = precision_recall_fscore_support(y_true_all, y_pred_all, labels=list(range(B, C)), average="macro", zero_division=0)
    y_t = [base_classes.index(r["label"]) for r, p, _ in te if p is not None and r["label"] in base_classes]
    y_p = [p for r, p, _ in te if p is not None and r["label"] in base_classes]
    acc9 = accuracy_score(y_t, y_p) if y_t else float("nan")
    mp9, mr9, mf9, _ = precision_recall_fscore_support(y_t, y_p, labels=list(range(B)), average="macro", zero_division=0)
    leaked = sum(1 for p in y_p if p >= B)
    return {"conf": conf, "per_class": per_class,
            "zeroshot6": {"n": n_zs, "accuracy": zs_acc, "macro_recall_new6": zs_macro_recall,
                          "macro_precision_new6_15way": float(p6), "macro_recall_new6_15way": float(r6), "macro_f1_new6_15way": float(f6),
                          "per_class": per_class},
            "test9": {"n": len(y_t), "accuracy": float(acc9), "macro_precision": float(mp9), "macro_recall": float(mr9),
                      "macro_f1": float(mf9), "sent_to_new_classes": leaked}}


def _read_reference_summary(path):
    """performance_summary CSV 의 4지표(있으면). 9 앵커 결과와 나란히 적기 위한 것."""
    out = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for parts in csv.reader(f):
                if parts and parts[0] in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"):
                    try:
                        out[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    return out


def save_and_print(json_stem, pred_stem, suffix, title, zs, te, metrics, base_classes, new_classes, class_names,
                   extra, rule_desc, t0, reference_summary=None):
    """산출물 세 종류(요약 json · 클립별 CSV 2개 · 혼동표 CSV)를 쓰고 화면에 요약을 찍는다. 이름의 앞부분만 다르고 형식은 같다."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    for name, res in (("zeroshot6", zs), ("test", te)):
        with open(os.path.join(PLOT_DIR, f"{pred_stem}_{name}{suffix}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Filename", "True Label", "Predicted Label"] + [f"Prob_{c}" for c in class_names])
            for r, pred, cal in res:
                w.writerow([os.path.basename(r["path"]), r["label"], class_names[pred] if pred is not None else ""]
                           + ([f"{float(v):.6f}" for v in cal] if cal is not None else []))
    base9 = _read_reference_summary(reference_summary)
    summary = dict(extra)
    summary.update({"anchors": class_names, "new_classes": new_classes,
                    "zeroshot6": metrics["zeroshot6"],
                    "test9_with_15_anchors": dict(metrics["test9"], reference_9_anchors=base9 or None),
                    "elapsed_sec": round(time.time() - t0, 1), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(os.path.join(PLOT_DIR, f"{json_stem}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    conf_stem = json_stem.replace("zeroshot_", "zeroshot_confusion_", 1) if json_stem.startswith("zeroshot_") else f"{json_stem}_confusion"
    with open(os.path.join(PLOT_DIR, f"{conf_stem}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + class_names)
        for i, c in enumerate(new_classes):
            w.writerow([c] + [int(v) for v in metrics["conf"][i]])

    z, t9, C = metrics["zeroshot6"], metrics["test9"], len(class_names)
    print("\n" + "=" * 78)
    print(f" 제로샷 결과 — {title} (앵커 {C}개, others 규칙 {rule_desc})")
    print("=" * 78)
    print(f" (1) 제로샷 6종 {z['n']}클립: accuracy {z['accuracy']:.4f} · 새 6종 macro recall {z['macro_recall_new6']:.4f} "
          f"· 15-way 새 6종 macro P {z['macro_precision_new6_15way']:.4f} / R {z['macro_recall_new6_15way']:.4f} / F1 {z['macro_f1_new6_15way']:.4f}")
    for c, v in metrics["per_class"].items():
        print(f"     {c:<13} n={v['n']:<4} recall {v['recall']:.3f}  → others {v['sent_to_others']:.3f}  가장 많이 간 곳: {v['top_wrong']} ({v['top_wrong_share']:.3f})")
    print(f" (2) 정규 test 9종 {t9['n']}클립({C} 앵커): Accuracy {t9['accuracy']:.4f} · Macro P {t9['macro_precision']:.4f} "
          f"· Macro R {t9['macro_recall']:.4f} · Macro F1 {t9['macro_f1']:.4f} · 새 클래스로 샌 클립 {t9['sent_to_new_classes']}")
    if base9:
        print(f"     참고(9 앵커, {os.path.basename(reference_summary)}): " + " · ".join(f"{k} {v:.4f}" for k, v in base9.items()))
    print(f"[INFO] 저장: {PLOT_DIR}/{json_stem}.json · {pred_stem}_{{zeroshot6,test}}{suffix}.csv · {conf_stem}.csv")


def main():
    ap = argparse.ArgumentParser(description="Mark5.2 제로샷 평가(앵커 15개)")
    ap.add_argument("--mark_version", type=str, default="mark5.2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--unlabeled_n", type=int, default=None)
    ap.add_argument("--labeled_n", type=int, default=None)
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--spill_labeled", action="store_true")
    ap.add_argument("--pseudo_source", type=str, default=None)
    ap.add_argument("--zeroshot_bank", type=str, default=os.path.join(SHARED_DIR, "resources", "zeroshot_prompt_bank.json"))
    ap.add_argument("--limit", type=int, default=None, help="동작 확인용: 세트마다 라벨별 앞 N 클립만(산출물 이름에 _limitN)")
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    config.experiment_seed = args.seed; config.unlabeled_n = None if args.ceiling else args.unlabeled_n
    config.ceiling = args.ceiling; config.labeled_n = args.labeled_n
    config.spill_labeled = args.spill_labeled; config.pseudo_source = args.pseudo_source
    device = config.device
    tag = config.run_tag
    suffix = f"_limit{args.limit}" if args.limit else ""

    encoder, branch, text_head, bg, model_path, sha = load_student(config, device)
    text_emb, new_classes, prompts_used = build_anchors(config, args.zeroshot_bank, device)
    base_classes = list(config.get_classes_for_evaluation())
    class_names = base_classes + new_classes
    rule_desc = apply_rule_from_json(config)
    print(f"[INFO] 제로샷: run_tag={tag} · 체크포인트 {os.path.basename(model_path)} · 앵커 {len(class_names)}개 "
          f"({len(base_classes)} + 새 {len(new_classes)}: {new_classes}) · others 규칙 {rule_desc}")
    parser = AudioParser(config, segment_mode=True)

    zs_rows = list(csv.DictReader(open(_resolve_csv("dataset_eval_zeroshot6.csv"), newline="", encoding="utf-8-sig")))
    te_rows = list(csv.DictReader(open(_resolve_csv("dataset_test.csv"), newline="", encoding="utf-8-sig")))
    if args.limit:                                                  # 라벨별 앞 N 클립 (6종·9종이 다 들어가야 동작 확인이 된다)
        def _per_label(rows, n):
            seen, out = {}, []
            for r in rows:
                if seen.get(r["label"], 0) < n:
                    out.append(r); seen[r["label"]] = seen.get(r["label"], 0) + 1
            return out
        zs_rows, te_rows = _per_label(zs_rows, args.limit), _per_label(te_rows, args.limit)
    bad = sorted({r["label"] for r in zs_rows if r["label"] not in new_classes})
    if bad:
        raise SystemExit(f"[ERROR] dataset_eval_zeroshot6.csv 의 라벨 {bad} 이 zeroshot 뱅크의 클래스 {new_classes} 에 없습니다.")

    t0 = time.time()
    print(f"[INFO] (1) 제로샷 세트 {len(zs_rows)}클립")
    zs = predict_set(zs_rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device)
    print(f"[INFO] (2) 정규 test {len(te_rows)}클립 (15 앵커)")
    te = predict_set(te_rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device)

    extra = {"run_tag": tag, "checkpoint": {"file": os.path.basename(model_path), "sha256": sha},
             "zeroshot_bank": os.path.relpath(args.zeroshot_bank, PROJECT_ROOT), "prompts_used": prompts_used,
             "others_rule": rule_desc, "limit": args.limit}
    metrics = compute_metrics(zs, te, base_classes, new_classes, class_names)
    save_and_print(f"zeroshot_{tag}{suffix}", f"zeroshot_predictions_{tag}", suffix, tag, zs, te, metrics,
                   base_classes, new_classes, class_names, extra, rule_desc, t0,
                   reference_summary=os.path.join(PLOT_DIR, f"performance_summary_{tag}.csv"))


if __name__ == "__main__":
    main()
