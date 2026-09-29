#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""제로샷 평가(학습 0회) — Mark5.2 의 텍스트 앵커 헤드에 학습에 없던 6종 문장을 추가하고 두 세트에서 잰다. (체크리스트 10절)

[신설 2026-09-17 / Mark5_EXP2]

기여 ②: "텍스트 앵커 헤드로 문장 하나에 클래스를 추가한다". softmax 헤드(비교모델)는 클래스를 늘리려면 재학습이 필요한데,
우리 헤드는 앵커 벡터를 하나 더 붙이면 된다. 그것이 실제로 되는지를 잰다.

앵커 = 기존 9종(prompt_bank.json) + 새 6종(zeroshot_prompt_bank.json — door·treadmill·golf_putting·violin·piano·cat).
같은 문장 인코더(all-MiniLM-L6-v2)·같은 템플릿 4개·동의어 평균이라 새 앵커는 기존 앵커와 같은 방식으로 만들어진다.

잰 것
  (1) 제로샷 세트 dataset_eval_zeroshot6.csv (631, 정답 = 새 6종) — others 는 9 앵커 규칙, 클래스는 15 앵커 argmax.
      새 클래스별 재현율(정답을 그 클래스로 보낸 비율) · 오분류 방향(가장 많이 간 곳) · 6종 평균.
  (2) 정규 test dataset_test.csv (430, 정답 = 9종) — 같은 15 앵커로 판정.
      9종 Accuracy · Macro P · Macro R · Macro F1. 새 앵커 6개가 붙은 뒤에도 기존 성능이 유지되는지
      (plots/performance_summary_<run_tag>.csv 의 9 앵커 결과와 나란히 적는다 — 있으면).
others 규칙은 그 회차의 val 자동 선택 json(plots/others_threshold_<run_tag>.json)을 그대로 쓴다. 없으면 config 값.
  ★ others 판정은 **기존 9 앵커의 softmax**(= eval.py 와 완전히 같은 계산)로 내린다. 15-way softmax 위에 9-way 에서 고른 임계값을 얹으면
    Prob_others 가 앵커 6개만큼 희석돼 운영점이 바뀌기 때문이다(2026-09-17 검수에서 잡음). 그래서
      (a) 9 앵커 경로: 세그먼트 softmax(9) → 집계 → pair 보정 → others 규칙 → "others 인가"       ← eval.py 그대로
      (b) 15 앵커 경로: 세그먼트 softmax(15) → 집계 → pair 보정 → others 를 뺀 argmax             ← 새 앵커가 붙은 클래스 판정
    (a) 가 others 면 others, 아니면 (b). CSV 의 Prob_* 열은 (b)의 15-way 확률이고 "Prob others (9-way)" 열이 (a)의 값이다.

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
from vild_model import LearnableBackgroundEmbedding, build_audio_encoder, build_student_head    # noqa: E402
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


def _student_records(parser, path, clip_mode):
    """[이식 2026-09-29 / A3] student 입력 기록. 클립 모드면 클립 전체(첫 48,000샘플 파형 + 클립 mel 뷰)를 조각 1개짜리 기록으로 —
    eval.py 의 클립 모드와 같은 계산(조각 가중 평균은 가중치 1). 조각 모드면 옛 대로 1초 조각 K개."""
    if clip_mode:
        out = parser.load_clip_with_segments(path)
        return [out["clip_record"]] if out is not None else []
    return parser.load_and_segment_with_metadata(path)


def load_student(config, device):
    if getattr(config, "student_head", "text_anchor") == "linear":
        raise SystemExit("[ERROR] --linear_head(선형 분류 헤드) 모델에는 텍스트 앵커가 없어 제로샷이 정의되지 않습니다(진단용 H 회차).")
    model_path = os.path.join(BASE_DIR, f"student_model_{config.run_tag}.pth")
    if not os.path.isfile(model_path):
        raise SystemExit(f"[ERROR] 체크포인트가 없습니다: {model_path}")
    ckpt = load_checkpoint(model_path, map_location=device)
    _enc = (ckpt.get("config_metadata") or {}).get("encoder_type")
    if _enc is not None and _enc != getattr(config, "encoder_type", "cnn"):
        raise SystemExit(f"[ERROR] 체크포인트 encoder_type={_enc!r} 과 config {config.encoder_type!r} 이 다릅니다.")
    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(ckpt, "model_state_dict", "encoder_state_dict", "model"))
    # [이식 2026-09-29 / PH] 학습과 같은 구조여야 branch state_dict 가 맞는다(head_pretrained_hidden 이면 supervised branch = Identity).
    branch = DualBranchStudentHead(config.embedding_dim, distill_final_relu=getattr(config, "distill_branch_final_relu", False),
                                   supervised_identity=bool(getattr(config, "head_pretrained_hidden", False))).to(device)
    bs = ckpt.get("branch_state_dict") or ckpt.get("head_state_dict") or ckpt.get("head")
    if bs is not None:
        branch.load_state_dict(bs, strict=False)
    text_head = build_student_head(config).to(device)   # text_anchor 만 여기 온다(선형은 위에서 막음) — eval.py 와 같은 함수
    # [이식 2026-09-26 / Mark5.2] classifier_state_dict 를 헤드에 로드한다. 학습형 로짓 스케일(LT)을 켠 뒤로는
    # 이 줄이 없으면 학습된 온도가 평가에서 조용히 무시된다.
    _cls_state = ckpt.get("classifier_state_dict")
    if getattr(text_head, "learnable", False):
        if not _cls_state or "logit_scale" not in _cls_state:
            raise RuntimeError("[ERROR] 체크포인트에 logit_scale 이 없습니다 — 학습형 로짓 스케일이 없던 "
                               "2026-09-24 이전 코드로 학습한 모델이거나 run_tag 가 어긋난 것입니다.")
        text_head.load_state_dict(_cls_state)
        print(f"[LT] 학습형 로짓 스케일 로드: T={text_head.temperature:.4f} (초기 {getattr(config, 'logit_temperature', 0.07)})")
    elif _cls_state:
        text_head.load_state_dict(_cls_state, strict=False)
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


def apply_rule_from_json(config, ckpt_sha=None, allow_config_rule=False):
    """val 자동 선택 json 의 규칙을 config 에 싣는다. eval.py 와 같이 체크포인트 sha256 을 대조한다(다르면 멈춤 — 재학습 뒤 val 을 다시 돌려야 한다).
    [검수 P5 / 2026-09-17] json 이 없으면 eval.py 처럼 멈춘다(전에는 config 기본 direct 0.09 로 조용히 떨어져 표 8 에 다른 운영점이
    섞일 수 있었다). 동작 확인용으로만 --allow_config_rule 로 config 값을 쓰게 하고, 그때는 화면과 json 의 others_rule 에 표시된다."""
    thr_json = os.path.join(PLOT_DIR, f"others_threshold_{config.run_tag}.json")
    if os.path.exists(thr_json):
        with open(thr_json, encoding="utf-8") as f:
            saved = json.load(f)
        saved_sha = (saved.get("checkpoint") or {}).get("sha256") or saved.get("checkpoint_sha256")
        # [2차 검수 P-2] eval.py 와 같은 강도: run_tag 도 대조하고 지문 없는 json 도 통과시키지 않는다.
        if saved.get("run_tag") != config.run_tag or not saved_sha or (ckpt_sha and saved_sha != ckpt_sha):
            raise SystemExit(f"[ERROR] {os.path.basename(thr_json)} 은 지금 체크포인트로 고른 값이 아닙니다"
                             f"(json run_tag={saved.get('run_tag')} sha256={str(saved_sha)[:12]} / 지금 {config.run_tag} sha256={str(ckpt_sha)[:12]})"
                             f" — eval.py --split val 을 다시 돌리십시오.")
        config.others_decision_mode = saved.get("rule", config.others_decision_mode)
        if config.others_decision_mode == "direct" and saved.get("threshold") is not None:
            config.others_direct_threshold = float(saved["threshold"])
        # [2026-09-22 / C] 두 헤드 섞는 가중치도 test 와 같게 json 에서(없으면 config 기본 0.5). 클래스 편향(A)은 제로샷에 걸지 않는다 —
        # 9클래스 val 로 고른 사전확률 보정을 새 앵커 6종이 든 15앵커 argmax 에 적용할 근거가 없다(비교모델 zeroshot_eval 도 같음).
        if saved.get("distill_eval_weight") is not None:
            config.distill_branch_eval_weight = float(saved["distill_eval_weight"])
        return f"val 선택 json({config.others_decision_mode}" + (f", thr {config.others_direct_threshold:g})" if config.others_decision_mode == "direct" else ")")
    if not allow_config_rule:
        raise SystemExit(f"[ERROR] {thr_json} 이 없습니다. eval.py --split val 을 먼저 돌려 others 규칙을 골라야 test·eval 과 같은 운영점으로 "
                         f"제로샷을 잴 수 있습니다(run_experiments.py 는 이 순서로 돕니다). 동작 확인만 하려면 --allow_config_rule.")
    print(f"[WARN] {os.path.basename(thr_json)} 이 없어 config 기본 규칙을 씁니다(--allow_config_rule). 표에 넣을 값이 아닙니다.")
    return f"config 기본(--allow_config_rule, {config.others_decision_mode}" + (f", thr {config.others_direct_threshold:g})" if config.others_decision_mode == "direct" else ")")


def predict_set(rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device, n_base=None):
    """클립마다 두 경로를 계산한다(모듈 머리말 ★ 참고).
      (a) 9 앵커(text_emb[:n_base]) 경로 = eval.py 와 같은 계산으로 others 판정
      (b) 15 앵커 경로 = 같은 계산을 15 앵커로 → others 를 뺀 argmax 가 클래스 판정
    반환 [(row, pred_idx|None, probs15|None, prob_others9|None)]."""
    use_bg_override = getattr(config, "use_background_override_at_eval", False)
    student_input = getattr(config, "student_input", "mel")
    clip_mode = bool(getattr(config, "student_clip_input", False))   # [A3] mark5.2 기본 = 클립 입력(--segment_input 이면 조각)
    w = getattr(config, "distill_branch_eval_weight", 0.5)
    oi = class_names.index("others")
    n_base = n_base or len(class_names)
    base_names = class_names[:n_base]
    out = []

    def _seg_probs(sup, dis, emb):
        sl, dl = text_head(sup, emb), text_head(dis, emb)
        if bg is not None and use_bg_override:
            bgn = F.normalize(bg(), dim=0)
            sl = sl.clone(); sl[:, oi] = torch.maximum(sl[:, oi], (F.normalize(sup, dim=1) @ bgn) / text_head.temperature)
            dl = dl.clone(); dl[:, oi] = torch.maximum(dl[:, oi], (F.normalize(dis, dim=1) @ bgn) / text_head.temperature)
        return ((1 - w) * torch.softmax(sl, dim=-1) + w * torch.softmax(dl, dim=-1)).squeeze(0).cpu().numpy()

    def _clip(seg_probs, sal, names):
        if config.enable_temporal_smoothing:
            seg_probs = apply_temporal_smoothing(seg_probs, config.temporal_smoothing_alpha)
        agg, _ = aggregate_segment_probs(seg_probs, sal, config)
        return apply_class_pair_calibration(agg, names, config)

    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            recs = _student_records(parser, r["path"], clip_mode)
            if not recs:
                out.append((r, None, None, None)); continue
            p9, p15, sal = [], [], [x["saliency"] for x in recs]
            for x in recs:
                if student_input == "waveform":
                    feat = encoder(x["wave"].unsqueeze(0).to(device))
                else:
                    seg = x["tensor"]; seg = seg.unsqueeze(0) if seg.ndim == 3 else seg
                    feat = encoder(seg.to(device))
                sup, dis = branch(feat)
                p9.append(_seg_probs(sup, dis, text_emb[:n_base]))
                p15.append(_seg_probs(sup, dis, text_emb))
            # (a) eval.py 와 같은 others 판정 (9 앵커)
            cal9 = _clip(p9, sal, base_names)
            cal9, pred9, _ = apply_others_calibration(cal9.copy(), base_names, config)
            cal9, pred9, _ = apply_abstention(cal9, base_names, config, pred9)
            # (b) 15 앵커 클래스 판정
            cal15 = _clip(p15, sal, class_names)
            if int(pred9) == oi:
                pred = oi
            else:
                masked = cal15.copy(); masked[oi] = -1.0
                pred = int(np.argmax(masked))
            out.append((r, int(pred), cal15, float(cal9[oi])))
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}")
    return out


def compute_metrics(zs, te, base_classes, new_classes, class_names):
    """(1) 제로샷 세트·(2) 정규 test 의 지표. zs·te 는 [(row, pred_idx|None, probs|None)] — zeroshot_eval 과 clap_zeroshot 이 같이 쓴다.

    - per_class[c]: n · recall(정답을 c 로 보낸 비율) · top_wrong(가장 많이 간 다른 곳) · sent_to_others
    - zeroshot6: accuracy(6종 전체 클립 기준) · macro P/R/F1(6종 세트 안, 새 6종 라벨 기준) · 15-way 라벨 공간에서 새 6종의 macro P/R/F1
      (예측이 기존 9종으로 새면 재현율이, 기존 클립이 새 종으로 오면 정밀도가 떨어진다 — test 클립도 함께 넣는다)
    - test9: 9종 Accuracy · Macro P · Macro R · Macro F1(15 앵커로 판정, 라벨은 9종) · 새 클래스로 샌 클립 수
    """
    C, B = len(class_names), len(base_classes)
    conf = np.zeros((len(new_classes), C), dtype=int)
    for r, pred, *_ in zs:
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
    # [검수 P8 / 2026-09-17] 6종 세트 안에서의 macro P·F1 — 지표 4개 세트 규칙(Accuracy 옆에 P·R·F1). 예측이 기존 9종·others 로
    # 새면 그 클립은 어느 새 클래스의 예측에도 안 들어가므로 재현율만 깎이고, 새 클래스끼리 헷갈리면 정밀도가 깎인다.
    y_true_zs = [new_classes.index(r["label"]) + B for r, p, *_ in zs if p is not None]
    y_pred_zs = [p for r, p, *_ in zs if p is not None]
    if y_true_zs:
        p6s, r6s, f6s, _ = precision_recall_fscore_support(y_true_zs, y_pred_zs, labels=list(range(B, C)), average="macro", zero_division=0)
    else:
        p6s = r6s = f6s = float("nan")
    y_true_all = [new_classes.index(r["label"]) + B for r, p, *_ in zs if p is not None] + \
                 [base_classes.index(r["label"]) for r, p, *_ in te if p is not None and r["label"] in base_classes]
    y_pred_all = [p for r, p, *_ in zs if p is not None] + [p for r, p, *_ in te if p is not None and r["label"] in base_classes]
    p6, r6, f6, _ = precision_recall_fscore_support(y_true_all, y_pred_all, labels=list(range(B, C)), average="macro", zero_division=0)
    y_t = [base_classes.index(r["label"]) for r, p, *_ in te if p is not None and r["label"] in base_classes]
    y_p = [p for r, p, *_ in te if p is not None and r["label"] in base_classes]
    acc9 = accuracy_score(y_t, y_p) if y_t else float("nan")
    mp9, mr9, mf9, _ = precision_recall_fscore_support(y_t, y_p, labels=list(range(B)), average="macro", zero_division=0)
    leaked = sum(1 for p in y_p if p >= B)
    return {"conf": conf, "per_class": per_class,
            "zeroshot6": {"n": n_zs, "accuracy": zs_acc, "macro_precision_new6": float(p6s), "macro_recall_new6": zs_macro_recall,
                          "macro_f1_new6": float(f6s), "macro_recall_new6_sklearn": float(r6s),
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
            w.writerow(["Filename", "True Label", "Predicted Label", "Prob others (9-way)"] + [f"Prob_{c}" for c in class_names])
            for item in res:
                r, pred, cal = item[0], item[1], item[2]
                p9 = item[3] if len(item) > 3 else None
                w.writerow([os.path.basename(r["path"]), r["label"], class_names[pred] if pred is not None else "",
                            (f"{p9:.6f}" if p9 is not None else "")]
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
    print(f" (1) 제로샷 6종 {z['n']}클립: accuracy {z['accuracy']:.4f} · 새 6종 macro P {z['macro_precision_new6']:.4f} / R {z['macro_recall_new6']:.4f} "
          f"/ F1 {z['macro_f1_new6']:.4f} · 15-way(+test 클립) 새 6종 macro P {z['macro_precision_new6_15way']:.4f} / R {z['macro_recall_new6_15way']:.4f} "
          f"/ F1 {z['macro_f1_new6_15way']:.4f}")
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
    # [신설 2026-09-25 / LR] 학습 때 준 --lr/--encoder_lr 과 같은 값. 안 주면 config 기본값의 태그를 찾는다(옛 동작).
    ap.add_argument("--lr", type=float, default=None, help="[LR] 학습 때 준 --lr 과 같은 값(run_tag _lr<값>)")
    ap.add_argument("--encoder_lr", type=float, default=None, help="[LR] 학습 때 준 --encoder_lr 과 같은 값(run_tag _elr<값>)")
    ap.add_argument("--zeroshot_bank", type=str, default=os.path.join(SHARED_DIR, "resources", "zeroshot_prompt_bank.json"))
    ap.add_argument("--limit", type=int, default=None, help="동작 확인용: 세트마다 라벨별 앞 N 클립만(산출물 이름에 _limitN)")
    ap.add_argument("--allow_config_rule", action="store_true", help="동작 확인용: val 선택 json 이 없어도 config 기본 others 규칙으로 진행")
    # [이식 2026-09-29 / Mark5.2] 학습(train_mark5.py) 때 준 것과 같은 ablation·묶음 플래그. run_tag 맞추기에 더해 PH 는 인코더·branch 구조,
    # clip/segment 는 student 입력 단위를 바꾼다. --linear_head 모델은 앵커가 없어 제로샷 불가(load_student 가 멈춤).
    ap.add_argument("--no_feature_kd", action="store_true", help="[A1] 학습 때 줬으면 같이(run_tag _noFKD)")
    ap.add_argument("--pl_min_gap", type=float, default=None, help="[A2] 학습 때 준 값(0 이면 run_tag _noGAP)")
    ap.add_argument("--no_distill_branch", action="store_true", help="[A4] 학습 때 줬으면 같이(run_tag _noDB)")
    ap.add_argument("--clip_input", action="store_true", help="[A3] 학습 때 줬으면 같이(run_tag _noSEG · mark5.2 는 기본이 클립 입력)")
    ap.add_argument("--linear_head", action="store_true", help="[H] 진단용 선형 헤드 — 제로샷 불가(멈춤)")
    ap.add_argument("--segment_input", action="store_true", help="학습 때 줬으면 같이(조각 1초×5 입력·이름 표식 없음)")
    ap.add_argument("--head_pretrained_hidden", action="store_true", help="[PH] 학습 때 줬으면 같이(run_tag _PH · 투영/branch 구조)")
    ap.add_argument("--lp_epochs", type=int, default=None, help="[LP-FT] 학습 때 준 값(run_tag _LP<N>)")
    ap.add_argument("--hard_weight", type=float, default=None, help="[HW] 학습 때 준 값(run_tag _HW<g>)")
    ap.add_argument("--others_from_reject", type=float, default=None, help="[OR] 학습 때 준 값(run_tag _OR<τ>)")
    ap.add_argument("--neg_pl", type=float, nargs=2, default=None, metavar=("TAU", "BETA"), help="[NEG] 학습 때 준 값(run_tag _NEG)")
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    config.experiment_seed = args.seed; config.unlabeled_n = None if args.ceiling else args.unlabeled_n
    config.ceiling = args.ceiling; config.labeled_n = args.labeled_n
    config.spill_labeled = args.spill_labeled; config.pseudo_source = args.pseudo_source
    _lr_txt = config.apply_cli_learning_rates(args.lr, args.encoder_lr)   # [LR 2026-09-25] run_tag 를 학습 때와 맞춘다
    if _lr_txt:
        print(f"[LR] 명령줄 학습률: {_lr_txt} (체크포인트 이름 맞추기용)")
    _ab_txt = config.apply_cli_ablation(no_feature_kd=args.no_feature_kd, pl_min_gap=args.pl_min_gap, no_distill_branch=args.no_distill_branch,
                                        clip_input=args.clip_input, linear_head=args.linear_head, segment_input=args.segment_input,
                                        head_pretrained_hidden=args.head_pretrained_hidden, lp_epochs=args.lp_epochs, hard_weight=args.hard_weight,
                                        others_from_reject=args.others_from_reject, neg_pl=args.neg_pl)
    if _ab_txt:
        print(f"[ABLATION] {_ab_txt} (체크포인트 이름·구조 맞추기용)")
    if getattr(config, "student_clip_input", False):
        print(f"[A3] 평가 단위 = 클립(파형 {getattr(config, 'student_clip_samples', 48000)}샘플 통째, 조각 1개 취급) — eval.py 와 같은 계산")
    device = config.device
    tag = config.run_tag
    suffix = f"_limit{args.limit}" if args.limit else ""

    encoder, branch, text_head, bg, model_path, sha = load_student(config, device)
    text_emb, new_classes, prompts_used = build_anchors(config, args.zeroshot_bank, device)
    base_classes = list(config.get_classes_for_evaluation())
    class_names = base_classes + new_classes
    rule_desc = apply_rule_from_json(config, sha, allow_config_rule=args.allow_config_rule)
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
    zs = predict_set(zs_rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device, n_base=len(base_classes))
    print(f"[INFO] (2) 정규 test {len(te_rows)}클립 (15 앵커)")
    te = predict_set(te_rows, config, parser, encoder, branch, text_head, bg, text_emb, class_names, device, n_base=len(base_classes))

    extra = {"run_tag": tag, "checkpoint": {"file": os.path.basename(model_path), "sha256": sha},
             "zeroshot_bank": os.path.relpath(args.zeroshot_bank, PROJECT_ROOT), "prompts_used": prompts_used,
             "others_rule": rule_desc, "limit": args.limit}
    metrics = compute_metrics(zs, te, base_classes, new_classes, class_names)
    save_and_print(f"zeroshot_{tag}{suffix}", f"zeroshot_predictions_{tag}", suffix, tag, zs, te, metrics,
                   base_classes, new_classes, class_names, extra, rule_desc, t0,
                   reference_summary=os.path.join(PLOT_DIR, f"performance_summary_{tag}.csv"))


if __name__ == "__main__":
    main()
