#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pseudo-label 출처 비교(체크리스트 8절)용 라벨 CSV 생성기 — "왜 하필 이진 전문가들의 합의인가"에 숫자로 답하기 위해
전문가 8개 합의(우리 방식) 대신 다른 주체가 붙인 pseudo-label 을 만든다.

[신설 2026-09-17 / Mark5_EXP2]

세 출처(조건은 labeled 200 + unlabeled 3,800: dataset_unlabeled 2,000 + 남은 labeled 1,800, --spill_labeled):
  ① 전문가 8개 합의  — 이 스크립트가 아니라 train_mark5.py 가 학습 중 온라인으로 붙인다(기본 경로).
  ② self             — 라벨 200 으로만 배운 student(u0, 같은 seed)가 스스로 붙인다.  --source self
  ③ gen              — 라벨 200 으로 파인튜닝한 단일 generalist(Baseline_EfficientAT_mn10 sup 조건, 같은 seed)가 붙인다.  --source gen

출력: <프로젝트 루트>/pseudo_labels_<source>_l<labeled_n>_s<seed>.csv   열: path,label,confidence,source,model,generated_at
  train_mark5.py --pseudo_csv <이 파일> --pseudo_source <self|gen> --labeled_n 200 --spill_labeled --seed <k>
  가 이것을 읽어 teacher 대신 쓴다(feature KD 는 없음 — teacher 특징이 없으므로).

라벨 규칙:
  self : eval.py 와 같은 클립 판정 — 세그먼트 5개 확률(두 갈래 혼합) 집계 → class-pair 보정 → others 규칙.
         others 규칙은 그 회차의 plots/others_threshold_<run_tag>.json(val 자동 선택)을 쓴다(체크포인트 sha256 대조, 없으면 멈춤. 동작 확인용 --allow_config_rule).
         confidence = 최종 확률 벡터의 최댓값.
  gen  : 비교모델 폴더의 model_def.build_model 로 만들고 체크포인트(model_state)를 실어 3.0초 클립을 통째로 넣는다.
         others 규칙은 비교모델 eval.py 와 같이 그 회차의 <baseline_dir>/plots/others_threshold_<tag>.json(val 자동 선택, 2026-09-17 B안)을
         쓴다(체크포인트 sha256 대조, 없으면 멈춤. --allow_config_rule 이면 argmax). confidence = softmax 최댓값. 클래스 이름으로 맞춘다.

같은 부분집합을 쓰기 위해 train_mark5.select_labeled_subset(같은 seed·labeled_n)로 "남은 labeled" 를 뽑는다.

사용:
  python model/make_pseudo_labels.py --source self --labeled_n 200 --seed 1 --spill_labeled \\
      --ckpt model/student_model_mark5.2_PL_SA_l200_u0_s1.pth
  python model/make_pseudo_labels.py --source gen  --labeled_n 200 --seed 1 --spill_labeled \\
      --ckpt ../Baseline_EfficientAT_mn10/model/efficientat_mn10_sup_lr5e-05_l200_s1.pth --baseline_dir ../Baseline_EfficientAT_mn10
  --limit 5 를 주면 동작 확인용(산출물 이름에 _limit5).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
SHARED_DIR = os.path.join(PROJECT_ROOT, "shared_vild")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from vild_config import AudioViLDConfig                                   # noqa: E402
from train_mark5 import select_labeled_subset, _resolve_csv_path          # noqa: E402


def _pool(config, labeled_n, seed, spill_labeled, limit=None):
    """unlabeled 풀 = dataset_unlabeled.csv (+ 남은 labeled). 학습이 쓰는 것과 같은 규칙."""
    rows = list(csv.DictReader(open(_resolve_csv_path("dataset_unlabeled.csv"), newline="", encoding="utf-8-sig")))
    pool = [{"path": r["path"]} for r in rows]
    n_unl = len(pool)
    if spill_labeled:
        if labeled_n is None:
            raise SystemExit("[ERROR] --spill_labeled 는 --labeled_n 이 필요합니다.")
        lab = list(csv.DictReader(open(_resolve_csv_path("dataset_labeled.csv"), newline="", encoding="utf-8-sig")))
        _, rest = select_labeled_subset(lab, labeled_n, seed, list(config.get_classes_for_evaluation()))
        pool += [{"path": r["path"]} for r in rest]
    print(f"[INFO] 풀: dataset_unlabeled {n_unl} + 남은 labeled {len(pool) - n_unl} = {len(pool)}개")
    if limit:
        pool = pool[:int(limit)]
        print(f"[INFO] --limit {limit}: 앞 {len(pool)}개만")
    return pool


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_with_self(config, ckpt_path, pool, device, allow_config_rule=False):
    """Mark5.2 student 로 라벨링 — eval.py 의 클립 판정과 같은 계산."""
    import torch.nn.functional as F
    from postprocess_utils import (aggregate_segment_probs, apply_abstention, apply_class_pair_calibration,
                                   apply_others_calibration, apply_temporal_smoothing)
    from vild_model import LearnableBackgroundEmbedding, ViLDTextHead, build_audio_encoder
    from vild_head import DualBranchStudentHead
    from vild_parser_teacher import AudioParser
    from checkpoint_utils import load_checkpoint, resolve_state_dict

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_sha = _sha256(ckpt_path)
    _ckpt_enc = (ckpt.get("config_metadata") or {}).get("encoder_type")
    if _ckpt_enc is not None and _ckpt_enc != getattr(config, "encoder_type", "cnn"):
        raise SystemExit(f"[ERROR] 체크포인트 encoder_type={_ckpt_enc!r} 과 config {config.encoder_type!r} 이 다릅니다.")
    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(ckpt, "model_state_dict", "encoder_state_dict", "model"))
    branch = DualBranchStudentHead(config.embedding_dim,
                                   distill_final_relu=getattr(config, "distill_branch_final_relu", False)).to(device)
    bs = ckpt.get("branch_state_dict") or ckpt.get("head_state_dict") or ckpt.get("head")
    if bs is not None:
        branch.load_state_dict(bs, strict=False)
    text_head = ViLDTextHead(config).to(device)
    text_emb = config.get_class_text_embeddings(for_evaluation=True).to(device)
    bg = None
    if config.use_background_embedding and ckpt.get("background_state_dict") is not None:
        bg = LearnableBackgroundEmbedding(config.embedding_dim).to(device)
        bg.load_state_dict(ckpt["background_state_dict"]); bg.eval()
    use_bg_override = getattr(config, "use_background_override_at_eval", False)
    encoder.eval(); branch.eval(); text_head.eval()
    class_names = config.get_classes_for_evaluation()
    student_input = getattr(config, "student_input", "mel")
    w = getattr(config, "distill_branch_eval_weight", 0.5)

    # others 규칙: 그 회차의 val 자동 선택 json 을 쓴다(eval.py test 와 같은 규칙).
    # [검수 P5 / 2026-09-17] json 이 없으면 멈추고, 있으면 체크포인트 sha256 을 대조한다(eval.py·zeroshot_eval.py 와 같은 규칙).
    # 전에는 json 이 없으면 config 기본(direct 0.09)으로 조용히 떨어졌고 지문 대조도 없었다. --allow_config_rule 은 동작 확인용.
    thr_json = os.path.join(PROJECT_ROOT, "plots", f"others_threshold_{config.run_tag}.json")
    if os.path.exists(thr_json):
        with open(thr_json, encoding="utf-8") as f:
            saved = json.load(f)
        saved_sha = (saved.get("checkpoint") or {}).get("sha256")
        # [2차 검수 P-2] eval.py 와 같은 강도로: run_tag 도 대조하고, 지문이 없는 json(옛 형식)도 통과시키지 않는다.
        if saved.get("run_tag") != config.run_tag or not saved_sha or saved_sha != ckpt_sha:
            raise SystemExit(f"[ERROR] {os.path.basename(thr_json)} 은 지금 체크포인트로 고른 값이 아닙니다"
                             f"(json run_tag={saved.get('run_tag')} sha256={str(saved_sha)[:12]} / 지금 {config.run_tag} sha256={ckpt_sha[:12]})"
                             f" — 재학습했으면 eval.py --split val 을 다시 돌리십시오.")
        config.others_decision_mode = saved.get("rule", config.others_decision_mode)
        if config.others_decision_mode == "direct" and saved.get("threshold") is not None:
            config.others_direct_threshold = float(saved["threshold"])
        rule_desc = f"val 선택 json({config.others_decision_mode}"
    elif allow_config_rule:
        print(f"[WARN] {os.path.basename(thr_json)} 이 없어 config 기본 규칙을 씁니다(--allow_config_rule). 실험용 라벨이 아닙니다.")
        rule_desc = f"config 기본(--allow_config_rule, {config.others_decision_mode}"
    else:
        raise SystemExit(f"[ERROR] {thr_json} 이 없습니다. self 라벨은 test 와 같은 운영점으로 매겨야 하므로 그 회차의 eval.py --split val 을 "
                         f"먼저 돌리십시오. 동작 확인만 하려면 --allow_config_rule.")
    if config.others_decision_mode == "direct":
        rule_desc += f", thr {config.others_direct_threshold:g})"
    else:
        rule_desc += ")"
    print(f"[INFO] self 라벨링: {os.path.basename(ckpt_path)} · 입력 {student_input} · others 규칙 {rule_desc}")

    parser = AudioParser(config, segment_mode=True)
    out = []
    with torch.no_grad():
        for i, item in enumerate(pool, 1):
            recs = parser.load_and_segment_with_metadata(item["path"])
            if not recs:
                out.append((item["path"], None, 0.0)); continue
            seg_probs, sal = [], [r["saliency"] for r in recs]
            for r in recs:
                if student_input == "waveform":
                    feat = encoder(r["wave"].unsqueeze(0).to(device))
                else:
                    seg = r["tensor"]; seg = seg.unsqueeze(0) if seg.ndim == 3 else seg
                    feat = encoder(seg.to(device))
                sup, dis = branch(feat)
                sl, dl = text_head(sup, text_emb), text_head(dis, text_emb)
                if bg is not None and use_bg_override:
                    oi = class_names.index("others"); bgn = F.normalize(bg(), dim=0)
                    sl = sl.clone(); sl[:, oi] = torch.maximum(sl[:, oi], (F.normalize(sup, dim=1) @ bgn) / text_head.temperature)
                    dl = dl.clone(); dl[:, oi] = torch.maximum(dl[:, oi], (F.normalize(dis, dim=1) @ bgn) / text_head.temperature)
                prob = ((1 - w) * torch.softmax(sl, dim=-1) + w * torch.softmax(dl, dim=-1)).squeeze(0).cpu().numpy()
                seg_probs.append(prob)
            if config.enable_temporal_smoothing:
                seg_probs = apply_temporal_smoothing(seg_probs, config.temporal_smoothing_alpha)
            agg, _ = aggregate_segment_probs(seg_probs, sal, config)
            cal = apply_class_pair_calibration(agg, class_names, config)
            cal, pred, _ = apply_others_calibration(cal.copy(), class_names, config)
            cal, pred, _ = apply_abstention(cal, class_names, config, pred)
            out.append((item["path"], class_names[int(pred)], float(np.max(cal))))
            if i % 200 == 0:
                print(f"  {i}/{len(pool)}")
    return out, os.path.basename(ckpt_path)


def _label_with_generalist(config, ckpt_path, baseline_dir, pool, device, allow_config_rule=False):
    """비교모델(EfficientAT sup) 로 라벨링 — 3.0초 클립 통째로, softmax 에 비교모델 eval.py 와 같은 others 규칙.
    [2차 검수 C-2 / 2026-09-17] 전에는 argmax 고정이라 B안(val 선택 규칙) 뒤의 비교모델 test 판정·self 라벨과 운영점이 달랐다."""
    bdir = os.path.abspath(baseline_dir)
    if not os.path.isfile(os.path.join(bdir, "model_def.py")):
        raise SystemExit(f"[ERROR] --baseline_dir 에 model_def.py 가 없습니다: {bdir}")
    sys.path.insert(0, bdir)
    import importlib
    model_def = importlib.import_module("model_def")
    bdata = importlib.import_module("common.data")
    bpaths = importlib.import_module("common.paths")
    brule = importlib.import_module("common.others_rule")
    tag = os.path.splitext(os.path.basename(ckpt_path))[0]           # 비교모델 체크포인트 이름 = run_tag (common/naming.artifact_paths)
    rule_json = brule.rule_json_path(os.path.join(bdir, "plots"), tag)
    if os.path.isfile(rule_json):
        rule, thr, _saved = brule.load_rule_json(rule_json, tag, ckpt_path)      # run_tag·sha256 다르면 RuntimeError
        rule_desc = f"val 선택 json {brule.describe(rule, thr)}"
    elif allow_config_rule:
        rule, thr = "argmax", None
        rule_desc = "argmax(--allow_config_rule, json 없음 — 실험용 라벨이 아닙니다)"
        print(f"[WARN] {os.path.basename(rule_json)} 이 없어 argmax 로 진행합니다(--allow_config_rule).")
    else:
        raise SystemExit(f"[ERROR] {rule_json} 이 없습니다. gen 라벨은 비교모델 test 와 같은 운영점으로 매겨야 하므로 그 회차의 "
                         f"eval.py --split val 을 먼저 돌리십시오(run_experiments.py 는 이 순서로 돕니다). 동작 확인만 하려면 --allow_config_rule.")
    model = model_def.build_model(num_classes=len(bpaths.CLASS_NAMES), load_pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.eval()
    bnames = list(bpaths.CLASS_NAMES)
    ours = list(config.get_classes_for_evaluation())
    missing = [c for c in ours if c not in bnames]
    if missing:
        raise SystemExit(f"[ERROR] 비교모델 클래스에 없는 이름 {missing} (비교모델 {bnames})")
    print(f"[INFO] gen 라벨링: {os.path.basename(ckpt_path)} · 비교모델 {getattr(model, 'MODEL_TAG', '?')} · 클래스 {bnames} · others 규칙 {rule_desc}")
    out = []
    with torch.no_grad():
        for i, item in enumerate(pool, 1):
            wav = torch.from_numpy(bdata.load_clip(item["path"])).unsqueeze(0).to(device)
            prob = torch.softmax(model(wav), dim=-1).squeeze(0).cpu().numpy()
            k = int(brule.predict_rule(prob[None, :], bnames, rule, thr)[0])      # eval.py 와 같은 판정
            out.append((item["path"], bnames[k], float(prob[k])))
            if i % 200 == 0:
                print(f"  {i}/{len(pool)}")
    return out, os.path.basename(ckpt_path)


def main():
    ap = argparse.ArgumentParser(description="pseudo-label 출처 비교용 라벨 CSV 생성")
    ap.add_argument("--source", type=str, required=True, choices=["self", "gen"])
    ap.add_argument("--ckpt", type=str, required=True, help="self: Mark5.2 student .pth / gen: 비교모델 .pth")
    ap.add_argument("--mark_version", type=str, default="mark5.2")
    ap.add_argument("--labeled_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1, help="학습에 쓴 seed(남은 labeled 부분집합·run_tag 를 맞추는 용도)")
    ap.add_argument("--unlabeled_n", type=int, default=None, help="self 의 학습 run_tag 를 맞추는 용도(보통 0)")
    ap.add_argument("--spill_labeled", action="store_true", help="남은 labeled 도 풀에 넣는다(8절 조건: 3,800)")
    ap.add_argument("--baseline_dir", type=str, default=os.path.join(PROJECT_ROOT, "..", "Baseline_EfficientAT_mn10"))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow_config_rule", action="store_true", help="동작 확인용: val 선택 json 이 없어도 진행(self: config 기본 규칙, gen: argmax)")
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    device = config.device
    if args.source == "self":
        # self 는 "labeled_n 으로만 배운 u0 student" 가 원칙. run_tag 를 그 회차와 맞춰 val 선택 json 을 찾는다.
        config.experiment_seed = args.seed
        config.labeled_n = args.labeled_n
        config.unlabeled_n = 0 if args.unlabeled_n is None else args.unlabeled_n
    pool = _pool(config, args.labeled_n, args.seed, args.spill_labeled, args.limit)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"[ERROR] 체크포인트가 없습니다: {ckpt}")
    t0 = time.time()
    if args.source == "self":
        rows, model_name = _label_with_self(config, ckpt, pool, device, allow_config_rule=args.allow_config_rule)
    else:
        rows, model_name = _label_with_generalist(config, ckpt, args.baseline_dir, pool, device, allow_config_rule=args.allow_config_rule)

    suffix = f"_limit{args.limit}" if args.limit else ""
    out = args.out or os.path.join(PROJECT_ROOT, f"pseudo_labels_{args.source}_l{args.labeled_n}_s{args.seed}{suffix}.csv")
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    n_ok = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "confidence", "source", "model", "generated_at"])
        for path, label, conf in rows:
            if label is None:
                continue
            w.writerow([path, label, f"{conf:.6f}", args.source, model_name, generated_at]); n_ok += 1
    dist = {}
    for _, label, _ in rows:
        if label is not None:
            dist[label] = dist.get(label, 0) + 1
    confs = [c for _, l, c in rows if l is not None]
    print(f"[INFO] 저장: {out} — {n_ok}/{len(rows)}개 라벨, 평균 확신도 {np.mean(confs):.3f} (>=0.5: {np.mean([c >= 0.5 for c in confs]):.3f}), "
          f"{time.time() - t0:.0f}s")
    print(f"[INFO] 클래스 분포: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
