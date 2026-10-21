import os
import sys
import csv
import argparse
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import build_audio_encoder, build_student_head
from vild_head import DualBranchStudentHead
from vild_parser_teacher import AudioParser
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import load_checkpoint, resolve_state_dict
from experiment_identity import dataset_identity


def _resolve_dataset_csv_path(split: str):
    # [2026-09-17] split 은 test·val 또는 eval_<세트>. 파일 이름은 늘 dataset_<split>.csv 다.
    """[변경 2026-08-17] test 고정이었던 것을 split 인자로 일반화했다.

    mark4.x 는 eval.py --split val 로 val 을 따로 평가해 운영점(threshold)을 고르고
    test 는 마지막에 한 번만 썼는데, mark5 에는 그 경로가 없어서 val 을 평가할 수단이
    아예 없었다(dataset_test.csv 만 읽었음).
    """
    filename = f"dataset_{split}.csv"
    candidates = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "preprocessing", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] {split} 인덱스 파일을 찾을 수 없습니다: {candidates}")


def evaluate(mark_version: str, split: str = "test", seed_value=None, unlabeled_n=None, ceiling=False,
             others_threshold=None, labeled_n=None, spill_labeled=False, pseudo_source=None, lr=None, encoder_lr=None,
             no_feature_kd=False, pl_min_gap=None, no_distill_branch=False, clip_input=False, linear_head=False,
             segment_input=False, head_pretrained_hidden=False, lp_epochs=None, hard_weight=None,
             others_from_reject=None, neg_pl=None, teacher_seg_mean=False, teacher_clean_view=False,
             bg_weight=None, student_raw_wave=False, joint_forward=False, plain_output=True):
    """[확장 2026-08-30 / Mark5_EXP] seed_value·unlabeled_n 을 받는다.
    [이식 2026-09-29 / Mark5.2] ablation(A1~A4·H)·격차 대응 묶음(PH·LP·HW·OR·NEG) 플래그도 받는다 — 학습 때와 같은 값을 줘야
    run_tag 가 맞고, PH·H·clip/segment 는 헤드·입력 구조까지 바꾸므로 이름 맞추기 이상의 뜻이 있다.

    eval 은 학습을 하지 않지만 **run_tag 로 체크포인트를 찾기 때문에** 학습과 같은 조건을
    넘겨줘야 한다. 예를 들어 --seed 3 --unlabeled_n 500 으로 학습했다면 체크포인트가
    student_model_mark5.0_PL_SA_u500_s3.pth 이므로, eval 에도 같은 인자를 줘야 그 파일을 읽는다.
    안 주면 예전처럼 태그 없는 이름을 찾는다(기존 동작 보존).
    """
    if not plain_output:
        raise ValueError("This model uses supervised argmax only; calibrated output is unsupported.")
    if plain_output and others_threshold is not None:
        raise ValueError("--plain_output 은 지도 갈래의 9-class argmax 를 사용하므로 --others_threshold 와 함께 쓸 수 없습니다.")
    config = AudioViLDConfig(mark_version=mark_version)
    config.experiment_seed = seed_value
    # [신설 2026-09-06] 천장 조건은 태그가 _ceil 이라 --ceiling 을 같이 줘야 그 체크포인트를 찾는다.
    if ceiling and unlabeled_n not in (None, 0):
        # [리뷰 반영] 학습 쪽(train_mark5.py)과 같은 규칙 — 조용히 한쪽을 버리지 않는다.
        raise ValueError(f"[ERROR] --ceiling 과 --unlabeled_n={unlabeled_n} 을 함께 줄 수 없습니다(천장은 unlabeled 를 쓰지 않음).")
    config.ceiling = bool(ceiling)
    config.unlabeled_n = None if ceiling else unlabeled_n
    config.labeled_n = labeled_n       # [신설 2026-09-17 / 8절] 학습 때 --labeled_n 을 줬으면 같은 값을 줘야 _l<n> 체크포인트를 찾는다
    config.spill_labeled = bool(spill_labeled)   # _sp
    config.pseudo_source = pseudo_source         # _ps<이름>
    # [신설 2026-09-25 / LR] 학습 때 준 --lr/--encoder_lr 과 같은 값을 줘야 run_tag(_lr/_elr)가 맞아 그 체크포인트를 찾는다.
    # 평가는 학습률을 쓰지 않는다 — 오직 파일 이름 때문이다.
    _lr_txt = config.apply_cli_learning_rates(lr, encoder_lr)
    # [신설 2026-09-28 / ablation A1·A2·A4] 학습 때 준 것과 같은 플래그를 줘야 run_tag(_noFKD/_noGAP/_noDB)가 맞아 그 체크포인트를 찾는다.
    # A1·A2 는 평가에서 이름 말고는 쓰이지 않는다. A4 는 증류 갈래 가중치 w 후보를 (0,) 으로 좁혀 지도 헤드만으로 판정한다.
    # [신설 2026-09-29 / A3] clip_input 은 이름 맞추기(_noSEG)에 더해 평가 입력도 바꾼다 — 클립마다 조각 5개 대신 3초 파형 한 번(아래 루프).
    # [신설 2026-09-29 / H] linear_head 는 이름 맞추기(_LIN)에 더해 헤드 구조도 바꾼다(build_student_head) — 학습 때와 같아야 classifier_state_dict 가 맞는다.
    # [2026-09-29 밤 / 묶음] PH 는 헤드 구조(투영·branch)까지 바꾸므로 학습과 같이 줘야 state_dict 가 맞는다. LP·HW·OR·NEG 는 이름 맞추기만.
    _ab_txt = config.apply_cli_ablation(no_feature_kd=no_feature_kd, pl_min_gap=pl_min_gap, no_distill_branch=no_distill_branch,
                                        clip_input=clip_input, linear_head=linear_head, segment_input=segment_input,
                                        head_pretrained_hidden=head_pretrained_hidden, lp_epochs=lp_epochs, hard_weight=hard_weight,
                                        others_from_reject=others_from_reject, neg_pl=neg_pl,
                                        teacher_seg_mean=teacher_seg_mean, teacher_clean_view=teacher_clean_view,
                                        bg_weight=bg_weight, student_raw_wave=student_raw_wave,
                                        joint_forward=joint_forward)   # [2026-10-08 / JF] 평가에선 이름(_JF) 맞추기만
    if _lr_txt:
        print(f"[LR] 명령줄 학습률: {_lr_txt} (체크포인트 이름 맞추기용) · run_tag {config.run_tag}")
    if _ab_txt:
        print(f"[ABLATION] {_ab_txt} · run_tag {config.run_tag}")
    _clip_mode = bool(getattr(config, "student_clip_input", False))
    if plain_output:
        if not _clip_mode:
            raise ValueError("--plain_output 은 클립 입력 모델에만 적용합니다(세그먼트 집계 정책을 바꾸지 않음).")
        # 학습과 같은 지도 갈래 분류기를 사용하고 평가 보정을 적용하지 않는다.
        config.distill_branch_eval_weight = 0.0
        config.distill_eval_weight_candidates = (0.0,)
        config.class_bias_grid = None
        config.class_pair_margin_overrides = {}
        config.others_decision_mode = "argmax"
        config.others_threshold_mode = "fixed"
        config.enable_abstention = False
        config.use_background_override_at_eval = False
        config.save_visual_explanations = False
        print("[OUTPUT] 단일 분류 경로: 지도 갈래의 9-class argmax · 증류 변환층 미실행 · 평가 보정 없음")
    if _clip_mode:
        print(f"[A3] 평가 단위 = 클립(파형 {getattr(config, 'student_clip_samples', 48000)}샘플 통째, 조각 1개 취급 → 조각 가중 평균은 가중치 1)")
    device = config.device
    parser = AudioParser(config, segment_mode=True)

    # [추가 2026-08-17] split 이 test 가 아니면 결과 파일명에 _{split} 을 붙여
    # test 결과를 덮어쓰지 않게 한다(mark4.x eval.py 의 out_tag 규칙과 동일).
    out_tag = config.run_tag if split == "test" else f"{config.run_tag}_{split}"
    if plain_output:
        out_tag += "_plain"

    csv_path = _resolve_dataset_csv_path(split)
    print(f"[INFO] 평가 대상: split={split} · 인덱스={csv_path}")
    test_files = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    model_path = os.path.join(BASE_DIR, f"student_model_{config.run_tag}.pth")
    checkpoint = load_checkpoint(model_path, map_location=device)
    if (checkpoint.get("config_metadata") or {}).get("classification_policy") != "supervised_argmax_v1":
        raise ValueError("Checkpoint was not trained with this single-classifier implementation.")
    # [신설 2026-09-07] 체크포인트 지문. val 이 고른 임계값 json 에 적어 두고 test 가 대조한다
    # (재학습 뒤 옛 json 을 조용히 읽는 사고 방지 — code-reviewer 지적 3).
    with open(model_path, "rb") as _f:
        _ckpt_sha = hashlib.sha256(_f.read()).hexdigest()
    ckpt_fingerprint = {"file": os.path.basename(model_path), "size": os.path.getsize(model_path),
                        "mtime": os.path.getmtime(model_path), "sha256": _ckpt_sha}

    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(checkpoint, "model_state_dict", "encoder_state_dict", "model"))
    # [신설 2026-09-17] student 입력 종류(config.student_input). 체크포인트에 encoder_type 이 적혀 있으면 config 와 대조한다.
    _student_input = getattr(config, "student_input", "mel")
    _ckpt_enc = (checkpoint.get("config_metadata") or {}).get("encoder_type")
    if _ckpt_enc is not None and _ckpt_enc != getattr(config, "encoder_type", "cnn"):
        raise RuntimeError(f"[ERROR] 체크포인트의 encoder_type={_ckpt_enc!r} 과 config 의 {getattr(config, 'encoder_type', 'cnn')!r} 이 다릅니다.")
    print(f"[INFO] student 인코더 {type(encoder).__name__} · 입력 {_student_input}")
    # [변경 2026-08-22] 학습 때와 같은 구조로 만들어야 branch state_dict 가 맞는다.
    branch_head = DualBranchStudentHead(
        config.embedding_dim,
        distill_final_relu=getattr(config, "distill_branch_final_relu", False),
        supervised_identity=bool(getattr(config, "head_pretrained_hidden", False)),   # [PH 2026-09-29 밤] 학습과 같은 구조
    ).to(device)
    # [수정 2026-08-17] resolve_state_dict 는 아는 키가 하나도 없으면 KeyError 를 던지므로,
    # 아래 else 의 경고는 도달할 수 없는 코드였다(구버전 .pth 를 넣으면 경고 대신 죽었다).
    # 직접 get 으로 받아 경고 분기가 실제로 살아나게 한다.
    branch_state = (
        checkpoint.get("branch_state_dict")
        or checkpoint.get("head_state_dict")
        or checkpoint.get("head")
    )
    if branch_state is not None:
        branch_head.load_state_dict(branch_state, strict=True)
    else:
        raise ValueError("Checkpoint has no trained branch_state_dict.")
    text_head = build_student_head(config).to(device)   # [H 2026-09-29] config.student_head 로 텍스트 앵커/선형 선택(학습과 같은 함수)
    # [이식 2026-09-26 / Mark5.2] classifier_state_dict 를 헤드에 로드한다. 학습형 로짓 스케일(LT)을 켠 뒤로는
    # 이 줄이 없으면 학습된 온도가 평가에서 조용히 무시된다.
    _cls_state = checkpoint.get("classifier_state_dict")
    if getattr(text_head, "learnable", False):
        if not _cls_state or "logit_scale" not in _cls_state:
            raise RuntimeError("[ERROR] 체크포인트에 logit_scale 이 없습니다 — 학습형 로짓 스케일이 없던 "
                               "2026-09-24 이전 코드로 학습한 모델이거나 run_tag 가 어긋난 것입니다.")
        text_head.load_state_dict(_cls_state)
        print(f"[LT] 학습형 로짓 스케일 로드: T={text_head.temperature:.4f} (초기 {getattr(config, 'logit_temperature', 0.07)})")
    elif getattr(config, "student_head", "text_anchor") == "linear":
        # [신설 2026-09-29 / H] 선형 헤드는 가중치가 전부라 strict 로 읽는다 — 아래 strict=False 분기로 흘러가면 키가 어긋나도
        # 조용히 무작위 초기값으로 평가된다.
        if not _cls_state or "linear.weight" not in _cls_state:
            raise RuntimeError("[ERROR] 체크포인트에 선형 헤드(linear.weight)가 없습니다 — --linear_head 없이 학습한 모델이거나 "
                               "run_tag(_LIN)가 어긋난 것입니다.")
        text_head.load_state_dict(_cls_state)
        print(f"[H] 선형 헤드 로드: linear.weight {tuple(_cls_state['linear.weight'].shape)} — 텍스트 앵커·AC/LT 미사용(2026-10-07 D0 본 레시피)")
    elif _cls_state:
        text_head.load_state_dict(_cls_state, strict=False)
    # 선형 분류기는 텍스트 앵커를 사용하지 않는다(체크포인트에도 None).
    text_emb = (None if getattr(config, "student_head", "text_anchor") == "linear"
                else config.get_class_text_embeddings(for_evaluation=True).to(device))

    encoder.eval()
    branch_head.eval()
    text_head.eval()

    class_names = config.get_classes_for_evaluation()
    label_map = config.get_target_label_map()
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_labels, all_preds, all_probs = [], [], []
    all_raw_probs = []
    all_prebias_probs = []   # [2026-09-22 / A] 편향 전 확률 — ROC AUC·ROC 곡선은 이것으로(운영점·편향과 무관한 모델 자체의 순위)
    calibration_rows = []
    prediction_rows = []   # [추가 2026-08-17] 샘플별 확률까지 남기는 행(운영점 재계산용)
    clip_records = []      # [변경 2026-09-07] 1차 통과: 클립마다 others 보정 전 확률만 모은다(임계값은 그 뒤에 정한다)
    skipped_label_counter = {}
    for row in test_files:
        path = row["path"]
        label = row["label"]
        if label not in label_map:
            # [추가] 9-class 밖 라벨은 조용히 넘기지 않고 집계
            skipped_label_counter[label] = skipped_label_counter.get(label, 0) + 1
            continue
        if _clip_mode:
            # [A3] 클립 전체를 조각 1개짜리 기록으로 다룬다: wave = 첫 48,000샘플(학습과 같은 자르기), tensor = 클립 mel 뷰(그림용).
            _out = parser.load_clip_with_segments(path)
            segment_records = [_out["clip_record"]] if _out is not None else []
        else:
            segment_records = parser.load_and_segment_with_metadata(path)
        if not segment_records:
            continue

        segment_probs = []
        with torch.no_grad():
            for record in segment_records:
                seg = record["tensor"]
                if seg.ndim == 3:
                    seg = seg.unsqueeze(0)
                seg = seg.to(device)
                # [변경 2026-09-17 / Mark5.2 6절] student 가 mn10(파형 입력)이면 같은 창의 파형(record["wave"])을 넣는다.
                if _student_input == "waveform":
                    base_features = encoder(record["wave"].unsqueeze(0).to(device))
                else:
                    base_features = encoder(seg)
                supervised_features = branch_head.forward_supervised(base_features)
                sup_logits = text_head(supervised_features, text_emb)
                # Only this classifier contributes to inference.
                segment_probs.append(torch.softmax(sup_logits, dim=-1).squeeze(0).cpu().numpy())

        # Clip input has exactly one prediction; no temporal or class calibration.
        if len(segment_probs) != 1:
            raise ValueError("Single-classifier evaluation requires one full-clip prediction.")
        calibrated_prob = segment_probs[0]
        clip_records.append({
            "path": path, "label": label, "raw_prob": calibrated_prob.copy(),
        })

    # [변경 2026-09-07] 라벨 경고와 "샘플 0개" 검사를 임계값 결정보다 앞에 둔다(리뷰 지적 21).
    if skipped_label_counter:
        print(
            f"[WARN] 9-class 밖 라벨로 건너뛴 {split} 파일: {skipped_label_counter} "
            f"(허용 클래스: {sorted(label_map.keys())})"
        )
    if len(clip_records) == 0:
        raise ValueError(
            f"[ERROR] 평가 가능한 {split} 샘플이 0개입니다. 건너뛴 라벨: {skipped_label_counter}. "
            f"dataset_{split}.csv 라벨이 9-class와 일치하는지 확인하세요."
        )

    for rec in clip_records:
        rec["raw_prob_prebias"] = rec["raw_prob"].copy()
    thr_info = {"mode": "plain_output", "rule": "argmax", "threshold": None,
                "distill_eval_weight": 0.0, "class_bias": {}, "checkpoint": ckpt_fingerprint,
                "classification_policy": "supervised_argmax_v1",
                "training_classification_policy": (checkpoint.get("config_metadata") or {}).get("classification_policy"),
                "run_tag": config.run_tag, "split": split,
                "dataset_identity": dataset_identity(csv_path)}
    if plain_output:
        with open(os.path.join(plot_dir, f"output_policy_{out_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(thr_info, f, ensure_ascii=False, indent=2)

    # 2차 통과: 정해진 임계값으로 판정하고, 기존과 같은 행을 만든다.
    for rec in clip_records:
        path, label = rec["path"], rec["label"]
        raw_prob = rec["raw_prob"]
        raw_pred = int(np.argmax(raw_prob))
        calibrated_prob, pred = raw_prob.copy(), raw_pred
        ordered = np.sort(raw_prob)
        calib_meta = {
            "forced_to_others": False,
            "raw_top_conf": float(raw_prob[raw_pred]),
            "raw_margin": float(ordered[-1] - ordered[-2]),
            "entropy": float(-(raw_prob * np.log(np.clip(raw_prob, 1e-8, 1.0))).sum() / np.log(len(class_names))),
            "prob_others": float(raw_prob[class_names.index("others")]),
        }
        abstained = False
        all_labels.append(label_map[label])
        all_preds.append(pred)
        all_probs.append(calibrated_prob)
        all_raw_probs.append(raw_prob)      # [변경 2026-09-07] ROC AUC 는 규칙 적용 전 확률로 잰다(리뷰 지적 7)
        all_prebias_probs.append(np.asarray(rec.get("raw_prob_prebias", raw_prob)))   # [2026-09-22 / A] 편향 전
        prediction_rows.append(
            {
                "Filename": os.path.basename(path),
                "True Label": label,
                "Raw Predicted Label": class_names[raw_pred],
                "Predicted Label": class_names[pred],
                "Forced To Others": calib_meta["forced_to_others"],
                "Abstained": abstained,
                "Raw Top Confidence": calib_meta["raw_top_conf"],
                "Raw Margin": calib_meta["raw_margin"],
                "Entropy": calib_meta["entropy"],
                # [추가 2026-08-22] direct 모드가 실제로 보는 값. 스윕을 다시 돌릴 때 쓴다.
                "Prob Others (raw)": calib_meta.get("prob_others", float("nan")),
                **{f"Prob_{n}": float(raw_prob[i]) for i, n in enumerate(class_names)},
                # [추가 2026-09-21 밤 / A] 편향 전 확률(w 만 적용). 편향을 안 골랐으면 Prob_* 와 같다.
                **{f"PreBias_Prob_{n}": float(rec.get("raw_prob_prebias", raw_prob)[i]) for i, n in enumerate(class_names)},
            }
        )
        calibration_rows.append({
            "path": path,
            "true_label": label,
            "pred_label": class_names[pred],
            "forced_to_others": calib_meta["forced_to_others"],
            "abstained": abstained,
            "raw_top_conf": calib_meta["raw_top_conf"],
            "raw_margin": calib_meta["raw_margin"],
            "entropy": calib_meta["entropy"],
        })

    print(f"[INFO] 유효 {split} 샘플: {len(all_labels)}개")

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {out_tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{out_tag}.png"))
    cm_df.to_csv(os.path.join(plot_dir, f"confusion_matrix_{out_tag}.csv"), encoding="utf-8")   # [2026-09-17 / 7절] 혼동 칸을 숫자로(행 = 정답, 열 = 판정)
    plt.close()

    # [추가] mark4 eval.py와 동일한 형태로 Accuracy/P/R/F1/ROC AUC를 CSV로 저장(기존엔 콘솔 print만 하고 파일에 안 남았음).
    all_probs_arr = np.array(all_prebias_probs)   # [변경 2026-09-07] 규칙 적용 전 확률(운영점과 무관한 AUC) · [2026-09-22] 편향 전 확률
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    # [신설 2026-08-30 / Mark5_EXP] macro 집계값. 2026-08-24 랩미팅에서 교수님이
    # "Precision 과 Recall 이 표에서 빠지면 안 된다"고 하셨다.
    #
    # 클래스가 9개라 P/R/F1 도 9개씩 나오는데, 논문 표에는 한 칸씩만 들어가므로 합쳐야 한다.
    # macro = 9개를 그냥 평균(모든 클래스를 동등하게 봄). weighted 는 샘플 많은 클래스가
    # 점수를 지배하고, micro 는 다중분류에서 Accuracy 와 같은 값이라 적을 이유가 없다.
    # 우리 val/test 는 48/48/.../47 로 거의 균형이라 macro 가 맞고, 지금까지 써온 것도 macro 다
    # (2026-08-23 확정한 test macro F1 0.8744 가 이 값이다).
    #
    # 기존에는 이 세 값이 콘솔 classification_report 에만 찍히고 CSV 에 안 남아서,
    # 25회분 결과를 자동으로 모아 평균·표준편차를 낼 수 없었다.
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", labels=list(range(len(class_names))), zero_division=0
    )
    # [수정] labels 명시 + try/except로 방어. 테스트셋에 9-class 중 일부가 아예 없으면
    # (1) labels 없이는 "열 개수 불일치"로 죽고, (2) labels를 줘도 그 클래스의 OvR 이진화가
    # "양성 샘플 0개"가 되어 AUC 자체가 수학적으로 정의 불가능해 여전히 ValueError가 난다.
    # 작은 test set에서는 흔히 벌어질 수 있는 상황이라, 죽이지 않고 NaN 처리 후 경고만 남긴다.
    try:
        roc_auc_macro = roc_auc_score(
            all_labels, all_probs_arr, multi_class="ovr", average="macro",
            labels=list(range(len(class_names))),
        )
        print(f"ROC AUC (macro, one-vs-rest): {roc_auc_macro:.4f}")
    except ValueError as e:
        roc_auc_macro = float("nan")
        print(f"[WARN] ROC AUC 계산 불가(테스트셋에 없는 클래스가 있을 수 있음): {e}")

    # [추가] others FPR: 실제 라벨이 "others"인데 8개 타겟 클래스 중 하나로 오분류된 비율.
    # = 1 - Recall(others). "조용한 배경음을 소음으로 잘못 경보하는 비율"이라는 실용적 의미를 명시하기 위해 별도 계산.
    others_idx = class_names.index("others")
    others_row = cm[others_idx]
    others_row_sum = int(others_row.sum())
    others_fpr = float((others_row_sum - others_row[others_idx]) / others_row_sum) if others_row_sum > 0 else float("nan")
    print(f"[others FPR] 실제 others인데 타겟 클래스로 오분류된 비율: {others_fpr:.4f}")

    # [추가] mark4 eval.py의 roc_curve 저장과 동일한 형태(9-class는 one-vs-rest로 클래스별 곡선).
    plt.figure(figsize=(7, 6))
    all_labels_arr = np.array(all_labels)
    for i, cname in enumerate(class_names):
        fpr_curve, tpr_curve, _ = roc_curve((all_labels_arr == i).astype(int), all_probs_arr[:, i])
        auc_i = auc(fpr_curve, tpr_curve)
        plt.plot(fpr_curve, tpr_curve, label=f"{cname} AUC={auc_i:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({out_tag})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"roc_curve_{out_tag}.png"))
    plt.close()

    summary_csv = os.path.join(plot_dir, f"performance_summary_{out_tag}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Performance Summary for {out_tag}\n\n")
        # [변경 2026-08-30 / Mark5_EXP] Macro Precision·Recall·F1 세 줄 추가.
        # 논문 표 1 이 Accuracy · Macro P · Macro R · Macro F1 · ROC AUC 다섯 칸이고,
        # 25회분을 집계 스크립트가 이 CSV 에서 긁어간다. 행 이름을 바꾸면 집계가 깨진다.
        # [추가 2026-09-07] 쓰인 others 임계값도 남긴다(aggregate_results 는 METRICS 만 읽으므로 무해).
        pd.DataFrame({
            "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                       "ROC AUC (Macro, OvR)", "Others FPR", "Others Threshold", "Others Rule"],
            "Score": [accuracy, macro_precision, macro_recall, macro_f1,
                      roc_auc_macro, others_fpr,
                      float(config.others_direct_threshold) if config.others_decision_mode == "direct" else float("nan"),
                      str(config.others_decision_mode)],
        }).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({
            "Class": class_names, "Precision": precision, "Recall": recall, "F1-Score": f1,
        }).to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장: {summary_csv}")

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            os.path.join(plot_dir, f"calibration_details_{out_tag}.csv"),
            index=False,
            encoding="utf-8",
        )

    # [추가 2026-08-17] 샘플별 예측 CSV. 클래스별 확률(others 보정 전)을 담고 있어
    # 이 파일 하나로 운영점(others_confidence/margin/entropy)을 바꿔가며 재계산할 수 있다.
    if prediction_rows:
        pred_csv = os.path.join(plot_dir, f"prediction_details_{out_tag}.csv")
        pd.DataFrame(prediction_rows).to_csv(pred_csv, index=False, encoding="utf-8")
        print(f"[INFO] 예측 상세 CSV 저장: {pred_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델의 성능을 평가합니다.")
    parser.add_argument("--mark_version", type=str, default="mark5.2")   # [변경 2026-09-17] 기본 mark5.2
    # [추가 2026-08-17] 평가 split 선택. 기본은 test(기존 동작 그대로).
    # 운영점은 val 로 고르고 test 는 마지막 확인 한 번만 쓴다(mark4.x 와 같은 원칙).
    # [변경 2026-09-17 / 10절] test·val 외에 평가 전용 세트 이름도 받는다: eval_far_heldout · eval_boundary_B (dataset_eval_*.csv,
    # preprocessing/generate_eval_index.py 가 만든다). 정답이 전부 others 인 세트라 결과의 "Others FPR" = 1 − open-set ①
    # (학습에 없던 소리를 others 로 보낸 비율). 운영점은 val 에서 고른 json 을 그대로 읽는다(test 와 같은 규칙).
    parser.add_argument("--split", type=str, default="test",
                        help="test | val | eval_far_heldout | eval_boundary_B (dataset_<split>.csv 를 읽는다)")
    # [신설 2026-08-30 / Mark5_EXP] 학습 때 준 것과 **같은 값**을 줘야 그 조건의
    # 체크포인트를 찾는다(run_tag 가 파일명을 가르기 때문).
    parser.add_argument("--seed", type=int, default=None,
                        help="학습 때 쓴 seed. 체크포인트 이름의 _s<k> 를 맞추는 용도.")
    parser.add_argument("--unlabeled_n", type=int, default=None,
                        help="학습 때 쓴 unlabeled 개수. 체크포인트 이름의 _u<n> 을 맞추는 용도.")
    parser.add_argument("--labeled_n", type=int, default=None,
                        help="학습 때 쓴 labeled 개수. 체크포인트 이름의 _l<n> 을 맞추는 용도.")
    parser.add_argument("--spill_labeled", action="store_true", help="학습 때 --spill_labeled 를 줬으면 같이(_sp 태그).")
    parser.add_argument("--pseudo_source", type=str, default=None, help="학습 때 --pseudo_source 를 줬으면 같이(_ps 태그).")
    parser.add_argument("--ceiling", action="store_true",
                        help="천장 조건으로 학습한 체크포인트(_ceil)를 평가할 때 준다.")
    parser.add_argument("--lr", type=float, default=None, help="[LR 2026-09-25] 학습 때 준 --lr 과 같은 값(run_tag _lr<값>)")
    parser.add_argument("--encoder_lr", type=float, default=None, help="[LR 2026-09-25] 학습 때 준 --encoder_lr 과 같은 값(run_tag _elr<값>)")
    # [신설 2026-09-07 / Mark5_EXP2] 기본은 val 에서 고른 임계값(config.others_threshold_mode=val_select).
    # 이 값을 주면 그 값으로 고정해 평가한다(옛 결과 재현·수동 점검용). json 은 건드리지 않는다.
    parser.add_argument("--others_threshold", type=str, default=None,
                        help="others 규칙을 고정: 실수(direct 임계값) 또는 'argmax'. 주지 않으면 val 에서 고른 값 사용")
    parser.add_argument("--plain_output", action="store_true", default=True,
                        help="클립 모델의 지도 갈래 9-class argmax. OVA·갈래 혼합·편향·임계값 선택 없이 평가하고 별도 _plain 결과에 저장(학습은 그대로)")
    # [신설 2026-09-28 / ablation A1·A2·A4] 학습(train_mark5.py) 때 준 것과 같은 플래그를 준다(run_tag 맞추기 + A4 는 w=0 고정).
    parser.add_argument("--no_feature_kd", action="store_true", help="[A1] 학습 때 --no_feature_kd 를 줬으면 같이(run_tag _noFKD)")
    parser.add_argument("--pl_min_gap", type=float, default=None, help="[A2] 학습 때 준 --pl_min_gap 과 같은 값(0 이면 run_tag _noGAP)")
    parser.add_argument("--no_distill_branch", action="store_true",
                        help="[A4] 학습 때 --no_distill_branch 를 줬으면 같이(run_tag _noDB · 증류 갈래 가중치 w=0 고정)")
    parser.add_argument("--clip_input", action="store_true",
                        help="[A3 2026-09-29] 학습 때 --clip_input 을 줬으면 같이(run_tag _noSEG · 평가 입력도 클립 3초 파형 통째)")
    parser.add_argument("--linear_head", action="store_true",
                        help="[H 2026-09-29] 학습 때 --linear_head 를 줬으면 같이(run_tag _LIN · 헤드도 nn.Linear 로 만들어 체크포인트를 읽는다)")
    # [2026-09-29 밤 / 묶음] 학습 때 준 것과 같은 플래그(run_tag 맞추기. PH 는 헤드 구조도 같이 바꿈).
    parser.add_argument("--segment_input", action="store_true", help="학습 때 --segment_input 을 줬으면 같이(조각 입력·이름 표식 없음)")
    parser.add_argument("--head_pretrained_hidden", action="store_true", help="[PH] 학습 때 줬으면 같이(run_tag _PH · 투영/branch 구조)")
    parser.add_argument("--lp_epochs", type=int, default=None, help="[LP-FT] 학습 때 준 값(run_tag _LP<N>)")
    parser.add_argument("--hard_weight", type=float, default=None, help="[HW] 학습 때 준 값(run_tag _HW<g>)")
    parser.add_argument("--others_from_reject", type=float, default=None, help="[OR] 학습 때 준 값(run_tag _OR<τ>)")
    parser.add_argument("--neg_pl", type=float, nargs=2, default=None, metavar=("TAU", "BETA"), help="[NEG] 학습 때 준 값(run_tag _NEG)")
    # [2026-09-30 밤 / 가짜 정답 재료] 가짜 정답 재료 플래그 — 평가에서는 이름(run_tag _SM·_CT) 맞추기에만 쓰인다.
    parser.add_argument("--teacher_seg_mean", action="store_true", help="[SM] 학습 때 줬으면 같이(run_tag _SM)")
    parser.add_argument("--teacher_clean_view", action="store_true", help="[CT] 학습 때 줬으면 같이(run_tag _CT)")
    # [2026-10-01 / 지도만 격차 진단] --bg_weight 는 이름(run_tag _BG<값>) 맞추기만, --student_raw_wave 는 이름(_RW)에 더해 평가 입력도 바꾼다
    # (파서가 student 클립 파형을 원본 진폭으로 준다 — 학습 때와 같아야 한다).
    parser.add_argument("--bg_weight", type=float, default=None, help="[BG] 학습 때 준 값(run_tag _BG<값>)")
    parser.add_argument("--student_raw_wave", action="store_true", help="[RW] 학습 때 줬으면 같이(run_tag _RW · 평가 입력도 원본 진폭)")
    parser.add_argument("--joint_forward", action="store_true", help="[Mark5.3 JF] 학습 때 줬으면 같이(run_tag _JF, 평가 동작은 같다)")
    args = parser.parse_args()
    # [신설 2026-09-26] logFiles/<run_tag>__eval_<split>.log 에도 남긴다.
    from run_logger import RunLogger
    _c = AudioViLDConfig(mark_version=args.mark_version)
    _c.experiment_seed = args.seed
    _c.ceiling = bool(args.ceiling)
    _c.unlabeled_n = None if args.ceiling else args.unlabeled_n
    _c.labeled_n = args.labeled_n
    _c.spill_labeled = bool(args.spill_labeled)
    _c.pseudo_source = args.pseudo_source
    _c.apply_cli_learning_rates(args.lr, args.encoder_lr)
    _bundle = dict(segment_input=args.segment_input, head_pretrained_hidden=args.head_pretrained_hidden, lp_epochs=args.lp_epochs,
                   hard_weight=args.hard_weight, others_from_reject=args.others_from_reject, neg_pl=args.neg_pl,
                   teacher_seg_mean=args.teacher_seg_mean, teacher_clean_view=args.teacher_clean_view,
                   bg_weight=args.bg_weight, student_raw_wave=args.student_raw_wave,
                   joint_forward=args.joint_forward)
    _c.apply_cli_ablation(no_feature_kd=args.no_feature_kd, pl_min_gap=args.pl_min_gap, no_distill_branch=args.no_distill_branch,
                          clip_input=args.clip_input, linear_head=args.linear_head, **_bundle)
    with RunLogger(PROJECT_ROOT, _c.run_tag, f"eval_{args.split}" + ("_plain" if args.plain_output else "")):
        evaluate(mark_version=args.mark_version, split=args.split,
                 seed_value=args.seed, unlabeled_n=args.unlabeled_n, ceiling=args.ceiling,
                 others_threshold=args.others_threshold, labeled_n=args.labeled_n,
                 spill_labeled=args.spill_labeled, pseudo_source=args.pseudo_source,
                 lr=args.lr, encoder_lr=args.encoder_lr,
                 no_feature_kd=args.no_feature_kd, pl_min_gap=args.pl_min_gap, no_distill_branch=args.no_distill_branch,
                 clip_input=args.clip_input, linear_head=args.linear_head, plain_output=args.plain_output, **_bundle)
