"""
shared_vild/checkpoint_utils.py
Mark4.x / Mark5.0 공통 Checkpoint 저장/로드 유틸리티.

저장 스키마:
  schema_version    : int
  model_type        : str  ("teacher_encoder", "teacher_classifier", "student_full", ...)
  mark_version      : str  ("mark4.1", "mark5.0", ...)
  config_metadata   : dict (encoder_type, visual_view_type, num_input_channels, segment 설정 등)
  model_state_dict  : OrderedDict
  branch_state_dict : OrderedDict | None
  classifier_state_dict : OrderedDict | None
  background_state_dict : OrderedDict | None
  text_embeddings   : Tensor | None
"""

import os
import torch

SCHEMA_VERSION = 1

_CONFIG_KEYS = [
    "encoder_type",
    "visual_view_type",
    "num_input_channels",
    "embedding_dim",
    "segment_length",
    "segment_hop",
    "max_segments",
    "n_mels",
    "sample_rate",
    "fft_size",
    "hop_length",
    "anchor_centering",        # [F 2026-09-24] provenance 용. teacher ckpt 에는 없어 apply_config_metadata 가 건너뛴다.
    "learnable_logit_scale",
    "learning_rate",           # [LR 2026-09-25] provenance 용 — 학습률 선정 뒤 "이 파일은 어느 학습률로 만들었나" 를 파일 안에 남긴다.
    "encoder_learning_rate",   #   apply_config_metadata 는 이 둘을 건너뛴다(아래 _PROVENANCE_ONLY_KEYS).
    "use_feature_kd",          # [ablation 2026-09-28] provenance 용 — 이 student 파일이 어느 부품을 끄고 만들어졌나(A1·A2·A4).
    "pseudo_label_min_gap",    #   run_tag(_noFKD/_noGAP/_noDB)와 같은 정보를 파일 안에도 남긴다. 되돌려 적용하지는 않는다.
    "distill_branch_classify",
    "classification_policy",
    "joint_forward",
    "student_clip_input",      # [ablation A3 2026-09-29] provenance 용 — student 가 클립 3초 통째(True)로 학습됐나. run_tag _noSEG 와 같은 정보.
    "student_head",            # [헤드 진단 H 2026-09-29] provenance 용 — "text_anchor"(본 모델) / "linear"(nn.Linear 진단). run_tag _LIN 과 같은 정보.
    # [2026-09-29 밤 / 격차 분석 대응 묶음] provenance 용 6키 — 어느 묶음으로 만든 student 파일인지. run_tag 표식(_PH·_LP·_HW·_OR·_NEG)과 같은 정보.
    "head_pretrained_hidden",
    "lp_epochs",
    "hard_weight",
    "others_from_reject_tau",
    "neg_pl_tau",
    "neg_pl_beta",
    "teacher_clip_input",      #   student 가 클립 단위 teacher(Mark4.x clip_input) 로 학습됐나(EnsembleTeacher 가 .pth 8개에서 판정한 값).
    "clip_merge_rule",         # [2026-09-30 밤 / 가짜 정답 재료] provenance 용 — 조각 teacher 출력을 클립으로 합친 규칙("vote"/"mean"). run_tag _SM 과 같은 정보.
    "spec_augment_teacher_view",   #   teacher 가 본 입력("augmented"/"clean"). run_tag _CT 와 같은 정보.
    "background_embedding_weight", # [2026-10-01 / 지도만 격차 진단] provenance 용 — BG 보조 손실 가중치(run_tag _BG<값> 과 같은 정보).
    "student_wave_normalize",      #   student 클립 파형을 최대 진폭 1 로 정규화했나(False = 원본 진폭, run_tag _RW 와 같은 정보).
]

# [LR 2026-09-25] 기록만 하고 되돌려 적용하지 않는 키. 학습률은 "이 파일을 만들 때 쓴 값"이지 "이 파일을 읽는 쪽이 써야 할 값"이
# 아니다. train_mark5.py 의 EnsembleTeacher 가 teacher ckpt 8개를 apply_config_metadata 로 읽는데, 앞으로 teacher 가 이 키를 갖고
# 저장되면 student 의 학습률(과 run_tag)이 teacher 값으로 조용히 바뀐다. 2026-09-25 실측: 지금 teacher ckpt 에는 이 키가 없다.
# [ablation 2026-09-28] 절제 플래그 세 개도 같은 이유로 기록만 한다 — teacher(Mark4.x) 파일이 이 키를 가져도 student 설정을 바꾸지 않는다.
# [2026-09-29 밤] Mark4.x teacher 가 새로 기록하는 clip_input·clip_frames 도 여기 — student config 에 그대로 흘러들면 안 된다(EnsembleTeacher 가 따로 읽는다).
_PROVENANCE_ONLY_KEYS = {"learning_rate", "encoder_learning_rate", "use_feature_kd", "pseudo_label_min_gap", "distill_branch_classify", "classification_policy",
                         "student_clip_input", "student_head", "head_pretrained_hidden", "lp_epochs", "hard_weight",
                         "others_from_reject_tau", "neg_pl_tau", "neg_pl_beta", "teacher_clip_input", "clip_input", "clip_frames",
                         "clip_merge_rule", "spec_augment_teacher_view",   # [2026-09-30 밤 / 가짜 정답 재료] 기록만(teacher 파일에 있어도 student 설정을 안 바꾼다)
                         "background_embedding_weight", "student_wave_normalize",   # [2026-10-01 / 지도만 격차 진단] 기록만
                         "joint_forward"}


def extract_config_metadata(config) -> dict:
    meta = {}
    for key in _CONFIG_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            meta[key] = val
    return meta


def apply_config_metadata(config, ckpt: dict):
    """체크포인트에 저장된 config 값을 config 객체에 되돌려 적용한다.

    [수정 2026-08-17] setter 가 없는 @property 는 건너뛴다.
    num_input_channels 는 AudioViLDConfig 에서 visual_view_type 으로부터 계산되는
    읽기 전용 property 인데, extract_config_metadata 가 _CONFIG_KEYS 에 그것을 포함해
    저장해 두기 때문에 setattr 에서 AttributeError 로 죽었다
    ("property 'num_input_channels' ... has no setter").
    이 함수는 mark5 의 EnsembleTeacher 에서만 호출되므로 mark4.x 에서는 드러나지 않았고,
    mark5.0 학습을 처음 돌린 2026-08-17 에 teacher 8개를 로드하는 첫 줄에서 터졌다.

    파생값을 건너뛰어도 잃는 정보는 없다. 계산의 근거인 visual_view_type 이 같은
    metadata 에 들어 있어 함께 적용되고, 그러면 property 가 같은 값을 다시 계산한다
    (teacher 8개 전부 visual_view_type='mel_delta' / num_input_channels=3 으로 확인).
    """
    meta = ckpt.get("config_metadata", {})
    for key, val in meta.items():
        if key in _PROVENANCE_ONLY_KEYS:      # [LR 2026-09-25] 학습률은 기록용 — 읽는 쪽 config 에 넣지 않는다
            continue
        if not hasattr(config, key):
            continue
        class_attr = getattr(type(config), key, None)
        if isinstance(class_attr, property) and class_attr.fset is None:
            continue
        setattr(config, key, val)


def save_checkpoint(
    path: str,
    model_type: str,
    mark_version: str,
    model_state=None,
    branch_state=None,
    classifier_state=None,
    background_state=None,
    text_embeddings=None,
    config=None,
):
    """Save the student encoder, feature adapter, and single classifier."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "mark_version": mark_version,
        "config_metadata": extract_config_metadata(config) if config is not None else {},
        "model_state_dict": model_state,
        "branch_state_dict": branch_state,
        "classifier_state_dict": classifier_state,
        "background_state_dict": background_state,
        "text_embeddings": text_embeddings,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[checkpoint_utils] Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        return {"model_state_dict": ckpt}
    return ckpt


def resolve_state_dict(ckpt: dict, *keys):
    for key in keys:
        if key in ckpt and ckpt[key] is not None:
            return ckpt[key]
    raise KeyError(
        f"[checkpoint_utils] None of the keys {keys} found in checkpoint. "
        f"Available keys: {list(ckpt.keys())}"
    )
