import os
import sys
import csv
import argparse
import hashlib
import random          # [추가 2026-08-30 / Mark5_EXP] unlabeled 누적 서브샘플링용 별도 난수기
import copy            # [신설 2026-10-07 / Mark5.3 FM] 학생 EMA 사본
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from feature_cache import get_feature_cache_path, get_metadata_cache_path
from vild_config import AudioViLDConfig
from vild_model import (
    ExpertBinaryHead,      # [신설 2026-10-07 / Mark5.3 OVA]
    LearnableBackgroundEmbedding,
    ViLDTextHead,
    build_audio_encoder,
    build_student_head,
    build_teacher_encoder,
)
from vild_head import DualBranchStudentHead
from vild_parser_teacher import AudioParser
from vild_parser_common import _normalize_visual_tensor   # [2026-09-29 밤] 클립 teacher 입력을 teacher_clip_frames 로 맞출 때
from teacher_fusion import WeightedTeacherFusion
from seed_utils import set_seed
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import apply_config_metadata, load_checkpoint, resolve_state_dict, save_checkpoint


def _resolve_csv_path(filename, required=True):
    """CSV 를 찾아 경로를 돌려준다.

    [변경 2026-08-09] required=False 를 추가했다. dataset_val.csv 는 있으면 쓰고 없으면
    옛 방식으로 폴백해야 해서, 없을 때 예외 대신 None 이 필요하다.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "preprocessing", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if not required:
        return None
    raise FileNotFoundError(f"[ERROR] CSV 파일을 찾을 수 없습니다: {candidates}")


def _resolve_resource_path(filename):
    candidates = [
        os.path.join(RESOURCES_DIR, filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] 리소스 파일을 찾을 수 없습니다: {candidates}")


def _align_teacher_arch(config, state_dict, class_name, ckpt_name):
    """[추가 2026-08-17] teacher .pth 안의 키 이름을 보고 어느 인코더 구조인지 판정해
    config.use_large_teacher_encoder 를 그것에 맞춘다.

    이렇게 하는 이유: 구조가 안 맞으면 load_state_dict 가 Missing/Unexpected 키 수십 줄만
    토하고 죽어서, 무엇이 문제인지 알아내는 데 시간이 걸린다(2026-08-17 실제로 그랬다).
    설정 플래그보다 파일에 든 실물이 사실이므로 파일 기준으로 맞춘다.

      conv_block1.*      -> SimpleAudioEncoder (student 인코더, 3블록·약 14.3만 파라미터.
                            2026-08-17 이전에는 2블록 4.5만이었다)
      model.3.0.weight   -> TeacherAudioEncoder (4블록, 약 49만 파라미터)
    """
    keys = set(state_dict.keys())
    if any(k.startswith("conv_block1.") for k in keys):
        detected_large = False
    elif "model.3.0.weight" in keys:
        detected_large = True
    else:
        raise RuntimeError(
            f"[ERROR] teacher 체크포인트의 인코더 구조를 판정할 수 없습니다: {ckpt_name} "
            f"(class={class_name}). 아는 구조는 SimpleAudioEncoder(conv_block1.*)와 "
            f"TeacherAudioEncoder(model.3.0.weight) 두 가지입니다. "
            f"실제 키 예시: {sorted(keys)[:5]}"
        )

    if getattr(config, "use_large_teacher_encoder", False) != detected_large:
        arch = "TeacherAudioEncoder(4블록)" if detected_large else "SimpleAudioEncoder(student 인코더)"
        print(
            f"[INFO] teacher 인코더 구조를 체크포인트에 맞춰 조정합니다 — "
            f"{class_name}: {arch} ({ckpt_name})"
        )
        config.use_large_teacher_encoder = detected_large
    return detected_large


# [변경 2026-09-05] teacher 8개 목록은 model/make_consensus_labels.py(비교모델용 합의 라벨 생성기)와 한 곳에서 공유해야 한다
# (목록이 둘로 갈리면 다른 teacher 조합으로 라벨이 만들어져도 아무도 모른다 — code-reviewer 지적).
# [이식 2026-09-29 / Mark5.2] 그 한 곳이 아래 specialist_config_for(config) 함수다(옛 모듈 상수 SPECIALIST_CONFIG 는 이 함수 안으로 들어갔다).


def select_labeled_subset(labeled_files, labeled_n, seed, classes):
    """[신설 2026-09-17 / 8절] labeled 서브샘플 — 라벨 효율 표(200·500·1,000)와 pseudo-label 출처 비교가 같은 부분집합을 쓴다.
    클래스별로 고르게(n//C 개씩, 나머지는 앞 클래스부터 1개씩), seed 로 클래스별 순서를 섞어 앞에서 자른다.
    반환: (뽑힌 행 목록, 남은 행 목록). 같은 seed·n 이면 어디서 불러도 같은 결과."""
    _ln = int(labeled_n)
    classes = list(classes)
    if _ln <= 0 or _ln > len(labeled_files):
        raise ValueError(f"[ERROR] labeled_n={_ln} 은 1 이상, 실제 labeled 개수({len(labeled_files)}) 이하여야 합니다.")
    by_cls = {c: [r for r in labeled_files if r.get("label") == c] for c in classes}
    rng = random.Random(int(seed))
    for c in classes:
        rng.shuffle(by_cls[c])
    base, extra = divmod(_ln, len(classes))
    picked, rest = [], []
    for i, c in enumerate(classes):
        k = base + (1 if i < extra else 0)
        if k > len(by_cls[c]):
            raise ValueError(f"[ERROR] labeled_n={_ln}: 클래스 {c} 에 {k}개가 필요한데 {len(by_cls[c])}개뿐입니다.")
        picked += by_cls[c][:k]
        rest += by_cls[c][k:]
    return picked, rest


def load_pseudo_csv(path, classes):
    """[신설 2026-09-17 / 8절] CSV(path,label,confidence,…) -> {basename: (label_idx, confidence)}. 라벨은 classes 안이어야 한다."""
    lookup = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in ("path", "label")):
        raise ValueError(f"[ERROR] pseudo-label CSV 형식이 아닙니다(path,label 열 필요): {path}")
    bad = sorted({r["label"] for r in rows if r["label"] not in classes})
    if bad:
        raise ValueError(f"[ERROR] pseudo-label CSV 에 클래스 밖 라벨 {bad}: {path}")
    for r in rows:
        conf = float(r.get("confidence", 1.0) or 1.0)
        lookup[os.path.basename(r["path"])] = (classes.index(r["label"]), conf)
    return lookup


class _CsvFusion:
    """[신설 2026-09-17 / 8절] CSV pseudo-label 을 fusion 출력과 같은 모양으로 감싼다(train 루프가 둘을 같은 코드로 다룬다).
    logits: one-hot [B, C] · features: None · uncertainty = 1 - confidence · disagreement 0 · specialist_weights 없음."""
    def __init__(self, logits, features, confidence):
        self.logits = logits
        self.features = features
        self.uncertainty = 1.0 - confidence
        self.disagreement = torch.zeros_like(confidence)
        self.specialist_weights = {}


def specialist_config_for(config):
    """[분리 2026-09-29 밤] teacher 8개 목록(train_mark5 안에 인라인이던 dict). Mark5_EXP2 의 같은 이름 함수와 맞춘다 — make_consensus_labels.py 가 쓴다.
    mark5.2 는 4.5 가 talking, mark5.0 은 media_talking."""
    talking_key = "talking" if "talking" in config.get_classes_for_evaluation() else "media_talking"
    return {
        "heavy_impact": {"mark_version": "mark4.1", "encoder_path": "best_teacher_encoder_mark4.1.pth", "classifier_path": "best_teacher_classifier_mark4.1.pth"},
        "dragging": {"mark_version": "mark4.2", "encoder_path": "best_teacher_encoder_mark4.2.pth", "classifier_path": "best_teacher_classifier_mark4.2.pth"},
        "construction": {"mark_version": "mark4.3", "encoder_path": "best_teacher_encoder_mark4.3.pth", "classifier_path": "best_teacher_classifier_mark4.3.pth"},
        "machine_noise": {"mark_version": "mark4.4", "encoder_path": "best_teacher_encoder_mark4.4.pth", "classifier_path": "best_teacher_classifier_mark4.4.pth"},
        talking_key: {"mark_version": "mark4.5", "encoder_path": "best_teacher_encoder_mark4.5.pth", "classifier_path": "best_teacher_classifier_mark4.5.pth"},
        "water_toilet": {"mark_version": "mark4.6", "encoder_path": "best_teacher_encoder_mark4.6.pth", "classifier_path": "best_teacher_classifier_mark4.6.pth"},
        "water_shower": {"mark_version": "mark4.7", "encoder_path": "best_teacher_encoder_mark4.7.pth", "classifier_path": "best_teacher_classifier_mark4.7.pth"},
        "dog_bark": {"mark_version": "mark4.8", "encoder_path": "best_teacher_encoder_mark4.8.pth", "classifier_path": "best_teacher_classifier_mark4.8.pth"},
    }


def detect_teacher_clip_input(specialist_config):
    """[신설 2026-09-29 밤 / 클립 전문가] teacher .pth 8개의 provenance(config_metadata.clip_input · clip_frames)를 읽어
    "이 전문가들은 클립 전체(64×301)를 보고 학습됐나" 를 판정한다. Mark4.x 가 2026-09-29 부터 clip_input=True 로 저장한다.
    8개가 섞여 있으면(일부만 클립) 학습을 멈춘다 — 조각 teacher 와 클립 teacher 를 한 앙상블에 섞으면 입력 모양이 다르다.
    돌려주는 값: (clip_input: bool, clip_frames: int). 옛 파일(키 없음)은 조각 단위(False, 301)."""
    flags, frames = {}, set()
    for name, paths in specialist_config.items():
        meta = load_checkpoint(_resolve_resource_path(paths["encoder_path"]), map_location="cpu").get("config_metadata") or {}
        flags[name] = bool(meta.get("clip_input", False))
        if flags[name]:
            frames.add(int(meta.get("clip_frames", 301)))
    if len(set(flags.values())) > 1:
        raise RuntimeError(f"[ERROR] teacher 8개의 입력 단위가 섞여 있습니다(클립={[k for k, v in flags.items() if v]} / "
                           f"조각={[k for k, v in flags.items() if not v]}). 8개를 같은 방식(Mark4.x clip_input)으로 다시 학습해 넣으십시오.")
    if len(frames) > 1:
        raise RuntimeError(f"[ERROR] teacher 의 clip_frames 가 서로 다릅니다: {sorted(frames)}")
    clip = all(flags.values())
    return clip, (frames.pop() if frames else 301)


class EnsembleTeacher:
    def __init__(self, specialist_config, device, others_rule="min", score_mode="margin",
                 use_distribution_alignment=False, dist_align_tau=0.5, dist_align_momentum=0.999):
        self.device = device
        self.specialists = {}
        self.embedding_dim = None
        # [2026-09-29 밤] teacher 입력 단위(클립/조각)는 .pth provenance 로 판정 — train_mark5 가 dataset 을 만들기 전에 같은 함수를 부른다.
        self.clip_input, self.clip_frames = detect_teacher_clip_input(specialist_config)
        self.fusion = None  # WeightedTeacherFusion은 배치마다 재생성하지 않고 1회 생성 후 재사용
        self.others_rule = others_rule  # [추가 2026-08-18] fusion 의 others 칸 조립 규칙
        self.score_mode = score_mode    # [추가 2026-08-22] 담당 칸 점수: "raw" | "margin"
        # [추가 2026-08-22] 분포 정렬. fusion 을 1회 생성해 재사용하므로 EMA 상태가 학습 내내 유지된다.
        self.use_distribution_alignment = use_distribution_alignment
        self.dist_align_tau = dist_align_tau
        self.dist_align_momentum = dist_align_momentum
        for class_name, paths in specialist_config.items():
            encoder_ckpt = load_checkpoint(_resolve_resource_path(paths["encoder_path"]), map_location=device)
            config = AudioViLDConfig(mark_version=paths["mark_version"])
            apply_config_metadata(config, encoder_ckpt)
            # [수정 2026-08-17] student 용 build_audio_encoder 가 아니라 teacher 용
            # build_teacher_encoder 를 쓴다. mark4.x teacher 8개는 2026-07-13 teacher 강화 이후
            # 4블록 TeacherAudioEncoder 로 학습됐는데, 여기서 student 용 SimpleAudioEncoder 를 만들어
            # 넣으려다 state_dict 가 안 맞아 죽었다. 작년 11월 더미 teacher(186KB)가 2블록이었기
            # 때문에 이 코드가 그때 기준으로 남아 있었던 것이다.
            encoder_state = resolve_state_dict(encoder_ckpt, "model_state_dict", "encoder_state_dict", "model")
            _align_teacher_arch(config, encoder_state, class_name, paths["encoder_path"])
            encoder = build_teacher_encoder(config).to(self.device)
            encoder.load_state_dict(encoder_state)
            encoder.eval()

            classifier_ckpt = load_checkpoint(_resolve_resource_path(paths["classifier_path"]), map_location=device)
            apply_config_metadata(config, classifier_ckpt)
            classifier = ViLDTextHead(config, learnable=False).to(self.device)   # [이식 2026-09-26] teacher 는 고정 T(저장본에 파라미터 없음)
            classifier.load_state_dict(resolve_state_dict(classifier_ckpt, "classifier_state_dict", "head_state_dict", "head"))
            classifier.eval()
            if "text_embeddings" in classifier_ckpt:
                text_emb = classifier_ckpt["text_embeddings"].to(device)
            else:
                text_emb = config.get_class_text_embeddings().to(device)
            self.specialists[class_name] = {
                "encoder": encoder,
                "classifier": classifier,
                "text_emb": text_emb,
            }
            if self.embedding_dim is None:
                self.embedding_dim = config.embedding_dim

    @torch.no_grad()
    def __call__(self, unlabeled_audio_batch, student_class_map):
        fusion_inputs = {}
        for class_name, models in self.specialists.items():
            region_emb = models["encoder"](unlabeled_audio_batch)
            logits = models["classifier"](region_emb, models["text_emb"])
            fusion_inputs[class_name] = {
                "logits": logits,
                "features": region_emb,
            }
        if self.fusion is None:
            # [변경 2026-08-18] others 조립 규칙을 config 에서 받는다. 기본은 min
            # (평균은 무관한 전문가 7명의 "내 거 아님" 표까지 섞여 others 가 유리해진다.
            #  근거 실측은 vild/teacher_fusion.py 의 docstring 참조).
            self.fusion = WeightedTeacherFusion(
                student_class_map, self.embedding_dim, self.device,
                others_rule=self.others_rule,
                score_mode=self.score_mode,
                use_distribution_alignment=self.use_distribution_alignment,
                dist_align_tau=self.dist_align_tau,
                dist_align_momentum=self.dist_align_momentum,
            )
        return self.fusion.fuse(fusion_inputs)


class SemiSupervisedDataset(Dataset):
    def __init__(self, file_path_list, parser, config, is_labeled=True):
        self.samples = []
        self.parser = parser
        self.config = config
        self.is_labeled = is_labeled
        # [신설 2026-09-29 / ablation A3] 클립 모드: 샘플 하나 = 클립 하나(teacher 조각 K개를 [K, C, n_mels, T] 로 쌓고,
        # student 는 클립 파형 [student_clip_samples] 하나). 조각 모드는 샘플 하나 = 조각 하나(2026-09-28 이전 동작 그대로).
        # [2026-09-29 밤 / 클립 전문가] teacher 가 클립 단위(config.teacher_clip_input)면 teacher 텐서는 조각 K개 대신 클립 전체 mel
        # [1, C, n_mels, teacher_clip_frames] 하나(K=1). 조각 모드 student 와 클립 teacher 의 조합은 train_mark5 가 앞에서 막는다.
        self.clip_mode = bool(getattr(config, "student_clip_input", False))
        self.teacher_clip = bool(getattr(config, "teacher_clip_input", False))
        self.teacher_frames = int(getattr(config, "teacher_clip_frames", 301) or 301)
        valid_labels = set(config.get_classes_for_evaluation()) if is_labeled else None

        skipped_label_counter = {}
        for item in file_path_list:
            path = item["path"]
            label = item.get("label")
            if is_labeled and label not in valid_labels:
                # [추가] 9-class 밖 라벨은 조용히 넘기지 않고 집계
                skipped_label_counter[label] = skipped_label_counter.get(label, 0) + 1
                continue
            try:
                if self.clip_mode:
                    out = parser.load_clip_with_segments(path)
                    if out is None:
                        continue
                    segs = out["segments"]
                    if self.teacher_clip:
                        # teacher 도 클립 전체를 본다: 클립 뷰 [C, n_mels, T] 를 teacher_clip_frames 로 맞춰(길면 자르고 짧으면 0) K=1 로 쌓는다.
                        clip_t = _normalize_visual_tensor(out["clip_record"]["tensor"],
                                                          (config.num_input_channels, config.n_mels, self.teacher_frames))
                        if clip_t is None:
                            print(f"[ERROR] 클립 텐서 모양 불일치: {path}")
                            continue
                        teacher_t = clip_t.unsqueeze(0)                                       # [1, C, n_mels, 301]
                        sal, starts = [out["clip_record"]["saliency"]], [0]
                    else:
                        teacher_t = torch.stack([r["tensor"] for r in segs], dim=0)         # [K, C, n_mels, T] teacher 조각 K개
                        sal, starts = [r["saliency"] for r in segs], [r["start_frame"] for r in segs]
                    metadata = {
                        "path": path,
                        "label": label if is_labeled else "unlabeled",
                        "segment_index": -1,                      # 클립 단위 표시
                        "start_frame": 0,
                        "end_frame": out["clip_record"]["end_frame"],
                        "saliency": sal,                                     # teacher 입력 단위별(조각 K개 또는 클립 1개)
                        "segment_start_frames": starts,
                    }
                    self.samples.append((teacher_t,                                          # teacher 용
                                         out["clip_wave"],                                   # [48000] student 용
                                         label if is_labeled else -1, metadata))
                    continue
                segment_records = parser.load_and_segment_with_metadata(path)
                for record in segment_records:
                    seg = record["tensor"]
                    if seg is not None:
                        metadata = {
                            "path": path,
                            "label": label if is_labeled else "unlabeled",
                            "segment_index": record["segment_index"],
                            "start_frame": record["start_frame"],
                            "end_frame": record["end_frame"],
                            "saliency": record["saliency"],
                        }
                        # [변경 2026-09-17 / Mark5.2 6절] 같은 창의 파형(record["wave"], 1.0초 16 kHz)도 같이 들고 간다.
                        # student 가 mn10(파형 입력)이면 이것을 쓰고 teacher 는 계속 mel 을 쓴다(config.student_input).
                        self.samples.append((seg, record["wave"], label if is_labeled else -1, metadata))
            except Exception as e:
                print(f"[ERROR] Failed to parse {path}: {e}")

        # [추가] 조용한 탈락 방지: 9-class 밖 라벨이 있으면 개수와 함께 보고하고,
        # labeled인데 유효 샘플이 0개면 즉시 중단(옛 6-class 데이터로 돌리는 사고 방지).
        if skipped_label_counter:
            print(
                f"[WARN] 9-class 밖 라벨로 건너뛴 labeled 파일: {skipped_label_counter} "
                f"(허용 클래스: {sorted(valid_labels)})"
            )
        if is_labeled and len(self.samples) == 0:
            raise ValueError(
                f"[ERROR] 유효한 labeled 세그먼트가 0개입니다. 입력 {len(file_path_list)}개가 모두 탈락했습니다. "
                f"건너뛴 라벨: {skipped_label_counter}. "
                "데이터 폴더/CSV 라벨이 9-class와 일치하는지 확인하세요."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SampleListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate(batch):
    # [변경 2026-09-17] (mel, wave, label, meta) — wave 는 [B, segment_samples] 로 쌓는다.
    mels, waves, labels, metadata = zip(*batch)
    return torch.stack(mels, dim=0), torch.stack(waves, dim=0), list(labels), list(metadata)


def dkd_loss(student_logits, teacher_logits, target_idx, T, dkd_alpha, dkd_beta):
    """DKD(Decoupled Knowledge Distillation, Zhao et al. 2022) — KD 를 TCKD + NCKD 로 분해.

        TCKD  타겟 클래스 vs 나머지 전체 의 **이진** 분포에 대한 KL
        NCKD  타겟을 뺀 나머지 클래스들끼리 재정규화한 분포에 대한 KL
        DKD   = dkd_alpha · TCKD + dkd_beta · NCKD

    vanilla KD 를 분해하면 KD = TCKD + (1 - p_t)·NCKD 가 되어, teacher 가 확신할수록
    NCKD(= dark knowledge 의 본체)가 눌린다. DKD 는 그 계수를 떼어내 독립적인 beta 로 세운다.

    target_idx 는 호출부에서 **teacher 앙상블의 argmax** 로 넘긴다(unlabeled 라 정답이 없음).
    """
    n_cls = student_logits.size(1)
    mask = F.one_hot(target_idx, num_classes=n_cls).bool()

    ps = F.softmax(student_logits / T, dim=1)
    pt = F.softmax(teacher_logits / T, dim=1)

    # --- TCKD: [타겟, 비타겟합] 2차원 분포 ---
    ps_t = ps[mask].unsqueeze(1)
    pt_t = pt[mask].unsqueeze(1)
    bs = torch.cat([ps_t, 1.0 - ps_t], dim=1).clamp_min(1e-8)
    bt = torch.cat([pt_t, 1.0 - pt_t], dim=1).clamp_min(1e-8)
    tckd = (bt * (bt.log() - bs.log())).sum(1).mean() * (T ** 2)

    # --- NCKD: 타겟을 제외하고 재정규화한 분포 ---
    # 타겟 위치를 -inf 로 눌러 softmax 에서 빠지게 한다. 그 자리는 log 가 -inf 가 되므로
    # 합산 전에 0 으로 덮어 nan(0 * -inf)을 막는다.
    s_nt = student_logits.masked_fill(mask, float("-inf"))
    t_nt = teacher_logits.masked_fill(mask, float("-inf"))
    log_ps_nt = F.log_softmax(s_nt / T, dim=1)
    log_pt_nt = F.log_softmax(t_nt / T, dim=1)
    pt_nt = log_pt_nt.exp()
    kl = (pt_nt * (log_pt_nt - log_ps_nt)).masked_fill(mask, 0.0)
    nckd = kl.sum(1).mean() * (T ** 2)

    return dkd_alpha * tckd + dkd_beta * nckd, tckd, nckd


class DistillationLoss(nn.Module):
    """[변경 2026-08-09] use_dkd 스위치 추가.

    False 면 기존 경로(단일 KL divergence)를 그대로 타므로 이전 학습과 동일하게 동작한다.
    True 면 soft 항만 DKD 로 바뀌고, hard/feature KD 와 alpha(hard vs soft 균형)는 건드리지 않는다.
    alpha 와 dkd_alpha/dkd_beta 는 **층이 다르다** — alpha 는 hard vs KD 의 균형이고
    dkd_alpha/beta 는 KD 내부에서 TCKD vs NCKD 의 균형이다.
    """

    def __init__(self, temperature, alpha, feature_kd_weight=0.0,
                 use_dkd=False, dkd_alpha=1.0, dkd_beta=1.0,
                 use_pseudo_label_ce=False, hard_weight=None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        # [신설 2026-09-29 밤 / HW] 정답 라벨 CE 의 가중치. None 이면 옛 (1−α)(Hinton KD 관례 — 같은 샘플에 hard·soft 를 함께 걸 때 두 항의
        # 합을 1 로 맞추는 것). 우리는 정답 라벨은 labeled 조각에, 가짜 정답은 unlabeled 조각에 따로 걸므로 그 관례가 뜻이 없고 지도 신호의
        # 기울기만 0.7 배로 줄인다(2026-09-29 격차 분석). --hard_weight 1.0 이면 α 는 unlabeled 항에만 곱힌다.
        self.hard_weight = None if hard_weight is None else float(hard_weight)
        self.feature_kd_weight = feature_kd_weight
        self.use_dkd = use_dkd
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta
        # [추가 2026-08-22] unlabeled 신호를 KL 대신 pseudo-label CE 로 거는 스위치.
        self.use_pseudo_label_ce = use_pseudo_label_ce
        self.hard = nn.CrossEntropyLoss()
        # per-sample 가중치를 곱하려면 배치 평균 전 값이 필요하다.
        self.pseudo = nn.CrossEntropyLoss(reduction="none")
        self.soft = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, hard_targets=None, teacher_logits=None,
                student_features=None, teacher_features=None, pseudo_weights=None, pseudo_targets=None):
        # [신설 2026-09-29 / A3] pseudo_targets 를 주면(클립 모드: 조각 다수결 라벨) teacher_logits 의 argmax 대신 그것으로 CE 를 건다.
        # 안 주면(조각 모드) 옛 동작(argmax) 그대로. pseudo-label CE 모드에서만 쓰인다.
        total = torch.tensor(0.0, device=student_logits.device)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        soft_loss = torch.tensor(0.0, device=student_logits.device)
        feat_loss = torch.tensor(0.0, device=student_logits.device)

        if hard_targets is not None and len(hard_targets) > 0:
            hard_loss = self.hard(student_logits, hard_targets)
            total = total + ((1 - self.alpha) if self.hard_weight is None else self.hard_weight) * hard_loss

        if teacher_logits is not None and len(teacher_logits) > 0:
            if self.use_pseudo_label_ce:
                # [신설 2026-08-22] teacher 에게서 분포가 아니라 정답만 받는다.
                # 근거: T=4.0 에서 soft target 의 정규화 엔트로피가 0.9704(균등=1.0)라
                # KL 이 나르는 정보가 0.0113 nats 뿐인데, 같은 teacher 의 argmax 는
                # non-others 에서 0.8940 로 맞는다. 즉 정보는 분포가 아니라 argmax 에 있다.
                # 분포를 따라하게 하면 teacher 의 애매함과 오류까지 함께 배운다.
                if pseudo_targets is None:
                    pseudo_targets = teacher_logits.argmax(dim=1)
                per_sample = self.pseudo(student_logits, pseudo_targets)
                if pseudo_weights is not None:
                    # 가중 평균. 분모가 0 에 가까우면(전부 저확신) 손실을 0 으로 둔다.
                    denom = pseudo_weights.sum().clamp_min(1e-6)
                    soft_loss = (per_sample * pseudo_weights).sum() / denom
                else:
                    soft_loss = per_sample.mean()
            elif self.use_dkd:
                # 타겟 축은 teacher 앙상블의 argmax. unlabeled 라 정답 라벨이 없다.
                target_idx = teacher_logits.argmax(dim=1)
                soft_loss, _, _ = dkd_loss(
                    student_logits, teacher_logits, target_idx,
                    self.temperature, self.dkd_alpha, self.dkd_beta)
            else:
                soft_loss = self.soft(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                ) * (self.temperature ** 2)
            total = total + self.alpha * soft_loss

        if (
            self.feature_kd_weight > 0
            and student_features is not None
            and teacher_features is not None
            and len(teacher_features) > 0
        ):
            sf = F.normalize(student_features, dim=1)
            tf = F.normalize(teacher_features, dim=1)
            feat_loss = 0.5 * F.l1_loss(sf, tf) + 0.5 * (1 - F.cosine_similarity(sf, tf, dim=1).mean())
            total = total + self.feature_kd_weight * feat_loss

        return total, hard_loss, soft_loss, feat_loss


def _supcon_loss(features, labels, temperature):
    """[신설 2026-10-07 / Mark5.3 SC] 지도 대조 손실(Khosla 외 2020 형식). 같은 클래스 쌍을 당기고 다른 클래스를 민다.
    양성 쌍이 하나도 없는 샘플은 평균에서 뺀다. 그런 샘플뿐이면 None."""
    f = F.normalize(features, dim=1)
    n = f.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=f.device)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    valid = pos.any(dim=1)
    if not bool(valid.any()):
        return None
    logits = (f @ f.t()) / float(temperature)
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_i = -(log_prob * pos.float()).sum(dim=1) / pos.float().sum(dim=1).clamp_min(1.0)
    return loss_i[valid].mean()


def merge_segments_to_clip(fusion_output, n_clips, n_seg, min_conf, min_gap, min_segments, rule="vote",
                           spec_probs=None, spec_cls_idx=None):
    """[신설 2026-09-29 / ablation A3] teacher 융합 출력(조각 n_clips×n_seg 개, 클립별로 조각이 이어져 나열된 순서)을 클립 단위로 합친다.

    rule="vote"(기본·옛 동작) — 비교모델 합의 라벨 스크립트(Mark5_EXP2/model/make_consensus_labels.py, 2026-09-05·09-22)의 기본 규칙과 같다.
    2026-09-28 드라이브 consensus_segments.csv(unlabeled 2,000클립×5조각) 실측으로 그 스크립트와 채택·라벨이 같음을 확인했다.
      조각 채택  : 확신도(1 − uncertainty) >= min_conf  그리고  1등−2등 전문가 확신도 차이(disagreement) >= min_gap  (0 이면 그 조건 없음)
      클립 채택  : 채택 조각 수 >= min_segments (5개 중 3 = 과반)
      클립 라벨  : 채택 조각 argmax 의 다수결, 동점이면 그 라벨 조각들의 확신도 합이 큰 쪽 (score = 표 수 + 확신도 합/(K+1) 의 argmax —
                   확신도 합/(K+1) < 1 이라 표 수를 뒤집지 못하고 동점만 가른다. 표 수·확신도 합까지 같으면 클래스 인덱스가 작은 쪽)
      클립 확신도: 채택 조각 확신도의 평균 (pseudo_label_confidence_weighting 의 가중치)
      클립 특징  : 채택 조각 융합 특징의 확신도 가중 평균 (feature KD 목표. 융합 특징 자체가 전문가 확신도 가중 평균이라 같은 방식)
      클립 로짓  : 채택 조각 융합 로짓의 확신도 가중 평균 — 참고용. 가짜 정답 CE 는 위 다수결 라벨(pseudo_targets)을 쓴다.

    [신설 2026-09-30 밤 / 가짜 정답 재료] rule="mean" — 전문가별 조각 평균(합의 라벨 스크립트의 --clip_rule mean 과 같은 규칙, config.clip_merge_rule).
      spec_probs [n_clips×n_seg, E] 는 전문가 E명의 조각별 확신도(fusion_output.specialist_weights 를 전문가 순서대로 이은 것),
      spec_cls_idx [E] 는 그 전문가들의 학생 클래스 인덱스다.
      전문가 평균: p̄[e] = 조각 n_seg 개의 확신도 평균
      클립 채택  : max p̄ >= min_conf  그리고  (1등 − 2등 p̄) >= min_gap   (조각 단위 채택과 min_segments 는 쓰지 않는다)
      클립 라벨  : p̄ 가 가장 큰 전문가의 클래스 (동률이면 전문가 순서가 앞선 쪽)
      클립 확신도: max p̄
      클립 특징  : 조각 융합 특징을 "그 조각의 전문가 확신도 합" 으로 가중 평균 — 융합 특징이 전문가 확신도 가중 평균이므로
                   조각·전문가 전체를 확신도로 가중 평균한 것과 같다(K=1 이면 융합 특징 그대로).
      클립 로짓  : 담당 칸 = logit(p̄), others 칸 = −max(담당 칸) — 조각 융합의 margin 과 같은 눈금. 참고용(가짜 정답 CE 는 라벨을 쓴다).
      실측(2026-09-30, 3차 조각 teacher·증강 없는 원본·unlabeled 2,000클립을 정답 라벨과 대조): vote 채택 1,288·오답 50 → mean 채택 1,339·오답 36.

    돌려주는 dict: keep [n] bool · label [n] long · conf [n] · features [n, D] · logits [n, C] · n_kept [n] · seg_keep [n, K] · seg_conf [n, K] · seg_gap [n, K]
    (rule="mean" 이면 clip_gap [n] · spec_mean [n, E] 도. n_kept·seg_keep 은 두 규칙 모두 조각 단위 채택 결과 — mean 에서는 기록용이다.)
    vote 에서 채택 조각이 하나도 없는 클립은 conf 0 · features/logits 0 이고 keep 이 False 라 학습에서 빠진다."""
    logits = fusion_output.logits.view(n_clips, n_seg, -1)
    feats = fusion_output.features.view(n_clips, n_seg, -1)
    conf = (1.0 - fusion_output.uncertainty).view(n_clips, n_seg).to(logits.device)
    gap = fusion_output.disagreement.view(n_clips, n_seg).to(logits.device)
    seg_keep = torch.ones_like(conf, dtype=torch.bool)
    if min_conf > 0:
        seg_keep = seg_keep & (conf >= min_conf)
    if min_gap > 0:
        seg_keep = seg_keep & (gap >= min_gap)
    n_kept = seg_keep.sum(dim=1)
    num_classes = logits.shape[-1]
    if rule == "mean":
        if spec_probs is None or spec_cls_idx is None:
            raise ValueError("[ERROR] merge_segments_to_clip(rule='mean') 에는 spec_probs·spec_cls_idx 가 필요합니다.")
        p = spec_probs.to(logits.device).to(logits.dtype).view(n_clips, n_seg, -1)          # [n, K, E] 전문가별 조각 확신도
        pbar = p.mean(dim=1)                                                               # [n, E] 전문가별 조각 평균
        top2 = torch.topk(pbar, k=min(2, pbar.shape[1]), dim=1).values
        clip_conf = top2[:, 0]
        clip_gap = (top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else torch.zeros_like(clip_conf)
        keep = torch.ones_like(clip_conf, dtype=torch.bool)
        if min_conf > 0:
            keep = keep & (clip_conf >= min_conf)
        if min_gap > 0:
            keep = keep & (clip_gap >= min_gap)
        cls_idx = spec_cls_idx.to(logits.device)
        label = cls_idx[pbar.argmax(dim=1)]
        w = p.sum(dim=2)                                                                   # [n, K] 조각별 전문가 확신도 합(융합 특징의 분모와 같은 양)
        clip_feat = (feats * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pc = pbar.clamp(1e-6, 1.0 - 1e-6)
        margin = torch.log(pc) - torch.log1p(-pc)                                          # logit(p̄)
        others = [i for i in range(num_classes) if i not in set(cls_idx.tolist())]
        if len(others) != 1:
            raise ValueError(f"[ERROR] 전문가가 맡지 않은 클래스(others)가 정확히 1개여야 합니다: {others}")
        clip_logits = torch.zeros(n_clips, num_classes, dtype=logits.dtype, device=logits.device)
        clip_logits[:, cls_idx] = margin
        clip_logits[:, others[0]] = -margin.max(dim=1).values
        return {"keep": keep, "label": label, "conf": clip_conf, "features": clip_feat, "logits": clip_logits,
                "n_kept": n_kept, "seg_keep": seg_keep, "seg_conf": conf, "seg_gap": gap, "clip_gap": clip_gap, "spec_mean": pbar}
    if rule != "vote":
        raise ValueError(f"[ERROR] merge_segments_to_clip 의 rule 은 'vote' 또는 'mean' 입니다: {rule!r}")
    keep = n_kept >= int(min_segments)
    seg_argmax = logits.argmax(dim=2)                                                  # [n, K]
    onehot = F.one_hot(seg_argmax, num_classes=num_classes).to(logits.dtype) * seg_keep.unsqueeze(-1).to(logits.dtype)
    votes = onehot.sum(dim=1)                                                          # [n, C] 채택 조각의 표 수
    conf_sum = (onehot * conf.unsqueeze(-1)).sum(dim=1)                                # [n, C] 그 라벨 조각들의 확신도 합
    label = (votes + conf_sum / (float(n_seg) + 1.0)).argmax(dim=1)
    w = conf * seg_keep.to(conf.dtype)                                                 # [n, K] 채택 조각만 확신도, 나머지 0
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    clip_conf = w.sum(dim=1) / n_kept.clamp_min(1).to(w.dtype)                         # 채택 조각 확신도 평균(채택 0 이면 0)
    clip_feat = (feats * w.unsqueeze(-1)).sum(dim=1) / denom
    clip_logits = (logits * w.unsqueeze(-1)).sum(dim=1) / denom
    return {"keep": keep, "label": label, "conf": clip_conf, "features": clip_feat, "logits": clip_logits,
            "n_kept": n_kept, "seg_keep": seg_keep, "seg_conf": conf, "seg_gap": gap}


def apply_spec_augment(mel_batch, num_freq_masks, freq_mask_param,
                       num_time_masks, time_mask_param):
    """SpecAugment(Park et al. 2019) — mel 배치의 주파수 대역·시간 구간을 무작위로 가린다.

    [신설 2026-08-23]

    입력: [B, C, n_mels, T]. 채널(mel·delta1·delta2)은 같은 위치를 함께 가린다 —
    같은 소리의 세 가지 표현이므로 위치가 어긋나면 가린 정보가 다른 채널로 샌다.

    ⚠마스크 값은 0 이 아니라 **그 샘플·그 채널의 평균**이다. 우리 mel 은 AmplitudeToDB 를
    거친 dB 스케일이고 별도 정규화가 없어서, 0 은 "무음"이 아니라 상당히 큰 소리다.
    0 으로 가리면 조용한 배경 위에 밝은 띠를 그려넣는 셈이 된다. 평균으로 채우면
    "정보만 지운다"에 가깝다.

    난수는 torch 전역 시드(42)를 따른다. 학습 루프에서만 호출할 것 — val 루프와 eval.py 는
    원본을 그대로 본다.
    """
    if mel_batch.dim() != 4:
        return mel_batch
    B, _, n_mels, n_frames = mel_batch.shape
    out = mel_batch.clone()
    fill = mel_batch.mean(dim=(2, 3), keepdim=True)      # [B, C, 1, 1] 채널별 평균
    for b in range(B):
        for _ in range(num_freq_masks):
            f = int(torch.randint(0, freq_mask_param + 1, (1,)).item())
            if f == 0 or f >= n_mels:
                continue
            f0 = int(torch.randint(0, n_mels - f + 1, (1,)).item())
            out[b, :, f0:f0 + f, :] = fill[b, :, 0, 0].view(-1, 1, 1)
        for _ in range(num_time_masks):
            t = int(torch.randint(0, time_mask_param + 1, (1,)).item())
            if t == 0 or t >= n_frames:
                continue
            t0 = int(torch.randint(0, n_frames - t + 1, (1,)).item())
            out[b, :, :, t0:t0 + t] = fill[b, :, 0, 0].view(-1, 1, 1)
    return out


def _stable_file_split(path, val_ratio=0.1):
    """[폴백 전용 / 2026-08-09] dataset_val.csv 가 없을 때만 쓰는 옛 방식.

    파일명 해시로 10% 를 val 로 뗀다. 재현은 되지만 **수집 단계에서 세션 단위로 갈라 둔
    split 을 무시한다** — 같은 세션(같은 녹음 시간대)의 클립이 train 과 val 에 함께 들어가
    val 성능이 낙관 편향된다. mark4.x 는 data/{split} 폴더로 나뉘어 있어 이 문제가 없었고,
    mark5 도 2026-08-09 부터 data_source_val -> data_val -> dataset_val.csv 를 쓴다.
    """
    key = os.path.basename(path).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def train_mark5(seed_value=None, mark_version="mark5.2", unlabeled_n=None, ceiling_csv=None, labeled_n=None,
                pseudo_csv=None, pseudo_source=None, spill_labeled=False, lr=None, encoder_lr=None,
                no_feature_kd=False, pl_min_gap=None, no_distill_branch=False, clip_input=False, linear_head=False,
                segment_input=False, head_pretrained_hidden=False, lp_epochs=None, hard_weight=None,
                others_from_reject=None, neg_pl=None, teacher_seg_mean=False, teacher_clean_view=False,
                bg_weight=None, student_raw_wave=False, m53=None, joint_forward=False):
    """[확장 2026-08-30 / Mark5_EXP] seed_value·unlabeled_n 을 실험 조건으로 받는다.
    [이식 2026-09-29 / Mark5.2] ablation A1~A4·H 와 격차 분석 대응 묶음(PH·LP-FT·HW·OR·NEG) 플래그를 받는다(vild_config.apply_cli_ablation).
    아무 플래그도 안 주면 mark5.2 기본(클립 입력·조각 teacher 또는 클립 teacher 자동 판정) 그대로다.

    두 인자 모두 None 이면 기존 Mark5 와 완전히 같은 동작이다
    (seed 42 · unlabeled 전량 · run_tag 에 조건 태그 없음).

    [신설 2026-09-06] ceiling_csv 를 주면 **천장(upper bound) 조건**이 된다.
        unlabeled 2,000개에 가려 두었던 진짜 라벨(model/make_ceiling_labels.py 가
        data_provenance.xlsx 에서 뽑은 CSV)을 붙여 labeled 와 합치고, unlabeled 는 0개로 두어
        teacher 를 한 번도 부르지 않는 순수 지도학습으로 돈다. Cances et al. (2022) 의
        "Supervised, 라벨 100%" 기준선이다. run_tag 는 _u<n> 대신 _ceil 이 붙어
        기존 25회 산출물과 겹치지 않는다. 이 조건의 모델은 제안 모델이 아니다.
    """
    # seed 를 명시하지 않으면 옛 기본값 42 를 쓰되, run_tag 에는 태그를 안 붙인다.
    effective_seed = 42 if seed_value is None else int(seed_value)
    is_ceiling = ceiling_csv is not None

    config = AudioViLDConfig(mark_version=mark_version)
    # run_tag 가 이 두 값을 읽어 산출물 파일명을 가른다. config 생성 직후에 넣어야
    # 이후의 모든 경로 계산(체크포인트·plots·CSV)이 같은 태그를 쓴다.
    config.experiment_seed = seed_value
    config.unlabeled_n = None if is_ceiling else unlabeled_n
    config.ceiling = is_ceiling
    config.labeled_n = labeled_n        # [신설 2026-09-17 / 8절] run_tag 에 _l<n>
    # [신설 2026-09-17 / 8절] pseudo-label 출처 비교. CSV 를 주면 teacher 대신 그 라벨을 쓴다(태그 _ps<이름>).
    if (pseudo_csv is None) != (pseudo_source is None):
        raise ValueError("[ERROR] --pseudo_csv 와 --pseudo_source 는 함께 줘야 합니다(태그와 파일이 짝).")
    if pseudo_csv is not None and is_ceiling:
        raise ValueError("[ERROR] --pseudo_csv 는 천장 조건과 함께 쓸 수 없습니다(천장은 unlabeled 를 쓰지 않음).")
    config.pseudo_source = pseudo_source
    config.spill_labeled = bool(spill_labeled)
    # [신설 2026-09-25 / LR] 명령줄 학습률(--lr 새 층 / --encoder_lr 백본). 기본값과 다르면 run_tag 에 _lr/_elr 이 붙으므로
    # 태그가 처음 쓰이기 전에 넣는다. 옵티마이저(아래 "[변경 2026-09-17] 백본 학습률 분리")는 config 값을 읽으니 그대로다.
    _lr_txt = config.apply_cli_learning_rates(lr, encoder_lr)
    # [신설 2026-09-28 / ablation A1·A2·A4] 우리 부품을 하나씩 끄는 플래그(vild_config.apply_cli_ablation). run_tag 에
    # _noFKD/_noGAP/_noDB 가 붙어 본 모델 산출물과 따로 쌓인다. 아래 criterion·keep_mask·distill 분류 손실은 config 값을 읽으니 그대로다.
    _ab_txt = config.apply_cli_ablation(no_feature_kd=no_feature_kd, pl_min_gap=pl_min_gap, no_distill_branch=no_distill_branch,
                                        clip_input=clip_input, linear_head=linear_head, segment_input=segment_input,
                                        head_pretrained_hidden=head_pretrained_hidden, lp_epochs=lp_epochs, hard_weight=hard_weight,
                                        others_from_reject=others_from_reject, neg_pl=neg_pl,
                                        teacher_seg_mean=teacher_seg_mean, teacher_clean_view=teacher_clean_view,
                                        bg_weight=bg_weight, student_raw_wave=student_raw_wave, m53=m53,
                                        joint_forward=joint_forward)
    if _lr_txt:
        print(f"[LR] 명령줄 학습률: {_lr_txt} · run_tag {config.run_tag}")
    if _ab_txt:
        print(f"[ABLATION] {_ab_txt} · run_tag {config.run_tag}")
    if spill_labeled and labeled_n is None:
        raise ValueError("[ERROR] --spill_labeled 는 --labeled_n 과 함께 써야 합니다(남은 labeled 를 unlabeled 로 돌리는 옵션).")
    if is_ceiling and unlabeled_n not in (None, 0):
        # 천장은 정의상 unlabeled 를 쓰지 않는다. 다른 값을 같이 주면 조건이 모호해지므로 멈춘다.
        raise ValueError(f"[ERROR] --ceiling_csv 와 --unlabeled_n={unlabeled_n} 을 함께 줄 수 없습니다. "
                         "천장 조건은 unlabeled 를 쓰지 않습니다(--unlabeled_n 을 빼십시오).")
    # [Mark5_EXP2] CSV 가짜 정답 경로(--pseudo_csv, 8절 출처 비교)에는 전문가 8명의 확신도가 없다. OR(전원 거절)·NEG(부정 라벨)는 그 확신도로
    # 정하는 장치라 이 경로에서는 정의되지 않는다 — 조용히 끄면 run_tag(_OR/_NEG)가 거짓이 되므로 멈춘다. PH·LP·HW 는 teacher 와 무관해 그대로 된다.
    if pseudo_csv is not None and (float(getattr(config, "others_from_reject_tau", 0.0) or 0.0) > 0
                                   or float(getattr(config, "neg_pl_tau", 0.0) or 0.0) > 0):
        raise ValueError("[ERROR] --pseudo_csv(CSV 가짜 정답) 경로에는 전문가 확신도가 없어 --others_from_reject / --neg_pl 을 쓸 수 없습니다. "
                         "두 플래그를 빼고 돌리십시오(PH·LP·HW 는 함께 쓸 수 있습니다).")

    set_seed(effective_seed, deterministic=getattr(config, "cudnn_deterministic", False))

    print(f"[AC/LT] anchor_centering={config.anchor_centering} learnable_logit_scale={config.learnable_logit_scale} · run_tag {config.run_tag}")
    # [신설 2026-09-29 / 헤드 진단 H] 선형 헤드면 위 AC/LT 값은 헤드가 안 쓰므로 뜻이 없다(텍스트 임베딩은 만들되 헤드가 무시한다).
    if getattr(config, "student_head", "text_anchor") == "linear":
        print(f"[H] student 분류 헤드 = 선형(nn.Linear {config.embedding_dim}→{len(config.get_classes_for_evaluation())}) — "
              f"텍스트 앵커·AC/LT 미사용, 제로샷 불가(2026-10-07 D0 본 레시피). 두 갈래(지도·증류)가 같은 Linear 를 공유한다")
    # [신설 2026-09-29 / A3] 클립 모드 — 학습 단위가 조각에서 클립으로 바뀐다(배치 16 = 클립 16개, 에폭당 스텝 수 1/K). teacher 는 조각 그대로.
    _clip_mode = bool(getattr(config, "student_clip_input", False))
    _clip_min_segments = int(getattr(config, "clip_min_segments", 3))
    # [2026-09-29 밤 / 클립 전문가] teacher .pth 8개의 provenance 로 입력 단위를 판정해 config 에 적는다(dataset 이 이 값으로 teacher 텐서를 만든다).
    specialist_config = specialist_config_for(config)
    _teacher_clip, _teacher_frames = detect_teacher_clip_input(specialist_config)
    config.teacher_clip_input = bool(_teacher_clip)
    config.teacher_clip_frames = int(_teacher_frames)
    if _teacher_clip:
        if not _clip_mode:
            raise ValueError("[ERROR] teacher 8개가 클립 단위(Mark4.x clip_input)로 학습됐는데 student 가 조각 입력(--segment_input)입니다. "
                             "클립 teacher 는 클립 student 와만 짝이 됩니다(조각마다 teacher 라벨을 매길 수 없음).")
        _clip_min_segments = 1
        print(f"[TEACHER] 전문가 8개 = 클립 단위(mel {config.n_mels}×{_teacher_frames}) → 클립당 teacher 출력 1개(K=1), "
              f"채택 = 확신도 ≥ {getattr(config, 'pseudo_label_min_confidence', 0.5)} & 1등−2등 차이 ≥ {getattr(config, 'pseudo_label_min_gap', 0.0):g}")
    else:
        print(f"[TEACHER] 전문가 8개 = 조각 단위(mel {config.n_mels}×{config.segment_length} × {config.max_segments})")
    # [신설 2026-09-30 밤 / 가짜 정답 재료] 조각 → 클립 합치기 규칙. "mean"(--teacher_seg_mean)은 조각 teacher 에서만 뜻이 있다(K=1 이면 vote 와 같다).
    # 클립 teacher(4차)가 resources/ 에 들어 있는데 이 플래그를 주면 이름(_SM)만 붙고 재료는 클립 teacher 가 되므로 멈춘다.
    _merge_rule = str(getattr(config, "clip_merge_rule", "vote"))
    if _merge_rule not in ("vote", "mean"):
        raise ValueError(f"[ERROR] clip_merge_rule 은 'vote' 또는 'mean' 이어야 합니다: {_merge_rule!r}")
    if _merge_rule == "mean" and _teacher_clip:
        raise ValueError("[ERROR] --teacher_seg_mean(조각 평균 합의)은 조각 단위 teacher 에서만 씁니다. 지금 resources/ 의 teacher 8개는 "
                         "클립 단위(Mark4.x clip_input)입니다 — Mark5.2 의 preprocessing/import_teachers.py --run_subdir <조각 판 보관 폴더> 로 "
                         "조각 teacher 16개를 넣은 뒤 다시 돌리십시오.")
    if _clip_mode and _merge_rule == "mean":
        print(f"[A3] 학습 단위 = 클립(파형 {getattr(config, 'student_clip_samples', 48000)}샘플 통째). teacher 조각 {config.max_segments}개 → "
              f"[SM] 조각 평균 합의: 전문가별로 조각 확신도를 평균 → 클립 채택 = 평균 확신도 최대 ≥ {getattr(config, 'pseudo_label_min_confidence', 0.5)} & "
              f"1등−2등 차이 ≥ {float(getattr(config, 'pseudo_label_min_gap', 0.0) or 0.0):g}, 라벨 = 평균 확신도 1등 전문가, 확신도 = 그 평균, "
              f"특징 = 조각 융합 특징의 전문가 확신도 합 가중 평균 (merge_segments_to_clip rule='mean')")
    elif _clip_mode:
        print(f"[A3] 학습 단위 = 클립(파형 {getattr(config, 'student_clip_samples', 48000)}샘플 통째). teacher 조각 {1 if _teacher_clip else config.max_segments}개 → "
              f"클립 합치기: 조각 채택(확신도·1등−2등 차이) ≥{_clip_min_segments}개면 채택, 라벨 = 채택 조각 다수결, "
              f"확신도 = 채택 조각 평균, 특징 = 채택 조각 확신도 가중 평균 (merge_segments_to_clip)")
    # [2026-09-29 밤 / 격차 분석 대응 묶음] 켜진 것만 로그에 남긴다(플래그 없으면 옛 동작).
    if getattr(config, "head_pretrained_hidden", False):
        print("[PH] 헤드 단순화: mn10 사전학습 은닉층(960→1280+Hardswish+Dropout) 유지 → Linear(1280→384) 투영 하나 → 앵커. supervised branch = Identity")
    _lp_epochs = int(getattr(config, "lp_epochs", 0) or 0)
    if _lp_epochs > 0:
        print(f"[LP-FT] 처음 {_lp_epochs}에폭은 백본(mn10 core) 고정(requires_grad·BN 통계) → 투영·branch·헤드·BG 만 학습 → 그 뒤 전체 해제. "
              f"LP 구간은 early stopping 에 세지 않고 FT 첫 에폭에서 best 를 다시 잰다")
    if getattr(config, "hard_weight", None) is not None:
        print(f"[HW] 정답 라벨 CE 가중치 {float(config.hard_weight):g} (옛 (1−α)={1 - float(getattr(config, 'distill_alpha', 0.3)):g})")
    _or_tau = float(getattr(config, "others_from_reject_tau", 0.0) or 0.0)
    if _or_tau > 0:
        print(f"[OR] 전원 거절 → others 가짜 정답: 전문가 8명 최대 확신도 < {_or_tau:g} 이면 others(가중치 1−최대). 합의 라벨에 없던 배경 신호")
    _neg_tau = float(getattr(config, "neg_pl_tau", 0.0) or 0.0)
    _neg_beta = float(getattr(config, "neg_pl_beta", 0.0) or 0.0)
    _use_neg = _neg_tau > 0 and _neg_beta > 0
    if _use_neg:
        print(f"[NEG] 부정 가짜 라벨: 미채택 클립에서 전문가 c 확신도 < {_neg_tau:g} 인 클래스에 −{_neg_beta:g}·log(1−p_student(c)) (others 제외)")
    if pseudo_csv is not None and _clip_mode:
        print("[INFO] CSV 가짜 정답 경로 + 클립 모드: 라벨이 이미 클립 단위라 조각 합치기(merge_segments_to_clip)를 거치지 않는다. teacher 조각은 읽되 쓰지 않음")
    if pseudo_csv is not None and (_merge_rule == "mean" or getattr(config, "spec_augment_teacher_view", "augmented") == "clean"):
        print("[INFO] CSV 가짜 정답 경로: --teacher_seg_mean / --teacher_clean_view 는 이 회차에서 쓰이지 않는다(전문가 출력을 쓰지 않음). "
              "run_tag 의 _SM·_CT 는 같은 묶음의 합의 회차와 이름을 맞추려고 그대로 붙는다(_G04 와 같은 규칙).")

    device = config.device
    parser = AudioParser(config, segment_mode=True)

    labeled_files = list(csv.DictReader(open(_resolve_csv_path("dataset_labeled.csv"), newline="", encoding="utf-8")))

    # ==========================================================================
    # [신설 2026-09-17 / 8절] labeled 서브샘플 — 라벨 효율 표(labeled 200·500·1,000)
    #   클래스별로 고르게 뽑는다: 클래스당 n//9 개, 나머지 n%9 개는 클래스 순서 앞에서부터 1개씩.
    #   같은 seed 면 같은 클립이 뽑히고, 작은 n 이 큰 n 의 부분집합(누적)이 되도록 클래스별로 섞은 순서의 앞에서 자른다.
    #   unlabeled 는 건드리지 않는다(2,000 전량 또는 --unlabeled_n). val·test 도 그대로 430.
    #   천장(ceiling) 조건과 같이 주면 "labeled n + 진짜 라벨 2,000" 이 되어 표의 '상한' 행과 뜻이 달라지므로 막는다.
    # ==========================================================================
    spilled_files = []
    if labeled_n is not None:
        if is_ceiling:
            raise ValueError("[ERROR] --labeled_n 과 --ceiling_csv 는 함께 쓸 수 없습니다(상한 행은 labeled 2,000 + 진짜 라벨 2,000 으로 정의).")
        _classes = list(config.get_classes_for_evaluation())
        labeled_files, _rest = select_labeled_subset(labeled_files, labeled_n, effective_seed, _classes)
        _cnt = {c: sum(1 for r in labeled_files if r["label"] == c) for c in _classes}
        print(f"[INFO] labeled 서브샘플: {len(labeled_files)}개 사용 (seed={effective_seed}, 클래스별 {_cnt})")
        if spill_labeled:
            # 남은 labeled 클립의 라벨을 떼어 unlabeled 풀에 합친다(체크리스트 8절 "labeled 200 + unlabeled 3,800").
            spilled_files = [{"path": r["path"]} for r in _rest]
            print(f"[INFO] 남은 labeled {len(spilled_files)}개를 라벨 없이 unlabeled 풀에 합칩니다(--spill_labeled)")

    # ==========================================================================
    # [신설 2026-09-06] 천장 조건 — 진짜 라벨 CSV 를 labeled 에 합친다
    # ==========================================================================
    # CSV 는 model/make_ceiling_labels.py 가 만든 것(열: path,label,confidence,source,generated_at).
    # 검사하는 것: 필수 열, 9-class 안의 라벨, labeled 와 같은 클립이 섞이지 않았는지,
    # dataset_unlabeled.csv 의 클립을 **전부** 덮는지(일부만 붙이면 '천장'이 아니다).
    n_ceiling = 0
    if is_ceiling:
        cpath = ceiling_csv if os.path.isabs(ceiling_csv) else os.path.join(PROJECT_ROOT, ceiling_csv)
        if not os.path.isfile(cpath):
            raise FileNotFoundError(f"[ERROR] 천장용 라벨 CSV 가 없습니다: {cpath}\n"
                                    "        model/make_ceiling_labels.py 로 먼저 만드십시오.")
        with open(cpath, newline="", encoding="utf-8-sig") as f:
            ceiling_rows = list(csv.DictReader(f))
        if not ceiling_rows or any(c not in ceiling_rows[0] for c in ("path", "label")):
            raise ValueError(f"[ERROR] 천장용 라벨 CSV 형식이 다릅니다(path,label 열 필요): {cpath}")
        valid = set(config.get_classes_for_evaluation())
        bad = sorted({r["label"] for r in ceiling_rows if r["label"] not in valid})
        if bad:
            raise ValueError(f"[ERROR] 천장용 라벨 CSV 에 9-class 밖 라벨이 있습니다: {bad}")
        lab_keys = {os.path.basename(r["path"]) for r in labeled_files}
        dup = [r["path"] for r in ceiling_rows if os.path.basename(r["path"]) in lab_keys]
        if dup:
            raise ValueError(f"[ERROR] 천장용 라벨 {len(dup)}개가 labeled 와 같은 클립입니다(예: {dup[0]}).")
        _unl_csv = _resolve_csv_path("dataset_unlabeled.csv", required=False)
        if _unl_csv:
            unl_keys = {os.path.basename(r["path"]) for r in csv.DictReader(open(_unl_csv, newline="", encoding="utf-8"))}
            ceil_keys = {os.path.basename(r["path"]) for r in ceiling_rows}
            if ceil_keys != unl_keys:
                raise ValueError(f"[ERROR] 천장용 라벨 CSV({len(ceil_keys)}개)가 dataset_unlabeled.csv({len(unl_keys)}개)와 "
                                 f"같은 클립 집합이 아닙니다(없음 {len(unl_keys - ceil_keys)}, 남음 {len(ceil_keys - unl_keys)}). "
                                 "천장은 unlabeled 전부에 라벨을 되돌려 준 조건이어야 합니다.")
        else:
            # [2026-09-06 리뷰 반영] 조용히 건너뛰지 않는다(비교모델 train.py 와 같은 경고).
            print("[WARN] dataset_unlabeled.csv 가 없어 천장 라벨이 unlabeled 전부를 덮는지 확인하지 못했습니다.")
        labeled_files = labeled_files + [{"path": r["path"], "label": r["label"]} for r in ceiling_rows]
        n_ceiling = len(ceiling_rows)
        _dist = {}
        for r in ceiling_rows:
            _dist[r["label"]] = _dist.get(r["label"], 0) + 1
        print(f"[INFO] 천장 조건: 진짜 라벨 {n_ceiling}개를 labeled 에 합쳤습니다 -> 지도학습 {len(labeled_files)}개 "
              f"(생성 시각 {ceiling_rows[0].get('generated_at', '?')})")
        print(f"[INFO] 천장 라벨 클래스 분포: {dict(sorted(_dist.items()))}")

    # [변경 2026-08-09] dataset_unlabeled.csv 가 없어도 터지지 않게 하되, **크게 경고한다.**
    # 이 파일이 없으면 unlabeled 샘플이 0개가 되고, 아래 학습 루프의
    #     if unlabeled_indices:  ...  ensemble_teacher(...)
    # 블록이 한 번도 실행되지 않는다. 즉 **8개 teacher 가 한 번도 불리지 않고 soft loss·feature KD 가
    # 전혀 없는 순수 supervised 학습**이 된다. mark5 의 핵심(Dynamic Online Ensemble KD)이
    # 통째로 빠지므로 조용히 넘어가면 안 된다.
    unlabeled_csv = _resolve_csv_path("dataset_unlabeled.csv", required=False)
    unlabeled_files = (list(csv.DictReader(open(unlabeled_csv, newline="", encoding="utf-8")))
                       if unlabeled_csv else [])
    if is_ceiling:
        # 천장은 unlabeled 를 쓰지 않는다(라벨을 되돌려 줘서 위에서 labeled 에 합쳤다).
        unlabeled_files = []

    # ==========================================================================
    # [신설 2026-08-30 / Mark5_EXP] unlabeled 서브샘플링 — 데이터효율곡선(랩미팅 8번)
    # ==========================================================================
    # "라벨 없는 데이터를 넣을수록 성능이 오른다"를 보이기 위해 0/500/1000/1500/2000 을
    # 돌린다. 뽑는 방식은 **누적(nested)** 이다 — 같은 seed 안에서 500 ⊂ 1000 ⊂ 1500 ⊂ 2000.
    #
    # 왜 누적인가: 조건마다 완전히 새로 뽑으면 성능 변화가 "데이터가 늘어서"인지
    # "이번에 뽑힌 표본이 운이 좋아서"인지 섞인다. 앞에서 잘라 쓰면 큰 조건이 작은 조건을
    # 그대로 포함하므로 증가분만 순수하게 보인다.
    #
    # 왜 별도 난수기(random.Random)인가: 전역 random 을 쓰면 여기서 소비한 난수만큼
    # 이후 흐름(가중치 초기화 등)이 밀려서, n 이 다르면 seed 가 같아도 초기 가중치가
    # 달라진다. 그러면 "unlabeled 양"과 "초기값" 두 변수가 섞인다. 별도 인스턴스를 쓰면
    # 같은 seed = 같은 초기 가중치가 보장되고, 달라지는 것은 unlabeled 양 하나뿐이다.
    if unlabeled_n is not None:
        _n = int(unlabeled_n)
        if _n < 0:
            raise ValueError(f"[ERROR] --unlabeled_n 은 0 이상이어야 합니다. 받은 값: {_n}")
        if _n > len(unlabeled_files):
            raise ValueError(
                f"[ERROR] --unlabeled_n={_n} 이 실제 unlabeled 개수({len(unlabeled_files)})보다 큽니다. "
                "인덱스 셀(generate_dataset_index_mark5.py --mode train)이 돌았는지 확인하십시오."
            )
        _rng = random.Random(effective_seed)
        _order = list(range(len(unlabeled_files)))
        _rng.shuffle(_order)
        _picked = sorted(_order[:_n])          # 앞에서 n 개 -> 누적 구조. 정렬은 재현성·가독성용.
        unlabeled_files = [unlabeled_files[i] for i in _picked]
        print(f"[INFO] unlabeled 서브샘플: {len(unlabeled_files)}개 사용 "
              f"(전체 {len(_order)}개 중, seed={effective_seed} 기준 누적 추출)")

    if spilled_files:
        unlabeled_files = unlabeled_files + spilled_files
        print(f"[INFO] unlabeled 풀 = dataset_unlabeled {len(unlabeled_files) - len(spilled_files)} + 남은 labeled {len(spilled_files)} = {len(unlabeled_files)}개")

    if not unlabeled_files and is_ceiling:
        print("[INFO] 천장 조건이므로 unlabeled 0개 · teacher 미사용 · 순수 지도학습입니다(의도된 동작).")
    elif not unlabeled_files:
        print("[WARN] " + "=" * 74)
        print("[WARN] unlabeled 데이터가 0개입니다 — 지식 증류가 작동하지 않습니다.")
        print("[WARN]   · 8개 teacher(ensemble_teacher)가 한 번도 호출되지 않습니다.")
        print("[WARN]   · soft loss / feature KD 가 전부 0 이고 hard loss 만 남습니다.")
        print("[WARN]   · 결과적으로 순수 supervised 학습이 됩니다(mark5 의 novelty 상실).")
        print("[WARN] data_source_unlabeled 를 채우고 run_all.py 의 unlabeled 단계를 돌리십시오.")
        print("[WARN] " + "=" * 74)

    labeled_dataset = SemiSupervisedDataset(labeled_files, parser, config, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(unlabeled_files, parser, config, is_labeled=False)

    # [변경 2026-08-09] val 을 dataset_val.csv 에서 읽는다.
    # 수집 단계(mark5_slicer.py)에서 SINS·CHiME 은 세션 단위로, AI Hub 는 클립 단위로
    # train/val/test 를 갈라 두었다. 그런데 이 함수가 dataset_labeled.csv 를 파일명 해시로
    # 10% 떼어 val 을 만들고 있어서 그 분리가 통째로 무시되고 있었다(같은 세션 클립이
    # train 과 val 에 섞임 -> val 성능이 낙관 편향). dataset_val.csv 가 있으면 그것을 쓰고,
    # 없을 때만 옛 해시 분할로 폴백하되 눈에 띄게 경고한다.
    val_csv = _resolve_csv_path("dataset_val.csv", required=False)
    if val_csv and os.path.exists(val_csv):
        val_files = list(csv.DictReader(open(val_csv, newline="", encoding="utf-8")))
        val_dataset_src = SemiSupervisedDataset(val_files, parser, config, is_labeled=True)
        train_samples = list(labeled_dataset.samples)
        val_samples = list(val_dataset_src.samples)
        print(f"[INFO] val 을 dataset_val.csv 에서 읽었습니다 — "
              f"train {len(train_samples)}개 / val {len(val_samples)}개 (폴더 기반 분리)")
    else:
        print("[WARN] " + "=" * 70)
        print("[WARN] dataset_val.csv 가 없어 파일명 해시로 train 의 10% 를 val 로 뗍니다(옛 방식).")
        print("[WARN] 이러면 수집 단계의 세션 단위 split 이 무시되어 val 성능이 낙관 편향됩니다.")
        print("[WARN] preprocessing/fix_audio_length.py 로 data_source_val -> data_val 을 만들고")
        print("[WARN] generate_dataset_index_mark5.py --mode val 을 먼저 돌리십시오.")
        print("[WARN] " + "=" * 70)
        train_samples = []
        val_samples = []
        for sample in labeled_dataset.samples:
            _, _, label, metadata = sample
            split_name = _stable_file_split(metadata["path"], val_ratio=0.1)
            if split_name == "val":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    train_dataset = SampleListDataset(train_samples)
    val_dataset = SampleListDataset(val_samples)

    final_train_dataset = ConcatDataset([train_dataset, unlabeled_dataset])

    train_loader = DataLoader(final_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate)

    student_encoder = build_audio_encoder(config).to(device)
    # [신설 2026-09-17 / Mark5.2 6절] student 입력 종류. "waveform" 이면 파서가 준 파형을 mn10 에 넣고,
    # SpecAugment 는 인코더 안(128 mel, bin 별 시간 평균 채움)에서 건다. "mel" 이면 옛 경로 그대로.
    _student_input = getattr(config, "student_input", "mel")
    _enc_params = sum(p.numel() for p in student_encoder.parameters() if p.requires_grad)
    print(f"[INFO] student 인코더: {type(student_encoder).__name__} (encoder_type={getattr(config, 'encoder_type', 'cnn')}, "
          f"입력={_student_input}, 학습 파라미터 {_enc_params:,}"
          + (f", 사전학습 {'로드' if getattr(student_encoder, 'pretrained_loaded', False) else '없음'}"
             if hasattr(student_encoder, "pretrained_loaded") else "") + ")")

    def run_student_encoder(mel_x, wave_x, augment=False):
        """student 인코더 호출 한 곳. mel 경로는 호출자가 이미 SpecAugment 를 걸어 준 mel 을 받고,
        파형 경로는 인코더가 augment 플래그로 자기 mel 에 건다."""
        if _student_input == "waveform":
            return student_encoder(wave_x, augment=augment)
        return student_encoder(mel_x)
    # [변경 2026-08-22] distill branch 마지막 ReLU 를 config 로 제어한다(기본 없음).
    # eval.py 도 같은 값을 읽어야 state_dict 구조가 맞는다.
    student_branch = DualBranchStudentHead(
        config.embedding_dim,
        distill_final_relu=getattr(config, "distill_branch_final_relu", False),
        supervised_identity=bool(getattr(config, "head_pretrained_hidden", False)),   # [PH 2026-09-29 밤] 지도 갈래 항등
    ).to(device)
    student_classifier = build_student_head(config).to(device)   # [H 2026-09-29] config.student_head 로 텍스트 앵커/선형 선택(eval 과 같은 함수)
    student_text_emb = config.get_class_text_embeddings(for_evaluation=True).to(device)
    student_label_map = config.get_target_label_map()

    # [추가] dummy_label 등이 student 차원에 새지 않았는지 즉시 검증(누수 시 학습 전에 터지게).
    eval_classes = config.get_classes_for_evaluation()
    assert student_text_emb.shape[0] == len(student_label_map) == len(eval_classes), (
        f"[ERROR] student 클래스 차원 불일치: text_emb={student_text_emb.shape[0]}, "
        f"label_map={len(student_label_map)}, eval_classes={len(eval_classes)} (현재 기대값 9). "
        "for_evaluation=True 누락 또는 dummy_label 누수/ config 불일치를 의심하세요."
    )

    # [추가] 학습형 background(others) 임베딩. use_background_embedding=False면 사용하지 않음(None).
    background_embedding = None
    if config.use_background_embedding:
        background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)

    # [신설 2026-10-07 / Mark5.3] 켠 부품(vild_config.M53_PARTS 순서). 빈 tuple 이면 아래 Mark5.3 분기를 모두 건너뛴다(옛 동작 그대로).
    _m53 = tuple(getattr(config, "m53_parts", None) or ())
    _m53_on = bool(_m53)
    # [신설 2026-10-08 / Mark5.3 JF] 정답·라벨 없는 클립을 학생에 한 번에 통과(vild_config m53_joint_forward 주석). 클립 파형 student 에서만.
    _m53_jf = bool(getattr(config, "m53_joint_forward", False))
    if _m53_jf and (not _m53_on or _student_input != "waveform"):
        raise ValueError("[ERROR] --joint_forward 는 Mark5.3(--m53) · 클립 파형 student 에서만 씁니다.")
    if _m53_jf:
        print("[M53/JF] 정답 클립과 라벨 없는 클립을 학생에 한 번에 통과시킨다(BN = 배치 전체). 한쪽만 있는 배치는 그 배치 전체로 통과(옛 경로와 같다).")
    ova_head = None
    if "OVA" in _m53:
        ova_head = ExpertBinaryHead(config.embedding_dim, len(specialist_config), getattr(config, "m53_ova_hidden", None)).to(device)

    # teacher 8개 목록(specialist_config)은 위에서 이미 만들었다(2026-09-29 밤 분리 — dataset 을 만들기 전에 provenance 를 읽기 위해).
    # make_consensus_labels.py 도 같은 specialist_config_for(config) 를 쓴다.
    _others_rule = getattr(config, "fusion_others_rule", "min")
    _score_mode = getattr(config, "fusion_score_mode", "margin")
    _use_da = getattr(config, "use_distribution_alignment", False)
    _da_tau = getattr(config, "dist_align_tau", 0.5)
    _da_mom = getattr(config, "dist_align_momentum", 0.999)
    pseudo_lookup = None
    if pseudo_csv is not None:
        # [신설 2026-09-17 / 8절] CSV 모드: teacher 8개를 아예 만들지 않는다. 라벨·확신도는 CSV 에서.
        _cpath = pseudo_csv if os.path.isabs(pseudo_csv) else os.path.join(PROJECT_ROOT, pseudo_csv)
        pseudo_lookup = load_pseudo_csv(_cpath, list(config.get_classes_for_evaluation()))
        _hit = sum(1 for r in unlabeled_files if os.path.basename(r["path"]) in pseudo_lookup)
        print(f"[INFO] pseudo-label 출처 = {pseudo_source} (CSV {_cpath}) — unlabeled {len(unlabeled_files)}개 중 {_hit}개에 라벨 있음, "
              f"teacher 8개 미사용, feature KD 없음")
        if _hit < len(unlabeled_files):
            print(f"[WARN] CSV 에 없는 unlabeled 클립 {len(unlabeled_files) - _hit}개는 확신도 0 으로 취급돼 학습에서 빠집니다.")
        ensemble_teacher = None
    else:
        ensemble_teacher = EnsembleTeacher(specialist_config, device,
                                           others_rule=_others_rule, score_mode=_score_mode,
                                           use_distribution_alignment=_use_da,
                                           dist_align_tau=_da_tau, dist_align_momentum=_da_mom)
    if pseudo_lookup is not None:
        pass
    elif _score_mode == "margin":
        print("[INFO] teacher fusion: 담당 칸 = margin(target-others), others 칸 = -max(margin)")
    else:
        print(f"[INFO] teacher fusion: 담당 칸 = raw logit, others 칸 조립 규칙 = {_others_rule}")
    if _use_da:
        print(f"[INFO] 분포 정렬: 적용 (tau={_da_tau}, EMA momentum={_da_mom}, 목표=균등)")
    else:
        print("[INFO] 분포 정렬: 미적용")
    # [추가 2026-08-23] SpecAugment 설정
    _use_sa = getattr(config, "use_spec_augment", False)
    _sa_f = getattr(config, "spec_augment_freq_mask_param", 8)
    _sa_nf = getattr(config, "spec_augment_num_freq_masks", 2)
    _sa_t = getattr(config, "spec_augment_time_mask_param", 16)
    _sa_nt = getattr(config, "spec_augment_num_time_masks", 2)
    _sa_teacher_view = getattr(config, "spec_augment_teacher_view", "augmented")
    if _use_sa:
        print(f"[INFO] SpecAugment: 적용 (주파수 {_sa_nf}x최대{_sa_f}/{config.n_mels}칸, "
              f"시간 {_sa_nt}x최대{_sa_t}/{config.segment_length}프레임, "
              f"teacher view={_sa_teacher_view}) — 학습에만, val/eval 은 원본")
    else:
        print("[INFO] SpecAugment: 미적용")
    teacher_feature_cache = []
    unlabeled_metadata_cache = []

    # [변경 2026-08-09] DKD 설정을 config 에서 읽는다. getattr 로 받아 옛 config 와도 호환.
    _use_dkd = getattr(config, "use_dkd", False)
    _dkd_a = getattr(config, "dkd_alpha", 1.0)
    _dkd_b = getattr(config, "dkd_beta", 1.0)
    # [변경 2026-08-17] alpha·T 를 하드코딩에서 config 로 뺐다. 값 자체는 그대로다
    # (0.7 / 4.0 은 컨퍼런스 코드 mark3.2 에서 검증한 값이고 mark5 는 그 손실 구조를 물려받았다).
    # mark4.x 가 쓰는 0.3 / 2.0 은 같은 샘플에 hard·soft 를 함께 거는 구조에서 soft 가 hard 를
    # 굶긴 문제를 잡느라 내린 값이므로, labeled→hard / unlabeled→soft 로 나뉜 mark5 에는
    # 그대로 옮겨오지 않는다. 다만 조정이 필요할 때 코드를 고치지 않고 config 만 바꾸도록 통로를 둔다.
    _alpha = getattr(config, "distill_alpha", 0.7)
    _temp = getattr(config, "distill_temperature", 4.0)
    # [추가 2026-08-22] pseudo-label CE 설정
    _use_pl = getattr(config, "use_pseudo_label_ce", False)
    _pl_min_conf = getattr(config, "pseudo_label_min_confidence", 0.5)
    _pl_weighting = getattr(config, "pseudo_label_confidence_weighting", True)
    _pl_min_gap = float(getattr(config, "pseudo_label_min_gap", 0.0) or 0.0)   # [신설 2026-09-21 밤 / B]
    if pseudo_csv is not None and _pl_min_gap > 0:
        # [Mark5_EXP2] CSV 로 들어온 가짜 정답(자기 학습·단일 generalist 출처 비교)은 합의 정보(1등−2등 차이)가 없다 — 조건을 끈다.
        print(f"[INFO] --pseudo_csv 경로: 1등−2등 차이 조건 {_pl_min_gap:g} 은 적용하지 않음(CSV 라벨엔 합의 정보가 없음)")
        _pl_min_gap = 0.0
    _distill_classify = getattr(config, "distill_branch_classify", False)
    if _m53_on:
        if pseudo_csv is not None:
            raise ValueError("[ERROR] --m53 은 전문가 8명의 확신도를 쓰므로 --pseudo_csv(CSV 가짜 정답) 경로와 함께 쓸 수 없습니다.")
        print(f"[M53] Mark5.3 부품 {'·'.join(_m53)}")
        if "PF" in _m53:
            _alpha = float(config.m53_pl_weight)
            _pl_weighting = False
            print(f"[M53/PF] 가짜 정답 CE 가중치 {_alpha:g}(옛 α 0.3), 확신도 가중 끔 — 합의 가짜 정답을 정답 라벨과 같은 세기로")
        if "SK" in _m53:
            print(f"[M53/SK] 합의에서 빠진 클립에 9칸 부드러운 목표: 전문가 확신도 1~{int(config.m53_soft_topk)}등 몫 + others 몫(1 − 최대 확신도), "
                  f"가중치 {float(config.m53_soft_weight):g}")
        if "OVA" in _m53:
            print(f"[M53/OVA] 전문가 이진 출력 {len(specialist_config)}개(MLP {config.embedding_dim}→{int(config.m53_ova_hidden)}→{len(specialist_config)}), "
                  f"BCE 가중치 {float(config.m53_ova_weight):g}, 평가 때 others 섞음 {float(config.m53_ova_eval_mix):g}")
        if "FM" in _m53:
            print(f"[M53/FM] 학생 EMA 사본(감쇠 {float(config.m53_fm_ema):g}) 무증강 답 → 증강 입력 CE, FreeMatch 클래스별 문턱, "
                  f"가중치 {float(config.m53_fm_weight):g}(처음 {int(config.m53_fm_rampup_epochs)}에폭 선형 증가)")
        if "CB" in _m53:
            print(f"[M53/CB] 가짜 정답 클래스 가중 = (평균 채택 수/클래스 채택 수)^{float(config.m53_cb_power):g} [0.5, 2.0]"
                  + (", FM 목표 분포 정렬(균등)" if "FM" in _m53 else ""))
        if "SC" in _m53:
            print(f"[M53/SC] 지도 대조 손실(정답 라벨 + 채택 가짜 정답, 온도 {float(config.m53_sc_temperature):g}) 가중치 {float(config.m53_sc_weight):g}")
    criterion = DistillationLoss(
        temperature=_temp, alpha=_alpha,
        feature_kd_weight=(config.feature_kd_weight if (config.use_feature_kd and pseudo_csv is None) else 0.0),
        use_dkd=_use_dkd, dkd_alpha=_dkd_a, dkd_beta=_dkd_b,
        use_pseudo_label_ce=_use_pl,
        hard_weight=getattr(config, "hard_weight", None))     # [HW 2026-09-29 밤] None 이면 옛 (1−α)
    if _use_pl:
        _mode_name = "pseudo-label CE"
    elif _use_dkd:
        _mode_name = "DKD"
    else:
        _mode_name = "vanilla KD"
    print(f"[INFO] KD 모드: {_mode_name}"
          + (f" (dkd_alpha={_dkd_a}, dkd_beta={_dkd_b}, 타겟축=teacher argmax)" if _use_dkd and not _use_pl else "")
          + (f" | 채택 임계값={_pl_min_conf}, 1등−2등 차이≥{_pl_min_gap:g}, 확신도 가중={_pl_weighting}" if _use_pl else f" | T={_temp}")
          + f" | alpha={_alpha}, feature_kd={config.feature_kd_weight if config.use_feature_kd else 0.0}"
          + f" | distill branch 분류손실={_distill_classify}")
    # [변경 2026-09-17] 백본 학습률 분리. config.encoder_learning_rate 가 있으면(mn10: 5e-5) 사전학습 백본만 그 값,
    # 새로 만드는 층(투영·branch·background)은 learning_rate(1e-4). None 이면 전부 learning_rate(옛 동작).
    # [검수 C5 / 2026-09-17] 투영층 proj(Dropout+Linear 960->384)는 Mn10AudioEncoder 안에 있어 전에는 인코더 그룹(5e-5)에
    # 같이 들어갔다. config 주석("투영은 1e-4")과 맞추려고 proj 만 골라 새 층 그룹으로 옮긴다.
    _enc_lr = getattr(config, "encoder_learning_rate", None)
    _head_params = list(student_branch.parameters())
    if background_embedding is not None:
        _head_params += list(background_embedding.parameters())
    # [수정 2026-10-01] 본체(Mark5.2 train_mark5.py)에는 있고 2026-09-26 이식에서 빠져 있던 줄. 이 줄이 없으면 분류 헤드의 파라미터가
    # 옵티마이저에 들어가지 않아 학습형 로짓 스케일(LT)이 초깃값 1/0.07 에서 한 번도 움직이지 않고(2026-10-01 의 70회 로그 전부 T=0.0700),
    # --linear_head 의 nn.Linear 도 무작위 초깃값 그대로 고정된다. 본체 로그는 '새 층 640,897개'(스케일 1개 포함), 이 줄이 없던 EXP2 는 640,896개.
    _head_params += list(student_classifier.parameters())   # 학습형 로짓 스케일·선형 헤드(텍스트 앵커+고정 T 면 빈 목록)
    if ova_head is not None:
        _head_params += list(ova_head.parameters())          # [Mark5.3 OVA] 새 층 학습률 그룹
    _enc_trainable = [p for p in student_encoder.parameters() if p.requires_grad]
    # [2026-09-29 밤 / LP-FT] 백본 = 인코더 파라미터 중 투영층(proj)을 뺀 것. 학습률 분리와 LP 고정이 같은 집합을 쓴다.
    _proj_ids = {id(p) for p in getattr(student_encoder, "proj", nn.Identity()).parameters()}
    _backbone_params = [p for p in _enc_trainable if id(p) not in _proj_ids]
    if _enc_lr is not None:
        _new_layer_params = [p for p in _enc_trainable if id(p) in _proj_ids] + _head_params
        optimizer_params = [{"params": _backbone_params, "lr": float(_enc_lr)},
                            {"params": _new_layer_params, "lr": config.learning_rate}]
        print(f"[INFO] 학습률: 백본 {float(_enc_lr):g} ({sum(p.numel() for p in _backbone_params):,}개) / "
              f"새 층(투영·branch·background) {config.learning_rate:g} ({sum(p.numel() for p in _new_layer_params):,}개)")
    else:
        optimizer_params = _enc_trainable + _head_params

    def _set_backbone_frozen(frozen: bool):
        """[LP-FT] 백본 파라미터의 requires_grad 를 끄고/켠다. Adam 은 grad 가 None 인 파라미터를 건너뛴다(weight decay 도)."""
        for p in _backbone_params:
            p.requires_grad_(not frozen)
    _backbone_module = getattr(student_encoder, "core", None)     # mn10 의 BN 통계를 LP 구간에 고정하기 위해(eval 모드)
    if _lp_epochs > 0 and _backbone_module is None:
        print("[WARN] LP-FT 는 mn10 백본(student_encoder.core)에서만 BN 통계를 고정합니다 — 이 인코더에는 core 가 없어 requires_grad 만 끕니다.")
    if _lp_epochs > 0:
        _set_backbone_frozen(True)
        print(f"[LP] 백본 고정 시작 — 학습 파라미터 {sum(p.numel() for p in student_encoder.parameters() if p.requires_grad) + sum(p.numel() for p in _head_params):,}개 "
              f"(백본 {sum(p.numel() for p in _backbone_params):,}개 제외)")
    # [변경 2026-08-17] weight_decay 와 LR 스케줄러를 mark4.x 와 같게 맞췄다.
    # mark4.x 는 weight_decay 가 전무해 teacher 가 val best 를 epoch 5 에 찍고 곧장 과적합하던 것을
    # 실측하고 1e-4 를 넣었고, ReduceLROnPlateau(factor 0.5, patience 3)도 함께 쓴다.
    # mark5 에는 둘 다 없어서 학습 절차가 달랐고, 그러면 mark4.x 결과와 같은 표에 올릴 수 없다.
    # [변경 2026-08-18] 기본값을 config 값(1e-4)과 같게 둔다. 0.0 이면 드라이브에 옛 config 가
    # 올라갔을 때 정규화 없이 조용히 학습된다.
    _weight_decay = getattr(config, "weight_decay", 1e-4)
    optimizer = optim.Adam(optimizer_params, lr=config.learning_rate, weight_decay=_weight_decay)
    # [신설 2026-10-07 / Mark5.3] FM 의 학생 EMA 사본(기울기 없음, 평가 모드)과 FreeMatch·분포 정렬·CB 의 누적 상태.
    _ema = None
    if "FM" in _m53:
        _ema = {"enc": copy.deepcopy(student_encoder), "branch": copy.deepcopy(student_branch), "cls": copy.deepcopy(student_classifier)}
        for _m in _ema.values():
            _m.eval()
            for _p in _m.parameters():
                _p.requires_grad_(False)
    _n_cls_student = len(config.get_classes_for_evaluation())
    _m53_pl_count = torch.zeros(_n_cls_student, device=device)                     # CB: 지금까지 채택된 가짜 정답의 클래스별 누적 수
    _fm_tau_g = 1.0 / _n_cls_student                                              # FM: FreeMatch 전역 문턱(평균 최대 확률의 EMA)
    _fm_p_local = torch.full((_n_cls_student,), 1.0 / _n_cls_student, device=device)   # FM: 클래스별 평균 확률의 EMA
    _da_mean = torch.full((_n_cls_student,), 1.0 / _n_cls_student, device=device)      # CB+FM: 분포 정렬용 EMA 평균 예측
    _m53_mom = float(getattr(config, "m53_fm_momentum", 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    _patience = getattr(config, "early_stopping_patience", 10)
    print(f"[INFO] 학습 절차: weight_decay={_weight_decay}, ReduceLROnPlateau(factor=0.5, patience=3), "
          f"early stopping patience={_patience}, best val 체크포인트 저장")
    # [추가 2026-08-30 / Mark5_EXP] 실험 조건을 시작 로그에 못박는다.
    # 25회를 돌리다 보면 "이 산출물이 어느 조건이었지"가 헷갈리므로, 로그만 봐도
    # run_tag 와 조건이 바로 확인되게 한다.
    print(f"[INFO] 실험 조건: seed={effective_seed}"
          f"{'(미지정 기본값)' if seed_value is None else ''}, "
          + (f"천장(진짜 라벨 {n_ceiling}개, unlabeled 미사용), " if is_ceiling else
             f"unlabeled={'전량' if unlabeled_n is None else unlabeled_n}개 (실제 {len(unlabeled_files)}개), ")
          + f"cudnn_deterministic={getattr(config, 'cudnn_deterministic', False)}, "
          f"run_tag={config.run_tag}")
    # [신설 2026-09-07] 이 run_tag 로 val 이 골라 둔 others 규칙 json 은 이전 체크포인트의 것이다. 새 학습이
    # 시작되면 지워서, test 가 옛 값을 읽는 일이 없게 한다(eval.py 는 지문도 대조한다 — 리뷰 지적 3).
    _stale = os.path.join(PROJECT_ROOT, "plots", f"others_threshold_{config.run_tag}.json")
    if os.path.exists(_stale):
        os.remove(_stale)
        print(f"[INFO] 이전 학습의 others 규칙 json 삭제: {_stale}")
    # [2026-09-18 / 3차 검수 P-3] 옛 완주 표식(loss_history)·체크포인트·val/test 요약도 지운다. 안 그러면 --force 재실행이 에폭 도중에
    # 죽었을 때 "새 중도 체크포인트 + 옛 완주 표식 + 옛 요약" 이 남아 run_experiments._needs 가 통째로 [건너뜀] 한다
    # (비교모델 common/train_loop.fit 이 09-05 에 고친 것과 같은 구멍).
    for _stale in (os.path.join(PROJECT_ROOT, "plots", f"loss_history_{config.run_tag}.csv"),
                   os.path.join(PROJECT_ROOT, "model", f"student_model_{config.run_tag}.pth"),
                   os.path.join(PROJECT_ROOT, "plots", f"performance_summary_{config.run_tag}_val.csv"),
                   os.path.join(PROJECT_ROOT, "plots", f"performance_summary_{config.run_tag}.csv")):
        if os.path.exists(_stale):
            os.remove(_stale)
            print(f"[INFO] 이전 학습의 산출물 삭제: {_stale}")

    train_hist, val_hist = [], []
    # [추가] train/val 직접 비교를 위해 컴포넌트별 손실 히스토리 분리 기록
    #  - train_hist        : Total(hard+soft+feat)  → 참고용
    #  - train_hard_hist   : raw hard CE            → val_hard_hist와 직접 비교 가능
    #  - train_soft/feat   : KD 컴포넌트            → 학습 진행 참고용
    #  - val_hard_hist     : val의 raw hard CE      → train_hard_hist와 직접 비교 가능
    train_hard_hist, train_soft_hist, train_feat_hist = [], [], []
    # [추가] background embedding 보조 loss(bg_loss) 히스토리
    train_bg_hist = []
    val_hard_hist = []
    print(f"[INFO] Student training ({mark_version}) started on {device}")
    # [추가 2026-08-17] best val 체크포인트 + early stopping 상태
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    stopped_early = False

    # [2026-09-29 밤 / teacher SpecAugment] teacher 입력이 클립(301프레임)이면 시간 마스크 폭을 프레임 수에 비례해 늘린다(16/101 → 48/301).
    _sa_t_teacher = _sa_t
    if _use_sa and _teacher_clip:
        _sa_t_teacher = int(round(_sa_t * float(_teacher_frames) / float(config.segment_length)))
        print(f"[INFO] teacher SpecAugment 시간 마스크 최대 폭 {_sa_t} → {_sa_t_teacher} (클립 {_teacher_frames}프레임 기준)")
    _others_idx_student = student_label_map["others"]
    _spec_names = list(specialist_config.keys())                               # fusion_output.specialist_weights 의 키 순서와 같게 쓴다
    _spec_cls_idx = torch.tensor([student_label_map[n] for n in _spec_names], dtype=torch.long, device=device)   # 전문가 → 학생 클래스 인덱스

    for epoch in range(config.num_epochs):
        student_encoder.train()
        student_branch.train()
        # [LP-FT] LP 구간: 백본 BN 통계 고정(eval 모드) · FT 진입 에폭: 백본 해제 + best/early stopping 초기화
        _in_lp = _lp_epochs > 0 and epoch < _lp_epochs
        if _in_lp and _backbone_module is not None:
            _backbone_module.eval()
        if _lp_epochs > 0 and epoch == _lp_epochs:
            _set_backbone_frozen(False)
            best_val, best_epoch, no_improve = float("inf"), -1, 0
            print(f"[FT] 백본 해제 — 에폭 {epoch + 1}부터 전체 파인튜닝. best val·early stopping 을 여기서 다시 잰다")
        total_loss = total_hard_loss = total_soft_loss = total_feat_loss = 0.0
        total_bg_loss = 0.0
        # [추가 2026-08-22] pseudo-label 채택 통계와 epoch 소요시간.
        total_distill_cls_loss = 0.0
        total_neg_loss = 0.0
        epoch_pseudo_total = 0
        epoch_pseudo_kept = 0
        epoch_pseudo_conf_sum = 0.0
        epoch_or_kept = 0          # [OR] 전원 거절로 others 가짜 정답이 된 샘플 수
        epoch_neg_samples = 0      # [NEG] 부정 라벨을 받은 샘플 수
        epoch_neg_labels = 0       # [NEG] 그 샘플들의 부정 라벨(클래스) 총수
        _m53_tot = {"SK": 0.0, "OVA": 0.0, "FM": 0.0, "SC": 0.0}   # [Mark5.3] 부품별 손실 합(에폭 로그용)
        _m53_sk_n = 0; _m53_fm_kept = 0.0; _m53_fm_n = 0
        epoch_started_at = time.time()
        # [추가 2026-08-17] 캐시를 에폭마다 비운다. teacher 는 eval·no_grad·가중치 고정이라
        # 같은 입력에 늘 같은 값을 내므로 에폭마다 쌓아봐야 같은 값이 num_epochs 벌 중복될 뿐이다.
        # 그대로 두면 80 에폭에서 리스트가 1GB 넘게 불어나고 torch.cat 이 그만큼 사본을 또 만든다.
        teacher_feature_cache.clear()
        unlabeled_metadata_cache.clear()

        for mel_batch, wave_batch, label_batch, metadata_batch in train_loader:
            mel = mel_batch.to(device)
            wave = wave_batch.to(device) if _student_input == "waveform" else None
            optimizer.zero_grad()

            labeled_indices = [i for i, lbl in enumerate(label_batch) if isinstance(lbl, str)]
            unlabeled_indices = [i for i, lbl in enumerate(label_batch) if lbl == -1]
            loss = torch.tensor(0.0, device=device)
            _sc_feats, _sc_lbls = [], []   # [Mark5.3 SC] 이 배치의 정답 라벨·채택 가짜 정답 특징
            # [신설 2026-10-08 / Mark5.3 JF] 정답·라벨 없는 클립을 학생에 한 번에 통과시키고(BN = 배치 16개 전체), 아래 정답 갈래와
            # Mark5.3 라벨 없는 갈래가 이 출력을 잘라 쓴다. 손실 식·증강(augment=_use_sa, 옛 두 호출과 같다)은 그대로다.
            # 옛 경로는 정답만 따로 통과시켜 정답 라벨 200 이면 지도 손실의 BN 이 1~2개 클립으로 계산됐다.
            _jf = None
            if _m53_jf and labeled_indices and unlabeled_indices:
                _jf_base = run_student_encoder(None, wave[labeled_indices + unlabeled_indices], augment=_use_sa)
                _jf_sup, _jf_dis = student_branch(_jf_base)
                _jf_logits = student_classifier(_jf_sup, student_text_emb)
                _jf_nl = len(labeled_indices)
                _jf = {"l_sup": _jf_sup[:_jf_nl], "l_logits": _jf_logits[:_jf_nl],
                       "u_sup": _jf_sup[_jf_nl:], "u_dis": _jf_dis[_jf_nl:], "u_logits": _jf_logits[_jf_nl:]}

            if labeled_indices:
                labeled_mel = mel[labeled_indices]
                # [추가 2026-08-23] SpecAugment. 정답 라벨은 그대로 두고 입력만 가린다.
                # [변경 2026-09-17] 파형 경로면 인코더가 자기 mel 에 건다(augment=_use_sa).
                if _use_sa and _student_input == "mel":
                    labeled_mel = apply_spec_augment(labeled_mel, _sa_nf, _sa_f, _sa_nt, _sa_t)
                labeled_targets = torch.tensor([student_label_map[label_batch[i]] for i in labeled_indices], dtype=torch.long, device=device)
                if _jf is not None:
                    supervised_features, student_logits = _jf["l_sup"], _jf["l_logits"]     # [JF] 한 번에 통과한 출력의 정답 부분
                else:
                    base_features = run_student_encoder(labeled_mel, wave[labeled_indices] if wave is not None else None, augment=_use_sa)
                    supervised_features, _ = student_branch(base_features)
                    student_logits = student_classifier(supervised_features, student_text_emb)
                hard_total, hard_loss, _, _ = criterion(student_logits, hard_targets=labeled_targets)
                loss = loss + hard_total
                total_hard_loss += hard_loss.item()
                if ova_head is not None:
                    # [Mark5.3 OVA] 정답 라벨 클립: 정답 클래스의 전문가 1, 나머지 0(others 는 전부 0).
                    _ova_t = (labeled_targets.view(-1, 1) == _spec_cls_idx.view(1, -1)).to(supervised_features.dtype)
                    _ova_l = F.binary_cross_entropy_with_logits(ova_head(supervised_features), _ova_t)
                    loss = loss + float(config.m53_ova_weight) * _ova_l
                    _m53_tot["OVA"] += _ova_l.item()
                if "SC" in _m53:
                    _sc_feats.append(supervised_features); _sc_lbls.append(labeled_targets)

                # [2026-10-01 / 지도만 격차 진단] 가중치 0(--bg_weight 0)이면 이 보조 손실을 아예 걸지 않는다(로그의 BG 칸은 0).
                if config.use_background_embedding and background_embedding is not None and float(config.background_embedding_weight) > 0:
                    others_idx = student_label_map["others"]
                    others_mask = (labeled_targets == others_idx)
                    if others_mask.any():
                        target_feat = F.normalize(supervised_features[others_mask], dim=1)
                        bg = F.normalize(background_embedding(), dim=0).unsqueeze(0).expand_as(target_feat)
                        bg_loss = (1 - F.cosine_similarity(target_feat, bg, dim=1)).mean()
                        loss = loss + config.background_embedding_weight * bg_loss
                        total_bg_loss += bg_loss.item()

            if unlabeled_indices:
                unlabeled_mel = mel[unlabeled_indices]
                # [신설 2026-09-29 / A3] 클립 모드: mel 은 [b, K, C, n_mels, T](클립당 teacher 조각 K개)라 teacher 용으로 [b*K, ...] 로 편다.
                # student 는 아래에서 클립 파형(wave, [b, 48000])만 쓴다. SpecAugment(teacher view)는 편 조각에 그대로 걸린다.
                if _clip_mode:
                    _n_u, _K = int(unlabeled_mel.shape[0]), int(unlabeled_mel.shape[1])
                    unlabeled_mel = unlabeled_mel.reshape(_n_u * _K, *unlabeled_mel.shape[2:])
                # [추가 2026-08-23] SpecAugment. teacher view 는 config 로 갈린다.
                #   augmented: teacher 도 증강 view 를 본다 — 진짜 online. 증강마다
                #              pseudo-label 이 재계산되고, 가려진 view 에서 앙상블 합의가
                #              깨지는 세그먼트는 아래 선별(keep_mask)이 걸러낸다.
                #   clean    : teacher 는 원본을 보고 student 만 증강 view 를 본다
                #              (FixMatch 식 일관성 정규화 — pseudo-label 품질 보존).
                # [변경 2026-09-17] teacher 는 어느 경우든 mel 을 본다. student 가 파형 경로면 student 의 증강은
                # 인코더 안에서 독립적으로 걸린다(mel 규격이 64 와 128 로 달라 같은 위치를 맞출 수 없다).
                if _use_sa:
                    aug_unlabeled_mel = apply_spec_augment(unlabeled_mel, _sa_nf, _sa_f, _sa_nt, _sa_t_teacher)
                    teacher_input = aug_unlabeled_mel if _sa_teacher_view == "augmented" else unlabeled_mel
                    student_input = aug_unlabeled_mel
                else:
                    teacher_input = unlabeled_mel
                    student_input = unlabeled_mel
                unlabeled_wave = wave[unlabeled_indices] if wave is not None else None
                if pseudo_lookup is not None:
                    # CSV 모드: one-hot 로짓(argmax 만 쓰이므로 충분)·확신도. 특징은 없다(feature KD 0).
                    _lab, _conf = [], []
                    for bi in unlabeled_indices:
                        _e = pseudo_lookup.get(os.path.basename(metadata_batch[bi]["path"]))
                        _lab.append(_e[0] if _e else 0); _conf.append(_e[1] if _e else 0.0)
                    _oh = F.one_hot(torch.tensor(_lab, device=device), num_classes=len(student_label_map)).float()
                    _cf = torch.tensor(_conf, device=device, dtype=torch.float32)
                    fusion_output = _CsvFusion(_oh, None, _cf)
                else:
                    fusion_output = ensemble_teacher(teacher_input, student_label_map)

                merged = None
                if _clip_mode and pseudo_lookup is None:
                    # [신설 2026-09-29 / A3] 조각 K개의 teacher 출력을 클립 하나로 합친다(규칙은 merge_segments_to_clip docstring).
                    # 이 아래의 keep_mask·pseudo_conf 는 클립 단위이고, 가짜 정답은 다수결 라벨(_t_labels)로 CE 를 건다.
                    # [2026-09-30 밤 / 가짜 정답 재료] rule="mean" 이면 전문가별 조각 확신도 [n*K, 8] 로 전문가별 평균을 내 클립을 채택한다(vote 는 이 값을 안 쓴다).
                    _sw_seg = torch.cat([fusion_output.specialist_weights[n].to(device).float() for n in _spec_names], dim=1)   # [n*K, 8]
                    merged = merge_segments_to_clip(
                        fusion_output, _n_u, _K,
                        min_conf=(_pl_min_conf if (_use_pl and _pl_min_conf > 0) else 0.0),
                        min_gap=(_pl_min_gap if (_use_pl and _pl_min_gap > 0) else 0.0),
                        min_segments=_clip_min_segments,
                        rule=_merge_rule, spec_probs=_sw_seg, spec_cls_idx=_spec_cls_idx)
                    pseudo_conf = merged["conf"]                          # [b] vote: 채택 조각 확신도 평균 / mean: 전문가별 조각 평균의 최댓값
                    keep_mask = merged["keep"]                            # [b] vote: 채택 조각 >= _clip_min_segments / mean: 평균 확신도·1등−2등 차이 조건
                    _t_logits, _t_feats, _t_labels = merged["logits"], merged["features"], merged["label"]
                else:
                    # (조각 모드, 또는 [Mark5_EXP2] CSV 가짜 정답 경로 — CSV 라벨은 이미 클립 단위라 클립 모드여도 합칠 조각이 없다.
                    #  _CsvFusion 의 uncertainty·disagreement 는 [b] 라 아래 식이 그대로 맞는다.)
                    # [신설 2026-08-22] 합의 기반 선별.
                    # specialist_weights[c] 는 전문가 c 가 "이건 내 클래스다"라고 본 확률이다.
                    # 그 최대값이 곧 "여덟 명 중 가장 강한 주장의 세기"이고, 이것이 낮으면
                    # 아무도 이 소리를 자기 것이라고 보지 않았다는 뜻이다. 그런 클립에 argmax 로
                    # 억지 정답을 붙이면 오답이 그대로 student 에게 주입되므로 학습에서 뺀다.
                    # (fusion_output.uncertainty 가 정확히 1 - 이 최대값이다.)
                    pseudo_conf = 1.0 - fusion_output.uncertainty       # [B]
                    if _use_pl and _pl_min_conf > 0:
                        keep_mask = pseudo_conf >= _pl_min_conf
                    else:
                        keep_mask = torch.ones_like(pseudo_conf, dtype=torch.bool)
                    # [신설 2026-09-21 밤 / B] 1등 전문가와 2등 전문가의 확신도 차이(fusion_output.disagreement = top1 − top2)가
                    # 작으면 "합의" 가 아니라 "둘이 다툼" 이다. 그런 세그먼트의 argmax 는 절반쯤 틀린다(vild_config 주석의 실측).
                    if _use_pl and _pl_min_gap > 0:
                        keep_mask = keep_mask & (fusion_output.disagreement.to(keep_mask.device) >= _pl_min_gap)
                    _t_logits, _t_feats, _t_labels = fusion_output.logits, fusion_output.features, None   # None → argmax(옛 동작)

                # [2026-09-29 밤 / OR·NEG] 전문가 8명의 확신도 행렬 [n, 8] (클립 모드는 조각 K개 평균, 조각 모드는 그대로).
                # [Mark5_EXP2] CSV 가짜 정답 경로(_CsvFusion)에는 전문가가 없다 → None. OR·NEG 는 그 경로에서 앞에서 막았다(ValueError).
                _sw = _spec_max = _reject = None
                if pseudo_lookup is None:
                    _sw = torch.stack([fusion_output.specialist_weights[n].to(device).float() for n in _spec_names], dim=1).squeeze(-1)   # [n*K 또는 n, 8]
                    # [수정 2026-10-01] specialist_weights[n] 는 [N,1] 이라 stack 이 [N,8,1] 을 만든다. 클립 모드는 아래 view 가 바로잡았지만 조각 모드
                    # (--segment_input)는 _spec_max 가 [N,1] → _reject 가 [N,N] 이 되어 OR·NEG 를 켜면 멈췄다. squeeze(-1) 로 [N,8]. 클립 모드 결과는 그대로.
                    if _clip_mode:
                        _sw = _sw.view(_n_u, _K, -1).mean(dim=1)                                                        # [n, 8]
                    _spec_max = _sw.max(dim=1).values                                                                    # [n]
                if _or_tau > 0:
                    # 전원 거절: 아무 전문가도 최대 확신도 τ_low 를 넘기지 못한 샘플 → others 가짜 정답(가중치 = 1 − 최대 확신도).
                    _reject = (~keep_mask) & (_spec_max < _or_tau)
                    if _reject.any():
                        if _t_labels is None:
                            _t_labels = _t_logits.argmax(dim=1)
                        _t_labels = _t_labels.clone(); _t_labels[_reject] = _others_idx_student
                        pseudo_conf = pseudo_conf.clone(); pseudo_conf[_reject] = (1.0 - _spec_max[_reject]).to(pseudo_conf.dtype)
                        if _clip_mode:
                            # 채택 조각이 없어 0 이던 특징·로짓을 조각 단순 평균으로 채운다(feature KD 목표가 0 벡터가 되지 않게).
                            _t_feats = _t_feats.clone(); _t_feats[_reject] = fusion_output.features.view(_n_u, _K, -1)[_reject].mean(dim=1)
                            _t_logits = _t_logits.clone(); _t_logits[_reject] = fusion_output.logits.view(_n_u, _K, -1)[_reject].mean(dim=1)
                        keep_mask = keep_mask | _reject
                        epoch_or_kept += int(_reject.sum().item())

                epoch_pseudo_total += int(keep_mask.numel())
                epoch_pseudo_kept += int(keep_mask.sum().item())
                epoch_pseudo_conf_sum += float(pseudo_conf.sum().item())

                if (not _m53_on) and _use_neg and (~keep_mask).any():
                    # [NEG] 미채택 샘플: 전문가 c 가 "내 것 아님" 을 확신(확신도 < τ_neg)하는 클래스마다 −β·log(1 − p_student(c)).
                    _neg_idx = (~keep_mask).nonzero(as_tuple=True)[0]
                    _neg_cls = (_sw[_neg_idx] < _neg_tau)                                                              # [m, 8]
                    if _neg_cls.any():
                        _nm = None if _clip_mode else student_input[_neg_idx]
                        _nw = unlabeled_wave[_neg_idx] if unlabeled_wave is not None else None
                        _nf = run_student_encoder(_nm, _nw, augment=_use_sa)
                        _nsup, _ = student_branch(_nf)
                        _np = torch.softmax(student_classifier(_nsup, student_text_emb), dim=1)                          # [m, C]
                        _p_neg = _np[:, _spec_cls_idx]                                                                   # [m, 8] 전문가 클래스만
                        _neg_loss = (-torch.log((1.0 - _p_neg).clamp_min(1e-6)) * _neg_cls.float()).sum() / _neg_cls.float().sum().clamp_min(1.0)
                        loss = loss + _neg_beta * _neg_loss
                        total_neg_loss += _neg_loss.item()
                        epoch_neg_samples += int(_neg_cls.any(dim=1).sum().item())
                        epoch_neg_labels += int(_neg_cls.sum().item())

                if (not _m53_on) and keep_mask.any():
                    kept_mel = None if _clip_mode else student_input[keep_mask]     # [A3] 클립 모드의 student 는 파형만 쓴다
                    kept_wave = unlabeled_wave[keep_mask] if unlabeled_wave is not None else None
                    kept_teacher_logits = _t_logits[keep_mask]
                    kept_teacher_feats = _t_feats[keep_mask] if _t_feats is not None else None   # CSV 모드(_CsvFusion)는 features 가 None → feature KD 0 (검수 C2)
                    kept_targets = _t_labels[keep_mask] if _t_labels is not None else None   # [A3] 클립 다수결 라벨(조각 모드·CSV 모드 None → argmax)
                    kept_weights = pseudo_conf[keep_mask] if (_use_pl and _pl_weighting) else None

                    base_features = run_student_encoder(kept_mel, kept_wave, augment=_use_sa)
                    supervised_features, distill_features = student_branch(base_features)
                    student_logits = student_classifier(supervised_features, student_text_emb)
                    soft_total, _, soft_loss, feat_loss = criterion(
                        student_logits,
                        teacher_logits=kept_teacher_logits,
                        student_features=distill_features,
                        teacher_features=kept_teacher_feats,
                        pseudo_weights=kept_weights,
                        pseudo_targets=kept_targets,
                    )
                    loss = loss + soft_total
                    total_soft_loss += soft_loss.item()
                    total_feat_loss += feat_loss.item()

                    # [신설 2026-08-22] distill branch 에도 분류 손실을 건다.
                    # 이 갈래는 지금까지 feature KD 만 받아 분류를 배운 적이 없는데,
                    # eval.py 는 distill_branch_eval_weight=0.5 로 최종 확률의 절반을
                    # 이 갈래에서 가져간다. 분류를 배운 적 없는 표현으로 분류를 하고 있었던 셈이다.
                    if _use_pl and _distill_classify:
                        distill_logits = student_classifier(distill_features, student_text_emb)
                        db_total, _, db_loss, _ = criterion(
                            distill_logits,
                            teacher_logits=kept_teacher_logits,
                            pseudo_weights=kept_weights,
                            pseudo_targets=kept_targets,
                        )
                        loss = loss + db_total
                        total_distill_cls_loss += db_loss.item()
                if _m53_on:
                    # [신설 2026-10-07 / Mark5.3] unlabeled 를 학생에 한 번만 통과시키고(증강 입력, 옛 경로와 같은 augment) 아래 손실이 모두 이 출력을 쓴다.
                    # NEG·채택 가짜 정답 CE·증류 갈래 CE·특징 증류는 옛 식 그대로이고, 가중치(PF·CB)와 새 손실(SK·OVA·FM·SC)만 다르다.
                    if _jf is not None:
                        _u_sup, _u_dis, _u_logits = _jf["u_sup"], _jf["u_dis"], _jf["u_logits"]   # [JF] 한 번에 통과한 출력의 라벨 없는 부분
                    else:
                        _u_base = run_student_encoder(None if _clip_mode else student_input, unlabeled_wave, augment=_use_sa)
                        _u_sup, _u_dis = student_branch(_u_base)
                        _u_logits = student_classifier(_u_sup, student_text_emb)
                    _cbw = None
                    if "CB" in _m53:
                        _seen = _m53_pl_count > 0
                        _cbw = torch.ones_like(_m53_pl_count)
                        if bool(_seen.any()):
                            _cbw[_seen] = (_m53_pl_count[_seen].mean() / _m53_pl_count[_seen]).pow(float(config.m53_cb_power)).clamp(0.5, 2.0)
                    if _use_neg and _sw is not None and (~keep_mask).any():
                        _neg_idx = (~keep_mask).nonzero(as_tuple=True)[0]
                        _neg_cls = (_sw[_neg_idx] < _neg_tau)
                        if _neg_cls.any():
                            _p_neg = torch.softmax(_u_logits[_neg_idx], dim=1)[:, _spec_cls_idx]
                            _neg_loss = (-torch.log((1.0 - _p_neg).clamp_min(1e-6)) * _neg_cls.float()).sum() / _neg_cls.float().sum().clamp_min(1.0)
                            loss = loss + _neg_beta * _neg_loss
                            total_neg_loss += _neg_loss.item()
                            epoch_neg_samples += int(_neg_cls.any(dim=1).sum().item())
                            epoch_neg_labels += int(_neg_cls.sum().item())
                    if keep_mask.any():
                        kept_teacher_logits = _t_logits[keep_mask]
                        kept_teacher_feats = _t_feats[keep_mask] if _t_feats is not None else None
                        kept_targets = _t_labels[keep_mask] if _t_labels is not None else kept_teacher_logits.argmax(dim=1)
                        kept_weights = pseudo_conf[keep_mask] if (_use_pl and _pl_weighting) else None
                        if _cbw is not None:
                            _w_cls = _cbw[kept_targets]
                            kept_weights = _w_cls if kept_weights is None else kept_weights * _w_cls
                            _m53_pl_count += torch.bincount(kept_targets, minlength=_n_cls_student).to(_m53_pl_count.dtype)
                        soft_total, _, soft_loss, feat_loss = criterion(
                            _u_logits[keep_mask], teacher_logits=kept_teacher_logits,
                            student_features=_u_dis[keep_mask], teacher_features=kept_teacher_feats,
                            pseudo_weights=kept_weights, pseudo_targets=kept_targets)
                        loss = loss + soft_total
                        total_soft_loss += soft_loss.item()
                        total_feat_loss += feat_loss.item()
                        if _use_pl and _distill_classify:
                            db_total, _, db_loss, _ = criterion(
                                student_classifier(_u_dis[keep_mask], student_text_emb), teacher_logits=kept_teacher_logits,
                                pseudo_weights=kept_weights, pseudo_targets=kept_targets)
                            loss = loss + db_total
                            total_distill_cls_loss += db_loss.item()
                        if "SC" in _m53:
                            _sc_feats.append(_u_sup[keep_mask]); _sc_lbls.append(kept_targets)
                    if "SK" in _m53 and _sw is not None and (~keep_mask).any():
                        # [Mark5.3 SK] 9칸 부드러운 목표: others 몫 o = 1 − 최대 확신도, 나머지 (1−o) 를 1~k등 전문가 확신도 비율로 나눈다.
                        _sk_idx = (~keep_mask).nonzero(as_tuple=True)[0]
                        _p = _sw[_sk_idx].clamp(0.0, 1.0)
                        _tv, _ti = torch.topk(_p, k=min(int(config.m53_soft_topk), _p.shape[1]), dim=1)
                        _o = (1.0 - _p.max(dim=1).values).clamp(0.0, 1.0)
                        _share = _tv / _tv.sum(dim=1, keepdim=True).clamp_min(1e-6) * (1.0 - _o).unsqueeze(1)
                        _q = torch.zeros(_sk_idx.numel(), _u_logits.shape[1], device=_u_logits.device, dtype=_u_logits.dtype)
                        _q.scatter_(1, _spec_cls_idx[_ti], _share.to(_q.dtype))
                        _q[:, _others_idx_student] = _o.to(_q.dtype)
                        _ce = -(_q * F.log_softmax(_u_logits[_sk_idx], dim=1)).sum(dim=1)
                        _w = _cbw[_q.argmax(dim=1)] if _cbw is not None else torch.ones_like(_ce)
                        _sk_l = (_ce * _w).sum() / _w.sum().clamp_min(1e-6)
                        loss = loss + float(config.m53_soft_weight) * _sk_l
                        _m53_tot["SK"] += _sk_l.item(); _m53_sk_n += int(_sk_idx.numel())
                    if ova_head is not None and _sw is not None:
                        # [Mark5.3 OVA] unlabeled 전부: 전문가 8명의 확신도(조각 평균)를 이진 출력이 따라 하게 한다.
                        _ova_l = F.binary_cross_entropy_with_logits(ova_head(_u_sup), _sw.clamp(0.0, 1.0).to(_u_sup.dtype))
                        loss = loss + float(config.m53_ova_weight) * _ova_l
                        _m53_tot["OVA"] += _ova_l.item()
                    if _ema is not None:
                        # [Mark5.3 FM] EMA 사본이 무증강 파형으로 낸 답(+CB 면 분포 정렬) → FreeMatch 클래스별 문턱을 넘는 것만 학생 증강 출력의 CE 목표로.
                        with torch.no_grad():
                            _eb = _ema["enc"](unlabeled_wave, augment=False)
                            _es, _ = _ema["branch"](_eb)
                            _pw = torch.softmax(_ema["cls"](_es, student_text_emb), dim=1)
                            if "CB" in _m53:
                                _da_mean.mul_(_m53_mom).add_(_pw.mean(dim=0), alpha=1.0 - _m53_mom)
                                _pw = _pw * (1.0 / _n_cls_student) / _da_mean.clamp_min(1e-6)
                                _pw = _pw / _pw.sum(dim=1, keepdim=True)
                            _maxp, _tgt = _pw.max(dim=1)
                            _fm_tau_g = _m53_mom * _fm_tau_g + (1.0 - _m53_mom) * float(_maxp.mean().item())
                            _fm_p_local.mul_(_m53_mom).add_(_pw.mean(dim=0), alpha=1.0 - _m53_mom)
                            _mask = (_maxp >= (_fm_tau_g * _fm_p_local / _fm_p_local.max())[_tgt]).float()
                        _ramp = min(1.0, float(epoch) / max(1, int(config.m53_fm_rampup_epochs)))
                        _fm_l = (F.cross_entropy(_u_logits, _tgt, reduction="none") * _mask).mean()
                        if _ramp > 0:
                            loss = loss + float(config.m53_fm_weight) * _ramp * _fm_l
                        _m53_tot["FM"] += _fm_l.item(); _m53_fm_kept += float(_mask.sum().item()); _m53_fm_n += int(_mask.numel())
                _fc = _t_feats if merged is not None else fusion_output.features
                if _fc is not None:                                   # CSV 모드(_CsvFusion)는 features 가 None → 캐시 없음
                    teacher_feature_cache.append(_fc.detach().cpu())
                for local_pos, batch_idx in enumerate(unlabeled_indices):
                    meta = dict(metadata_batch[batch_idx])
                    if merged is not None:
                        # [A3] 클립 단위 기록: 조각 K개의 평균 차이·불확실성, 전문가 가중치의 조각 평균, 채택 조각 수, 다수결 라벨.
                        meta["teacher_disagreement"] = float(merged["seg_gap"][local_pos].mean().item())
                        meta["teacher_uncertainty"] = float((1.0 - merged["seg_conf"][local_pos]).mean().item())
                        meta["teacher_specialist_weights"] = {
                            name: float(weight.view(_n_u, _K)[local_pos].mean().item())
                            for name, weight in fusion_output.specialist_weights.items()
                        }
                        meta["n_segments_kept"] = int(merged["n_kept"][local_pos].item())
                        meta["pseudo_label_index"] = int(_t_labels[local_pos].item())          # [OR] 전원 거절 others 까지 반영된 라벨
                    else:
                        meta["teacher_disagreement"] = float(fusion_output.disagreement[local_pos].item())
                        meta["teacher_uncertainty"] = float(fusion_output.uncertainty[local_pos].item())
                        meta["teacher_specialist_weights"] = {
                            name: float(weight[local_pos].item())
                            for name, weight in fusion_output.specialist_weights.items()
                        }
                    meta["pseudo_from_reject"] = bool(_reject is not None and _reject[local_pos].item())   # [OR] 이 샘플이 전원 거절로 others 가 됐나
                    meta["specialist_max_conf"] = (float(_spec_max[local_pos].item())                      # [OR·NEG] 전문가 8명 최대 확신도
                                                   if _spec_max is not None else float("nan"))              # (CSV 경로는 전문가 없음)
                    # [추가 2026-08-22] 이 세그먼트(클립 모드면 클립)가 pseudo-label 로 채택됐는지와 그때의 확신도.
                    # 나중에 "어떤 소리가 계속 탈락하는가"를 분석할 재료가 된다.
                    meta["pseudo_confidence"] = float(pseudo_conf[local_pos].item())
                    meta["pseudo_kept"] = bool(keep_mask[local_pos].item())
                    unlabeled_metadata_cache.append(meta)

            if "SC" in _m53 and _sc_feats:
                _sc_l = _supcon_loss(torch.cat(_sc_feats), torch.cat(_sc_lbls), float(config.m53_sc_temperature))
                if _sc_l is not None:
                    loss = loss + float(config.m53_sc_weight) * _sc_l
                    _m53_tot["SC"] += _sc_l.item()
            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if _ema is not None:
                    # [Mark5.3 FM] EMA 갱신: 파라미터는 감쇠 평균, BN 통계 등 버퍼는 그대로 복사.
                    with torch.no_grad():
                        _d = float(config.m53_fm_ema)
                        for _k, _src in (("enc", student_encoder), ("branch", student_branch), ("cls", student_classifier)):
                            for _pe, _ps in zip(_ema[_k].parameters(), _src.parameters()):
                                _pe.mul_(_d).add_(_ps.detach(), alpha=1.0 - _d)
                            for _be, _bs in zip(_ema[_k].buffers(), _src.buffers()):
                                _be.copy_(_bs)

        avg_loss = total_loss / max(1, len(train_loader))
        avg_hard = total_hard_loss / max(1, len(train_loader))
        avg_soft = total_soft_loss / max(1, len(train_loader))
        avg_feat = total_feat_loss / max(1, len(train_loader))
        avg_bg = total_bg_loss / max(1, len(train_loader))
        train_hist.append(avg_loss)
        train_hard_hist.append(avg_hard)
        train_soft_hist.append(avg_soft)
        train_feat_hist.append(avg_feat)
        train_bg_hist.append(avg_bg)

        student_encoder.eval()
        student_branch.eval()
        val_total = 0.0
        val_hard_total = 0.0
        with torch.no_grad():
            for mel_batch, wave_batch, label_batch, _ in val_loader:
                mel = mel_batch.to(device)
                targets = torch.tensor([student_label_map[lbl] for lbl in label_batch], dtype=torch.long, device=device)
                base_features = run_student_encoder(mel, wave_batch.to(device) if _student_input == "waveform" else None, augment=False)
                supervised_features, _ = student_branch(base_features)
                logits = student_classifier(supervised_features, student_text_emb)
                # [수정] total((1-α)·hard)뿐 아니라 raw hard CE도 따로 기록해 train_hard와 동일 척도로 비교
                val_loss, val_hard, _, _ = criterion(logits, hard_targets=targets)
                val_total += val_loss.item()
                val_hard_total += val_hard.item()
        avg_val = val_total / max(1, len(val_loader))
        avg_val_hard = val_hard_total / max(1, len(val_loader))
        val_hist.append(avg_val)
        val_hard_hist.append(avg_val_hard)

        # [추가 2026-08-22] pseudo-label 채택률·평균 확신도·epoch 소요시간.
        # 채택률이 계속 낮으면(예: 0.3 미만) unlabeled 를 대부분 버리고 있다는 뜻이라
        # pseudo_label_min_confidence 를 내려야 한다. 반대로 1.0 에 붙어 있으면 선별이
        # 아무 일도 안 하고 있다는 뜻이다.
        _keep_ratio = (epoch_pseudo_kept / epoch_pseudo_total) if epoch_pseudo_total else 0.0
        _mean_conf = (epoch_pseudo_conf_sum / epoch_pseudo_total) if epoch_pseudo_total else 0.0
        _elapsed = time.time() - epoch_started_at
        _extra = ""
        if _use_pl:
            _extra = f" | PL채택 {_keep_ratio:.3f}({epoch_pseudo_kept}/{epoch_pseudo_total}) conf {_mean_conf:.3f}"
            if _distill_classify:
                _avg_db = total_distill_cls_loss / max(1, len(train_loader))
                _extra += f" | DistillCls {_avg_db:.4f}"
        if _or_tau > 0:
            _extra += f" | OR→others {epoch_or_kept}"
        if _use_neg:
            _extra += f" | NEG {total_neg_loss / max(1, len(train_loader)):.4f}({epoch_neg_samples}샘플·{epoch_neg_labels}라벨)"
        if _lp_epochs > 0:
            _extra += " | LP" if _in_lp else " | FT"
        if getattr(student_classifier, "learnable", False):
            _extra += f" | T={student_classifier.temperature:.4f}"   # [LT] 학습형 스케일이 움직이는지 본다
        if _m53_on:
            _nb = max(1, len(train_loader))
            _extra += " | M53 " + " ".join(f"{k} {_m53_tot[k] / _nb:.4f}" for k in ("SK", "OVA", "FM", "SC") if k in _m53)
            if "SK" in _m53:
                _extra += f" SK클립 {_m53_sk_n}"
            if "FM" in _m53:
                _extra += f" FM통과 {(_m53_fm_kept / max(1, _m53_fm_n)):.3f} τ {_fm_tau_g:.3f}"
            if "CB" in _m53 and bool((_m53_pl_count > 0).any()):
                _extra += " CB채택 " + "/".join(str(int(x)) for x in _m53_pl_count.tolist())
        print(f"[Epoch {epoch+1}] Total {avg_loss:.4f} | Hard {avg_hard:.4f} | Soft {avg_soft:.4f} | Feat {avg_feat:.4f} | BG {avg_bg:.4f} | Val(hard*) {avg_val_hard:.4f}"
              + _extra + f" | {_elapsed:.1f}s")

        # [추가 2026-08-17] LR 스케줄 · best 저장 · early stopping (mark4.x 와 같은 절차)
        scheduler.step(avg_val)
        if avg_val < best_val:
            print(f"[EarlyStopping] Val loss {best_val:.6f} -> {avg_val:.6f}. Saving...")
            best_val = avg_val
            best_epoch = epoch + 1
            no_improve = 0
            save_checkpoint(
                os.path.join(BASE_DIR, f"student_model_{config.run_tag}.pth"),
                model_type="student_full",
                mark_version=mark_version,
                model_state=student_encoder.state_dict(),
                branch_state=student_branch.state_dict(),
                classifier_state=student_classifier.state_dict(),
                background_state=background_embedding.state_dict() if background_embedding is not None else None,
                # [추가 2026-08-17] config 를 넘겨 config_metadata 를 남긴다. 전에는 인자가 빠져 있어
                # mark5 산출물만 provenance 가 빈 dict 였다(teacher 파일에는 10개가 들어 있다).
                config=config,
                extra_states=({"ova_state_dict": ova_head.state_dict()} if ova_head is not None else None),   # [Mark5.3 OVA]
            )
        elif _in_lp:
            print(f"[EarlyStopping] LP 구간 — counter 를 세지 않는다 ({epoch + 1}/{_lp_epochs})")
        else:
            no_improve += 1
            print(f"[EarlyStopping] counter: {no_improve}/{_patience}")
            if no_improve >= _patience:
                stopped_early = True
                print(f"[INFO] Early stopping. best epoch {best_epoch} (val {best_val:.6f})")
                break

    print("Training finished."
          + (f" (early stop, best epoch {best_epoch}/{config.num_epochs}, val {best_val:.6f})" if stopped_early
             else f" (best epoch {best_epoch}/{config.num_epochs}, val {best_val:.6f})"))
    # [변경 2026-08-18] 저장된 체크포인트가 없는 경우의 처리를 손실곡선 저장 뒤로 미룬다.
    # 여기서 바로 raise 하면(예: val loss 가 첫 에폭부터 NaN 이라 한 번도 개선되지 않은 경우)
    # 이번 수정으로 지키려던 손실 히스토리를 또 잃는다. 원인 분석에 필요한 것이 바로 그 곡선이다.
    no_checkpoint = best_epoch < 0
    if no_checkpoint:
        print("[ERROR] val 이 한 번도 개선되지 않아 저장된 체크포인트가 없습니다. "
              "손실 곡선만 저장하고 종료합니다(val loss 가 NaN 인지 확인하십시오).")

    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # [수정] train 총손실(hard+soft+feat)과 val(hard만)을 한 축에서 직접 비교하면
    #        측정 대상이 달라 과적합을 오판할 수 있음.
    #   -> 직접 비교 가능한 'Train Hard(raw CE) vs Val Hard(raw CE)'를 주 곡선(굵게)으로,
    #      Train Total/Soft/Feat은 학습 진행 참고용(점/파선)으로 함께 표기.
    # [2026-09-06] 그림 안의 글자는 전부 영문으로 둔다. 코랩·서버에는 한글 폰트가 없어
    # 한글 라벨이 네모(□)로 깨진 채 저장되고, 논문 그림은 어차피 영문이다.
    plt.figure(figsize=(9, 6))
    plt.plot(train_hist, label="Train total (hard+soft+feat, reference)", linestyle="--", alpha=0.55)
    plt.plot(train_soft_hist, label="Train soft KD (reference)", linestyle=":", alpha=0.55)
    plt.plot(train_feat_hist, label="Train feature KD (reference)", linestyle=":", alpha=0.55)
    plt.plot(train_hard_hist, label="Train hard (raw CE)", linewidth=2)
    plt.plot(val_hard_hist, label="Val hard (raw CE)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Student loss ({config.run_tag})\nCompare train hard vs. val hard (same raw CE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"loss_curve_student_{config.run_tag}.png"))

    # [추가] 손실곡선 raw 숫자를 CSV로도 저장. PNG만 있으면 다른 모델(CED-Tiny 등)과
    # 겹쳐 그리는 비교 그래프를 다시 만들 수 없어서, 비교 실험용으로 숫자 그대로 남긴다.
    loss_history_csv = os.path.join(plot_dir, f"loss_history_{config.run_tag}.csv")
    with open(loss_history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "train_hard", "train_soft", "train_feat", "train_bg", "val_total", "val_hard"])
        for i in range(len(train_hist)):
            writer.writerow([
                i + 1, train_hist[i], train_hard_hist[i], train_soft_hist[i],
                train_feat_hist[i], train_bg_hist[i], val_hist[i], val_hard_hist[i],
            ])
    print(f"[INFO] 손실곡선 CSV 저장: {loss_history_csv}")

    # [이동·수정 2026-08-17] teacher 특징·메타데이터 캐시 저장.
    # 전에는 이 블록이 손실곡선 저장보다 앞에 있었고 cache 디렉터리를 아무도 만들지 않아서,
    # 80 에폭을 다 돌고 여기서 "Parent directory ... does not exist" 로 죽으면서
    # 손실곡선 PNG·CSV 를 통째로 잃었다(학습·체크포인트는 이미 끝난 뒤라 더 아깝다).
    # 그래서 (1) 디렉터리를 만들고 (2) 진단용인 이 저장을 맨 뒤로 돌리고 (3) 실패해도
    # 학습 결과를 잃지 않도록 경고만 남기고 넘어간다.
    try:
        cache_dir = os.path.join(PROJECT_ROOT, "cache", mark_version)
        os.makedirs(cache_dir, exist_ok=True)
        if teacher_feature_cache:
            torch.save(torch.cat(teacher_feature_cache, dim=0), get_feature_cache_path(PROJECT_ROOT, mark_version, "unlabeled_train"))
        if unlabeled_metadata_cache:
            torch.save(unlabeled_metadata_cache, get_metadata_cache_path(PROJECT_ROOT, mark_version, "unlabeled_train"))
        if teacher_feature_cache or unlabeled_metadata_cache:
            print(f"[INFO] teacher 캐시 저장: {cache_dir} (마지막 에폭분)")
    except Exception as e:
        print(f"[WARN] teacher 캐시 저장 실패(학습 결과에는 영향 없음): {e}")

    if no_checkpoint:
        raise RuntimeError(
            "[ERROR] val 이 한 번도 개선되지 않아 student_model 체크포인트가 없습니다. "
            f"손실 곡선은 {plot_dir} 에 저장했습니다."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark5.2 Student 모델을 학습합니다(반복실험판).")
    parser.add_argument("--mark_version", type=str, default="mark5.2")   # [변경 2026-09-17] 기본 mark5.2
    # [변경 2026-08-30 / Mark5_EXP] default 42 -> None.
    # 값을 주면 그 seed 로 돌고 산출물 이름에 _s<k> 가 붙는다(반복실험).
    # 안 주면 예전처럼 seed 42 로 돌고 태그도 안 붙는다(기존 동작 보존).
    parser.add_argument("--seed", type=int, default=None,
                        help="반복실험용 seed. 지정하면 산출물 이름에 _s<k> 가 붙는다. 미지정 시 42.")
    # [신설 2026-08-30 / Mark5_EXP] 데이터효율곡선용 unlabeled 개수.
    parser.add_argument("--unlabeled_n", type=int, default=None,
                        help="unlabeled 를 몇 개만 쓸지(0/500/1000/1500/2000). "
                             "같은 seed 안에서 누적(nested)으로 뽑는다. 미지정 시 전량.")
    # [신설 2026-09-06] 천장 조건. CSV 는 model/make_ceiling_labels.py 가 만든다.
    parser.add_argument("--ceiling_csv", type=str, default=None,
                        help="천장(upper bound) 조건: unlabeled 에 진짜 라벨을 붙인 CSV 경로(프로젝트 루트 기준). "
                             "주면 unlabeled 를 쓰지 않고 지도학습으로 돌며 run_tag 에 _ceil 이 붙는다.")
    # [신설 2026-09-17 / 8절] 라벨 효율 표용 labeled 개수.
    parser.add_argument("--labeled_n", type=int, default=None,
                        help="labeled 를 몇 클립만 쓸지(200/500/1000). 클래스별 고르게, seed 기준 누적. 미지정 시 전량(2,000).")
    # [신설 2026-09-17 / 8절] pseudo-label 출처 비교·unlabeled 풀 확장.
    parser.add_argument("--pseudo_csv", type=str, default=None,
                        help="teacher 대신 쓸 pseudo-label CSV(path,label,confidence). model/make_pseudo_labels.py 가 만든다.")
    parser.add_argument("--pseudo_source", type=str, default=None,
                        help="--pseudo_csv 의 출처 이름(self | gen). run_tag 에 _ps<이름> 이 붙는다.")
    parser.add_argument("--spill_labeled", action="store_true",
                        help="--labeled_n 으로 뽑고 남은 labeled 를 라벨 없이 unlabeled 풀에 합친다(태그 _sp).")
    # [신설 2026-09-25 / LR] 학습률 선정(model/select_lr.py)용. 안 주면 config 기본값(새 층 1e-4 · 백본 5e-5)·옛 태그 그대로.
    # run_experiments.py 는 plots/lr_selected.json 이 있으면 이 두 값을 자동으로 넘긴다.
    parser.add_argument("--lr", type=float, default=None,
                        help="[LR 2026-09-25] 새로 만드는 층(투영·branch·background·로짓 스케일)의 학습률. 기본값과 다르면 run_tag 에 _lr<값>")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="[LR 2026-09-25] 사전학습 백본(mn10)의 학습률. 기본값과 다르면 run_tag 에 _elr<값>. eval 에도 같은 값을 줄 것")
    # [신설 2026-09-28 / ablation A1·A2·A4] 우리 부품 끄기. 셋 다 eval.py 에도 같은 플래그를 줘야 체크포인트(run_tag)를 찾는다.
    parser.add_argument("--no_feature_kd", action="store_true",
                        help="[A1] 특징 증류(feature KD, 가중치 0.3)를 끈다. run_tag 에 _noFKD")
    parser.add_argument("--pl_min_gap", type=float, default=None,
                        help="[A2] 가짜 정답 채택의 1등−2등 차이 조건(기본 0.4). 0 이면 조건을 끈다(run_tag 에 _noGAP, _G04 는 안 붙음)")
    parser.add_argument("--no_distill_branch", action="store_true",
                        help="[A4] 증류 갈래를 끈다 — 분류 손실을 걸지 않고 eval 에서 w=0 고정(지도 헤드만). run_tag 에 _noDB")
    # [신설 2026-09-29 / ablation A3] student 가 1초 조각 5개 대신 클립 3초 파형을 통째로 본다. teacher 는 조각 그대로(클립 규칙으로 합침).
    parser.add_argument("--clip_input", action="store_true",
                        help="[A3] 세그먼트 분할을 끈다 — student 입력 = 클립 3초 파형 통째, teacher 조각 출력은 다수결로 합침. run_tag 에 _noSEG. eval 에도 같이 줄 것")
    # [신설 2026-09-29 / 헤드 진단 H] 텍스트 앵커 헤드를 nn.Linear(384→9) 로 바꾼다(비교모델 헤드와 같은 방식). 제로샷이 없어진다. [2026-10-07] 지도만 격차 진단 D0 로 본 레시피에 채택.
    parser.add_argument("--linear_head", action="store_true",
                        help="[H] 텍스트 앵커 헤드 대신 nn.Linear 분류 헤드(2026-10-07 D0 본 레시피, 제로샷 불가). run_tag 에 _LIN. eval 에도 같이 줄 것")
    # [신설 2026-09-29 밤 / 격차 분석 대응 묶음] 전부 켜는 것이 실험 기본. 하나씩 끄는 것은 묶음 뒤 ablation 용. eval 에도 같은 플래그를 줄 것.
    parser.add_argument("--segment_input", action="store_true",
                        help="mark5.2 기본(클립 입력)을 끄고 조각(1초×5) 입력으로 되돌린다(2026-09-28 이전 본 모델과 같은 이름). 클립 teacher 와는 함께 못 씀")
    parser.add_argument("--head_pretrained_hidden", action="store_true",
                        help="[PH] mn10 사전학습 은닉층 유지 → Linear(960→384) 투영 하나 → 앵커, branch LN+ReLU 제거. run_tag _PH")
    parser.add_argument("--lp_epochs", type=int, default=None,
                        help="[LP-FT] 처음 N 에폭 백본 고정(투영·branch·헤드·BG 만 학습) 뒤 전체 해제. run_tag _LP<N>")
    parser.add_argument("--hard_weight", type=float, default=None,
                        help="[HW] 정답 라벨 CE 가중치(옛 (1−α)=0.7 대신). 1.0 권장. run_tag _HW1")
    parser.add_argument("--others_from_reject", type=float, default=None,
                        help="[OR] 전문가 8명 최대 확신도 < 이 값이면 others 가짜 정답(가중 1−최대). 예 0.2 → run_tag _OR02")
    parser.add_argument("--neg_pl", type=float, nargs=2, default=None, metavar=("TAU", "BETA"),
                        help="[NEG] 미채택 클립에서 전문가 확신도 < TAU 인 클래스에 부정 라벨 −BETA·log(1−p). 예 0.1 0.3 → run_tag _NEG")
    # [신설 2026-09-30 밤 / 가짜 정답 재료] 가짜 정답 재료. eval 에도 같은 플래그를 줄 것(run_tag).
    parser.add_argument("--teacher_seg_mean", action="store_true",
                        help="[SM] 조각 teacher 출력을 전문가별로 평균한 뒤 클립 단위로 채택(옛 방식 = 조각 다수결). 조각 teacher 에서만. run_tag _SM")
    parser.add_argument("--teacher_clean_view", action="store_true",
                        help="[CT] teacher 는 증강 없는 원본을 본다(SpecAugment 는 student 에만). run_tag _CT")
    # [신설 2026-10-01 / 지도만 격차 진단] teacher 없이 도는 지도만(u0) 조건에서 EfficientAT 파인튜닝과의 차이를 하나씩 맞춰 보는 스위치. eval 에도 같이 줄 것(run_tag).
    parser.add_argument("--bg_weight", type=float, default=None,
                        help="[BG] others 특징을 배경 임베딩 쪽으로 당기는 보조 손실의 가중치(기본 0.1). 0 이면 끔. 기본값과 다르면 run_tag _BG<값>(0 → _BG0)")
    parser.add_argument("--student_raw_wave", action="store_true",
                        help="[RW] student 클립 파형을 정규화하지 않은 원본 진폭으로(teacher mel 은 그대로). 비교모델 입력과 같다. run_tag _RW")
    # [신설 2026-10-07 / Mark5.3] 부품 묶음. eval 에도 같은 값을 줄 것(run_tag·OVA 헤드 구조).
    parser.add_argument("--m53", type=str, default=None,
                        help="[Mark5.3] all 또는 PF,SK,OVA,FM,CB,SC 중 일부(쉼표). D0 레시피(--linear_head·클립 입력) 위에서만. run_tag _M53 / _M53-PF-…")
    # [신설 2026-10-08 / Mark5.3 JF] eval 에도 같이 줄 것(run_tag _JF).
    parser.add_argument("--joint_forward", action="store_true",
                        help="[Mark5.3 JF] 정답·라벨 없는 클립을 학생에 한 번에 통과(BN = 배치 전체). --m53 과 함께만. run_tag _JF")
    args = parser.parse_args()
    # [신설 2026-09-26] 화면에 찍히는 것을 logFiles/<run_tag>__train.log 에도 그대로 남긴다.
    # run_tag 를 알아야 파일 이름이 정해지므로 config 를 먼저 한 번 만든다(텍스트 임베딩은 이때 안 만들어진다).
    from run_logger import RunLogger
    _c = AudioViLDConfig(mark_version=args.mark_version)
    _c.experiment_seed = args.seed
    _is_ceil = bool(args.ceiling_csv)
    _c.ceiling = _is_ceil
    _c.unlabeled_n = None if _is_ceil else args.unlabeled_n
    _c.labeled_n = args.labeled_n
    _c.pseudo_source = args.pseudo_source
    _c.spill_labeled = bool(args.spill_labeled)
    _c.apply_cli_learning_rates(args.lr, args.encoder_lr)
    _bundle = dict(segment_input=args.segment_input, head_pretrained_hidden=args.head_pretrained_hidden, lp_epochs=args.lp_epochs,
                   hard_weight=args.hard_weight, others_from_reject=args.others_from_reject, neg_pl=args.neg_pl,
                   teacher_seg_mean=args.teacher_seg_mean, teacher_clean_view=args.teacher_clean_view,
                   bg_weight=args.bg_weight, student_raw_wave=args.student_raw_wave, m53=args.m53,
                   joint_forward=args.joint_forward)
    _c.apply_cli_ablation(no_feature_kd=args.no_feature_kd, pl_min_gap=args.pl_min_gap, no_distill_branch=args.no_distill_branch,
                          clip_input=args.clip_input, linear_head=args.linear_head, **_bundle)
    with RunLogger(PROJECT_ROOT, _c.run_tag, "train"):
        train_mark5(seed_value=args.seed, mark_version=args.mark_version,
                    unlabeled_n=args.unlabeled_n, ceiling_csv=args.ceiling_csv, labeled_n=args.labeled_n,
                    pseudo_csv=args.pseudo_csv, pseudo_source=args.pseudo_source, spill_labeled=args.spill_labeled,
                    lr=args.lr, encoder_lr=args.encoder_lr,
                    no_feature_kd=args.no_feature_kd, pl_min_gap=args.pl_min_gap, no_distill_branch=args.no_distill_branch,
                    clip_input=args.clip_input, linear_head=args.linear_head, **_bundle)
