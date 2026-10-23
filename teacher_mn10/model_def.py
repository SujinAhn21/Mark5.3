# -*- coding: utf-8 -*-
"""mn10 이진 교사 — EfficientAT mn10(AudioSet 사전학습)을 '자기 클래스 대 others' 2-class 로 전 층 파인튜닝한다.

[신설 2026-10-22 / mn10 교사 B+] 설계 문서 '교사(Mark4.1~4.8) 설계' 절.

비교모델 Baseline_EfficientAT_mn10/model_def.py 의 계산을 그대로 옮겼다(정답 2,000 에서 0.9633 을 낸 구현).
  - 입력 16 kHz 3초 파형 -> 32 kHz 리샘플 -> preemphasis -> STFT(1024, hop 320, win 800) -> Kaldi mel 128
    -> log(x+1e-5) -> (x+4.5)/5. 학습 배치에만 SpecAugment(augment.py, 비교모델과 같은 비율).
  - 머리는 저자 방식대로 get_model(num_classes=2, head_type="mlp") 로 마지막 Linear 만 새로 만든다.
  - 클래스 순서 = [자기 클래스, others] (Mark4.x 의 옛 교사와 같다). 양성 확률 = softmax(logits)[:, 0].
Mark5.3 은 이 패키지(teacher_mn10/ 7개 파일)의 사본으로 이 체크포인트를 읽는다(model/make_teacher_scores.py). 사본은 md5 가 같아야 한다.
"""
import contextlib
import io
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .augment import spec_augment

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)
THIRD = os.path.join(REPO_ROOT, "third_party", "efficientat")
if THIRD not in sys.path:
    sys.path.insert(0, THIRD)

from models.mn.model import get_model          # noqa: E402  (third_party/efficientat, 원본 그대로)

PRETRAINED_PATH = os.path.join(REPO_ROOT, "weights", "mn10_as_mAP_471.pt")

SAMPLE_RATE = 32000
N_MELS = 128
WIN_LENGTH = 800
HOP_SIZE = 320
N_FFT = 1024
FMIN = 0.0
FMAX = 15000.0
HEAD_KEYS = ("classifier.5.weight", "classifier.5.bias")
NUM_CLASSES = 2
MODEL_KIND = "mn10_binary_clip"


class Mn10BinaryTeacher(nn.Module):
    """파형 [B, 48000] (16 kHz) -> 로짓 [B, 2]. embed() 는 분류 직전 특징 [B, 960] 을 돌려준다."""

    def __init__(self, load_pretrained=True, pretrained_path=PRETRAINED_PATH):
        super().__init__()
        with contextlib.redirect_stdout(io.StringIO()):      # get_model 이 모델 전체를 print 한다
            self.core = get_model(num_classes=NUM_CLASSES, pretrained_name=None, width_mult=1.0,
                                  head_type="mlp", se_dims="c")
        self.pretrained_loaded = False
        if load_pretrained:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"[ERROR] EfficientAT 가중치가 없습니다: {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if HEAD_KEYS[1] not in sd:
                raise RuntimeError(f"[ERROR] mlp 헤드(classifier.5.*)가 든 EfficientAT 가중치가 아닙니다: {pretrained_path}")
            for k in HEAD_KEYS:
                sd.pop(k, None)
            res = self.core.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if k not in HEAD_KEYS]
            if res.unexpected_keys or missing:
                raise RuntimeError(f"[ERROR] EfficientAT 가중치 키 불일치 — missing={missing}, "
                                   f"unexpected={list(res.unexpected_keys)}")
            self.pretrained_loaded = True
        self.register_buffer("window", torch.hann_window(WIN_LENGTH, periodic=False), persistent=False)
        self.register_buffer("preemphasis", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        self.register_buffer("mel_basis", F.pad(mel_basis, (0, 1), mode="constant", value=0), persistent=False)

    def frontend(self, wave16k):
        x = torchaudio.functional.resample(wave16k, 16000, SAMPLE_RATE)
        x = F.conv1d(x.unsqueeze(1), self.preemphasis).squeeze(1)
        spec = torch.stft(x, N_FFT, hop_length=HOP_SIZE, win_length=WIN_LENGTH, center=True,
                          normalized=False, window=self.window, return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.matmul(self.mel_basis, power)
        mel = (mel + 0.00001).log()
        return (mel + 4.5) / 5.0

    def forward_with_features(self, wave16k, augment=False):
        mel = self.frontend(wave16k)
        if augment and self.training:
            mel = spec_augment(mel)
        logits, features = self.core(mel.unsqueeze(1))
        return logits, features

    def forward(self, wave16k, augment=False):
        return self.forward_with_features(wave16k, augment=augment)[0]

    def positive_prob(self, wave16k):
        """평가 모드에서 자기 클래스 확률 [B]."""
        return torch.softmax(self.forward(wave16k, augment=False), dim=1)[:, 0]


def build_teacher(load_pretrained=True):
    return Mn10BinaryTeacher(load_pretrained=load_pretrained)
