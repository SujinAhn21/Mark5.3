# -*- coding: utf-8 -*-
"""SpecAugment — Mark5 의 apply_spec_augment(train_mark5.py, 2026-08-23)와 같은 규칙을 비율로 옮긴 것.

[신설 2026-09-05] 조건표 2절 '증강' 항목.

Mark5 는 64 mel 중 최대 8칸짜리 주파수 마스크 2개, 101 프레임 중 최대 16칸짜리 시간 마스크
2개를 건다. 비교모델은 mel 수와 프레임 수가 다르므로(128 mel, 3초 = 약 300 프레임) 칸 수가
아니라 **비율**(8/64 = 12.5%, 16/101 = 15.8%)을 맞춘다. 모델에 내장된 증강(PANNs 의
SpecAugmentation, EfficientAT 의 freqm/timem)은 전부 끄고 이 함수 하나만 쓴다.

채움값은 0 이 아니라 **그 샘플·채널·주파수 bin 의 시간 평균**이다(Mark5 는 샘플·채널 전체 평균).
0 으로 가리면 dB·log 스케일에서 밝은 띠가 그려지는 것을 피하려는 뜻은 Mark5 와 같다. 그런데 비교모델
중 PANNs·EfficientAT 는 16 kHz 소리를 32 kHz 규격 mel 에 넣으므로 8 kHz 위 bin 들이 거의 비어 있어
(PANNs 64 bin 중 9개, EfficientAT 128 bin 중 24개, code-reviewer 실측), 전체 평균 하나로 채우면 그 빈
대역에 원래 값보다 수십 dB 밝은 띠가 생긴다. bin 별 시간 평균으로 채우면 주파수 마스크는 그 bin 의
평소 값으로 평평해지고, 시간 마스크는 평균 스펙트럼 기둥이 되어 어느 대역에서도 "정보만 지운다"에
가깝다. [2026-09-05 저녁, code-reviewer 지적 반영. Mark5 는 64 bin 이 전부 활성이라 두 채움값의 차이가 작다.]
난수는 torch 전역 seed 를 따른다. 학습 루프에서만 부르고 val·test 는 원본을 본다.
"""
import torch

FREQ_RATIO = 8.0 / 64.0      # Mark5: spec_augment_freq_mask_param 8 / n_mels 64
TIME_RATIO = 16.0 / 101.0    # Mark5: spec_augment_time_mask_param 16 / segment_length 101
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2


def mask_params(n_mels, n_frames, freq_ratio=FREQ_RATIO, time_ratio=TIME_RATIO):
    """이 입력 크기에서 마스크 최대 폭(칸 수). 64 mel -> 8, 128 mel -> 16, 300~301 프레임 -> 48."""
    f = max(1, int(round(n_mels * freq_ratio)))
    t = max(1, int(round(n_frames * time_ratio)))
    return f, t


def spec_augment(mel, freq_ratio=FREQ_RATIO, time_ratio=TIME_RATIO,
                 num_freq_masks=NUM_FREQ_MASKS, num_time_masks=NUM_TIME_MASKS):
    """입력 [B, F, T] 또는 [B, C, F, T]. 같은 모양으로 돌려준다."""
    squeeze = False
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
        squeeze = True
    if mel.dim() != 4:
        return mel.squeeze(1) if squeeze else mel
    B, _, n_mels, n_frames = mel.shape
    f_param, t_param = mask_params(n_mels, n_frames, freq_ratio, time_ratio)
    out = mel.clone()
    fill = mel.mean(dim=3, keepdim=True)               # [B, C, F, 1]  bin 별 시간 평균
    for b in range(B):
        for _ in range(num_freq_masks):
            f = int(torch.randint(0, f_param + 1, (1,)).item())
            if f == 0 or f >= n_mels:
                continue
            f0 = int(torch.randint(0, n_mels - f + 1, (1,)).item())
            out[b, :, f0:f0 + f, :] = fill[b, :, f0:f0 + f, :]
        for _ in range(num_time_masks):
            t = int(torch.randint(0, t_param + 1, (1,)).item())
            if t == 0 or t >= n_frames:
                continue
            t0 = int(torch.randint(0, n_frames - t + 1, (1,)).item())
            out[b, :, :, t0:t0 + t] = fill[b, :, :, :]
    return out.squeeze(1) if squeeze else out
