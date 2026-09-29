# model/profile_efficiency.py
"""
Mark 모델의 효율성 지표(파라미터 수, 모델 용량, FLOPs, CPU 추론 지연시간)를 측정한다.
비교모델(PANNs/AST/BEATs/EfficientAT/ConvNeXt-audio/EAT/CED-Tiny)과의
효율성-정확도 산점도에 쓸 숫자를 낸다.

주의:
- 텍스트 임베딩(SentenceTransformer)은 9개 클래스에 대해 배포 시 1회만 계산해
  캐싱하는 게 정상적인 엣지 배포 방식이므로, 이 스크립트는 그 비용을 추론 지연시간에서 제외한다
  (오디오 1개가 들어올 때마다 반복되는 비용이 아니기 때문).
- FLOPs는 encoder+branch_head(둘 다 nn.Linear/Conv2d 등 thop이 인식하는 레이어로 구성)만 측정한다.
  text_head(ViLDTextHead)는 nn.Linear 없이 raw matmul(cosine 유사도)만 쓰는데, 연산량이
  384×9 수준으로 미미해 무시 가능.
"""

import argparse
import json
import os
import sys
import time

import torch
from thop import profile as thop_profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import LearnableBackgroundEmbedding, build_audio_encoder, build_student_head
from vild_head import DualBranchStudentHead


def count_parameters(*modules):
    total = 0
    for m in modules:
        if m is None:
            continue
        total += sum(p.numel() for p in m.parameters())
    return total


class _EncoderBranchWrapper(torch.nn.Module):
    """encoder+branch_head를 하나로 묶어 thop이 한 번의 forward로 FLOPs를 잴 수 있게 함."""

    def __init__(self, encoder, branch_head):
        super().__init__()
        self.encoder = encoder
        self.branch_head = branch_head

    def forward(self, x):
        feat = self.encoder(x)
        return self.branch_head(feat)


def profile(mark_version: str, num_warmup: int = 10, num_runs: int = 100, clip_input: bool = False, linear_head: bool = False,
            segment_input: bool = False, head_pretrained_hidden: bool = False):
    device = torch.device("cpu")  # 엣지 배포 타겟이 CPU이므로 지연시간은 항상 CPU 기준으로 측정
    config = AudioViLDConfig(mark_version=mark_version)
    # [신설 2026-09-29 / ablation A3] 클립 입력(mark5.2 기본)이면 student 가 클립 3초 파형을 한 번에 보므로 클립당 연산 = 1회(조각 5회가 아님).
    # [신설 2026-09-29 / 헤드 진단 H] --linear_head 면 헤드가 nn.Linear(384→9)라 파라미터 수가 달라지고 파일 이름에 _LIN 이 붙는다.
    # [2026-09-29 밤 / PH] --head_pretrained_hidden 이면 사전학습 은닉층(960→960)이 남고 branch 지도 갈래가 항등이라 파라미터 수가 달라진다(_PH).
    _ab_txt = config.apply_cli_ablation(clip_input=clip_input, linear_head=linear_head, segment_input=segment_input,
                                        head_pretrained_hidden=head_pretrained_hidden)
    if _ab_txt:
        print(f"[ABLATION] {_ab_txt} · run_tag {config.run_tag}")
    _clip_mode = bool(getattr(config, "student_clip_input", False))
    n_segments_per_clip = 1 if _clip_mode else int(config.max_segments)
    _sup_identity = bool(getattr(config, "head_pretrained_hidden", False))

    encoder = build_audio_encoder(config).to(device).eval()
    branch_head = DualBranchStudentHead(config.embedding_dim, supervised_identity=_sup_identity).to(device).eval()
    text_head = build_student_head(config).to(device).eval()   # [H 2026-09-29] 학습·평가와 같은 헤드 선택 함수
    background_embedding = None
    if config.use_background_embedding:
        background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device).eval()

    total_params = count_parameters(encoder, branch_head, text_head, background_embedding)
    model_size_mb = total_params * 4 / (1024 ** 2)  # fp32 가정

    num_classes = len(config.get_classes_for_evaluation())
    # [변경 2026-09-17 / Mark5.2 6절] student 입력이 파형(mn10)이면 세그먼트 1.0초 파형으로 잰다.
    # mn10 은 mel 프론트엔드(리샘플·STFT·mel)까지 인코더 안에 있으므로 FLOPs·지연시간에 그 몫이 포함된다
    # (비교모델과 같은 기준). cnn 경로는 옛날처럼 mel [1, C, 64, 101] 을 넣는다(mel 계산은 파서 몫이라 제외).
    if _clip_mode:
        dummy_segment = torch.randn(1, int(getattr(config, "student_clip_samples", 48000)), device=device)
        _input_desc = f"waveform 1x{dummy_segment.shape[1]} (16 kHz, 클립 {getattr(config, 'student_clip_seconds', 3.0)}s 통째 · A3)"
    elif getattr(config, "student_input", "mel") == "waveform":
        dummy_segment = torch.randn(1, int(getattr(config, "segment_samples", 16000)), device=device)
        _input_desc = f"waveform 1x{dummy_segment.shape[1]} (16 kHz, {getattr(config, 'segment_duration', 1.0)}s)"
    else:
        dummy_segment = torch.randn(1, config.num_input_channels, config.n_mels, config.segment_length, device=device)
        _input_desc = f"mel 1x{config.num_input_channels}x{config.n_mels}x{config.segment_length}"
    print(f"[INFO] 측정 입력: {_input_desc} · 인코더 {type(encoder).__name__}")
    dummy_text_emb = torch.randn(num_classes, config.embedding_dim, device=device)

    # [추가] FLOPs 측정 (encoder+branch_head, 세그먼트 1개 기준).
    # thop.profile은 프로파일링 대상 모듈에 total_ops/total_params 훅을 남겨 이후 재사용 시
    # 충돌(AttributeError)을 일으키므로, 지연시간 측정용 encoder/branch_head와 별개의
    # 인스턴스를 새로 만들어 여기서만 쓰고 버린다.
    flops_encoder = build_audio_encoder(config).to(device).eval()
    flops_branch_head = DualBranchStudentHead(config.embedding_dim, supervised_identity=_sup_identity).to(device).eval()
    wrapper = _EncoderBranchWrapper(flops_encoder, flops_branch_head)
    macs, _ = thop_profile(wrapper, inputs=(dummy_segment,), verbose=False)
    gmacs_per_segment = macs / 1e9
    gflops_per_segment = gmacs_per_segment * 2  # 통상 FLOPs ≈ 2 x MACs 관례

    def run_once():
        with torch.no_grad():
            base_features = encoder(dummy_segment)
            supervised_features, distill_features = branch_head(base_features)
            _ = text_head(supervised_features, dummy_text_emb)
            _ = text_head(distill_features, dummy_text_emb)

    for _ in range(num_warmup):
        run_once()

    start = time.perf_counter()
    for _ in range(num_runs):
        run_once()
    elapsed = time.perf_counter() - start

    ms_per_segment = (elapsed / num_runs) * 1000
    # 클립 전체 = n_segments_per_clip 회 (조각 모드: 1.01초 창을 0.5초 간격으로 겹쳐 약 3초를 덮는 5회 · A3 클립 모드: 1회)
    ms_per_clip = ms_per_segment * n_segments_per_clip

    result = {
        "mark_version": mark_version,
        "encoder_type": getattr(config, "encoder_type", "cnn"),       # [추가 2026-09-17]
        "input": _input_desc,
        "n_segments_per_clip": n_segments_per_clip,                     # [추가 2026-09-29 / A3]
        "ablation_tag": config.ablation_tag,
        "total_parameters": total_params,
        "model_size_mb": round(model_size_mb, 4),
        "gmacs_per_segment": round(gmacs_per_segment, 6),
        "gflops_per_segment_approx": round(gflops_per_segment, 6),
        "gflops_per_clip_approx": round(gflops_per_segment * n_segments_per_clip, 6),
        "cpu_latency_ms_per_segment": round(ms_per_segment, 4),
        "cpu_latency_ms_per_clip": round(ms_per_clip, 4),
        "num_warmup": num_warmup,
        "num_runs": num_runs,
        "note": (
            "텍스트 임베딩(SentenceTransformer) 계산 비용은 배포 시 1회 캐싱 가정으로 제외됨. "
            "FLOPs는 encoder+branch_head만 측정(text_head는 미미해 제외), FLOPs≈2×MACs 근사."
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # [2026-09-29 / A3] 절제 표식이 있으면 파일 이름에 붙여 본 모델 값을 덮지 않는다(예: efficiency_profile_mark5.2_noSEG.json).
    out_path = os.path.join(plot_dir, f"efficiency_profile_{mark_version}{config.ablation_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 효율성 프로파일 저장: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark 모델의 파라미터 수·모델 용량·FLOPs·CPU 추론 지연시간을 측정합니다.")
    parser.add_argument("--mark_version", type=str, default="mark5.2")   # [변경 2026-09-17] 기본 mark5.2
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--clip_input", action="store_true",
                        help="[A3 2026-09-29] student 입력을 클립 3초 파형 통째로 재고(클립당 1회) 파일 이름에 _noSEG 를 붙인다")
    parser.add_argument("--linear_head", action="store_true",
                        help="[H 2026-09-29] 헤드를 nn.Linear 로 만들어 재고(파라미터 수 반영) 파일 이름에 _LIN 을 붙인다")
    parser.add_argument("--segment_input", action="store_true", help="클립 입력(기본)을 끄고 조각 1초 × 5회로 잰다(이름 표식 없음)")
    parser.add_argument("--head_pretrained_hidden", action="store_true", help="[PH] 사전학습 은닉층 유지·branch 항등 구조로 잰다(_PH)")
    args = parser.parse_args()
    profile(mark_version=args.mark_version, num_warmup=args.num_warmup, num_runs=args.num_runs, clip_input=args.clip_input,
            linear_head=args.linear_head, segment_input=args.segment_input, head_pretrained_hidden=args.head_pretrained_hidden)
