import os
import sys

import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

VILD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VILD_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
for p in (UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from parser_utils import load_audio_file


def _normalize_visual_tensor(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        return None

    target_channels, target_mels, target_time = target_shape
    channels, n_mels, time_dim = tensor.shape
    if channels != target_channels:
        if channels > target_channels:
            tensor = tensor[:target_channels]
        else:
            pad_channels = torch.zeros((target_channels - channels, n_mels, time_dim), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_channels], dim=0)

    pad_mel = max(0, target_mels - tensor.shape[1])
    pad_time = max(0, target_time - tensor.shape[2])
    if pad_mel > 0 or pad_time > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_time, 0, pad_mel))
    tensor = tensor[:, :target_mels, :target_time]
    if tensor.shape != target_shape:
        return None
    return tensor


class BaseAudioParser:
    def __init__(self, config):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.resampler_cache = {}
        # [수정 2026-07-11] torchaudio 최신버전(코랩 GPU 환경 등)에서 set_audio_backend가 아예
        # 제거되어 AttributeError가 남 — RuntimeError만 잡던 기존 코드가 이를 못 잡고 죽었음(실측).
        # 최신 torchaudio는 백엔드 자동선택이라 이 호출 자체가 불필요하므로, 있을 때만 시도한다.
        if hasattr(torchaudio, "set_audio_backend"):
            try:
                torchaudio.set_audio_backend("soundfile")
            except Exception:
                pass

    def _to_visual_mel(self, waveform):
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        if mel_db.ndim != 3 or mel_db.shape[1] != self.config.n_mels:
            return None
        return mel_db

    def _build_visual_views(self, mel_segment):
        base = mel_segment.squeeze(0)
        if self.config.visual_view_type == "mel":
            stacked = base.unsqueeze(0)
        elif self.config.visual_view_type == "mel_delta":
            delta1 = AF.compute_deltas(base.unsqueeze(0)).squeeze(0)
            delta2 = AF.compute_deltas(delta1.unsqueeze(0)).squeeze(0)
            stacked = torch.stack([base, delta1, delta2], dim=0)
        elif self.config.visual_view_type == "mel_energy":
            energy = (base - base.mean()) / (base.std() + 1e-6)
            stacked = torch.stack([base, energy], dim=0)
        else:
            stacked = base.unsqueeze(0)
        return stacked

    def compute_saliency_score(self, tensor):
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        base = tensor[0] if tensor.ndim == 3 else tensor
        energy = float(base.abs().mean().item())
        flux = float((base[:, 1:] - base[:, :-1]).abs().mean().item()) if base.shape[-1] > 1 else 0.0
        return energy + 0.5 * flux

    def load_and_segment_with_metadata(self, file_path):
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Parser] Skipped unreadable file: {file_path}")
            return []
        return self._segments_from_waveform(waveform, file_path)

    def load_clip_with_segments(self, file_path):
        """[신설 2026-09-29 / ablation A3] 파일을 한 번 읽어 클립 단위 student 입력과 teacher 조각을 함께 돌려준다.
            clip_wave   : [student_clip_samples] float32 16 kHz — 파일 첫 48,000 샘플(3.0초), 모자라면 0 으로 채움.
                          [2026-10-01] config.student_wave_normalize=False 면 원본 진폭(정규화 안 함), 기본은 최대 진폭 1 로 정규화된 파형.
                          비교모델 Baseline_*/common/data.py 의 자르기(wav[:CLIP_SAMPLES] + 0 채움)와 같다.
            clip_record : 클립 전체 mel 의 시각화 뷰 하나 — eval.py 가 조각 1개짜리 기록으로 다룬다(tensor [C, n_mels, T(=301)],
                          segment_index 0, start_frame 0, end_frame T, saliency, wave=clip_wave).
            segments    : load_and_segment_with_metadata 와 완전히 같은 조각 기록(teacher 8개용, 101프레임 × 최대 max_segments 개).
        못 읽거나 조각이 없으면 None. mel 은 한 번만 계산해 조각과 클립 뷰가 같은 값을 쓴다."""
        # [2026-10-01 / 지도만 격차 진단] config.student_wave_normalize=False(--student_raw_wave)면 student 클립 파형만 원본 진폭으로 되돌린다.
        # teacher 조각·클립 mel 은 아래에서 정규화 파형으로 그대로 만든다(전문가 8개가 그렇게 학습됐다).
        _raw = not getattr(self.config, "student_wave_normalize", True)
        _peak = 1.0
        if _raw:
            _loaded = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache, return_peak=True)
            waveform, _peak = (None, 1.0) if _loaded is None else _loaded
        else:
            waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Parser] Skipped unreadable file: {file_path}")
            return None
        mel_db = self._to_visual_mel(waveform)
        if mel_db is None:
            print(f"[Parser] Unexpected mel shape from {file_path}")
            return None
        segments = self._segments_from_waveform(waveform, file_path, mel_db=mel_db)
        if not segments:
            return None
        n = int(getattr(self.config, "student_clip_samples", int(self.config.sample_rate * 3.0)))
        wave_1d = waveform.reshape(-1)[:n]
        if wave_1d.numel() < n:
            wave_1d = torch.nn.functional.pad(wave_1d, (0, n - wave_1d.numel()))
        clip_wave = wave_1d.clone().to(torch.float32)
        if _raw:
            clip_wave = clip_wave * float(_peak)        # 정규화 전 진폭(비교모델 common/data.py load_clip 과 같은 값, float32 반올림 차이뿐)
        views = self._build_visual_views(mel_db)            # [C, n_mels, T] — 조각과 같은 view 종류(mel_delta 면 3채널)
        clip_record = {
            "segment_index": 0,
            "start_frame": 0,
            "end_frame": int(views.shape[-1]),
            "saliency": self.compute_saliency_score(views),
            "tensor": views,
            "wave": clip_wave,
        }
        return {"clip_wave": clip_wave, "clip_record": clip_record, "segments": segments}

    def _segments_from_waveform(self, waveform, file_path, mel_db=None):
        """[분리 2026-09-29] 파형 하나에서 teacher 조각 기록 목록을 만든다 — 옛 load_and_segment_with_metadata 본문 그대로.
        mel_db 를 주면 다시 계산하지 않는다(load_clip_with_segments 가 클립 뷰와 공유)."""
        try:
            if mel_db is None:
                mel_db = self._to_visual_mel(waveform)
            if mel_db is None:
                print(f"[Parser] Unexpected mel shape from {file_path}")
                return []

            _, _, total_time = mel_db.shape
            window = self.config.segment_length
            stride = self.config.segment_hop
            max_segments = getattr(self.config, "max_segments", 5)
            if total_time < window:
                print(f"[Parser] Mel too short for segmentation: {total_time} < {window} in {file_path}")
                return []

            # [신설 2026-09-17 / Mark5.2 6절] mel 세그먼트와 같은 창의 **파형**도 함께 낸다(record["wave"]).
            # student 인코더가 mn10(파형 입력)이면 이것을 쓰고, teacher 8개는 계속 mel(record["tensor"])을 쓴다.
            # mel 프레임 f 는 hop_length 간격(center=True)이라 프레임 [start, start+window) 은 샘플
            # [start*hop, start*hop + segment_samples) 에 해당한다(window 101·hop 160 -> 16,000 샘플 = 1.0 초).
            # 3.0 초 클립(48,000)이면 마지막 세그먼트 start=200 -> 32,000~48,000 으로 딱 맞는다. 짧으면 0 으로 채운다.
            hop = int(self.config.hop_length)
            seg_samples = int(getattr(self.config, "segment_samples", hop * (window - 1)))
            wave_1d = waveform.reshape(-1)
            candidates = []
            segment_counter = 0
            for start in range(0, total_time - window + 1, stride):
                segment = mel_db[:, :, start:start + window]
                views = self._build_visual_views(segment)
                normed = _normalize_visual_tensor(
                    views,
                    (self.config.num_input_channels, self.config.n_mels, self.config.segment_length),
                )
                if normed is not None:
                    s0 = start * hop
                    wave_seg = wave_1d[s0:s0 + seg_samples]
                    if wave_seg.numel() < seg_samples:
                        wave_seg = torch.nn.functional.pad(wave_seg, (0, seg_samples - wave_seg.numel()))
                    candidates.append({
                        "segment_index": segment_counter,
                        "start_frame": start,
                        "end_frame": start + window,
                        "saliency": self.compute_saliency_score(normed),
                        "tensor": normed,
                        "wave": wave_seg.clone().to(torch.float32),   # [segment_samples] 16 kHz
                    })
                    segment_counter += 1

            if not candidates:
                print(f"[Parser] No valid segments from: {file_path}")
                return []

            if self.config.segment_selection_mode == "salient_topk":
                selected = sorted(candidates, key=lambda x: x["saliency"], reverse=True)[:max_segments]
                selected.sort(key=lambda x: x["start_frame"])
            else:
                selected = candidates[:max_segments]

            while len(selected) < max_segments:
                last = selected[-1]
                selected.append({
                    "segment_index": last["segment_index"],
                    "start_frame": last["start_frame"],
                    "end_frame": last["end_frame"],
                    "saliency": last["saliency"],
                    "tensor": last["tensor"].clone(),
                    "wave": last["wave"].clone(),
                })
            return selected
        except Exception as e:
            print(f"[Parser] Exception while parsing {file_path}: {e}")
            return []

    def load_and_segment(self, file_path):
        return [record["tensor"] for record in self.load_and_segment_with_metadata(file_path)]

    def parse_sample(self, file_path, label_text):
        segments = self.load_and_segment(file_path)
        segments = [seg for seg in segments if isinstance(seg, torch.Tensor)]
        if not segments:
            raise ValueError(f"[Parser] No mel segments from {file_path}")
        mel_tensor = torch.cat(segments, dim=0)
        label_idx = self.config.get_class_index(label_text)
        return mel_tensor, label_idx
