# vild_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAudioEncoder(nn.Module):
    """
    Mel-Spectrogram 입력을 고정된 차원의 임베딩으로 변환하는 CNN 기반 오디오 인코더

    특징:
    - Dropout 추가로 과적합 방지
    - AdaptiveAvgPool + LayerNorm 구조로 소형 모델에서도 안정성 확보
    - 최소 입력 크기 추론을 위한 get_min_input_shape() 제공
    """

    def __init__(self, config):
        super().__init__()
        in_channels = getattr(config, 'num_input_channels', 1)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # [추가 2026-07-12 / 가설2] conv 2블록(32->64ch, 약 4.5만 파라미터)에서는 train hard
        # loss가 랜덤 근처(~0.60)에 고착되어 훈련 데이터조차 못 맞추는 과소적합이 관찰됨.
        # 용량 한계인지 진단하기 위해 3블록(32->64->128ch, 약 14.4만 파라미터)으로 확장.
        # 여전히 엣지 배포 가능한 초경량 수준. 입력 64x101 -> 3회 MaxPool 후 8x12.
        # 주의: 이 변경으로 기존 .pth 체크포인트와는 구조가 안 맞으므로 전체 재학습 필요.
        #
        # [Mark5 로 가져옴 2026-08-17] mark4.x 는 위 진단 뒤 3블록으로 갔는데 mark5 만 2블록으로
        # 남아 있었다. mark5 는 2-class 가 아니라 9-class 라 더 어려운 문제인데 더 작은 인코더로
        # 배우려던 셈이고, 효율성 표의 student 도 8개 버전과 다른 모델이 된다. 그래서 맞췄다.
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, config.embedding_dim)
        )

        self.model = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.head
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C, 64, 101] 형태의 visualized audio tensor

        Returns:
            Tensor: [B, embedding_dim] 형태의 오디오 임베딩 벡터
        """
        return self.model(x)

    @staticmethod
    def get_min_input_shape(config=None):
        """
        모델 구조를 기반으로 한 최소 입력 크기 반환

        Returns:
            Tuple[int, int, int, int]: (B, C, H, W)
        """
        h = getattr(config, 'n_mels', 64) if config else 64
        w = getattr(config, 'min_time_frames', 101) if config else 101
        c = getattr(config, 'num_input_channels', 1) if config else 1
        return (1, c, h, w)


class ResidualAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = getattr(config, "num_input_channels", 1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, config.embedding_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        residual = x
        x = self.block1(x)
        x = F.relu(x + residual)
        x = self.downsample(x)
        residual = x
        x = self.block2(x)
        x = F.relu(x + residual)
        return self.head(x)


class TeacherAudioEncoder(nn.Module):
    """
    [추가 2026-07-13 / teacher 강화] teacher 전용 대형 인코더.

    배경: 기존 teacher는 student와 완전히 같은 SimpleAudioEncoder(약 14.3만 파라미터)를 써서
    KD인데도 teacher의 용량 우위가 전혀 없었음(teacher val loss가 가장 약한 고리였던 구조적 원인).
    teacher는 엣지에 배포되지 않고 오프라인 soft label 생성에만 쓰이므로 크기 제약이 없다.

    구조: conv 4블록(32->64->128->256) + 같은 head 구성. 약 49만 파라미터(student의 약 3.4배).
    입력/출력 인터페이스는 SimpleAudioEncoder와 동일([B, C, 64, 101] -> [B, embedding_dim])이라
    ViLDTextHead·Feature KD(384차원)와 그대로 호환된다. 입력 64x101 -> 4회 MaxPool 후 4x6.
    주의: 기존 teacher .pth와 구조가 안 맞으므로 teacher부터 전체 재학습 필요.

    [Mark5 로 가져옴 2026-08-17] Mark4.5/vild/vild_model.py:85 의 정의를 글자 그대로 복사한 것이다.
    mark5 의 EnsembleTeacher 가 mark4.x teacher .pth 8개를 로드하는데, mark5 에는 이 클래스가
    없어서 student 용 SimpleAudioEncoder 에 4블록 가중치를 넣으려다 실패했다
    ("Missing key(s) conv_block1.* / Unexpected key(s) model.3.*"). 구조가 한 곳이라도
    달라지면 다시 안 맞으므로 Mark4.5 원본과 동일하게 유지해야 한다.
    """

    def __init__(self, config):
        super().__init__()
        in_channels = getattr(config, "num_input_channels", 1)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.model = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, config.embedding_dim),
        )

    def forward(self, x):
        return self.model(x)


def build_teacher_encoder(config):
    """teacher 인코더 선택 지점(단일 스위치). teacher_train.py와 extract_soft_labels.py가
    반드시 같은 함수를 써야 체크포인트 구조가 일치한다.
    config.use_large_teacher_encoder=False(또는 없음)면 기존 동작(SimpleAudioEncoder) 그대로."""
    if getattr(config, "use_large_teacher_encoder", False):
        return TeacherAudioEncoder(config)
    return SimpleAudioEncoder(config)


class ViLDTextHead(nn.Module):
    """
    region embedding과 텍스트 임베딩 간 cosine similarity 로짓을 계산하는 헤드 모듈

    특징:
    - Temperature scaling 적용
    - background class는 사용하지 않음 (ex. binary 분류나 multi-class 분류 시 직접 사용)
    - CrossEntropyLoss 또는 soft label 학습 시 softmax 적용은 외부에서 처리
    """

    def __init__(self, config, learnable=None):
        super().__init__()
        t = float(getattr(config, 'logit_temperature', 0.07))
        # [신설 2026-09-24 / F] 학습형 로짓 스케일(CLIP·ViLD 원본 방식). 전에는 temperature 가 파이썬 float 고정값이라
        # 학습되지 않았다. 2026-09-24 실측: 같은 백본·같은 데이터에서 Mark5.2 의 log(1등/2등) 평균이 3.48 로
        # EfficientAT(7.90)의 절반 — 확률이 평평해 정답을 2등에 두고 1등을 놓치는 경우가 83.7%(EfficientAT 64.0%).
        # log 공간 파라미터, 초기값 log(1/T) 라 첫 스텝은 고정 T 와 완전히 같다. 상한 log(100)(CLIP 과 동일).
        # learnable=None 이면 config.learnable_logit_scale 을 따른다. teacher(2-class 전문가)는 저장된 상태에 이 파라미터가
        # 없으므로 train_mark5.py 가 learnable=False 를 명시해 부른다.
        if learnable is None:
            learnable = bool(getattr(config, 'learnable_logit_scale', False))
        self.learnable = bool(learnable)
        if self.learnable:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / t), dtype=torch.float32))
            self._temperature = None
        else:
            self._temperature = t

    @property
    def temperature(self):
        """실효 temperature. 고정이면 config 값, 학습형이면 1/exp(logit_scale)(상한 반영). eval.py 의 background
        override 경로가 text_head.temperature 로 나누므로 학습형에서도 float 을 돌려줘야 한다."""
        if self.learnable:
            return 1.0 / float(self.logit_scale.detach().clamp(max=math.log(100.0)).exp())
        return self._temperature

    def forward(self, region_embeddings, class_text_embeddings):
        """
        Args:
            region_embeddings (Tensor): [B, D], student or teacher region embeddings
            class_text_embeddings (Tensor): [C, D], 사전 학습된 텍스트 임베딩

        Returns:
            Tensor: [B, C] 형태의 로짓 벡터 (softmax 적용 전)
        """
        region_norm = F.normalize(region_embeddings, dim=1)
        text_norm = F.normalize(class_text_embeddings, dim=1)

        logits = torch.matmul(region_norm, text_norm.T)  # [B, C]
        if self.learnable:
            logits = logits * self.logit_scale.clamp(max=math.log(100.0)).exp()
        else:
            logits = logits / self._temperature
        return logits


class LinearClassHead(nn.Module):
    """[신설 2026-09-29 / 헤드 진단 H] 텍스트 앵커 대신 nn.Linear(embedding_dim, num_classes) 로 로짓을 만드는 헤드.

    왜: ablation A1·A2·A4(2026-09-28)·A3(2026-09-29, seed 1)가 전부 본 모델과 1클립 차이라, EfficientAT-pl 과의 2%p 차이의
    남은 후보가 분류 헤드(텍스트 앵커 코사인 + AC/LT)였다. 비교모델 EfficientAT 의 mlp 헤드 마지막 층(nn.Linear(960, 9))과
    같은 방식으로 바꿔 1회 학습해 보는 진단용으로 만들었고, 2026-10-07 지도만 격차 진단(D0)으로 본 레시피에 채택했다. 우리 경로는 mn10 960 → proj Linear(960→384) → branch Linear(384→384)+LN+ReLU
    까지 있으므로 마지막 Linear(384→9) 하나만 더하면 깊이가 비교모델과 비슷해진다.

    ViLDTextHead 와 같은 호출 모양 forward(region_emb, class_text_emb) 을 받되 class_text_emb 는 쓰지 않는다 — 학습·평가·효율 측정의
    호출처를 바꾸지 않기 위해서다. 제로샷(텍스트 임베딩만 바꿔 새 클래스 판정)·앵커 중심화·학습형 로짓 스케일은 이 헤드에서 뜻이 없다.
    learnable=False · temperature=1.0 은 eval.py 의 logit_scale 로드 분기·background override 경로가 그대로 지나가게 하는 값이다.
    """

    learnable = False

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(int(embedding_dim), int(num_classes))

    @property
    def temperature(self):
        return 1.0

    def forward(self, region_embeddings, class_text_embeddings=None):
        return self.linear(region_embeddings)   # [B, C]


class LearnableBackgroundEmbedding(nn.Module):
    """
    'others'(배경) 클래스를 위한 학습형 임베딩.

    ViLDTextHead가 계산하는 cosine similarity 로짓 공간과 동일한 embedding_dim을 사용하며,
    eval 시 'others' 로짓을 max-override 방식으로 보정하는 데 쓰인다.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.background_emb = nn.Parameter(torch.randn(embedding_dim) * 0.01)

    def forward(self):
        return self.background_emb


class Mn10AudioEncoder(nn.Module):
    """[신설 2026-09-17 / Mark5.2 6절] 사전학습 오디오 백본 EfficientAT mn10 을 student 인코더로 쓴다.

    왜 바꾸나(2026-09-08 결정): 기존 SimpleAudioEncoder 는 사전학습 없는 3층 CNN(143,488 파라미터)이라
    표현이 병목이었다. 같은 val 운영점에서 반지도 이득이 0 이었고(09-07), 비교모델(mn10·CED·PANNs)이
    0.89~0.95 인데 우리는 0.86 이었다. 백본은 노벨티(이진 전문가 합의·텍스트 앵커 헤드)와 무관하므로
    인코더와 그 앞 mel 프론트엔드만 바꾸고 나머지(세그먼트 5개·teacher 합의·두 갈래 헤드·텍스트 앵커·
    others 규칙·SpecAugment·feature KD)는 그대로 둔다. 비교모델과 같은 백본이라 표에서 백본 효과가 상쇄된다.

    입력: [B, T] 16 kHz 파형(세그먼트 1.0 초 = 16,000 샘플. 파서가 mel 세그먼트와 같은 창으로 잘라 준다).
          SimpleAudioEncoder 는 mel [B, C, 64, 101] 을 받지만 mn10 은 32 kHz·128 mel 규격의 자기 프론트엔드를
          쓰므로 파형에서 시작해야 한다. 어느 쪽을 줄지는 config.student_input("waveform"/"mel")이 정한다.
    출력: [B, embedding_dim]  (기존과 같아 DualBranchStudentHead·ViLDTextHead·feature KD 가 그대로 붙는다)

    구조: frontend(Baseline_EfficientAT_mn10/model_def.py 와 같은 계산 — 16k→32k 리샘플, preemphasis -0.97,
          STFT 1024/hop 320/win 800, Kaldi mel 128(fmax 15 kHz), log(x+1e-5), (x+4.5)/5)
          → [학습 중·augment=True 면] SpecAugment(비율 8/64·16/101, 채움값 = 그 bin 의 시간 평균 — 8 kHz 위
            빈 대역에 밝은 띠가 생기지 않게. Baseline common/augment.py 규칙)
          → mn10 백본(third_party/efficientat, weights/mn10_as_mAP_471.pt, 전 층 파인튜닝) → 풀링 960
          → Dropout(0.3) → Linear(960, embedding_dim)
    사전학습 가중치는 저자 mlp 헤드(classifier.5.*)만 빼고 strict 하게 맞춘다. 헤드는 쓰지 않는다(features 만 씀).
    """

    SAMPLE_RATE = 32000
    N_MELS = 128
    WIN_LENGTH = 800
    HOP_SIZE = 320
    N_FFT = 1024
    FMIN = 0.0
    FMAX = 15000.0     # 저자 코드: fmax=None -> sr//2 - fmax_aug_range//2 = 16000 - 1000
    HEAD_KEYS = ("classifier.5.weight", "classifier.5.bias")

    def __init__(self, config):
        super().__init__()
        import contextlib, io, os, sys
        import torchaudio
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third = os.path.join(project_root, "third_party", "efficientat")
        if third not in sys.path:
            sys.path.insert(0, third)
        from models.mn.model import get_model     # third_party/efficientat (MIT, 원본 그대로)
        with contextlib.redirect_stdout(io.StringIO()):      # get_model 이 모델 전체를 print 한다
            self.core = get_model(num_classes=2, pretrained_name=None, width_mult=1.0,
                                  head_type="mlp", se_dims="c")
        self.pretrained_loaded = False
        self.pretrained_source = None
        weights_path = getattr(config, "mn10_pretrained_path", None)
        if weights_path is None:
            weights_path = os.path.join(project_root, "weights", "mn10_as_mAP_471.pt")
        if getattr(config, "mn10_load_pretrained", True):
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"[ERROR] EfficientAT mn10 가중치가 없습니다: {weights_path}")
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            for k in self.HEAD_KEYS:
                sd.pop(k, None)                                   # 저자 527-class 헤드는 버린다
            res = self.core.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if k not in self.HEAD_KEYS]
            if res.unexpected_keys or missing:
                raise RuntimeError(f"[ERROR] mn10 가중치 키 불일치 — missing={missing}, unexpected={list(res.unexpected_keys)}")
            self.pretrained_loaded = True
            self.pretrained_source = weights_path
        feat_dim = int(self.core.classifier[2].in_features)       # mlp 헤드 첫 Linear 의 입력 = 960 (mn10)
        # [신설 2026-09-29 밤 / PH] head_pretrained_hidden=True 면 저자 mlp 헤드의 사전학습 은닉층(AdaptiveAvgPool → Flatten →
        # Linear(960→1280) → Hardswish → Dropout)을 살리고 527-class 마지막 Linear 만 뗀다. _forward_impl 의 첫 반환값이 그 은닉 출력
        # [B, 1280] 이 되고, 그 위에 무작위 층은 텍스트 공간 투영 Linear(1280→384) 하나뿐이다(EfficientAT 비교모델이 은닉층 위에 새
        # Linear 하나만 얹는 것과 같은 깊이). 2026-09-29 격차 분석: 옛 구조는 이 은닉층을 버리고 무작위 층 둘(proj·branch)+ReLU 병목을
        # 끼워 넣었었다(Kumar et al. ICLR 2022 — 무작위 헤드가 사전학습 특징을 왜곡). 사전학습 Dropout(0.2)이 이미 있어 proj 에는 Dropout 을 안 둔다.
        self.use_pretrained_hidden = bool(getattr(config, "head_pretrained_hidden", False))
        if self.use_pretrained_hidden:
            hidden_dim = int(self.core.classifier[2].out_features)                       # 960
            self.core.classifier = nn.Sequential(*list(self.core.classifier.children())[:-1])   # 마지막 Linear(→527) 만 제거
            self.proj = nn.Sequential(nn.Linear(hidden_dim, config.embedding_dim))
        else:
            # 저자 mlp 헤드는 쓰지 않으므로 통째로 떼어낸다(state_dict·파라미터 수에 1.23M 이 헛되이 남지 않게).
            # _forward_impl 이 self.classifier(x) 를 부르므로 Identity 로 바꿔 둔다(그 출력은 버린다).
            self.core.classifier = nn.Identity()
            self.proj = nn.Sequential(
                nn.Dropout(getattr(config, "mn10_proj_dropout", 0.3)),
                nn.Linear(feat_dim, config.embedding_dim),
            )
        self.register_buffer("window", torch.hann_window(self.WIN_LENGTH, periodic=False), persistent=False)
        self.register_buffer("preemphasis", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.N_MELS, self.N_FFT, self.SAMPLE_RATE, self.FMIN, self.FMAX,
            vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        self.register_buffer("mel_basis", F.pad(mel_basis, (0, 1), mode="constant", value=0), persistent=False)
        self.input_sample_rate = int(getattr(config, "sample_rate", 16000))
        # SpecAugment 비율(Mark5 규칙 8/64·16/101 을 mel 수·프레임 수에 비례해 옮김) — 학습 루프가 augment=True 로 켠다.
        self.sa_freq_ratio = float(getattr(config, "spec_augment_freq_mask_param", 8)) / float(getattr(config, "n_mels", 64))
        self.sa_time_ratio = float(getattr(config, "spec_augment_time_mask_param", 16)) / float(getattr(config, "segment_length", 101))
        self.sa_num_freq = int(getattr(config, "spec_augment_num_freq_masks", 2))
        self.sa_num_time = int(getattr(config, "spec_augment_num_time_masks", 2))

    def frontend(self, wave):
        """[B, T] 16 kHz 파형 -> mel [B, 128, T']. Baseline_EfficientAT_mn10/model_def.py 와 같은 계산."""
        import torchaudio
        if wave.dim() == 3:
            wave = wave.squeeze(1)
        if self.input_sample_rate != self.SAMPLE_RATE:
            # Resample 모듈 대신 함수형 — 커널 버퍼가 state_dict 에 들어가 torchaudio 버전이 갈리면 로드가 깨진다.
            wave = torchaudio.functional.resample(wave, self.input_sample_rate, self.SAMPLE_RATE)
        x = F.conv1d(wave.unsqueeze(1), self.preemphasis).squeeze(1)
        spec = torch.stft(x, self.N_FFT, hop_length=self.HOP_SIZE, win_length=self.WIN_LENGTH, center=True,
                          normalized=False, window=self.window, return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.matmul(self.mel_basis, power)
        mel = (mel + 0.00001).log()
        return (mel + 4.5) / 5.0

    def spec_augment(self, mel):
        """[B, F, T] 에 주파수·시간 마스크. 채움값은 그 샘플·그 bin 의 시간 평균(비교모델과 같은 규칙).
        16 kHz 소리를 32 kHz 규격에 넣어 8 kHz 위 24 bin 이 거의 비어 있으므로 전체 평균 하나로 채우면
        그 대역에 밝은 띠가 생긴다(2026-09-05 code-reviewer 실측). 난수는 torch 전역 seed."""
        B, n_mels, n_frames = mel.shape
        f_param = max(1, int(round(n_mels * self.sa_freq_ratio)))
        t_param = max(1, int(round(n_frames * self.sa_time_ratio)))
        out = mel.clone()
        fill = mel.mean(dim=2, keepdim=True)               # [B, F, 1]
        for b in range(B):
            for _ in range(self.sa_num_freq):
                f = int(torch.randint(0, f_param + 1, (1,)).item())
                if f == 0 or f >= n_mels:
                    continue
                f0 = int(torch.randint(0, n_mels - f + 1, (1,)).item())
                out[b, f0:f0 + f, :] = fill[b, f0:f0 + f, :]
            for _ in range(self.sa_num_time):
                t = int(torch.randint(0, t_param + 1, (1,)).item())
                if t == 0 or t >= n_frames:
                    continue
                t0 = int(torch.randint(0, n_frames - t + 1, (1,)).item())
                out[b, :, t0:t0 + t] = fill[b, :, :]
        return out

    def forward(self, x, augment=False):
        """x: [B, T] 파형(또는 [B, 1, T]). augment 는 학습 루프가 SpecAugment 를 켤 때만 True 로 준다."""
        mel = self.frontend(x)
        if augment and self.training:
            mel = self.spec_augment(mel)
        head_out, features = self.core(mel.unsqueeze(1))         # features: 풀링된 [B, 960] · head_out: classifier 출력
        # [PH] 사전학습 은닉층을 살렸으면 classifier 출력(Hardswish·Dropout 뒤 [B, 960])을, 아니면 풀링 특징을 투영한다.
        x = head_out if self.use_pretrained_hidden else features
        # 원본 _forward_impl 이 .squeeze() 하므로 배치가 1이면 [960] 이 된다(원본은 자기 헤드 출력으로 배치 차원을
        # 되살리는데 헤드가 Identity 면 그 분기가 안 탄다). eval 은 늘 배치 1 이라 여기서 되살린다.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.proj(x)

    @staticmethod
    def get_min_input_shape(config=None):
        """파형 입력 크기 (B, T). 세그먼트 1.0 초."""
        sr = getattr(config, "sample_rate", 16000) if config else 16000
        dur = getattr(config, "segment_duration", 1.0) if config else 1.0
        return (1, int(sr * dur))


def build_audio_encoder(config):
    encoder_type = getattr(config, "encoder_type", "cnn")
    if encoder_type == "residual_cnn":
        return ResidualAudioEncoder(config)
    if encoder_type == "mn10":
        # [신설 2026-09-17] Mark5.2 기본 백본. config.student_input 이 "waveform" 이어야 파서가 파형 세그먼트를 준다.
        return Mn10AudioEncoder(config)
    return SimpleAudioEncoder(config)


def build_student_head(config):
    """[신설 2026-09-29 / 헤드 진단 H] student 분류 헤드 선택 지점(단일 스위치). config.student_head 가 "linear" 면
    LinearClassHead(--linear_head, 2026-10-07 D0 본 레시피), 아니면 ViLDTextHead(텍스트 앵커, config 기본값). train_mark5.py·eval.py·profile_efficiency.py 가
    같은 함수를 써야 체크포인트의 classifier_state_dict 구조가 맞는다.
    teacher(train_mark5.EnsembleTeacher)의 헤드는 여기를 거치지 않는다 — Mark4.x 저장본은 ViLDTextHead(learnable=False) 고정."""
    if getattr(config, "student_head", "text_anchor") == "linear":
        return LinearClassHead(config.embedding_dim, len(config.get_classes_for_evaluation()))
    return ViLDTextHead(config)
    
