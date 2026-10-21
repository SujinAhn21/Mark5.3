# vild_config.py
# 뭐가 많아질수록 힘들구나.  
# [12:41]

import math
import torch
from sentence_transformers import SentenceTransformer
import os

SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_vild"))
# [변경] append -> insert(0): prompt_bank 등 공용 모듈이 vild/ 등 다른 경로보다 항상 우선
# 해석되도록 shared_vild를 sys.path 앞쪽에 둔다. (구버전 vild/prompt_bank.py shadow 방지)
if SHARED_DIR not in os.sys.path:
    os.sys.path.insert(0, SHARED_DIR)

from prompt_bank import get_class_synonyms, get_prompt_templates, get_prompt_texts_for_class



class AudioViLDConfig:
    def __init__(self, mark_version="mark5.0"):
        self.mark_version = mark_version

        # ==============================================================================
        # 1. 클래스 설정 (Mark Version 별)
        # ==============================================================================
        if self.mark_version == "mark4.1":
            self.classes = ["heavy_impact", "others"]
        elif self.mark_version == "mark4.2":
            self.classes = ["dragging", "others"] 
        elif self.mark_version == "mark4.3":
            self.classes = ["construction", "others"] 
        elif self.mark_version == "mark4.4":
            self.classes = ["machine_noise", "others"]
        elif self.mark_version == "mark4.5":
            self.classes = ["media_talking", "others"]
        elif self.mark_version == "mark4.6":
            self.classes = ["water_toilet", "others"]
        elif self.mark_version == "mark4.7":
            self.classes = ["water_shower", "others"]
        elif self.mark_version == "mark4.8":
            self.classes = ["dog_bark", "others"]
        elif self.mark_version == "mark5.0":
            # [변경] 'dummy_label' 제거. mark5 학습/평가 경로는 load_and_segment_with_metadata만
            # 사용하고 unlabeled는 label=-1로 처리되어 dummy_label이 실제로 쓰이지 않음(잔재).
            # student 출력/loss는 아래 9개 클래스로 고정한다.
            self.classes = [
                "heavy_impact", "dragging", "construction", "machine_noise",
                "media_talking", "water_toilet", "water_shower", "dog_bark",
                "others",
            ]
        elif self.mark_version == "mark5.2":
            # [신설 2026-09-15] Mark5.2 — 데이터 재구성판. media_talking(TV 만) 을 talking(TV + 사람 말소리
            # 50:50) 으로 바꾸고 others 를 "층간소음이 아닌 먼 소리" 로 다시 정의했다(09-07·09-11·09-15 결정).
            # 클래스 순서는 mark5.0 과 같다(split 배분·teacher 순서가 이 순서를 따른다).
            self.classes = [
                "heavy_impact", "dragging", "construction", "machine_noise",
                "talking", "water_toilet", "water_shower", "dog_bark",
                "others",
            ]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"
                f"지원되는 값: ['mark4.1', 'mark4.2', 'mark4.3', 'mark4.4', 'mark4.5', 'mark4.6', 'mark4.7', 'mark4.8', 'mark5.0', 'mark5.2']"
            )

        # === 기존 속성 유지 ===
        # labeled_classes는 이제 파서가 사용할 모든 클래스를 의미하게 됨
        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)


        # ==============================================================================
        # 2. 오디오 및 모델 공통 파라미터
        # ==============================================================================
        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)
        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === Segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher가 사용할 최대 segment 수
        # [신설 2026-09-29 / ablation A3] student 가 조각 5개 대신 클립 3초를 통째로 본다(--clip_input, run_tag _noSEG).
        #   teacher 8개는 그대로 조각(101프레임)을 본다 — Mark4.x 가 그 길이로 학습됐다. 조각별 teacher 출력은
        #   train_mark5.merge_segments_to_clip 이 비교모델 합의 라벨(Mark5_EXP2/model/make_consensus_labels.py)과 같은 규칙으로
        #   클립 하나로 합친다(조각 채택 = 확신도·1등−2등 차이 조건, 클립 채택 = 채택 조각 clip_min_segments 개 이상,
        #   라벨 = 채택 조각 argmax 다수결, 확신도 = 채택 조각 평균, 특징 = 채택 조각 확신도 가중 평균).
        #   student 클립 파형 = 파일 첫 student_clip_samples 샘플(모자라면 0) — 비교모델 common/data.py 의 자르기와 같다.
        #   False(기본)면 지금까지의 조각 단위 학습·평가 그대로. mn10(파형 입력) 에서만 뜻이 있다.
        # [변경 2026-09-29 밤 / 격차 분석] mark5.2 는 클립 입력이 기본이다. 성긴 클립(소리가 3초 중 일부) 59개에서 조각 방식의 오답률이
        #   11.5%(EfficientAT 2.4%)였고 그 29/34 가 →others 였다(조각에 클립 라벨을 붙이는 라벨 잡음 + 배경 조각의 확신도² 가중).
        #   run_tag 표식은 옛 산출물(조각 방식 = 표식 없음)과 겹치지 않게 클립 모드에 _noSEG 를 그대로 붙인다. --segment_input 으로 되돌림.
        self.student_clip_input = (self.mark_version == "mark5.2")
        self.student_clip_seconds = 3.0
        self.student_clip_samples = int(self.sample_rate * self.student_clip_seconds)
        self.clip_min_segments = 3      # 클립 채택에 필요한 채택 조각 수(5개 중 과반). 합의 라벨 스크립트 --min_segments 기본값과 같다
        # [신설 2026-09-30 밤 / 가짜 정답 재료] 조각 teacher 의 출력 K개를 클립 하나로 합치는 규칙(train_mark5.merge_segments_to_clip 의 rule).
        #   "vote"(기본·옛 동작): 조각마다 채택(확신도·1등−2등 차이) → 채택 조각 ≥ clip_min_segments 면 클립 채택 → 라벨 = 채택 조각 다수결.
        #   "mean"              : 전문가별로 조각 K개의 확신도를 평균 → 그 평균 8개로 클립을 한 번에 채택(최대 ≥ 0.5 & 1등−2등 ≥ gap),
        #                         라벨 = 평균 확신도 1등 전문가의 클래스, 확신도 = 그 평균. --teacher_seg_mean 으로 켠다(run_tag _SM).
        #   근거(2026-09-30 실측 — 3차 조각 teacher, 증강 없는 원본, unlabeled 2,000클립을 정답 라벨과 대조): vote 채택 1,288·오답 50
        #   (정밀도 0.9612·회수율 0.6963) → mean 채택 1,339·오답 36(0.9731·0.7328). 4차 클립 teacher 는 1,281·오답 50. unlabeled 부트스트랩
        #   F1 차이 mean − vote +2.86%p [+2.00, +3.73]. 조각 teacher 에서만 뜻이 있다 — 클립 teacher(K=1)는 두 규칙이 같아 train_mark5 가 멈춘다.
        self.clip_merge_rule = "vote"
        # [신설 2026-09-29 밤] teacher(Mark4.x)가 클립 단위(64×301)로 학습됐는지 — None 이면 EnsembleTeacher 가 .pth 8개의 provenance
        #   (clip_input) 를 읽어 정한다(8개가 섞이면 오류). True 면 teacher 입력이 클립 mel 이고 조각 수 K=1, 클립 채택 조각 수 1.
        self.teacher_clip_input = None
        self.teacher_clip_frames = 301

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True
        # [추가] background embedding 보조 loss 가중치 (use_background_embedding=True일 때만 의미 있음)
        self.background_embedding_weight = 0.1
        # [신설 2026-10-01 / 지도만 격차 진단] 기본값 스냅샷 — --bg_weight 로 바꿨을 때만 run_tag 에 _BG<값> 이 붙게 한다(ablation_tag).
        self._default_bg_weight = float(self.background_embedding_weight)
        # [신설 2026-10-01 / 지도만 격차 진단] student 에게 주는 클립 파형을 최대 진폭 1 로 정규화할지.
        #   True(기본·옛 동작) : utils/parser_utils.load_audio_file 이 파일마다 |x| 최댓값으로 나눈 파형을 teacher mel 과 student 가 같이 쓴다.
        #   False(--student_raw_wave, run_tag _RW) : student 클립 파형만 원본 진폭으로 되돌린다(정규화 파형 × 그 파일의 최댓값).
        #        teacher 8개는 정규화 파형의 mel 로 학습됐으므로 그대로 둔다. 비교모델(Baseline_*/common/data.py load_clip)은 정규화하지 않는다.
        #   2026-10-01 실측: teacher 를 한 번도 부르지 않는 지도만(u0) 조건에서 같은 mn10 백본인데 test Accuracy 가 EfficientAT 0.9633 대
        #        우리 0.9349(5회 평균). 두 학습 경로를 대조해 남은 차이가 헤드·새 층 학습률·LP-FT·BG 보조 손실·이 정규화였다(전처리·SpecAugment·
        #        배치·옵티마이저·스케줄·체크포인트 기준은 같음). 정규화는 소리 크기 정보를 지운다 — others 를 '먼 소리'로 다시 정한 뒤로는
        #        크기가 단서일 수 있다. 원인인지는 진단 회차(_RW)로 확인한다. 클립 입력 mn10 student 에서만 쓴다.
        self.student_wave_normalize = True
        # [신설 2026-08-22] 평가에서 others 로짓에 background embedding 을 max 로 덮어쓸지 여부.
        # False 가 기본이다(=덮어쓰지 않는다).
        #
        # 1차 학습(val 430) 결과에서 accuracy 0.1209 인데 ROC AUC 는 0.8871 이 나와 원인을 추적한
        # 결과 이 override 였다. prediction_details CSV 실측: 430클립 중 417개(97.0%)에서 others 가
        # 1등이었고, others 가 아닌 383개 중 316개(82.5%)는 정답이 정확히 2등이었다. others 열을
        # 빼고 argmax 만 다시 계산하면 8클래스 정확도가 0.859 였다. 즉 모델은 제대로 배웠는데
        # others 하나가 모든 클립 위에 일률적으로 얹혀 있었다(Prob_others 평균 0.81, 정답이
        # others 인 클립 0.95 와 거의 차이가 없음).
        #
        # 구조상 그럴 수밖에 없다. 학습(train_mark5.py:533)은 background 를 others 샘플 feature 의
        # 평균 방향으로 당기는 보조 loss 만 걸고 **분류 로짓에는 넣지 않는다.** 그런데 평가만
        # others_logit = max(text_logit, cos(feature, bg)/logit_temperature) 로 덮었다. max 라
        # others 를 올리기만 하고 내리지는 못하는데다, 같은 인코더가 뽑은 feature 라 다른 클래스도
        # bg 와 방향이 겹치고, logit_temperature(0.07)로 나누면서 14.3배로 증폭된다.
        # (학습 종료 시 bg_loss 0.1040 = cos(others feature, bg) 약 0.896.)
        #
        # mark4.8 이 2026-07-12 에 잡은 background embedding override 와 같은 물건이다. 다만
        # 거기서는 Hard loss 가 랜덤 근처에서 안 움직여 "학습부터 꺼야 한다"는 결론이었고,
        # mark5 는 override 가 평가에만 있어 Hard loss 가 1.396->0.495 로 정상 수렴했다.
        # 그래서 재학습 없이 평가만 다시 돌리면 된다.
        #
        # 보조 loss 자체(use_background_embedding)는 끄지 않는다. others feature 를 한 방향으로
        # 모으는 것은 해롭지 않고, 파괴적인 것은 평가 시점의 max 덮어쓰기뿐이다.
        self.use_background_override_at_eval = False
        self.use_text_aligned_student = True
        self.use_feature_kd = True
        self.feature_kd_weight = 0.3
        self.feature_kd_loss_type = "cosine_l1"

        # === DKD (Decoupled Knowledge Distillation, Zhao et al. 2022) ===
        # [신설 2026-08-09] soft loss 계산 방식을 vanilla KD <-> DKD 로 갈아끼우는 스위치.
        # False 면 기존 경로(단일 KL divergence)를 그대로 타므로 지금까지의 학습과 동일하다.
        #
        # 왜 mark5 에서야 넣는가: DKD 는 KD 손실을 TCKD(타겟 vs 비타겟 이진 분포)와
        # NCKD(비타겟 클래스들끼리의 분포)로 분해한다. mark4.x 는 2-class 라 비타겟이 1개뿐이라
        # NCKD 가 항상 0 이고 DKD 가 vanilla 의 상수배(no-op)였다(2026-07-15 확인). 9-class 인
        # mark5 에서 처음으로 의미를 갖는다.
        #
        # ⚠타겟 클래스는 **teacher 앙상블의 argmax** 를 쓴다. 원논문은 supervised 세팅이라
        # ground-truth 라벨로 타겟을 정하는데, mark5 의 KD 는 unlabeled 에 걸리므로 정답이 없다.
        # 이미 teacher 의 판정을 믿고 배우는 구조라 신뢰 구조가 새로 나빠지지는 않지만,
        # **원논문 그대로가 아닌 변형**이므로 논문에 밝혀야 한다.
        #
        # beta=1.0 인 이유: vanilla KD 를 분해하면 KD = TCKD + (1 - p_t)·NCKD 라서 teacher 가
        # 확신할수록(p_t -> 1) NCKD 가 눌린다. beta=1.0 은 "그 눌림만 푼다"는 뜻이고 그 이상
        # 증폭하지 않는다. 원논문 권장값(CIFAR-100 beta=8, ImageNet beta=2)은 클래스가 100~1000개
        # 일 때의 값이라, 9-class 인 여기서는 정보가 아니라 노이즈를 키울 위험이 크다.
        self.use_dkd = False
        self.dkd_alpha = 1.0
        self.dkd_beta = 1.0
        self.visual_view_type = "mel_delta"
        self.segment_selection_mode = "salient_topk"
        self.max_visual_segments = self.max_segments
        self.logit_temperature = 0.07
        self.segment_aggregation_mode = "confidence_saliency"
        self.segment_confidence_power = 2.0
        self.segment_saliency_power = 1.0
        # === [신설 2026-08-22] others 결정 방식 ===
        # "direct"     : Prob_others >= others_direct_threshold 이면 others (기본)
        # "confidence" : 옛 방식 — top_conf/margin/entropy 세 조건 중 하나라도 걸리면 others 로 강제
        #
        # 옛 방식은 "모르겠으면 others" 라서 모델의 전반적인 확신도에 결과가 휘둘린다.
        # pseudo-label CE 로 student 확신도가 0.4384 -> 0.8047 로 오르자 같은 임계값에서
        # 강제 건수가 204 -> 22 로 줄며 others recall 이 0.574 -> 0.191 로 무너졌는데,
        # 정작 others 단독 AUC 는 0.9754 -> 0.9871 로 올라 있었다. 규칙이 정보를 못 꺼낸 것이다.
        # 근거 실측과 스윕표는 vild/postprocess_utils.py 의 apply_others_calibration docstring 참조.
        self.others_decision_mode = "direct"
        # [갱신 2026-08-23] 0.14 -> 0.09. 최종 채택 모델이 mark5.0_PL_SA(SpecAugment 포함)로
        # 바뀌면서 확률 분포가 이동해 val 430클립 스윕을 다시 돌린 값이다
        # (SA 모델: thr 0.09 -> acc 0.8814 / macroF1 0.8820. thr 0.14 였다면 0.8674/0.8667).
        # 0.14 는 SpecAugment 없는 mark5.0_PL 의 최적값이었다.
        # ⚠ val 에서 고른 값이라 낙관 편향이 있고, 재학습하면 반드시 다시 스윕한다(모델 종속).
        self.others_direct_threshold = 0.09
        # === [신설 2026-09-16, Mark5_EXP2 2026-09-07 방식 그대로] others 임계값을 어떻게 정하나 ===
        # 위 0.09 는 2026-08-23 에 mark5.0_PL_SA 하나의 val 로 사람이 고른 값이다. Mark5_EXP2 에서 조건마다
        # val 에서 고른 값이 u0 0.18~0.20 · ceil 0.18~0.24 · u2000 0.06~0.11 로 갈렸고(2026-09-07 실측),
        # 운영점은 모델마다 val 에서 고르는 것이 비교의 전제라 회차마다 자동으로 고른다.
        #   "val_select" (기본): eval.py --split val 이 후보(argmax + direct 격자)를 훑어 val Macro F1 최대값을 골라
        #                 plots/others_threshold_<run_tag>.json 에 저장(체크포인트 sha256 포함)하고,
        #                 --split test 는 그 파일을 읽고 지문이 맞을 때만 쓴다. run_all.py 가 val -> test 순서로 돈다.
        #   "fixed"     : 위 others_direct_threshold 를 그대로 쓴다(옛 결과 재현용). eval.py --others_threshold X 로도 고정 가능.
        self.others_threshold_mode = "val_select"
        self.others_threshold_grid = (0.02, 0.95, 0.01)     # (시작, 끝, 간격). 끝 포함
        self.others_threshold_criterion = "macro_f1"        # val 에서 최대화할 지표(현재 macro_f1 만 지원)
        # 아래 셋은 others_decision_mode="confidence" 일 때만 쓰인다.
        self.others_confidence_threshold = 0.45
        self.others_margin_threshold = 0.05
        # [삭제 2026-07-11] others_entropy_threshold 하드코딩(0.82) 제거, 아래 property로 대체.
        # mark4_refactored에서 발견: 이 값이 클래스 수와 무관한 절대값으로 고정돼 있으면,
        # 클래스 수가 다른 mark_version 사이에서 confidence_threshold와 어긋난 엄격도로 작동함
        # (mark4.x 2-class에서 confusion matrix 완전 붕괴의 근본 원인이었음, 자세한 내용은
        # mark4_refactored/vild/vild_config.py의 동일 property 주석 참조).
        # mark5(9-class)는 기존 하드코딩값(0.82)이 우연히 confidence_threshold(0.45)와 거의
        # 같은 엄격도(top_conf≈0.47 지점)였어서 이 변경으로도 동작이 거의 그대로 유지된다.
        # [비활성 2026-08-21] class-pair 보정을 끈다(빈 dict = 아무 짝도 손대지 않음).
        # 이 보정은 두 클래스의 확률 격차가 threshold(0.03) 미만이면 둘을 평균내 동점으로
        # 만드는데, 동점은 margin=0 이라 바로 뒤 others 보정(margin<0.05 면 others 강제,
        # postprocess_utils.py:38)에 반드시 걸린다. 즉 이 보정이 예측을 실제로 바꾸는 경우
        # (그 짝이 top-2 인 경우)마다 그 결정이 others 로 덮여, 결정 기구로서는 죽어 있었다.
        # 게다가 동점을 argmax 로 풀면 클래스 목록에서 인덱스가 앞선 쪽(construction,
        # water_toilet)이 항상 이기는데, 이건 측정으로 정한 우선순위가 아니라 코드에 적힌 순서다.
        # 끄는 결정적 이유는 스윕 오염이다. 보정이 걸린 클립은 prediction_details CSV 의
        # Raw Margin 이 실제 값(0.03 미만의 어떤 값)이 아니라 인위적인 0 으로 기록되어,
        # 나중에 others margin 임계값을 0~0.03 구간에서 스윕할 때 정작 제일 궁금한 짝 클래스
        # 구간의 숫자를 믿을 수 없게 된다. 빈 dict 면 apply_class_pair_calibration 이
        # 루프를 한 번도 안 돌고 입력을 그대로 반환한다.
        self.class_pair_margin_overrides = {}
        self.enable_temporal_smoothing = True
        self.temporal_smoothing_alpha = 0.65
        self.enable_abstention = False
        self.abstention_confidence_threshold = 0.40
        # 단일 분류 경로: 증류 변환층은 feature KD 학습에만 사용한다.
        self.distill_branch_eval_weight = 0.0
        # 분류 출력은 지도 갈래로 고정하며 평가 시 혼합 비중·클래스 편향을 탐색하지 않는다.
        self.distill_eval_weight_candidates = (0.0,)
        self.class_bias_grid = None
        self.class_bias_rounds = 3
        self.explain_topk_segments = 3
        self.save_visual_explanations = False   # [Mark5_EXP2] 반복실험에서는 세그먼트 그림을 안 남긴다
        # === [변경 2026-09-17 / Mark5.2 6절] student 인코더 선택 ===
        #   "cnn"  : SimpleAudioEncoder(3층 CNN, mel 64x101 입력) — mark5.0 까지의 인코더. 사전학습 없음.
        #   "mn10" : Mn10AudioEncoder(EfficientAT mn10, AudioSet 사전학습, 파형 입력) — Mark5.2 기본.
        # 2026-09-08 결정: 백본은 노벨티와 무관하고 3층 CNN(143,488 파라미터)이 표현 병목이라 비교모델과
        # 같은 사전학습 백본으로 바꾼다. "0.44M 가볍다" 주장은 접는다(mn10 은 4.2M·17MB 안팎).
        # 인코더 외(세그먼트 5개·teacher 합의·두 갈래 헤드·텍스트 앵커·others 규칙·SpecAugment·feature KD)는 그대로.
        self.encoder_type = "mn10" if self.mark_version == "mark5.2" else "cnn"
        # mn10 세부(encoder_type == "mn10" 일 때만 쓰인다). 규격 자체(32 kHz·128 mel·hop 320)는 사전학습을
        # 따라야 하므로 vild_model.Mn10AudioEncoder 의 상수이고, 여기서는 가중치·헤드·학습률만 고른다.
        self.mn10_pretrained_path = None      # None 이면 <저장소>/weights/mn10_as_mAP_471.pt (저장소 릴리스 v0.0.1, MIT)
        self.mn10_load_pretrained = True      # False 는 "사전학습 없이 같은 구조" 절제 실험용
        self.mn10_proj_dropout = 0.3          # 풀링 960 -> Dropout -> Linear(embedding_dim). SimpleAudioEncoder 헤드의 0.3 과 같게
        # 백본 학습률. 비교모델이 2026-09-05 학습률 선정 3회로 고른 5e-5 를 사전학습 백본에 쓰고, 새로 만드는 층
        # (투영·branch·classifier·background)은 아래 learning_rate(1e-4) 그대로(2026-09-17 결정). None 이면 전부 learning_rate.
        self.encoder_learning_rate = 5e-5 if self.encoder_type == "mn10" else None
        # [추가 2026-08-17] mark4.x teacher 8개는 2026-07-13 teacher 강화 이후 4블록
        # TeacherAudioEncoder(약 49만 파라미터)로 학습돼 있다. mark5 의 EnsembleTeacher 가
        # 그 .pth 를 읽을 때 같은 구조로 만들어야 하므로 True 로 둔다.
        # False 로 두면 student 용 SimpleAudioEncoder(3블록)를 만들어 state_dict 가 안 맞는다.
        # (student 자신의 인코더는 encoder_type 이 정하고 이 값의 영향을 받지 않는다.)
        self.use_large_teacher_encoder = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80
        self.learning_rate = 1e-4
        # [추가 2026-08-17] mark4.x 와 학습 절차를 같게 맞추기 위한 값들.
        # mark4.x 는 weight_decay 가 없어 val best 를 이른 에폭에 찍고 곧장 과적합하던 것을
        # 실측하고 1e-4 를 넣었다. early stopping patience 10 도 mark4.x 와 같은 값이다.
        self.weight_decay = 1e-4
        self.early_stopping_patience = 10
        # [추가 2026-08-17] KD 하이퍼파라미터. 값은 컨퍼런스 코드(mark3.2)에서 검증한 0.7 / 4.0 그대로이고,
        # train_mark5.py 에 하드코딩돼 있던 것을 config 로 뺀 것뿐이다.
        # mark4.x 의 0.3 / 2.0 은 같은 샘플에 hard·soft 를 함께 거는 구조에서 나온 값이라 옮겨오지 않는다.
        # [변경 2026-08-22] 0.7 -> 0.3. 아래 use_pseudo_label_ce 로 unlabeled 신호가
        # "teacher 분포 따라하기(KL)" 에서 "teacher 가 고른 정답으로 CE" 로 바뀌면서
        # alpha 의 의미가 달라졌다. 이제 두 항이 모두 CE 이고, unlabeled 쪽 정답은
        # 사람 라벨이 아니라 정확도 0.894 짜리 pseudo-label 이다. 정확도가 낮은 쪽이
        # 사람 라벨보다 2.33 배 무거우면 안 되므로, mark4.x 와 같은 0.3 으로 내린다.
        self.distill_alpha = 0.3
        self.distill_temperature = 4.0
        # [추가 2026-08-18] teacher fusion 의 others 칸 조립 규칙: "min" | "mean" | "max".
        # fusion_score_mode 가 "raw" 일 때만 쓰인다("margin" 이면 others 칸이 -max(margin) 로
        # 정해져 이 값이 무시된다). 실측 근거는 vild/teacher_fusion.py docstring 참조.
        self.fusion_others_rule = "min"

        # === [신설 2026-08-22] teacher 조립 방식과 pseudo-label 학습 ===
        # 배경: 1차 학습 뒤 실측에서 (a) fused teacher 자신의 val 정확도가 0.7488 로
        # student(0.8395)보다 낮고, (b) T=4.0 에서 soft target 의 정규화 엔트로피가 0.9704 로
        # 거의 균등분포라 KL 이 나르는 정보가 0.0113 nats 뿐이고, (c) 그런데도 teacher 의
        # argmax 는 non-others 에서 0.8940 로 꽤 맞는다는 것이 확인됐다.
        # 결론: teacher 에게서 분포를 받지 말고 **정답만** 받는다. 그게 mark3.0.0 의 원래
        # hybrid(soft label 의 argmax 를 hard label 로 쓰는 방식)이기도 하다.

        # 담당 칸에 target 로짓 원본을 넣을지(raw) margin(target-others)을 넣을지(margin).
        # margin 이 전문가별 스케일 차이(평균 2.07, 표준편차 3.8배)를 상쇄한다.
        self.fusion_score_mode = "margin"
        # unlabeled 손실을 KL(분포 따라하기) 대신 pseudo-label CE 로 건다.
        self.use_pseudo_label_ce = True
        # pseudo-label 채택 임계값. 담당 전문가의 "내 클래스" 확률(specialist positive weight)
        # 최대값이 이 값 미만이면 그 샘플은 학습에서 통째로 뺀다. 8명 중 아무도 확신하지 못한
        # 클립에 억지로 정답을 붙이면 그 오답이 그대로 student 에게 주입되기 때문이다.
        # 0.5 는 "적어도 한 명은 자기 것이라고 말한다"는 뜻이다.
        self.pseudo_label_min_confidence = 0.5
        # 채택된 샘플에 per-sample 가중치를 준다. 가중치는 그 최대 확률값 자체.
        # 확신이 0.95 인 클립과 0.52 인 클립을 같은 무게로 배우지 않게 한다.
        self.pseudo_label_confidence_weighting = True
        # [신설 2026-09-21 밤 / B] 채택 조건에 "1등 − 2등 전문가 확신도 차이 ≥ 이 값" 을 더한다(teacher_disagreement).
        # 근거(2차 학습의 teacher 캐시 10,000 세그먼트를 provenance 정답과 대조): 확신도 ≥0.5 만 걸면 채택 86% 에
        # 타겟 정확도 75%(water_toilet 51%·dragging 41%·machine_noise 56%) — 전문가 2명 이상이 0.5 넘게 손드는 세그먼트가
        # 58% 였고 확신도로는 못 거른다(0.9~0.95 구간 79%). 차이 ≥0.4 를 더하면 채택 37%·정확도 92.5%.
        # 0 이면 옛 동작. 태그 _G<값> 이 붙는다(예: 0.4 → _G04).
        self.pseudo_label_min_gap = 0.4
        # 정답 CE와 가짜 정답 CE는 지도 갈래의 분류기에만 적용한다.
        self.distill_branch_classify = False
        self.classification_policy = "supervised_argmax_v1"
        # distill branch 마지막 ReLU. 기본은 뺀다(근거는 vild/vild_head.py docstring).
        self.distill_branch_final_relu = False

        # === [신설 2026-08-22] 분포 정렬 (distribution alignment) ===
        # pseudo-label 의 클래스 분포가 균등에서 벗어난 것을 학습 단계에서 잡는다.
        # 실측(unlabeled 300클립 1500세그먼트): construction 19.2% / others 2.4% (균등 11.1%).
        # 그 편향이 test 혼동행렬에 그대로 — construction FP 21건(precision 0.687),
        # others recall 0.660. test 에서 클래스 가중치를 직접 최적화하면 acc 0.9000 까지 오르고
        # 그때 필요한 가중치가 construction 0.25 / others 2.2 로 이 편향의 역수 방향이다.
        #
        # ⚠사후 보정은 이미 실패했다 — val 430클립으로 클래스 가중치를 맞추니 val 은
        # 0.8814 -> 0.9023 인데 test 는 0.8581 -> 0.8581 로 전혀 안 움직였다(파라미터 10개를
        # 430샘플로 맞춰 과적합). 그래서 학습 단계에서 고친다. 여기서는 unlabeled 세그먼트
        # 10,000개 위에서 분포를 추정하므로 표본이 23배 크고 추정·적용 대상이 같다.
        # ★[결론 2026-08-23] 끈다. 재학습으로 검증한 결과 student 성능이 개선되지 않았다.
        #
        # 1) teacher 는 확실히 좋아졌다. val 430클립에 teacher 를 통과시킨 스윕
        #    (model/sweep_dist_align.py):
        #      tau    off     0.25    0.5     0.75    1.0
        #      acc    0.8744  0.8860  0.8860  0.8930  0.8977   (+2.33%p)
        #      othR   0.362   0.404   0.404   0.426   0.447    (+8.5%p)
        #      균등L1  0.1736  0.1550  0.1550  0.1354  0.1261
        #    실측 편향(teacher 관측 분포, 균등 = 0.1111):
        #      construction 0.1479(x0.75) · machine_noise 0.1479(x0.75) · others 0.0539(x2.06)
        #
        # 2) 그런데 tau=1.0 으로 재학습한 student 는 나아지지 않았다. 각자의 최적 others
        #    임계값에서 공정 비교한 val 430 결과:
        #      PL       acc 0.8814 / macroF1 0.8807  (thr 0.14)
        #      PL+DA1   acc 0.8791 / macroF1 0.8772  (thr 0.11)
        #    차이가 1클립이고, 정답이 top-2 안에 드는 비율은 0.9442 로 **완전히 같다**.
        #    바뀐 것은 클래스 배분뿐이다 — others +0.128(0.787->0.915) 를 얻고
        #    machine_noise -0.083(0.750->0.667) 을 잃어 총합이 상쇄됐다.
        #
        # 3) 왜 전달되지 않았나: student 가 이미 teacher 와 같은 수준이었다
        #    (student 0.8814 vs teacher 0.8744). 선생이 좋아져도 학생이 이미 그만큼 하고 있으면
        #    배울 것이 없다. 게다가 teacher 신호는 alpha=0.3 으로 가중되고 나머지 0.7 은
        #    labeled hard CE 다. 같은 이유로 DKD 도 접었다(2026-08-23).
        #
        # 코드는 남겨둔다. teacher 와 student 의 격차가 큰 상황(예: labeled 를 크게 줄인
        # 저라벨 실험)에서는 다시 의미가 생길 수 있고, 그때 True 로 켜면 된다.
        self.use_distribution_alignment = False
        # 보정 강도. 0=no-op, 1=관측 분포를 목표 분포로 완전 정렬.
        # 1.0 은 위 스윕에서 teacher 품질이 최고였던 값이라 켤 때의 기본값으로 남겨둔다.
        self.dist_align_tau = 1.0
        # 관측 분포 EMA. 0.999 면 최근 수천 배치의 분포에 수렴한다.
        self.dist_align_momentum = 0.999

        # === [신설 2026-08-23] SpecAugment (학습 시 온라인 증강) ===
        # mel 배치의 주파수 대역·시간 구간을 무작위로 가린다. 학습 루프에서만 걸리고
        # (Dataset 은 파싱 결과를 초기화 때 고정하므로 파서에 넣으면 epoch 마다 같은 마스크가
        # 나와 증강이 아니게 된다), val 루프와 eval.py 는 원본을 그대로 본다.
        #
        # 왜: (1) 클래스당 학습 클립이 222개뿐이라 결정 경계가 얇다 — test 에서 오답의
        # 대부분(53클립)이 "정답이 2위"였다. 마스킹은 한 가지 단서에 매달리는 지름길을 막아
        # 그 경계를 두껍게 만드는 표준 기법이다(SpecAugment, Park et al. 2019).
        # (2) 증강이 없으면 teacher 가 같은 입력에 늘 같은 값을 내므로 online KD 가
        # offline 과 수치적으로 동일했다(train_mark5.py 의 캐시 주석이 스스로 밝히던 사실).
        # 증강이 들어가면 매 epoch 다른 view 가 teacher 에 들어가 실시간 재계산이 실제로
        # 의미를 갖는다.
        self.use_spec_augment = True
        # 보수적인 표준 강도. 64 mel 중 최대 8칸짜리 주파수 마스크 2개,
        # 101 프레임 중 최대 16칸짜리 시간 마스크 2개.
        self.spec_augment_freq_mask_param = 8
        self.spec_augment_num_freq_masks = 2
        self.spec_augment_time_mask_param = 16
        self.spec_augment_num_time_masks = 2
        # teacher 가 어느 view 를 보는가.
        #   "augmented" : student 와 같은 증강 view (기본) — 진짜 online. 증강마다 pseudo-label 이
        #                 재계산되고, 가려진 view 에서 앙상블 합의가 깨지는 클립은 선별
        #                 (pseudo_label_min_confidence)이 걸러낸다.
        #                 ★ mn10(파형) 경로에서는 "같은 view" 가 아니다: teacher 는 64-mel 에 건 마스크를 보고, student 는
        #                 인코더 안 128-mel 에 따로 뽑은 마스크를 본다(규격이 달라 위치를 맞출 수 없다). 둘 다 증강 view 이지만
        #                 마스크는 독립이다 — 논문에는 "teacher·student 각각 독립 SpecAugment" 로 쓴다. (09-17 검수 P4)
        #   "clean"     : teacher 는 원본, student 만 증강 view (FixMatch 식 일관성 정규화) —
        #                 pseudo-label 품질(0.894)이 보존되는 안전한 대안.
        #                 [2026-09-30 밤 / 가짜 정답 재료] --teacher_clean_view 로 켠다(run_tag _CT). 실측(4차 클립 teacher, unlabeled 2,000클립):
        #                 원본 입력이면 타겟 가짜 정답 1,281개 중 오답 50(정밀도 0.961), 증강 입력(학습 마지막 에폭 캐시)이면 1,289개 중
        #                 오답 86(0.933) · 전원 거절→others 는 111개 중 진짜 others 107 → 133개 중 106. 비교모델이 받는 합의 라벨 CSV 는
        #                 원본 입력으로 만든 것이라, 원본으로 두면 Mark5.2 와 비교모델이 같은 재료를 받는다. 대신 가짜 정답이 에폭마다 같아진다.
        self.spec_augment_teacher_view = "augmented"

        # ==============================================================================
        # [신설 2026-08-30 / Mark5_EXP 전용 → 2026-09-17 Mark5.2 코드 기준으로 재정렬] 반복실험 설정
        # ==============================================================================
        # 세 값 모두 CLI 인자로 덮어쓰며, 아무 인자도 안 주면 Mark5.2 본체와 완전히 같은 동작이 된다.
        #   experiment_seed : 이 학습에 쓴 seed. run_tag 에 반영돼 산출물이 안 덮인다. None = 태그 없음(seed 42).
        #   unlabeled_n     : unlabeled 를 몇 개만 쓸지(데이터효율곡선). None = 전량. 0 이면 지식증류가 통째로 꺼진다.
        #   labeled_n       : [신설 2026-09-17 / 8절] labeled 를 몇 클립만 쓸지(라벨 효율 표: 200·500·1,000).
        #                     None = 전량(2,000). 클래스별로 고르게(n/9 씩, 나머지는 앞 클래스부터 1개씩) seed 로 뽑는다.
        #                     val·test 는 그대로 430. 태그 _l<n> 이 _u<n> 앞에 붙는다.
        #   cudnn_deterministic : 같은 seed 로 GPU 연산 순서까지 고정(10~20% 느림, 완전 재현).
        self.experiment_seed = None
        self.unlabeled_n = None
        self.labeled_n = None
        self.cudnn_deterministic = True
        # [신설 2026-09-17 / 8절 pseudo-label 출처 비교] "왜 하필 이진 전문가들의 합의인가"에 숫자로 답하는 실험.
        #   pseudo_source : None = 전문가 8개 온라인 합의(우리 방식, 태그 없음)
        #                   "self" = 라벨 200으로 배운 student 가 스스로 붙인 라벨 CSV / "gen" = 단일 generalist(EfficientAT) CSV
        #                   CSV 모드에서는 teacher 를 부르지 않고 feature KD 가 없다(teacher 특징이 없으므로). 태그 _ps<이름>.
        #   spill_labeled : True 면 labeled_n 으로 뽑고 남은 labeled 클립(라벨을 떼고)을 unlabeled 풀에 합친다.
        #                   labeled 200 이면 unlabeled 2,000 + 1,800 = 3,800(체크리스트 8절 조건). 태그 _sp.
        self.pseudo_source = None
        self.spill_labeled = False
        # [신설 2026-09-06 / Mark5_EXP] 천장(상한) 조건. True 면 unlabeled 에 가려 둔 진짜 라벨을 붙여 지도학습으로 돈다.
        # teacher 는 부르지 않는다. Cances et al. (2022) 의 "Supervised, 라벨 100%" 기준선. run_tag 에 _ceil(_u<n> 대신).
        self.ceiling = False
        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 데이터 경로 ===
        self.audio_dir = os.path.join("data_wav")
        self.prompt_bank_path = os.path.join(SHARED_DIR, "resources", "prompt_bank.json")

        # === [신설 2026-09-25 / LR] 학습률 기본값 스냅샷 ===
        # run_tag 의 lr_tag 가 "명령줄(--lr/--encoder_lr)이 기본값과 다른 값을 줬는가" 를 판단하는 기준이다.
        # 위에서 정한 learning_rate(1e-4)·encoder_learning_rate(mn10: 5e-5) 를 그대로 복사한다.
        # 이 줄 뒤에서 두 값을 바꾸는 코드가 생기면 스냅샷도 그 뒤로 옮겨야 한다(2026-09-25 기준 없음).
        self._default_lrs = (float(self.learning_rate),
                             None if getattr(self, "encoder_learning_rate", None) is None else float(self.encoder_learning_rate))

        # [이식 2026-09-26 / Mark5.2] 앵커 중심화(AC)·학습형 로짓 스케일(LT)은 본 모델의 구성 요소다. 끄는 경로는 없고
        # run_tag 에는 언제나 _AC_LT 가 붙는다(AC/LT 없이 돌린 2026-09-24 이전 산출물과 이름이 겹치지 않게).
        #  anchor_centering    : 텍스트 앵커에서 평가용 9클래스 앵커의 평균 벡터를 뺀다(제로샷 목록에도 같은 벡터를 뺀다).
        #  learnable_logit_scale: ViLDTextHead 의 temperature 를 CLIP 처럼 학습 파라미터로(vild_model.py 주석 참조).
        self.anchor_centering = True
        self.learnable_logit_scale = True
        # [이식 2026-09-29 / Mark5.2 → Mark5_EXP2] 헤드 진단 H · 격차 분석 대응 묶음(PH·LP-FT·HW·OR·NEG). 본체와 같은 필드·같은 기본값.
        # [신설 2026-09-29 / 헤드 진단 H] student 분류 헤드. "text_anchor"(기본) = ViLDTextHead(텍스트 앵커 코사인 + 위 AC/LT).
        #   "linear" = vild_model.LinearClassHead(nn.Linear(embedding_dim, 9)) — 비교모델 EfficientAT 의 mlp 헤드 마지막 층과 같은 방식.
        #   --linear_head 로만 켠다(run_tag _LIN). 제로샷이 없어진다. 처음(09-29)에는 "2%p 차이가 헤드에서 나는가" 를 보는 진단용이었고, 2026-10-07 지도만 격차 진단(D0)으로
        #   본 레시피에 채택했다(config 기본값은 그대로 text_anchor — 노트북·실행기가 --linear_head 로 켠다).
        #   이때 anchor_centering·learnable_logit_scale 은 헤드가 안 쓰므로 뜻이 없다(run_tag 의 _AC_LT 는 이름 규칙상 그대로 붙는다).
        self.student_head = "text_anchor"
        # === [신설 2026-09-29 밤 / 격차 분석 대응 묶음] 아래 다섯 가지는 --플래그로 켠다(기본 = 옛 동작). 근거는 Downloads
        #     Mark5.2_격차분석_2.4%p_원인과대응_20260929.txt. run_tag 표식은 ablation_tag 에 _PH·_LP<N>·_HW1·_OR<τ>·_NEG 순서.
        #   head_pretrained_hidden : mn10 의 사전학습 은닉층(classifier: Linear 960→1280 + Hardswish + Dropout, MobileNetV3 last_channel)을
        #                            버리지 않고 살려 그 위에 Linear(1280→384) 투영 하나만 둔다. supervised branch 의 LayerNorm+ReLU 는 뺀다(Identity).
        #                            EfficientAT 는 이 은닉층 위에 새 Linear 하나뿐인데 우리는 은닉층을 버리고 무작위 층 둘(proj·branch)+
        #                            ReLU 병목을 끼웠었다(Kumar et al. ICLR 2022: 무작위 헤드가 사전학습 특징을 왜곡).
        #   lp_epochs             : LP-FT. 처음 N 에폭 백본(mn10 core)을 고정(requires_grad·BN 통계)하고 투영·branch·헤드·BG 만 학습한 뒤
        #                            전부 푼다. LP 구간은 early stopping 에 세지 않고 FT 첫 에폭에서 best 를 다시 잰다.
        #   hard_weight           : 정답 라벨 CE 의 가중치. None 이면 옛 (1−α)=0.7(Hinton KD 관례 — 같은 샘플에 두 손실을 걸 때의 것이라
        #                            labeled/unlabeled 가 갈린 우리 구조에서는 지도 신호 기울기를 0.7배로 줄이는 효과뿐). 1.0 권장.
        #   others_from_reject_tau: 전문가 8명의 최대 확신도가 이 값 미만이면(전원 거절) others 가짜 정답으로 채택(가중 1−최대). 0 이면 끔.
        #                            합의 라벨은 구조상 others 가 0 개라(진짜 others 222클립 전부 미채택) 배경의 반지도 신호가 없었다.
        #                            OpenMatch(NeurIPS 2021)의 one-vs-all 전원 거절 = outlier 와 같은 발상.
        #   neg_pl_tau / neg_pl_beta: 미채택 클립에서 전문가 c 의 확신도 < tau 인 클래스에 "c 아님" 부정 라벨 −beta·log(1−p_student(c))
        #                            (UPS, ICLR 2021 의 negative pseudo-label). others 는 전문가가 없어 제외. tau 0 이면 끔.
        self.head_pretrained_hidden = False
        self.lp_epochs = 0
        self.hard_weight = None
        self.others_from_reject_tau = 0.0
        self.neg_pl_tau = 0.0
        self.neg_pl_beta = 0.0
        # (Mark5.2 본체의 run_tag_suffix 는 여기 없다 — Mark5_EXP2 는 실험 조건 태그 _l/_u/_ceil/_sp/_ps/_s 로 회차를 가른다.)

        # Joint forward keeps labeled/unlabeled clips in the same BatchNorm batch.
        self.joint_forward = False

        # === 내부 캐시 ===
        self._text_emb = None
        self._eval_text_emb = None # 평가용 텍스트 임베딩 캐시 추가
        self._eval_text_emb_raw = None   # [이식 2026-09-26] 중심화 전 원본(평가용 9클래스). 평균 벡터의 근거.
        self._anchor_mean = None         # [이식 2026-09-26] 앵커 중심화용 평균 벡터 [1, D]. 한 번 정하면 고정.
        self._prompt_texts = None
        self.prompt_templates = get_prompt_templates(self.prompt_bank_path)
        self.class_synonyms = get_class_synonyms(self.prompt_bank_path)


    # ==============================================================================
    # 3. 클래스 및 텍스트 관련 메서드
    # ==============================================================================
    def get_class_index(self, class_name: str) -> int:
        """주어진 클래스 이름의 인덱스를 반환. Unlabeled의 경우 -1을 반환."""
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"=> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_parser(self) -> list:
        """
        데이터 파싱 시 유효한 모든 레이블 목록을 반환. (dummy_label 포함)
        학습 데이터셋 구성 시 사용됨.
        """
        return self.labeled_classes

    def get_classes_for_text_prompts(self) -> list:
        """
        [기존 호환성 유지] 텍스트 프롬프트 생성에 사용될 클래스 목록을 반환.
        기본적으로 파서용 클래스 목록과 동일하게 동작함.
        """
        return self.labeled_classes
    
    def get_classes_for_evaluation(self) -> list:
        """
        [추가된 메서드] 모델 성능 평가에 사용될 실제 타겟 클래스 목록을 반환.
        'dummy_label'과 같이 평가에 사용되지 않는 레이블은 제외됨.
        """
        # self.classes 리스트에서 'dummy_label'을 필터링하여 반환
        return [cls for cls in self.classes if cls != 'dummy_label']

    def get_target_label_map(self) -> dict:
        """
        [수정] 평가용 클래스 목록을 기준으로 라벨-인덱스 맵을 생성함.
        모델의 최종 출력과 매칭시킬 때 사용됨.
        """
        # 평가용 클래스 목록을 사용하도록 변경
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_evaluation())}

    @property
    def run_tag(self) -> str:
        """[신설 2026-08-21 / 확장 2026-08-22] 산출물 파일명에 붙일 실행 태그.

        서로 다른 학습 설정이 같은 이름으로 저장돼 앞선 결과를 덮는 것을 막는다.
        붙는 순서는 아래 코드 그대로이고, 조합되면 이어붙는다(예: mark5.0_PL_DKD).

            (없음) : 1차 학습과 완전히 같은 설정 — vanilla KD, raw 조립
            _PL    : use_pseudo_label_ce=True  (unlabeled 를 KL 대신 pseudo-label CE 로)
            _RAW   : fusion_score_mode="raw"   (PL 을 쓰면서 옛 조립으로 되돌린 경우만)
            _SA    : use_spec_augment=True     (학습 시 SpecAugment 온라인 증강)
            _DA<t> : use_distribution_alignment=True (t = tau 를 소수점 없이, 예: tau 0.5 -> _DA05)
            _DKD   : use_dkd=True
            _G<g>  : pseudo_label_min_gap=g>0 이고 _PL 일 때 (예: 0.4 -> _G04)
            _AC_LT : 언제나 (2026-09-26 확정)
            _noFKD / _noGAP / _noDB : ablation A1(특징 증류 끔) / A2(gap 조건 끔, 이때 _G 없음) / A4(증류 갈래 끔) — ablation_tag
            _noSEG : ablation A3(student 가 조각 5개 대신 클립 3초 통째, 2026-09-29) — ablation_tag
            _LIN   : 헤드 진단 H(텍스트 앵커 헤드 대신 nn.Linear, 2026-09-29) — ablation_tag 의 맨 뒤
            _SM / _CT : [2026-09-30 밤 / 가짜 정답 재료] 조각 평균 합의(clip_merge_rule="mean") / teacher 원본 입력(spec_augment_teacher_view="clean") — ablation_tag 의 _NEG 뒤
            _PH / _LP<N> / _HW<g> / _OR<τ> / _NEG : 격차 분석 대응 묶음(2026-09-29 밤: 헤드 단순화 / LP-FT N에폭 / 정답 CE 가중 / 전원 거절→others / 부정 라벨) — ablation_tag
            _BG<g> / _RW : [2026-10-01 / 지도만 격차 진단] BG 보조 손실 가중치가 기본값(0.1)과 다를 때 / student 원본 진폭 파형 — ablation_tag 의 _CT 뒤
            _lr<g>_elr<g> : 학습률이 기본값과 다를 때 (lr_tag)
            그 뒤에 실험 조건 태그 _l<n> / _u<n>|_ceil / _sp / _ps<이름> / _s<k> (Mark5_EXP2 전용, 아래 코드 순서)

        태그가 갈리는 산출물: student 체크포인트 · 손실곡선 PNG · loss_history CSV ·
        혼동행렬 · ROC · performance_summary · calibration_details ·
        prediction_details · 설명 PNG 폴더.

        일부러 태그를 안 붙이는 것: cache/{mark_version}/ 의 teacher feature 캐시,
        resources/ 의 teacher .pth, dataset_*.csv 인덱스.

        ⚠ eval.py 도 이 태그로 체크포인트를 찾으므로, 학습과 평가의 config 가 같아야 한다.
        """
        tag = self.mark_version
        if getattr(self, "use_pseudo_label_ce", False):
            tag += "_PL"
            if getattr(self, "fusion_score_mode", "margin") == "raw":
                tag += "_RAW"
        if getattr(self, "use_spec_augment", False):
            tag += "_SA"
        if getattr(self, "use_distribution_alignment", False):
            # tau 를 태그에 넣는다. tau 를 바꿔가며 돌릴 때 산출물이 서로를 덮지 않게.
            _t = getattr(self, "dist_align_tau", 0.5)
            tag += "_DA" + f"{_t:g}".replace(".", "")
        if self.use_dkd:
            tag += "_DKD"
        # [수정 2026-09-22 / 검수] 1등−2등 차이 조건(_G<값>)은 가짜 정답 CE 를 쓸 때만 실제로 걸리므로(train_mark5.py 의
        # `_use_pl and _pl_min_gap > 0`) 그때만 붙인다. 가짜 정답 CE 없이 돌리는 절제 회차에 _G04 가 붙는 일이 없게.
        _gap = float(getattr(self, "pseudo_label_min_gap", 0.0) or 0.0)
        if _gap > 0 and getattr(self, "use_pseudo_label_ce", False):
            tag += "_G" + f"{_gap:g}".replace(".", "")
        # [이식 2026-09-26 / Mark5.2] 앵커 중심화·학습형 로짓 스케일은 본 모델의 구성 요소라 언제나 붙는다.
        # AC/LT 없이 돌린 2026-09-24 이전 산출물(mark5.2_PL_SA_G04 …)과 이름이 겹치지 않게 하는 구실도 겸한다.
        tag += "_AC_LT"
        # [이식 2026-09-29 / ablation·묶음] 우리 부품을 끈 회차(_noFKD·_noGAP·_noDB·_noSEG·_LIN)와 격차 대응 묶음(_PH·_LP<N>·_HW<g>·
        # _OR<τ>·_NEG) 표식(아래 ablation_tag). 자리는 _AC_LT 뒤·학습률 표식 앞·실험 조건 태그 앞 — aggregate_results._TAG_RE 는
        # 이 앞부분을 통째로 '설정(base)' 으로 읽으므로 묶음 회차는 본 모델 산출물과 다른 base 로 따로 집계된다(이름이 겹치지 않는다).
        # 표식은 전부 대문자·숫자라 _TAG_RE 의 _l<n>(소문자 l+숫자)·_u<n>·_s<k> 와 헷갈리지 않는다(2026-09-29 확인).
        tag += self.ablation_tag
        # [신설 2026-09-25 / LR] 학습률 선정(model/select_lr.py). 명령줄로 기본값과 **다른** 학습률을 줬을 때만
        # _lr<새 층>·_elr<백본> 이 붙는다(기본값이면 옛 태그 그대로 — 지금까지의 산출물과 재개 판정이 안 깨진다).
        # 표기는 비교모델 common/naming.py 와 같은 f"{lr:g}"(예: 5e-05 · 0.0002). 자리는 설정 태그의 맨 끝, 실험 조건 태그
        # (_l/_u/_ceil/_sp/_ps/_s) 앞 — aggregate_results._TAG_RE 는 이 앞부분을 통째로 '설정(base)' 으로 읽는다.
        tag += self.lr_tag
        # [추가 2026-08-30 / Mark5_EXP] 실험 조건 태그. None 이면 안 붙는다(0 도 유효 조건이라 "is not None").
        # [추가 2026-09-06] 천장 조건은 _u<n> 자리에 _ceil. [추가 2026-09-17] labeled_n 은 _l<n> 으로 그 앞에.
        #   예: mark5.2_PL_SA_l200_u2000_s1 · mark5.2_PL_SA_l200_ceil_s3 · mark5.2_PL_SA_u0_s2
        #   집계(aggregate_results.py)는 이 접미사를 보고 조건을 나눈다.
        _l = getattr(self, "labeled_n", None)
        if _l is not None:
            tag += f"_l{int(_l)}"
        if getattr(self, "ceiling", False):
            tag += "_ceil"
        else:
            _n = getattr(self, "unlabeled_n", None)
            if _n is not None:
                tag += f"_u{int(_n)}"
        # [추가 2026-09-17 / 8절] unlabeled 풀 확장(_sp)·pseudo-label 출처(_ps<이름>) 태그. _u<n> 뒤, _s<k> 앞.
        if getattr(self, "spill_labeled", False):
            tag += "_sp"
        _ps = getattr(self, "pseudo_source", None)
        if _ps:
            tag += f"_ps{_ps}"
        _s = getattr(self, "experiment_seed", None)
        if _s is not None:
            tag += f"_s{int(_s)}"
        return tag

    def apply_cli_learning_rates(self, lr=None, encoder_lr=None) -> str:
        """[신설 2026-09-25 / LR] 명령줄 --lr(새로 만드는 층)·--encoder_lr(사전학습 백본) 을 config 에 넣는다. None 은 그대로(기본값).
        run_tag 가 만들어지기 전(학습·평가 모두 config 생성 직후)에 불러야 산출물 이름이 맞는다. 기본값과 같은 값을 줘도 태그는
        안 바뀐다(lr_tag 는 값으로 비교). 돌려주는 값: 바뀐 것이 있으면 로그용 한 줄, 없으면 ''."""
        changed = []
        if lr is not None:
            if float(lr) <= 0:
                raise ValueError(f"[ERROR] --lr 은 양수여야 합니다: {lr}")
            self.learning_rate = float(lr)
            changed.append(f"새 층 {float(lr):g}")
        if encoder_lr is not None:
            if float(encoder_lr) <= 0:
                raise ValueError(f"[ERROR] --encoder_lr 은 양수여야 합니다: {encoder_lr}")
            if getattr(self, "encoder_type", "cnn") != "mn10":
                raise ValueError("[ERROR] --encoder_lr 은 사전학습 백본(encoder_type=mn10, mark5.2)에서만 뜻이 있습니다. "
                                 f"지금 encoder_type={getattr(self, 'encoder_type', 'cnn')!r} — 백본 학습률 분리가 없는 구성입니다.")
            self.encoder_learning_rate = float(encoder_lr)
            changed.append(f"백본 {float(encoder_lr):g}")
        return " / ".join(changed)

    @property
    def lr_tag(self) -> str:
        """[신설 2026-09-25 / LR] 학습률이 __init__ 기본값과 다를 때만 '_lr<g>'(새 층)·'_elr<g>'(백본) 를 돌려준다. 같으면 ''.
        train_mark5.py / eval.py 가 --lr / --encoder_lr 로 config 값을 덮어쓰면 run_tag 가 이 값으로 갈린다.
        기본값 스냅샷(_default_lrs)이 없는 옛 config 객체에서는 항상 '' (옛 동작)."""
        d_lr, d_elr = getattr(self, "_default_lrs", (None, None))
        out = ""
        cur = getattr(self, "learning_rate", None)
        if d_lr is not None and cur is not None and float(cur) != d_lr:
            out += f"_lr{float(cur):g}"
        enc = getattr(self, "encoder_learning_rate", None)
        if d_elr is not None and enc is not None and float(enc) != d_elr:
            out += f"_elr{float(enc):g}"
        return out

    def apply_cli_ablation(self, no_feature_kd=False, pl_min_gap=None, no_distill_branch=False, clip_input=False,
                           linear_head=False, segment_input=False, head_pretrained_hidden=False, lp_epochs=None,
                           hard_weight=None, others_from_reject=None, neg_pl=None,
                           teacher_seg_mean=False, teacher_clean_view=False,
                           bg_weight=None, student_raw_wave=False, joint_forward=False) -> str:
        """[신설 2026-09-28 / ablation A1·A2·A4] 명령줄 --no_feature_kd / --pl_min_gap / --no_distill_branch 를 config 에 넣는다.
        EfficientAT-pl(같은 백본·같은 라벨)과의 2%p 차이가 우리가 얹은 부품 중 어디서 나는지 하나씩 빼 보는 실험용.
          A1 no_feature_kd     : use_feature_kd=False — teacher 특징을 따라가는 손실(feature_kd_weight 0.3)을 뺀다.
          A2 pl_min_gap=0      : pseudo_label_min_gap=0 — 가짜 정답 채택의 "1등−2등 차이 ≥0.4" 조건을 뺀다(확신도 ≥0.5 만 남음).
                                 0 이 아닌 값을 주면 그 값으로 바꾼다(태그 _G<값>).
          A4 no_distill_branch : distill_branch_classify=False — 증류 갈래에 분류 손실을 걸지 않고, eval 의 w 후보를 (0,) 으로
                                 좁혀 최종 확률에서 증류 갈래를 뺀다(지도 헤드만 쓴다). 갈래 자체는 남아 feature KD 만 받는다(A1 과 독립).
          A3 clip_input        : [2026-09-29] student_clip_input=True — student 가 1초 조각 5개 대신 클립 3초 파형을 통째로 본다
                                 (teacher 는 조각 그대로, 조각별 출력을 클립 규칙으로 합침. __init__ 의 student_clip_input 주석 참조).
                                 A1·A2·A4 결과(2026-09-28, seed 1)가 셋 다 본 모델과 1클립 차이라 남은 후보가 이것이었다.
          H  linear_head         : [2026-09-29] student_head="linear" — 텍스트 앵커 헤드 대신 nn.Linear(embedding_dim, 9)(vild_model.LinearClassHead).
                                 A3 도 1클립 차이(seed 1)라 남은 후보가 헤드였다. 제로샷이 없어진다(run_tag _LIN). [2026-10-07] D0 본 레시피로 채택.
          SM teacher_seg_mean  : [2026-09-30 밤 / 가짜 정답 재료] clip_merge_rule="mean" — 조각 teacher 출력을 전문가별로 평균한 뒤 클립 단위로 채택한다
                                 (run_tag _SM). 클립 입력 student 에서만(조각 입력에는 합치는 단계가 없다), 조각 teacher 에서만(train_mark5 가 검사).
          CT teacher_clean_view: [2026-09-30 밤 / 가짜 정답 재료] spec_augment_teacher_view="clean" — teacher 는 증강 없는 원본을 본다(run_tag _CT).
          BG bg_weight         : [2026-10-01 / 지도만 격차 진단] background_embedding_weight — others 특징을 배경 임베딩 쪽으로 당기는 보조 손실의
                                 가중치(기본 0.1). 0 이면 그 손실을 걸지 않는다. 기본값과 다르면 run_tag _BG<값>(0 → _BG0).
          RW student_raw_wave  : [2026-10-01 / 지도만 격차 진단] student_wave_normalize=False — student 클립 파형을 정규화하지 않은 원본 진폭으로
                                 (teacher mel 은 그대로). 비교모델과 같은 입력. 클립 입력 mn10 student 에서만(run_tag _RW).
        run_tag 가 만들어지기 전(학습·평가 모두 config 생성 직후)에 불러야 산출물 이름(ablation_tag)이 맞는다. eval.py 는 학습 때와
        같은 플래그를 받아야 같은 체크포인트를 찾는다. 돌려주는 값: 바뀐 것이 있으면 로그용 한 줄, 없으면 ''."""
        changed = []
        if clip_input and segment_input:
            raise ValueError("[ERROR] --clip_input 과 --segment_input 을 함께 줄 수 없습니다.")
        if segment_input:
            # [2026-09-29 밤] mark5.2 는 클립 입력이 기본이라 조각 방식(2026-09-28 이전 본 모델)으로 되돌리는 플래그가 이것이다.
            self.student_clip_input = False
            changed.append(f"조각 입력(1초×{self.max_segments}, 2026-09-28 이전 방식)으로 되돌림")
        if clip_input and not getattr(self, "student_clip_input", False):
            if getattr(self, "encoder_type", "cnn") != "mn10":
                raise ValueError("[ERROR] --clip_input 은 파형 입력 백본(encoder_type=mn10, mark5.2)에서만 뜻이 있습니다. "
                                 f"지금 encoder_type={getattr(self, 'encoder_type', 'cnn')!r} (mel 조각 입력) 입니다.")
            self.student_clip_input = True
            changed.append(f"A3 세그먼트 분할 끔(student 클립 {self.student_clip_seconds:g}초 통째 · teacher 조각 {self.max_segments}개 그대로 "
                           f"→ 채택 조각 ≥{self.clip_min_segments} 다수결로 합침)")
        if head_pretrained_hidden:
            if getattr(self, "encoder_type", "cnn") != "mn10":
                raise ValueError("[ERROR] --head_pretrained_hidden 은 mn10 백본에서만 뜻이 있습니다(사전학습 은닉층이 mn10 의 것).")
            self.head_pretrained_hidden = True
            changed.append("PH 헤드 단순화(mn10 사전학습 은닉층 유지 → Linear(960→384) 하나 · branch LN+ReLU 제거)")
        if lp_epochs is not None:
            if int(lp_epochs) < 0:
                raise ValueError(f"[ERROR] --lp_epochs 는 0 이상이어야 합니다: {lp_epochs}")
            self.lp_epochs = int(lp_epochs)
            if self.lp_epochs > 0:
                changed.append(f"LP-FT(처음 {self.lp_epochs}에폭 백본 고정 뒤 전체 해제)")
        if hard_weight is not None:
            if float(hard_weight) <= 0:
                raise ValueError(f"[ERROR] --hard_weight 는 양수여야 합니다: {hard_weight}")
            self.hard_weight = float(hard_weight)
            changed.append(f"정답 라벨 CE 가중치 {self.hard_weight:g}(옛 (1−α)={1 - float(getattr(self, 'distill_alpha', 0.3)):g} 대신)")
        if others_from_reject is not None:
            if not (0.0 <= float(others_from_reject) < 1.0):
                raise ValueError(f"[ERROR] --others_from_reject 는 0 이상 1 미만이어야 합니다: {others_from_reject}")
            self.others_from_reject_tau = float(others_from_reject)
            if self.others_from_reject_tau > 0:
                changed.append(f"전원 거절 → others 가짜 정답(최대 확신도 < {self.others_from_reject_tau:g})")
        if neg_pl is not None:
            _tau, _beta = float(neg_pl[0]), float(neg_pl[1])
            if not (0.0 <= _tau < 1.0) or _beta < 0:
                raise ValueError(f"[ERROR] --neg_pl 은 (tau in [0,1), beta >= 0) 이어야 합니다: {neg_pl}")
            self.neg_pl_tau, self.neg_pl_beta = _tau, _beta
            if _tau > 0 and _beta > 0:
                changed.append(f"부정 가짜 라벨(전문가 확신도 < {_tau:g} 인 클래스, 가중치 {_beta:g})")
        if no_feature_kd:
            self.use_feature_kd = False
            changed.append("A1 특징 증류 끔(use_feature_kd=False)")
        if pl_min_gap is not None:
            _g = float(pl_min_gap)
            if _g < 0:
                raise ValueError(f"[ERROR] --pl_min_gap 은 0 이상이어야 합니다: {pl_min_gap}")
            self.pseudo_label_min_gap = _g
            changed.append("A2 1등−2등 차이 조건 끔(pseudo_label_min_gap=0)" if _g == 0 else f"1등−2등 차이 조건 {_g:g}")
        if no_distill_branch:
            self.distill_branch_classify = False
            self.distill_branch_eval_weight = 0.0
            self.distill_eval_weight_candidates = (0.0,)
            changed.append("A4 증류 갈래의 분류 끔(feature KD 유지 · eval w=0 고정)")
        if linear_head:
            self.student_head = "linear"
            changed.append(f"H 텍스트 앵커 헤드 → 선형 헤드(nn.Linear {self.embedding_dim}→{len(self.get_classes_for_evaluation())}, "
                           "제로샷 없음 · 2026-10-07 D0 본 레시피)")
        # [신설 2026-09-30 밤 / 가짜 정답 재료] 가짜 정답 재료: 조각 평균 합의(_SM) · teacher 원본 입력(_CT). 둘 다 teacher 쪽 설정이라 student 구조는 그대로다.
        if teacher_seg_mean:
            self.clip_merge_rule = "mean"
            changed.append("조각 평균 합의(전문가별 조각 확신도 평균 → 클립 단위 채택, clip_merge_rule=mean)")
        if teacher_clean_view:
            self.spec_augment_teacher_view = "clean"
            changed.append("teacher 입력 = 원본(SpecAugment 는 student 에만)")
        # [신설 2026-10-01 / 지도만 격차 진단] teacher 없이 도는 지도만(u0) 조건에서 EfficientAT 파인튜닝과의 차이를 하나씩 맞춰 보는 스위치 둘(둘 다 student 쪽 설정).
        if bg_weight is not None:
            if float(bg_weight) < 0:
                raise ValueError(f"[ERROR] --bg_weight 는 0 이상이어야 합니다: {bg_weight}")
            self.background_embedding_weight = float(bg_weight)
            changed.append("BG 보조 손실 끔(background_embedding_weight=0)" if self.background_embedding_weight == 0
                           else f"BG 보조 손실 가중치 {self.background_embedding_weight:g}")
        if student_raw_wave:
            if getattr(self, "encoder_type", "cnn") != "mn10" or not getattr(self, "student_clip_input", False):
                raise ValueError("[ERROR] --student_raw_wave 는 클립 입력 mn10 student(mark5.2 기본)에서만 씁니다 — "
                                 "조각 입력(--segment_input)·mel 입력 인코더에는 이 경로가 없습니다.")
            self.student_wave_normalize = False
            changed.append("student 파형 정규화 끔(원본 진폭 — 비교모델과 같은 입력 · teacher mel 은 정규화 파형 그대로)")
        if getattr(self, "clip_merge_rule", "vote") == "mean" and not getattr(self, "student_clip_input", False):
            raise ValueError("[ERROR] --teacher_seg_mean(조각 평균 합의)은 클립 입력 student(mark5.2 기본)에서만 뜻이 있습니다 — "
                             "조각 입력(--segment_input)에는 조각을 클립으로 합치는 단계가 없습니다.")
        if joint_forward:
            if not self.student_clip_input or self.encoder_type != "mn10":
                raise ValueError("--joint_forward requires a clip-input mn10 student.")
            self.joint_forward = True
            changed.append("JF labeled and unlabeled clips share one encoder forward")
        return " / ".join(changed)

    @property
    def ablation_tag(self) -> str:
        """[신설 2026-09-28 / ablation] run_tag 에 붙는 표식. 전부 켠 본 모델(기본값)은 ''.
          _noFKD : use_feature_kd=False            _noGAP : 가짜 정답 CE 를 쓰면서 pseudo_label_min_gap=0 (이때 _G<값> 은 안 붙는다)
          _noDB  : distill_branch_classify=False
          _noSEG : student_clip_input=True (A3, 2026-09-29). 기존 세 표식의 이름·순서는 그대로다.
          _LIN   : student_head="linear" (헤드 진단 H, 2026-09-29) — 맨 뒤.
        옛 config 객체(속성 없음)에서는 켠 것으로 보고 '' (옛 동작)."""
        out = ""
        if not getattr(self, "use_feature_kd", True):
            out += "_noFKD"
        if getattr(self, "use_pseudo_label_ce", False) and float(getattr(self, "pseudo_label_min_gap", 0.0) or 0.0) <= 0:
            out += "_noGAP"
        if not getattr(self, "distill_branch_classify", True):
            out += "_noDB"
        if getattr(self, "student_clip_input", False):
            out += "_noSEG"
        if getattr(self, "student_head", "text_anchor") == "linear":
            out += "_LIN"
        # [2026-09-29 밤 / 격차 분석 대응 묶음] 표식 순서 고정: _PH → _LP<N> → _HW<g> → _OR<τ 소수점 없이> → _NEG
        if getattr(self, "head_pretrained_hidden", False):
            out += "_PH"
        _lp = int(getattr(self, "lp_epochs", 0) or 0)
        if _lp > 0:
            out += f"_LP{_lp}"
        _hw = getattr(self, "hard_weight", None)
        if _hw is not None:
            out += "_HW" + f"{float(_hw):g}".replace(".", "")
        _or = float(getattr(self, "others_from_reject_tau", 0.0) or 0.0)
        if _or > 0:
            out += "_OR" + f"{_or:g}".replace(".", "")          # 0.2 → _OR02, 0.15 → _OR015 (_G04 와 같은 표기)
        if float(getattr(self, "neg_pl_tau", 0.0) or 0.0) > 0 and float(getattr(self, "neg_pl_beta", 0.0) or 0.0) > 0:
            out += "_NEG"
        # [2026-09-30 밤 / 가짜 정답 재료] 가짜 정답 재료 표식: _SM(조각 평균 합의) → _CT(teacher 원본 입력). 기존 표식의 이름·순서는 그대로다.
        if getattr(self, "clip_merge_rule", "vote") == "mean":
            out += "_SM"
        if getattr(self, "use_spec_augment", False) and getattr(self, "spec_augment_teacher_view", "augmented") == "clean":
            out += "_CT"
        # [2026-10-01 / 지도만 격차 진단] _BG<g>(BG 보조 손실 가중치가 기본값과 다를 때, 0 → _BG0) → _RW(student 원본 진폭 파형). 기존 표식의 이름·순서는 그대로다.
        _dbg, _bg = getattr(self, "_default_bg_weight", None), getattr(self, "background_embedding_weight", None)
        if _dbg is not None and _bg is not None and float(_bg) != float(_dbg):
            out += "_BG" + f"{float(_bg):g}".replace(".", "")
        if not getattr(self, "student_wave_normalize", True):
            out += "_RW"
        if self.joint_forward:
            out += "_JF"
        return out + "_SINGLE"

    @property
    def others_entropy_threshold(self) -> float:
        """
        [추가 2026-07-11] others_confidence_threshold와 "같은 엄격도(같은 top_conf 지점)"가
        되도록 클래스 수 기반으로 자동 역산한다. top_conf=confidence_threshold이고 나머지
        확률이 (클래스수-1)개에 균등분산되는 최악의 경우를 가정해 그때의 정규화 entropy를
        threshold로 삼는다. 클래스 수가 다른 mark_version 사이에서 entropy 조건이
        confidence 조건보다 부당하게 엄격/느슨해지는 것을 방지한다.
        """
        p = self.others_confidence_threshold
        n = self.num_distinct_labeled_classes
        if n <= 1:
            return 1.0
        rest = 1.0 - p
        probs = [p] + [rest / (n - 1)] * (n - 1)
        entropy = -sum(x * math.log(x, 2) for x in probs if x > 1e-12)
        return entropy / math.log2(n)

    @property
    def student_input(self) -> str:
        """[신설 2026-09-17] student 인코더에 무엇을 넣나 — "waveform"(mn10: 파서가 잘라 준 1.0초 파형) / "mel"(cnn).
        teacher 8개는 어느 경우든 mel 64x101 을 받는다. 파서는 항상 둘 다 만들고, 학습·평가 루프가 이 값으로 고른다."""
        return "waveform" if getattr(self, "encoder_type", "cnn") == "mn10" else "mel"

    @property
    def num_input_channels(self) -> int:
        if self.visual_view_type == "mel":
            return 1
        if self.visual_view_type == "mel_delta":
            return 3
        if self.visual_view_type == "mel_energy":
            return 2
        return 1

    def get_prompt_texts_for_class(self, class_name: str) -> list:
        return get_prompt_texts_for_class(class_name, self.prompt_bank_path)

    def _encode_class_prompts(self, classes) -> torch.Tensor:
        """[이식 2026-09-26 / Mark5.2] 클래스 목록을 프롬프트 평균 임베딩 [C, D] 로. 중심화는 하지 않는다."""
        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        aggregated = []
        for class_name in classes:
            prompts = self.get_prompt_texts_for_class(class_name)
            emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
            aggregated.append(emb.mean(dim=0))
        return torch.stack(aggregated, dim=0)

    def get_anchor_mean(self) -> torch.Tensor:
        """[이식 2026-09-26 / Mark5.2] 앵커 중심화용 평균 벡터 [1, D]. 평가용 9클래스(get_classes_for_evaluation) 앵커의
        평균으로 한 번 정해 캐시한다. 학습·평가·제로샷 어느 목록을 임베딩하든 이 같은 벡터를 빼야 공간이 일치한다."""
        if self._anchor_mean is None:
            if self._eval_text_emb_raw is None:
                self._eval_text_emb_raw = self._encode_class_prompts(self.get_classes_for_evaluation())
            self._anchor_mean = self._eval_text_emb_raw.mean(dim=0, keepdim=True)
        return self._anchor_mean

    def _apply_anchor_centering(self, emb: torch.Tensor) -> torch.Tensor:
        """[이식 2026-09-26 / Mark5.2] 언제나 적용한다(끄는 경로 없음)."""
        return emb - self.get_anchor_mean().to(emb.device)

    @staticmethod
    def _mean_offdiag_cosine(emb: torch.Tensor) -> float:
        n = torch.nn.functional.normalize(emb.detach().float(), dim=1)
        S = n @ n.T
        C = S.shape[0]
        return float((S.sum() - S.diagonal().sum()) / max(C * (C - 1), 1))

    def get_class_text_embeddings(self, for_evaluation: bool = False) -> torch.Tensor:
        """
        클래스 이름에 대한 텍스트 임베딩을 생성하여 반환함.
        :param for_evaluation: True일 경우, 평가용 클래스 목록을 사용하여 임베딩을 생성함.
        [이식 2026-09-26 / Mark5.2] 평가용 9클래스 평균 벡터를 뺀 값을 돌려준다(앵커 중심화는 언제나 적용).
        정규화는 ViLDTextHead 가 하므로 여기서는 하지 않는다.
        """
        if for_evaluation:
            # 평가용 임베딩 생성 및 캐싱
            if self._eval_text_emb is None:
                if self._eval_text_emb_raw is None:
                    self._eval_text_emb_raw = self._encode_class_prompts(self.get_classes_for_evaluation())
                raw = self._eval_text_emb_raw
                self._eval_text_emb = self._apply_anchor_centering(raw)
                print(f"[AC] 앵커 중심화: 평가용 {raw.shape[0]}클래스 코사인 평균 {self._mean_offdiag_cosine(raw):+.4f} "
                      f"→ {self._mean_offdiag_cosine(self._eval_text_emb):+.4f}")
            return self._eval_text_emb
        else:
            # 학습용(기존) 임베딩 생성 및 캐싱
            if self._text_emb is None:
                raw = self._encode_class_prompts(self.get_classes_for_text_prompts())
                self._text_emb = self._apply_anchor_centering(raw)
            return self._text_emb
