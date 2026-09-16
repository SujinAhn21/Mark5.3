# 출처
- 저장소: https://github.com/fschmid56/EfficientAT (MIT)
- 커밋: a425fdce92572e602a1d5634799bd9f1f2efa806  (2026-09-05 clone)
- 가져온 파일: models/mn/{model,block_types,utils,attention_pooling}.py, models/preprocess.py, helpers/utils.py
  — 수정 없음(원본 그대로). **예외: helpers/utils.py 는 NAME_TO_WIDTH 함수만 남긴 축약본**(원본이 import 시점에
  metadata/class_labels_indices.csv 를 상대경로로 열어 저장소 밖에서 죽기 때문. 파일 머리에 적어 둠). 원본 import 경로(models.mn.*, helpers.utils)를 살리려고 폴더 구조를 그대로 두고
  model_def.py 가 이 폴더를 sys.path 에 넣는다.
- 함께 필요한 패키지: torchvision (ConvNormActivation)
- 가중치: weights/mn10_as_mAP_471.pt  <- https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt
- 전처리는 preprocess.py 의 AugmentMelSTFT 와 같은 계산을 model_def.py 안에 다시 적었다
  (내장 증강 off, return_complex=True). 원본 파일은 대조용으로 남겨 둔다.

## fmax 15,000 의 근거 (2026-09-05 저녁 보강)
- 저자 파인튜닝 스크립트 ex_esc50.py 의 argparse 기본값은 `--resample_rate 32000 --n_mels 128 --window_size 800 --hop_size 320
  --n_fft 1024 --fmin 0 --fmax None --fmin_aug_range 10 --fmax_aug_range 2000` (2026-09-05 저장소 README·스크립트 열람).
  `fmax=None` 이면 preprocess.py:17-19 가 `sr//2 - fmax_aug_range//2 = 16000 - 1000 = 15000` 을 쓰고, 평가 모드에서는
  그 값이 그대로 fmax 가 된다(preprocess.py:52-56). 그래서 model_def.py 는 15,000 을 고정값으로 쓴다.
- torchvision 의존: models/mn/model.py 와 block_types.py 가 `torchvision.ops.misc.ConvNormActivation` 을 쓴다.
  0.19~0.20 에서는 남아 있으나 직접 생성 시 UserWarning 을 낸다. 코랩에서 학습 전에
  `python -c "from torchvision.ops.misc import ConvNormActivation"` 로 한 번 확인한다(코랩 실행순서 txt 셀 4).
