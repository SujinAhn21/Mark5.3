# Mark5.3 — 단일 분류기와 전문가 특징 증류

2026-10-22 통합판. PF·SK·OVA·FM·CB·SC의 구현, 설정, 명령줄 옵션을 제거했다.
`--m53`은 더 이상 받지 않는다. 기존 Mark4.x 전문가 8명의 인코더·분류기 체크포인트
16개는 그대로 teacher로 사용한다.

## 학습과 추론

학생 mn10 인코더의 일반 특징 → 하나의 9클래스 분류기가 최종 출력을 만든다.
정답 CE와 teacher 합의 pseudo-label CE 모두 이 분류기에만 적용된다.
별도 증류 변환층은 teacher 특징을 맞추는 학습 보조 모듈이며, 분류 손실과 추론에 참여하지 않는다.
특징 증류의 기울기는 학생 인코더와 변환층에 전달되고 teacher 가중치는 고정된다.

아래 실행 레시피의 손실은 정답 CE × 1.0 + 채택 pseudo-label CE × 0.3 + 특징 KD × 0.3
+ 부정 pseudo-label 손실 × 0.3이다. 채택 CE의 기존 확신도 가중, 조각 teacher 평균 합의,
전원 거절→others, NEG는 유지한다. 이들은 이번에 제거한 여섯 요소와 구분된다.
JF는 정답/무라벨 클립을 한 번에 인코더에 넣어 같은 BatchNorm 통계를 쓰는 처리다.

검증·최종 추론은 동일한 일반 특징 분류기의 argmax다. 증류 출력 혼합, OVA 융합,
클래스 편향, others 임계값 탐색과 과거 보정 JSON 로딩은 없다.
`model/profile_efficiency.py`도 같은 추론 경로로 측정한다. 훈련 전용 변환층은 배포 파라미터에서 제외한다.
thop의 MACs는 지원되는 연산만 집계하며 전체 프런트엔드의 정확한 FLOPs가 아니다.

## 실행

코드 식별자 `--mark_version mark5.2`는 기존 mn10 설정 이름이어서 유지한다.
논문의 제안모델 이름은 Mark5.3이다. 새 결과에는 `_SINGLE`이 붙는다.

```bash
python model/run_experiments.py --mark_version mark5.2 \
  --seeds 1 --labeled_ns 2000 --unlabeled_ns 2000 \
  --lr 0.0001 --encoder_lr 0.0001 --ignore_selected_lr --skip_test \
  --extra "--linear_head --head_pretrained_hidden --hard_weight 1.0 --others_from_reject 0.2 --neg_pl 0.1 0.3 --teacher_seg_mean --teacher_clean_view --bg_weight 0 --student_raw_wave --joint_forward" \
  --dry_run
```

`--dry_run`을 빼면 선택한 한 조건의 학습과 val 평가가 실행된다. test를 최종 평가할 때만
`--skip_test`를 뺀 동일 명령을 실행한다. 같은 체크포인트의 학습·val은 재사용하고 test만 추가한다.
학습률은 이 수정판의 실행 설정으로 1e-4를 명시하며 자동 탐색·과거 선정 JSON 읽기를 하지 않는다.
비교모델의 기존 학습률과 코드는 변경하지 않았다. 공정성은 동일한 학습률 하나만으로 정해지지 않으며,
데이터 분할, 사전학습, teacher가 사용한 추가 지도 정보와 선정 절차를 함께 명시해야 한다.

Colab은 [notebooks/Mark5.3 제안모델 실험.ipynb](notebooks/Mark5.3%20제안모델%20실험.ipynb)를 사용한다.
기본은 계획 확인이며 `RUN_TRAINING=True`로 선택 조건을 실행한다. 학습률 탐색이나 여러 조건의 자동 확장은 없다.
`EVALUATE_TEST=True`는 최종 test 평가를 실행한다. 이전 아블레이션 l200 결과를 새 결과로 복사하지 않는다.

## 파일과 환경

- `dataset_labeled.csv`, `dataset_unlabeled.csv`, `dataset_val.csv`, `dataset_test.csv`: 기존 분할을 유지한다.
- `resources/teacher_manifest.json`과 `best_teacher_{encoder,classifier}_mark4.*.pth` 16개: `_run3_20260923` teacher.
- `weights/mn10_as_mAP_471.pt`: 학생의 AudioSet 사전학습 가중치.
- `third_party/efficientat/`: vendored mn10 구현.
- 텍스트 앵커는 학생 선형 분류기에 불필요하지만 기존 teacher 초기화에는 SentenceTransformer 캐시가 필요하다.
- 로컬 CPU 검증 환경은 Python 3.8 / PyTorch 2.4.1+cpu이며 `requirements.txt`는 이 환경용이다.
  Colab에서는 CPU 전용 torch 요구사항을 설치하지 않고 노트북의 의존성 셀을 사용한다.

노트북은 기존 Drive 경로 `/content/drive/MyDrive/NewProject/codes/experiments/Mark5.3`를 사용한다.
데이터 목록이나 teacher를 다른 실험 폴더에서 덮어쓰지 않는다. 실행 전
`python model/verify_release.py`가 배포 소스·데이터 목록·가중치 해시를 검사한다.

## 결과와 재사용

- 체크포인트: `model/student_model_<run_tag>.pth`, `classification_policy=supervised_argmax_v1` 기록.
- val 결과: `plots/*_<run_tag>_val_plain.*`; test 결과: `plots/*_<run_tag>_plain.*`.
- `output_policy_*.json`에 체크포인트 SHA-256을 기록하며 실행기와 집계가 일치 여부를 확인한다.
- 이전 분류 구조의 체크포인트는 새 평가기가 거부한다. 기존 체크포인트/결과 파일은 과거 기록으로 보존한다.
- 집계: `python model/aggregate_results.py --split val --no_tests`.
  산출물은 `aggregate_*_mark5.2_val_plain.csv`이며 완료한 새 결과만 포함한다.

## 검증 범위

`tests/verify_single_classifier.py`는 실제 teacher·사전학습 mn10·로컬 음성으로 짧은 CPU 학습,
저장/재로딩, 특징 KD 기울기, teacher 고정, 일반 특징만 분류, 추론 시 증류 변환층 미실행,
보정 JSON 미사용, stale 결과 제외를 확인한다. 각 학습은 9개 정답 + 7개 무라벨, 1 epoch이다.
검증용 test 호출에도 val의 9클립만 넣으며 실제 held-out test를 읽지 않는다.

```bash
python tests/verify_single_classifier.py --source-root . --audio-root ../Mark5.2
python tests/verify_release_contract.py
```

로그와 보고서는 `verification/`에 저장하며 Git에는 포함하지 않는다.
이 검증은 구현의 일관성을 확인하는 절차다. 새 구조의 본학습 성능과 EfficientAT 초과 여부는 아직 검증되지 않았다.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted.
