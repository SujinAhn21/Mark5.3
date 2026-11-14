# Mark5.3 — 단일 분류기와 전문가 특징 증류

2026-11-14 교수님 피드백의 [5번 통계 자료와 1~4번 추가실험 노트북](feedback/20261114/README.md)을 추가했다. 추가실험은 사용자 실행 단계이며, 기존 논문 결과와 입력 버전을 보존한다.


2026-10-22 통합판. PF·SK·OVA·FM·CB·SC의 구현, 설정, 명령줄 옵션을 제거했다.
`--m53`은 더 이상 받지 않는다. 기존 Mark4.x 전문가 8명의 인코더·분류기 체크포인트
16개는 그대로 teacher로 사용한다.

## 2026-10-23 판: 검증 결합기 설계 (`--preset cmb`, 80회)

설계 문서 "Mark5.3 개선 설계: 검증된 교사 신호와 mn10 교사"를 따른다. 학습 중에 교사를 부르지 않고, 교사 점수 캐시 →
검증 결합기 → 가짜 정답 CSV 를 미리 만든 뒤 학생이 그 CSV 를 읽는다. 학생 헤드는 선형 헤드(`--linear_head`)라 제로샷은 없다.

| 순서 | 명령 | 하는 일 |
| --- | --- | --- |
| 1 | `python model/import_teachers.py --kind mn10 --source_root ../Mark4_series` | mn10 교사 8개 반입(schema·클래스·학습 후 판정 '통과' 확인) → `resources/teacher_manifest_mn10.json` |
| 1' | `python model/import_teachers.py --source_root ../Mark4_series` | B0 조각 교사 16파일 반입(기본 `--kind segment`) → `resources/teacher_manifest.json` |
| 2 | `python model/make_teacher_scores.py --teacher_kind mn10` (B0 는 `segment`) | `plots/teacher_scores_<kind>.{csv,json}` (mn10 은 특징 캐시 `teacher_feats_mn10.pt` 도) |
| 3 | `python model/teacher_gate_report.py --candidate plots/teacher_scores_mn10.csv --reference plots/teacher_scores_segment.csv` | 채택 관문(9클래스 4지표 · 결합기 5겹 교차검증 · 손들기 행렬) → `plots/teacher_gate_*.json` |
| 4 | `python model/make_combiner_labels.py --scores plots/teacher_scores_<채택>.csv` | `pseudo_labels_{cmb,cmbone,fix}_l<n\|all>_s<seed>.csv` + `plots/combiner/*.json` |
| 5 | `python model/label_freshness.py` | 교사 → 점수 캐시 → 결합기 CSV 해시 사슬 대조(어긋나면 멈춤) |
| 6 | `python model/run_experiments.py --mark_version mark5.2 --preset cmb --seeds 1 2 3 4 5` | 16조합 × 5 = 80회(본실험 4 · 곡선 3 · 지도만 3 · 아블레이션 6). 10-25 에 지도만 정답 1,000·500 을 더해 표 4 의 네 수준 모두 지도만 행이 있다(정답 200 의 지도만은 곡선의 u0) |
| 7 | `python model/aggregate_results.py --split test --main_pseudo_source cmb --bases <본실험 설정>` | 본표·라벨 효율 표(지도만 u0 회차 포함). 비교모델 짝은 `plc` |
| 8 | `python model/make_paper_tables.py --split test --main_pseudo_source cmb --baseline_raw ...` | 논문 표 |

- 아블레이션 표식: `_noTWO`(두 후보 끔) · `_psfix`(고정 규칙) · `_TC` + `_OR02_NEG`(옛 교사 경로 복원: 교사 점수·특징 캐시로 합의 + 특징 증류 + 전원 거절 + 부정 라벨, 증류 갈래 384→960).
- 공정한 관문 비교는 같은 새 데이터로 학습한 B0 다. 10-23 이전 3차 교사는 교사 train 이 Mark5.3 정답 라벨과 녹음을 공유해 정답 2,000 지표가 좋게 나온다.
- 시험: `tests/verify_two_candidate.py` · `tests/verify_teacher_cache.py` · `tests/verify_teacher_gate.py` · `tests/verify_import_mn10.py` · `tests/verify_cmb_aggregation.py`.
- 코드를 고친 뒤 커밋 전에 `python model/verify_release.py --refresh_source --source_only` 로 `source_manifest.json` 의 source 해시를 다시 계산한다(노트북의 release 검사가 이 해시를 대조한다).
- 아래 "학습과 추론"·"실행" 절은 10-22 판(교사 합의 가짜 정답, 학습 중 교사 실행)이다. 10-23 판에서는 `_TC` 아블레이션이 이 경로를 캐시로 재현한다.

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
결과물 4개(`model/make_four_results.py`)의 비교모델 숫자도 같은 규칙이다(2026-10-26): 비교모델 eval 은 val 에서 고른 클래스 편향·others 규칙을 test 에 걸어
성능 요약에 남기므로, 그 요약 대신 test 예측 CSV 의 보정 전 확률(`PreBias_Prob_*`)의 argmax 로 4지표를 다시 계산한다(재학습·재평가 없음).
같은 날 비교모델 eval 의 판정 규칙도 보정 없는 argmax 로 바꿔 60회를 다시 평가한다(노트북 5) — 그 뒤로는 성능 요약과 이 재계산 값이 같다.

라벨 없음 0(u0) 회차: 10-22 판의 데이터 검사가 라벨 없는 쪽 0개에서도 멈춰 u0 5회가 실패했다. 2026-10-26 에 정답 라벨 쪽만 0개일 때 멈추게 되돌렸다
(`model/train_mark5.py` `SemiSupervisedDataset`). 이 파일은 회차 신원 해시에 들어가므로 노트북 2 의 끝난 35회는 다시 돌리지 않고(노트북 2 [셀 7] 이 멈춘다)
노트북 4 가 u0 5회만 돈다. 35회의 체크포인트 신원은 고치기 전 판(dce8543) 코드 해시다 — 바뀐 줄은 라벨 없는 클립이 0개일 때만 실행된다.
`model/profile_efficiency.py`도 같은 추론 경로로 측정한다. 훈련 전용 변환층은 배포 파라미터에서 제외한다.
thop의 MACs는 지원되는 연산만 집계하며 전체 프런트엔드의 정확한 FLOPs가 아니다.

## 실행

코드 식별자 `--mark_version mark5.2`는 기존 mn10 설정 이름이어서 유지한다.
논문의 제안모델 이름은 Mark5.3이다. 새 결과에는 `_SINGLE`이 붙는다.

```bash
python model/run_experiments.py --mark_version mark5.2 \
  --seeds 1 --labeled_ns 2000 --unlabeled_ns 2000 --skip_test \
  --extra "--linear_head --head_pretrained_hidden --hard_weight 1.0 --others_from_reject 0.2 --neg_pl 0.1 0.3 --teacher_seg_mean --teacher_clean_view --bg_weight 0 --student_raw_wave --joint_forward" \
  --dry_run
```

`--dry_run`을 빼면 선택한 한 조건의 학습과 val 평가가 실행된다. test를 최종 평가할 때만
`--skip_test`를 뺀 동일 명령을 실행한다. 같은 체크포인트의 학습·val은 재사용하고 test만 추가한다.
옵티마이저는 Adam이다(2026-10-25 A안). 학습률은 다시 고르지 않고 `plots/lr_selected.json`(2026-10-08, 후보 1e-4·5e-5·2e-5,
seed 0, val Macro F1)이 고른 새 층 1e-4 · 백본 1e-4로 고정한다(10-08 최종 모델과 같은 값, `vild/vild_config.py`).
결합형 L2 weight decay 1e-4, `ReduceLROnPlateau(0.5, 3)`과 early stopping 10은 그대로다.
`--lr`·`--encoder_lr`을 주면 멈추고 `model/select_lr.py`도 실행되지 않는다.
새 결과의 run_tag에는 학습률 자리에 `_ADAM`이 붙어, 2026-10-21까지의 결과(`_elr0.0001` 등)와 섞이지 않는다.
2026-10-22~10-25에는 자동 학습률 Prodigy(run_tag `_PD`)로 바꿨다가 걷어냈다. 보폭이 Adam 학습률의 수십~수백 배로 커져
같은 mn10 백본의 EfficientAT에서 사전학습 가중치의 60.7%가 0 근처로 줄었고(정답 2,000, Adam 4.5%) test Acc가 6~9%p 낮았다.
비교모델(EfficientAT·PANNs·CED와 표준 반지도 4방법)도 같은 날 Adam·선정값 고정으로 되돌렸다(run_tag `_adam`).
Mark4.1~4.8 교사는 Adam 1e-4 그대로다 — 교사는 비교 대상이 아니라 제안모델과 비교모델이 함께 쓰는 재료다.

10-22 판의 한 조건 노트북(`Mark5.3 제안모델 실험.ipynb`)은 2026-10-23 에 아래 Colab 노트북으로 바뀌었다(git 이력에 남아 있다).

## Colab 노트북 (2026-10-23 판)

모두 **'모두 실행'** 한 번으로 끝나고, L4 로 돌린다(Blackwell 이면 첫 셀이 멈춘다 · CPU 런타임 허용 · 시간 제한 없음).
앞 단계를 기다리는 셀은 해시로 최신 판을 확인하므로 동시에 시작해도 된다.

| 노트북 | 하는 일 |
| --- | --- |
| `../Mark4.X/notebooks/Mark4.X 교사 학습(B+·B0).ipynb` × 8 | 교사마다 B+(mn10)·B0(조각, `--recipe run3`) 학습. 지금 데이터 인덱스로 끝난 판은 건너뜀 |
| `notebooks/1 교사 점수·채택 관문.ipynb` | 교사 16개 대기 → 반입(`--kind mn10`·`segment`) → 점수 캐시 → 관문(후보 B+ · 기준 B0) |
| `notebooks/2 Mark5.3 본실험 40회.ipynb` | [10-26] 관문 '채택(B+)' 이면 결합기 가짜 정답 → 논문 결과물 4개용 8조합 × seed 5 = 40회(`--preset cmb --cmb_paper`). 35회 완료·라벨 없음 0 의 5회 실패로 끝났다. 라벨 없음 0 수정 판에서는 [셀 3] 이 멈춘다(다시 돌리면 35회를 다시 학습하므로) |
| `notebooks/3 비교모델 60회.ipynb` | [10-26] 비교모델 3개 × 정답 라벨만 × 정답 200·500·1,000·전량 × seed 1~5. 지식 융합 구조가 아니므로 우리 가짜 정답(plc)을 쓰지 않음 |
| `notebooks/4 Mark5.3 라벨 없음 0 5회·결과물 4개.ipynb` | [10-26] 노트북 2 에서 실패한 정답 200·라벨 없음 0 의 5회만 다시 돈다(`--cmb_paper --cmb_unlabeled_n 0`). 나머지 35회는 요약 완전·지금 체크포인트 평가만 확인하고 다시 돌리지 않는다. 끝나면 결과물 4개(`model/make_four_results.py`) |
| `notebooks/5 비교모델 평가만 60회(argmax).ipynb` | [10-26] 비교모델 eval 의 판정 규칙을 Mark5.3 과 같은 보정 없는 argmax 로 바꾼 뒤 60회의 val·test 평가만 다시 돈다(학습이 끼면 멈춤). 끝나면 성능 요약 = argmax 값인지 확인하고 `make_argmax_summary.py` 표를 다시 만든다. 노트북 4 의 마지막 셀이 이 60회를 기다린다 |

- 결과물 4개(10-26): ① 데이터 효율 곡선 ② 라벨 없는 클립 수 곡선(정답 200) ③ 비교표 ④ 지식 융합 이득 표(정답 200: 평범한 구조 · 전문가 의사라벨만(고정 규칙) · 제안모델). test 4지표, seed 1~5 평균 ± 표준편차.
- `tests/verify_release_contract.py` 가 다섯 노트북의 셀 문법·L4 규칙·시간 제한 없음·80회 계획 태그·8조합(40회) 부분집합·plc 없음을 확인한다.

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
