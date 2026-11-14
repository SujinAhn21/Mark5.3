# 교수님 피드백 반영 자료 · 2026-11-14 v2

5번은 기존 예측의 통계 재계산 자료이며, 1~4번은 사용자가 Colab에서 실행할 추가실험 코드다. 새 연구 학습은 아직 실행하지 않았다.

| 위치 | 내용 |
|---|---|
| `statistics/` | 5번의 95% CI 결과, 40개 예측 CSV, 출처 그룹, 독립 재계산 코드 |
| `notebooks/01_피드백1~3_추가실험.ipynb` | 메타 결합기·두 후보 손실 분리, 전체 분포 증류, 교사 TRAIN 정답 통합 비교 |
| `notebooks/02_피드백4_출처별_재분할검증.ipynb` | 출처별 재분할 3개 × 학습 seed 5개 × KAMPA/EfficientAT |
| `src/` | 노트북에 포함되는 실행 코드와 입력·출처 메타데이터 |
| `tests/`, `review/` | 구현 검증과 독립 code-reviewer 검토·입력 확인 기록 |
| `tools/` | 노트북 재생성, 배포 검증, 출처 메타데이터 재생성 도구 |

노트북 한 파일을 Colab에 업로드하여 위에서 아래로 실행한다. CPU도 지원한다. GPU를 사용하면 기존 프로젝트 규칙대로 L4를 우선 사용하며 Blackwell 계열은 제외한다. Torch 확인과 모델 실행은 별도 Python 프로세스에서 수행하고 실행 시간 제한은 두지 않는다.

프로젝트 경로는 `/content/drive/MyDrive/NewProject/codes/experiments/Mark5.3`이다. 결과는 형제 폴더 `KAMPA_feedback_20261114_v2/<환경해시>/<1_3 또는 4>`에 저장한다. 하드웨어·라이브러리 환경이 달라지면 결과 폴더가 분리되고 서로 다른 환경의 결과는 합산하지 않는다.

기본 N=[200,2000]으로 1~3번은 90회, 4번은 60회, 총 150회를 실행한다. 같은 설정으로 다시 실행하면 마지막 저장 에폭에서 재개하고, 11개 완료 산출물의 해시가 모두 맞는 회차는 건너뛴다. 완료 표식만으로 건너뛰지 않는다.

모든 조건은 같은 실행기로 새로 학습한다. fixed 결합은 odds 기반 비학습 9클래스 규칙으로, 기존 논문의 raw-score 규칙과 별도 조건이다. full-q 증류는 T=1이며 `soft_selected`는 채택 대상·가중치를 고정하고 `soft_all`은 모든 U를 균등 사용한다. Oracle은 출처로 복원한 교사 TRAIN 라벨만 추가하며 숨겨진 U 정답을 쓰지 않는다. 각 조건의 상세 설명과 한계는 노트북 첫 셀과 전체 반영계획에 보존했다.

실제 모델 입력과 원본 파일은 WAV 인코딩이 달라 해시가 다를 수 있다. `audio_inputs.csv`는 실제 Drive 입력과 원본의 해시를 따로 보관한다. 실제 입력 4,860개·교사 8개·점수 CSV와 JSON의 입력 버전을 확인한 뒤 기존 교사 점수를 재사용한다. 각 새 지도 부분집합에서는 메타 결합기만 다시 적합한다.

`unlabeled_1358.wav`는 사용자의 지시에 따라 로컬/Drive 원본을 기존 실험에 사용한 교체 전 구간으로 복원한다. 교체 후 8~11초 파일과 이전 출처 장부는 별도 백업하고 복원 이력을 남긴다. 해당 음원의 낮은 유효음 비율은 이력에 유지한다. 기존 체크포인트·점수·가짜 정답·시험 예측을 변경하거나 재학습하지 않는다. 복원 및 분할 검토 기록은 `review/restoration_record.json`, `review/restoration_split_audit.json`에서 확인한다.

5번 CI는 학습된 5개 seed 모델을 고정한 시험 출처 그룹 표집의 불확실성이다. 새로운 학습/분할 전체의 CI나 새로운 집·기기에 대한 외부 검증으로 해석하지 않는다. EfficientAT 과거 실행은 동일 입력 경로·실행 로그·예측 정확도로 연결했고, 당시 음원별 SHA 기록은 없다는 한계도 검토 기록에 명시했다.

재현 명령(이 폴더 기준):

```bash
python statistics/재계산_paired_statistics.py
python -m pytest -q tests
python tools/verify_package.py
```

검증에는 Python 3.10 이상, 호환되는 torch/torchaudio/torchvision, `src/requirements.txt`, pytest, nbformat이 필요하다. 출처 장부 재생성 도구에는 openpyxl이 추가로 필요하다. 노트북 사용에는 pytest/nbformat/openpyxl이 필요하지 않다. `tools/build_notebooks.py`는 현재 src와 바이트가 같은 ZIP을 포함한 두 노트북을 만든다. 배포 파일을 변경한 뒤에는 검토·검증을 마친 다음 `tools/verify_package.py --refresh`로 매니페스트를 갱신한다.
