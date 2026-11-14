# KAMPA 추가실험 실행 코드 최종 검토

검토일: 2026-11-14. 독립 검토자: code-reviewer 하위 에이전트.

문서 상태: 원본 복원과 마지막 노트북 재생성 후 최종 해시 확인 완료.

검토한 최종 파일에서 **남아 있는 P1/P2 결함을 발견하지 않았다.** 이는 아래 범위의 코드 검토와 수행한 검증에 대한 판단이며, GPU에서 전체 연구 실험을 완료했다는 뜻은 아니다.

적용한 스킬: [code-reviewer SKILL.md](/home/ahnsujin/.codex/plugins/cache/awesome-codex-subagents/codex-quality-security/0.1.0+codex.20260613115031/skills/code-reviewer/SKILL.md). 실행 코드·노트북·테스트는 수정하지 않았으며 이 최종 검토 문서만 작성했다.

## 발견 사항과 잔여 검증 범위

| 구분 | 최종 판단 |
|---|---|
| P1 | 미해결 사항 발견 없음 |
| P2 | 미해결 사항 발견 없음 |
| 실제 Colab/GPU 실행 | 이 검토에서는 미실행. Drive 마운트, L4 연산, 실제 GPU 중단·재개와 장시간 연구 학습은 사용자의 실행에서 확인해야 한다. |
| CPU 재개·완료 | 작은 합성 모델을 이용한 중단 주입 검증을 수행했다. 실제 연구 음원을 이용한 학생 학습은 수행하지 않았다. |
| 과학적 결과 | 추가실험의 정확도·효과 크기·유의성은 아직 산출되지 않았다. 구현 검증으로 그러한 결과를 대신할 수 없다. |
| 환경 간 재현성 | 런타임 정보에 따른 결과 분리와 재개 거부를 검토했다. 서로 다른 CPU/GPU에서 동일 수치가 나온다는 보장은 하지 않는다. |

검토 도중 `unlabeled_1358.wav`의 로컬 원본과 기존 모델 입력의 버전 차이를 확인했다. 이후 사용자의 지시에 따라 원본과 출처 기록이 기존 실험에 사용된 버전으로 복원되었다. 최종 원본 해시는 `f6caae62f7ae06c3e705231a8ec3f69724fcc0ce6f3fea71f82a0da4f1615515`이며, PCM_16 변환 결과는 기존 입력 `667f1910ac25365ea4c5f6383dec129eb594a98e4545d34ee342515df57eaa69`와 정확히 같다. 최종 `audio_source_revisions.json`은 빈 객체이고 백업 변환 예외는 0개다. 추가실험 실행기는 원본 파일을 바꾸지 않는다. 이 발견과 복원은 기존 KAMPA의 즉시 전면 재학습이 필요하다는 결론을 뜻하지 않는다.

## 검토 범위와 이전 지적의 반영

초기 전달 ZIP의 `feedback_core.py`와 최종 소스의 실제 diff를 비교하고, 다음 파일의 전체 관련 코드를 읽었다.

- `src/feedback_core.py`
- `tools/build_notebooks.py`
- 재생성한 노트북 2개 및 포함 ZIP
- `tests/test_protocol.py`, `tests/test_resume_integrity.py`
- 입력 판정에 사용되는 `audio_inputs.csv`, `audio_source_revisions.json`, `input_hashes.json`, `teacher_hashes.json` 및 기존 점수 메타데이터

| 초기 문제 / 확인 대상 | 최종 구현과 확인 위치 | 검토 결과 |
|---|---|---|
| 원본 FLOAT와 모델 입력 PCM_16의 해시를 같은 것으로 취급 | `feedback_core.py:146`의 `load_inputs`, `:363`의 `verified_audio` | `source_sha256`과 실제 입력 `sha256`을 분리한다. 원본을 변환하더라도 최종 바이트 해시가 고정된 Drive 입력과 정확히 같아야 통과한다. 과거 백업 처리 경로도 해시 검증을 요구하지만 최종 manifest에서 사용하는 예외는 0개다. |
| 교사 점수와 실제 입력의 연결 검증 누락 | `feedback_core.py:433`의 `verified_teacher_scores`, `:570`의 `run_suite` | 교사 8개 파일 해시, 고정된 점수 CSV, 메타데이터의 교사 목록·행 수·전체 입력 identity·확률 범위를 확인한다. 음원 검증 후 점수를 읽으며, 새 교사 추론이나 재학습을 실행하지 않는다. |
| 인코딩 변경으로 Oracle 중복 제거가 무력화될 가능성 | `feedback_core.py:187`의 `oracle_pool`, `:579`의 Oracle 구성 | 교사 파일 간 중복과 학생 지도 데이터 중복은 원본 해시로 처리한다. 평가 출처 그룹과 원본 해시를 제외하며, 교사 TRAIN만 사용한다. |
| GPU 강제 실행 | `feedback_core.py:23`, `:471`, `build_notebooks.py:51`, `:78` | CPU를 허용한다. CUDA 사용 시 GPU 이름·capability·CUDA·드라이버 등을 기록한다. sm_100/sm_120 계열은 거부하며 노트북은 L4 사용을 안내한다. |
| 다른 환경에서 재개하거나 결과를 혼합할 가능성 | `feedback_core.py:472`, `:596`, `build_notebooks.py:52` | 런타임·코드·입력·targets·학습 설정이 실행 identity에 포함된다. 노트북 결과 경로를 런타임별로 나누고, 집계에서도 다른 런타임이 섞이면 중단한다. |
| 마지막 체크포인트 저장 직후 중단하면 history/best가 불완전 | `feedback_core.py:505`, `:547`, `:552` | 현재 가중치·최적 가중치·optimizer·scheduler·history·RNG를 하나의 원자적 `last.pt`에 묶는다. 재개 후 추가 에폭이 없어도 history와 best를 다시 내보낸다. |
| 완료 판정에서 manifest/targets/meta 누락 | `feedback_core.py:20`, `:43`, `:483`, `:567`, `:599` | 완료에 필요한 11개 파일을 해시로 검증한다. 완료 스킵과 집계가 동일한 검증 함수를 사용한다. |
| 파형 캐시의 부분 저장·내용 변경 | `feedback_core.py:393` | 임시 파일을 완성한 후 교체한다. shape·유한성·입력 해시·전처리 식별자·파형 해시를 검증한다. 손상된 기존 캐시는 오류를 내며 임의로 사용하지 않는다. |
| 원라벨 재현 테스트의 `scores=None` 회귀 | `test_protocol.py:13` | 원래 점수의 수학적 재현 검사에는 `legacy_scores=True`를 명시했다. 연구 실행은 별도의 엄격한 점수 검증 함수를 거친다. |

## 실험 통제와 데이터 누출 검토

메타 결합기는 각 조건에서 선택한 지도 샘플만으로 적합한다. 라벨 없는 데이터의 정답은 `load_inputs`에서 비우며 학습 targets를 만들 때 사용하지 않는다. 시험 라벨은 모델 선택이나 임계값 선택에 사용하지 않는다.

2×2 비교는 fixed/meta 결합과 single/set 손실을 나눈다. `meta_top1_matched`와 `soft_selected`는 KAMPA와 같은 채택 샘플·가중치를 유지한다. `soft_all`은 U 전체와 균등 가중치를 사용하므로 샘플 범위와 가중치까지 달라지는 대조군임을 노트북에 밝힌다. 해당 대조군은 학습 배치의 U를 제거하지 않아 joint-forward/BatchNorm 구성 변화와 손실의 차이를 구분한다.

Oracle은 교사 TRAIN의 출처를 이용해 9개 클래스를 복원한 집합을 사용한다. 동일 파일 제거 후 12,477개라는 수치를 독립적인 원본 정답 개수나 보장된 성능 상한으로 설명하지 않는다. 증강을 포함한 특권적 지도학습 비교라는 노트북 설명을 확인했다.

4번 재분할은 교사 TRAIN/VAL 및 기존 U에 노출된 그룹을 새 VAL/TEST에서 제외한다. 같은 split·N·학습 seed의 두 모델은 같은 지도 샘플과 VAL/TEST를 사용한다. 기존 데이터의 사후 재분할 검사이며, 새 집·새 기기에서의 외부 검증 또는 교사까지 다시 학습하는 전체 파이프라인 교차검증은 아니라는 한계를 명시한다.

## 직접 수행한 검증

1. 최종 `load_inputs`를 읽기 실행해 L=2,000, U=2,000, VAL=430, TEST=430과 학생 분할 간 출처 그룹 겹침 0을 확인했다.
2. 최종 `verified_teacher_scores`를 실제 고정 입력과 교사 파일에 대해 실행했다. 4,860행의 입력 identity는 `50d0d674753a715ea00def01345d3f63e6c875efae6442538c83dd4b506ed2cf`이며 기존 점수 메타데이터와 정확히 일치했다.
3. 복원 후 최종 `verified_audio`가 현재 `unlabeled_1358.wav` 원본을 직접 이용해 `667f1910ac25365ea4c5f6383dec129eb594a98e4545d34ee342515df57eaa69` 입력을 재현함을 확인했다. 원본 파일 해시와 manifest의 `source_sha256`은 모두 `f6caae62f7ae06c3e705231a8ec3f69724fcc0ce6f3fea71f82a0da4f1615515`다. `audio_source_revisions.json`은 `{}`이고 4,860행 중 `conversion_source_path`가 있는 행은 0개다.
4. CPU 런타임 선택을 확인했다. CUDA capability 조회를 모의한 sm_100/sm_120 거부 검사도 통과했다. 이는 실제 GPU 실행 검증과 구분한다.
5. 노트북 두 개에 `nbformat.validate`를 적용하고 모든 코드 셀을 AST로 파싱했다. 저장된 실행 출력이 없고, 내장 ZIP·별도 실행 ZIP·현재 `src`의 17개 파일이 모두 바이트 단위로 일치함을 확인했다.
6. 이 세션에서 CPU 합성 회귀 테스트 4개를 직접 실행해 통과했다. 중간 에폭/마지막 에폭의 체크포인트 저장 직후 중단을 주입한 뒤 재개하면 무중단 실행과 최종 가중치·best·시간을 제외한 history가 일치했다. 완료 스킵에서 모델을 다시 만들지 않으며, 완료 파일 11개 각각의 변조와 런타임 변경을 거부했다. 이 테스트는 작은 Linear/Dropout 모델과 인공 파형을 사용한다.

재실행 가능한 합성 검증 명령:

```text
python -m pytest -q tests/test_resume_integrity.py
```

`tests/test_protocol.py`는 기존 20개 가짜 라벨의 재현, 2×2/soft 통제, 손실 식, Oracle 출처, 재분할, 합성 입력 모델 전방/역방향 연산을 검사한다. 상위 작업자가 수행한 전체 pytest 기록 `/tmp/kampa-feedback-pytest.log`를 직접 읽었으며 결과는 `13 passed, 2 warnings in 43.60s`이다. 원본·메타데이터 복원 후에는 모델 연산 검사 1개를 제외한 메타데이터 관련 검사를 다시 수행했고, `/tmp/kampa-feedback-metadata-pytest.log`에서 `8 passed, 1 deselected, 1 warning in 13.57s`를 확인했다. 코어·빌더·두 테스트 파일의 해시는 이 과정에서 바뀌지 않았다. 이 두 로그는 검토자의 별도 합성 검증과 구분되는 상위 작업자의 실행 기록이다.

상위 작업자가 수행한 grep 전수 검색의 `review/grep_inventory.json`과 `review/grep_full_matches.txt`도 읽었다. inventory에 기록된 Python 파일 17개의 해시는 모두 현재 파일과 일치한다. 노트북 2개는 디코딩 검사 대상으로 기록되어 있으며, 내장 소스 일치는 위 5번에서 독립적으로 재확인했다. 옛 경로·placeholder 검색의 3개 일치는 EfficientAT vendor 코드가 지원하지 않는 head/model/SE 옵션을 거부하는 `NotImplementedError`이며, 현재 실행 조건의 미구현 부분을 뜻하지 않는다.

관련 근거: `review/audio_manifest_audit.json`, `review/input_encoding_audit.json`, `review/efficientat_input_provenance.json`, `review/unlabeled_1358_impact.json`. 최종 전체 음원 감사에는 현재 원본 4,860개, 과거 백업 변환 0개가 기록되어 있다. 이는 상위 작업자의 전수 검사 기록이며, 이 최종 검토에서는 전체 입력 identity와 복원된 원본의 변환을 별도로 직접 재확인했다. `unlabeled_1358_impact.json`은 복원 결정 이전에 수행한 전후 입력 비교 기록이므로 최종 manifest 상태와 구분한다.

## 검토한 파일 SHA-256

| 파일 | SHA-256 |
|---|---|
| 초기 전달 ZIP 내부 `feedback_core.py` | `7c9fe10814e1bf79a7cca6c6d8ab2e586eb533da2677f7f7aab48dfc5637fcb4` |
| `src/feedback_core.py` | `ecee7a1461a492c64153ca3a919ea4e9249955c897b1ba5b546d6821caf6239b` |
| `tools/build_notebooks.py` | `48313f720b899c685f8a6bd2ea94db14cf295381e0694ed0cdf347840e8e067e` |
| `tests/test_protocol.py` | `7a3e5b9b1d0cfdf583fd87329359bf1c447561cc56d73ea1b55a23984a577e09` |
| `tests/test_resume_integrity.py` | `f2920a806f20695b64a5839ab6d3f42d3de1b41873a3a38f0f56dd33b4b07bb6` |
| `src/provenance_active.csv` | `2680d470f84092e6d20aecd03fa902d9d71af0682bd9d5416fac55afba4e628f` |
| `src/audio_inputs.csv` | `c0099de5b7b6be63709ac4bf0d8b3dd7987dd13aa0322dff5a8b072eab298d32` |
| `src/audio_source_revisions.json` | `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356` |
| `notebooks/01_피드백1~3_추가실험.ipynb` | `1e90566cd1fd4ef5935ff80c5b756479c15ff7e7ff14cbb03b0225d5f357be58` |
| `notebooks/02_피드백4_출처별_재분할검증.ipynb` | `ddcdaaf3de5ae50399f68ee70bdd4e0fa0953b9081edd5a3c495e98a5011025f` |
| `notebooks/KAMPA_추가실험_실행코드.zip` | `45342d1b0e8cf9f4f8fa946daa6c4e6ef39b4ffbb9225c6f0602ff4281592bf3` |

검토는 위 해시의 파일에 적용된다. 실제 연구 학습, 원본 연구 결과 덮어쓰기, 기존 KAMPA 재학습은 이 검토에서 수행하지 않았다.
