# 재현 기록 (reproducibility)

이 폴더는 교사(Mark4.1~4.8)와 Mark5.3 학습 데이터를 **같은 클립으로 다시 만들 수 있게** 하는 작은 기록이다.
지금은 오디오와 데이터를 만드는 전처리 도구가 형제 저장소 `Mark5.2/`(데이터 창고)에 있다. 논문용으로 다듬을 때 데이터 목록·도구를 이 저장소로 모은다.
오디오 자체는 넣지 않는다. 원본 데이터셋마다 이용조건이 달라 재배포할 수 없는 것이 섞여 있기 때문이다(아래 표).
오디오는 각 데이터셋의 원래 사이트에서 받고, 여기 있는 목록·선택 결과·스크립트로 같은 구간을 잘라 쓴다.

## 무엇이 들어 있나

| 위치 | 내용 |
|---|---|
| `external_sources/tools/` | 큰 압축 파일에서 필요한 파일만 HTTP Range 로 꺼내는 도구 — `fsd_range.py`(FSD50K 분할 zip) · `bedavin_range.py` · `clotho_pick.py`(Clotho 7z) · `dcase_t5_range.py`(DCASE2018 Task5 eval zip 23개) |
| `external_sources/lists/` | 받은 파일 목록 — FSD50K `split,id` 목록(`ids_*.txt`), SINS 평가 세트 파일 목록(`list_sins_eval.txt`) |
| `external_sources/selection/` | 후보를 고른 결과 — SINS 평가 세트에서 고른 920구간(`pick_eval.json`: 세션·구간·노드), 10-22 외부 후보 목록(`external_metadata_candidates_20261022.csv`) |
| `external_sources/gate_reports/` | 품질·내용 게이트 결과(클립마다 출처 ID, 자른 시작 초, 통과 여부, sha256) — 10-23 판(`gate_report.csv`)·10-24 판(`gate_report_20261024.csv`)과 내용 검사 기준값(`content_check*.json`). 두 판 모두 talking 에 아이 말소리가 들어 있던 때의 결과다(아래 '10-25 talking 정의 변경') |

클립 하나하나의 최종 출처·자른 위치·sha256 은 `../../data_provenance.xlsx`(저장소 밖, 구글 드라이브 `NewProject/codes/experiments/`)에 있고,
교사 데이터 판의 오디오 해시 목록은 각 `Mark4.x/data_release_mark4.x.json` 에 있다.

## 외부 자료 폴더 구성 (게이트가 기대하는 위치)

`gate_external_candidates.py --ext_root <폴더>` 는 아래 구성을 기대한다. 우리 환경에서는 `C:\Users\user\journal_external_raw\` 였다.
아래 도구는 `external_sources/` 에서 `<폴더>` 를 작업 위치로 두고 돌린다.

| 하위 폴더 | 내용 | 받는 법 |
|---|---|---|
| `fsd50k_meta/` | FSD50K `FSD50K.metadata/`·`FSD50K.ground_truth/`·`FSD50K.doc/` | Zenodo 4060432 의 `FSD50K.metadata.zip`·`FSD50K.ground_truth.zip`·`FSD50K.doc.zip` 를 받아 푼다 |
| `fsd50k_audio/<id>.wav` | FSD50K 클립 | `tools/fsd_range.py lists/ids_*.txt fsd50k_audio` (분할 zip 에서 필요한 파일만 Range 로) |
| `clotho_meta/` · `clotho/audio/<split>/` | Clotho 2.1 메타데이터 CSV · 샤워 21개 | Zenodo 4783391 의 `clotho_metadata_*.csv`·`clotho_captions_*.csv` · `tools/clotho_pick.py clotho` (`pip install py7zr` 필요) |
| `bedavin/<cls>__<id>.wav` | BeDAViN 샤워·변기 원본 10개 | `tools/bedavin_range.py bedavin` |
| `bbc_sfx/<id>.wav` | BBC 효과음 샤워 6개 | 도구 없음 — BBC Sound Effects(RemArc) 사이트에서 id 로 받는다. id 는 `gate_external_candidates.py` 의 `BBC_SHOWER` |
| `snu_b36_50e/cd_wav/<폴더><번호>.wav` | SNU-B36-50E 의자 끌기(폴더 027~035, 450개) | Zenodo 3257142 압축을 풀고 `audio/027~035/*.m4a` 를 ffmpeg 로 모노·44.1kHz·PCM16 wav 로 바꾼다(예: `ffmpeg -i audio/027/027000.m4a -ac 1 -ar 44100 -c:a pcm_s16le cd_wav/027000.wav`) |
| `audioset_meta/class_labels_indices.csv` | AudioSet 527 라벨 이름(내용 검사용) | EfficientAT 저장소의 metadata 와 같은 파일 |
| `dcase2018t5/` | DCASE2018 Task5 eval `meta.txt`·`evaluation_setup/map.txt`, 위치표 `eval_index.json`, 오디오 `eval_audio/` | 메타는 Zenodo 1964758 `DCASE2018-task5-eval.meta.zip` · 위치표는 `tools/dcase_t5_range.py index dcase2018t5/eval_index.json` · 오디오는 `tools/dcase_t5_range.py fetch dcase2018t5/eval_index.json lists/list_sins_eval.txt dcase2018t5/eval_audio` |
| `chime6_meta/` · `chime6_audio/` | CHiME-6 전사 JSON · dev S02·S09 / eval S01·S21 의 Kinect 배열 CH1 | OpenSLR 150 `CHiME6_transcriptions.tar.gz` · 오디오는 압축을 흘려받으며 필요한 파일만 푼다: `curl -sL <미러>/resources/150/CHiME6_dev.tar.gz \| tar -xz -C chime6_audio --wildcards '*S02_U0?.CH1.wav' '*S09_U0?.CH1.wav'` (eval 은 `CHiME6_eval.tar.gz` 에서 S01·S21) |

SNU 처럼 압축 형식을 바꾸는 출처는 디코더 판에 따라 wav 바이트가 조금 달라질 수 있다. 최종 확인은 아래 4단계의 오디오 해시 대조로 한다.

## 다시 만드는 순서

1. 원본을 받는다. 외부 자료는 `tools/` 의 도구에 `lists/` 의 목록을 넣어 필요한 파일만 받는다
   (예: `python tools/fsd_range.py lists/ids_human_nonspeech_all.txt fsd50k_audio`). AI Hub 는 사이트에서 직접 신청해 받는다.
2. `../../Mark5.2/preprocessing/gate_external_candidates.py --tag <꼬리표>` 로 같은 후보에 같은 게이트를 건다. 목록이 외부 자료 폴더에 없으면 이 폴더의 사본을 읽는다.
3. `../../Mark5.2/preprocessing/` 의 재배치·적용 스크립트(`resplit_teacher_positives.py`·`replace_house_background.py`·`reallocate_hard_negatives.py`·`rebuild_talking_m45.py`·`remove_child_speech.py`)와
   증강·인덱스 스크립트로 교사 데이터를 만든다. 각 스크립트 머리말에 순서와 기대값이 있다.
4. `make_data_release.py` 로 만든 오디오 해시를 저장소의 `data_release_mark4.x.json` 과 대조한다. 같으면 같은 데이터다.

## 10-25 talking 정의 변경 (아이 말소리 제외)

2026-10-25 에 talking 을 "어른 말소리 + TV" 로 바꿨다. `remove_child_speech.py` 가 Mark5.3 의 아이 말소리 CHiME-Home 클립 129개
(정답 53 · 라벨 없는 54 · val 10 · test 12)를 같은 split 쪽 CHiME-Home 청크(어른 말소리·TV)로 바꾸고, 교사 Mark4.5 의 talking val·test
(각 215)를 바뀐 Mark5.3 학습 쪽 talking 에서 세션이 겹치지 않게 다시 나눈다. 파일 이름·클립 수·데이터 목록은 그대로이고 오디오 내용만 바뀌므로,
같은 데이터인지는 4단계의 오디오 해시(`data_release_mark4.5.json`)와 provenance 의 sha256 으로 확인한다.

`gate_reports/` 의 10-23·10-24 판은 이 변경 전에 돌린 것이다. talking 내용 검사 기준값(`content_check_20261024.json` 의 talking)은
그때의 Mark4.5 talking val·test(아이 말소리 포함)로 계산했고, 라벨 목록에도 아이 말소리(`Child speech, kid speaking`·`Babbling`)가 있었다.
지금 `gate_external_candidates.py` 는 이 두 라벨을 talking 에서 빼고 others 검사에만 쓰므로, 같은 후보로 다시 돌리면 talking 기준값과
talking 통과 여부가 저장된 보고서와 다를 수 있다. 10-24 에 고른 talking train(SINS 평가 세트 TV·대화 · CHiME-6 어른 대화)은 그대로 쓴다.

## 원본 데이터셋과 이용조건 (오디오를 여기 넣지 않는 이유)

| 데이터셋 | 이용조건 | 재배포 |
|---|---|---|
| AI Hub (층간소음·생활환경 소음 등) | AI Hub 데이터 이용정책·이용약관 — 승인받지 않은 제3자에게 열람·제공·배포 금지, 국외 반출은 별도 합의, 이용 시 "한국지능정보사회진흥원의 사업결과" 표시 | 불가 — 각자 신청해 받는다 |
| SINS (DCASE2018 Task5 개발 세트, Zenodo 1247090) | CC BY-NC 4.0 | 비상업·출처 표시 조건으로 가능 |
| SINS 평가 세트 (DCASE2018 Task5 eval, Zenodo 1964758) | CC BY-NC-ND 4.0 | 잘라 만든 클립은 불가(변경 금지) |
| CHiME-6 (OpenSLR 150) | CC BY-SA 4.0 | 같은 조건으로 공개해야 함 |
| CHiME-Home | 원 배포처 조건 | 원 배포처 확인 |
| FSD50K (Zenodo 4060432) | 클립마다 CC0 · CC BY · CC BY-NC · Sampling+ | 클립별 조건(게이트 보고서의 license 열) |
| Clotho 2.1 | Freesound 원 라이선스(메타데이터 CSV) | 클립별 조건 |
| BeDAViN (Zenodo 14486235) | CC BY 4.0 | 출처 표시 조건으로 가능 |
| BBC Sound Effects (RemArc) | 연구·교육용 | 불가 |
| SNU-B36-50E (Zenodo 3257142) | GPL-3.0 | 같은 조건 |
| ESC-50 | CC BY-NC 3.0 | 비상업·출처 표시 조건으로 가능 |
| VOiCES (rm4 tele) | CC BY 4.0 | 출처 표시 조건으로 가능 |

논문·발표에서는 위 데이터셋을 모두 인용하고, AI Hub 데이터는 이용정책대로 사업결과임을 밝힌다.
모델 가중치·결과(성능 표·예측 CSV)는 우리가 만든 결과물이라 공개할 수 있다. 모델이 확정되면 코랩 노트북도 이 저장소에 함께 둔다.
