> **Mark5_EXP2 (2026-09-07 신설, 2026-09-17 Mark5.2 코드 기준으로 재정렬)** — Mark5.2 의 실험판(반복실험·라벨 효율·상한·pseudo-label 출처 비교).
> 코드 본체는 Mark5.2 와 같고(mn10 백본·파형 파서·운영점 자동 선택), 여기에 실험 인자가 붙는다.
>
> - 학습 `model/train_mark5.py --seed k --unlabeled_n n --labeled_n l [--ceiling_csv ceiling_labels.csv] [--spill_labeled] [--pseudo_csv … --pseudo_source self|gen]`
>   run_tag = `<설정>[_l<n>](_u<n>|_ceil)[_sp][_ps<이름>]_s<k>` (예 `mark5.2_PL_SA_l200_u2000_s1`). 평가 `model/eval.py` 는 같은 인자로 같은 태그를 찾는다.
> - `--labeled_n`: 라벨 효율 표(200·500·1,000). 클래스별 고르게, seed 기준 누적. `--spill_labeled`: 남은 labeled 를 unlabeled 로 합침(200 → 3,800).
> - `--pseudo_csv/--pseudo_source`: teacher 8개 대신 CSV 라벨(자기학습 self / 단일 generalist gen). `model/make_pseudo_labels.py` 가 만든다. feature KD 없음.
> - 실행기 `model/run_experiments.py --preset paper`(55회: labeled 전량·200·500·1,000 × u0/u2000/ceil + l200 곡선 u500·u1000) ·
>   `--preset plsource`(15회: l200+3,800 에서 합의/self/gen) · 또는 `--labeled_ns … --unlabeled_ns … --seeds …`.
> - 집계 `model/aggregate_results.py [--baseline_raw <비교모델 aggregate_raw csv>…]`: 표 1·곡선·천장·검정(기존) +
>   라벨 효율 표(`paper_table_label_eff_*`) · l200 곡선 · 출처 비교 표(`paper_table_plsource_*`) · 비교모델 짝지은 검정(`aggregate_tests_vs_baseline_*`).
> - 상한(ceil) 라벨 CSV 는 새 데이터 기준으로 `model/make_ceiling_labels.py` 로 다시 만들어야 한다(저장소의 ceiling_labels.csv 는 mark5.0 시절 것).
> - mn10 가중치 `weights/`·코드 `third_party/`·torchvision 요구는 Mark5.2 README 와 같다.
> - **10절(2026-09-17) open-set·제로샷·CLAP — 전부 학습 0회.** 평가 세트 인덱스 `dataset_eval_{zeroshot6,boundary_B,far_heldout}.csv` 는
>   Mark5.2 `preprocessing/generate_eval_index.py` 가 만들고 여기 루트에 복사해 쓴다.
>   `model/eval.py --split eval_far_heldout|eval_boundary_B` (① 학습에 없던 소리→others, Others FPR = 1 − ①) ·
>   `model/openset_report.py` (네 모델 ①·② 표: prediction_details CSV 만 읽는다, 옆의 Baseline_* 자동 탐색) ·
>   `model/zeroshot_eval.py` (앵커 9 + 새 6종 `shared_vild/resources/zeroshot_prompt_bank.json`, 제로샷 세트 + test) ·
>   `model/clap_zeroshot.py` (LAION-CLAP 630k-audioset-best, 같은 문장·같은 지표·같은 산출물 형식, `pip install laion-clap`; 체크포인트는 mmap 으로 읽고 추론은 no_grad — 로컬 CPU 에서 2.1 GB 로 돈다).
>   `shared_vild/resources/prompt_bank.json` 은 Mark5.2 것(talking 항목·others 동의어)으로 맞췄다(전에는 Mark5 옛 뱅크가 남아 있었다).
> 아래는 Mark5 원본 README 그대로.

# Mark5.0
- Based on Mark3.2 but much better with vision-like perspectives

# License
- This project is licensed under the PolyForm Noncommercial License 1.0.0.  
- Commercial use is not permitted.
- See the [LICENSE](./LICENSE) file for details.


