> **Mark5.3 (2026-10-07, Mark5_EXP2 `e2ebcd5` 에서 분기)** — D0 레시피 위에 전문가 8개의 정보를 더 쓰는 부품 6개를 얹은 판. `--m53 all`(또는 `PF,SK,OVA,FM,CB,SC` 중 일부).
> 근거는 2026-10-07 D0 결과 분석(라벨 2,000 격차의 49% 가 others 오판, 라벨 200 격차의 47% 가 machine_noise · 합의에서 빠진 487클립의 정답이 1·2등 전문가 안 89.7% ·
> 전문가 거절 신호의 others AUROC 0.967 · 가짜 정답 클래스 쏠림 dragging 86 / dog_bark 216).
>
> | 부품 | 하는 일 | 선행연구 |
> |---|---|---|
> | PF | 합의 가짜 정답 CE 가중치 1(옛 0.3 × 확신도). `--hard_weight` 필수(없으면 멈춤 — 정답 라벨 CE 가 1−α=0 이 되는 것을 막음) | SoftMatch (ICLR 2023) |
> | SK | 합의에서 빠진 클립에 9칸 부드러운 목표 = 1·2등 전문가 몫 + others 몫(1 − 최대 확신도), 가중치 0.5 | IOMatch (ICCV 2023), SoftMatch |
> | OVA | 학생에 전문가 이진 출력 8개(MLP 384→384→8), 전문가 확신도를 BCE 로 따라 함 · 평가 때 others 확률에 0.5 섞음 | IOMatch, SSB (ICCV 2023) |
> | FM | 학생 EMA 사본(0.99)의 무증강 답 → 증강 입력 CE, FreeMatch 클래스별 문턱(감쇠 0.99), 처음 2에폭 선형 증가, 가중치 0.5 | FixMatch 오디오 (EUSIPCO 2023), FreeMatch (ICLR 2023) |
> | CB | 가짜 정답 클래스 가중 (평균 채택 수/클래스 채택 수)^0.5 [0.5, 2] · FM 목표 분포 정렬 | ACR (CVPR 2023) |
> | SC | 지도 대조 손실(정답 라벨 + 채택 가짜 정답, 온도 0.1) 가중치 0.1 | You 외 (Sci. Rep. 2024) |
>
> - run_tag: D0 태그 뒤 `_M53`(전부) 또는 `_M53-PF-SK…`(일부). `--m53` 을 안 주면 Mark5_EXP2 와 텐서 단위로 같은 학습(2026-10-07 회귀 시험).
> - 실행기 `model/run_experiments.py --preset pl4`(pl 20회: 정답 라벨 2,000·1,000·500·200 × u2000 × seed 5). 학습률은 `model/select_lr.py --tied --extra "… --m53 all"`.
> - 체크포인트에 `ova_state_dict` 가 더해진다(eval.py 가 읽음). 효율 측정 `profile_efficiency.py --m53 all` 은 OVA 헤드 +150,920 파라미터를 넣는다.
> - teacher(resources/) 와 인덱스 CSV 는 D0 가 돈 드라이브 Mark5_EXP2 와 같은 판(3차 조각 teacher `_run3_20260923`, Mark5.2 인덱스).
> - 노트북: `Mark5.3 실험.ipynb`(현재실험_노트북) — 학습률 3회 → pl 20회 → 효율 → 비교표(Mark5.3 · D0 · 비교모델 4개, 4지표 + Others FPR, 짝지은 검정).

> **Mark5_EXP2 (2026-09-07 신설, 2026-09-17 Mark5.2 코드 기준으로 재정렬)** — Mark5.2 의 실험판(반복실험·라벨 효율·상한·pseudo-label 출처 비교).
> 코드 본체는 Mark5.2 와 같고(mn10 백본·파형 파서·운영점 자동 선택), 여기에 실험 인자가 붙는다.
>
> - 학습 `model/train_mark5.py --seed k --unlabeled_n n --labeled_n l [--ceiling_csv ceiling_labels.csv] [--spill_labeled] [--pseudo_csv … --pseudo_source self|gen]`
>   run_tag = `<설정>[_l<n>](_u<n>|_ceil)[_sp][_ps<이름>]_s<k>` (예 `mark5.2_PL_SA_l200_u2000_s1`). 평가 `model/eval.py` 는 같은 인자로 같은 태그를 찾는다.
> - `--labeled_n`: 라벨 효율 표(200·500·1,000). 클래스별 고르게, seed 기준 누적. `--spill_labeled`: 남은 labeled 를 unlabeled 로 합침(200 → 3,800).
> - `--pseudo_csv/--pseudo_source`: teacher 8개 대신 CSV 라벨(자기학습 self / 단일 generalist gen). `model/make_pseudo_labels.py` 가 만든다. feature KD 없음.
> - 실행기 `model/run_experiments.py --preset paper`(55회: labeled 전량·200·500·1,000 × u0/u2000/ceil + l200 곡선 u500·u1000) ·
>   `--preset plsource`(15회: l200+3,800 에서 합의/self/gen) · 또는 `--labeled_ns … --unlabeled_ns … --seeds …`.
> - **학습률 선정(2026-09-25)** `model/select_lr.py`: 새 층 {2e-4,1e-4,5e-5} × 백본 {1e-4,5e-5,2e-5} = 9회를 seed 0 · `--unlabeled_n 2000` 으로 학습해
>   val Macro F1 로 고른다. 비교모델 `--phase sweep` 과 같은 것은 **축당 후보 3개 · seed 0 분리 · val Macro F1 · test 미사용**이고, 다른 것은
>   **축 개수(1축 3회 vs 2축 9회)** 와 **선정 조건(비교모델은 cond=sup, 제안모델은 본실험 조건 u2000)** 이다 — 논문에 둘 다 적는다. 결과 `plots/lr_selected.json` 이 있으면 `run_experiments.py` 가
>   자동으로 `--lr/--encoder_lr` 을 넘긴다(`--ignore_selected_lr` 로 끔). 학습률이 config 기본값과 다를 때만 run_tag 에 `_lr<값>`·`_elr<값>` 이 붙는다.
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


