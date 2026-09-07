# seed_utils.py

import random
import numpy as np
import torch
import os

def set_seed(seed_value=42, deterministic=False):
    """
    Reproducibility(재현성)를 위해 전역 random seed를 설정하는 함수입니다.

    [확장 2026-08-30 / Mark5_EXP] deterministic 인자 신설.

    seed 와 deterministic 은 서로 다른 층위이고, 둘은 충돌하지 않는다.
      - seed          : "어떤 무작위를 쓸 것인가"  (가중치 초기화 · 배치 순서 · SpecAugment 마스크)
      - deterministic : "GPU 가 덧셈을 어떤 순서로 할 것인가"

    GPU 는 수천 개 계산을 동시에 처리하고, 그 결과를 합칠 때 먼저 끝난 것부터 더한다.
    소수점 연산은 더하는 순서에 따라 마지막 자리가 달라지므로, 같은 seed 로 돌려도
    최종 정확도가 미세하게 흔들린다. deterministic=True 는 그 순서를 고정한다.

    둘을 같이 쓰면 "seed 마다 다른 결과가 나오되, 각 seed 는 언제 돌려도 재현된다" 가 되어
    논문에 seed 목록(예: 1~5)을 적으면 심사자가 그대로 재현할 수 있다.
    대가는 속도로, 보통 10~20% 느려진다.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    if deterministic:
        # cudnn 이 입력 크기별로 가장 빠른 알고리즘을 자동 탐색(benchmark)하면 실행마다
        # 다른 알고리즘이 뽑혀 결과가 달라진다. 탐색을 끄고 결정론적 알고리즘만 쓰게 한다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ※ torch.use_deterministic_algorithms(True) 는 일부러 쓰지 않는다.
        #    결정론 구현이 없는 연산을 만나면 예외를 던져 학습이 죽는다.
        #    cudnn 플래그만으로 CNN 학습의 재현성은 확보된다.
        # ※ PYTHONHASHSEED 는 파이썬이 시작되기 '전'에 정해져야 효과가 있어서
        #    여기서 os.environ 에 넣어도 이미 뜬 프로세스에는 영향이 없다.
        #    우리 코드는 dict/set 순서에 성능이 의존하지 않으므로 문제되지 않는다.

    print(f"Global seed set to {seed_value}"
          f"{' (cudnn deterministic)' if deterministic else ''}")