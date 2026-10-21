"""[신설 2026-10-22] 자동 학습률 옵티마이저 Prodigy 를 만드는 곳.

학습률 선정 실험(model/select_lr.py, 후보 3개 x seed 0)을 없애고, 제안모델과 비교모델이 모두 같은 자동 학습률 절차를 쓰게 한다.
소스는 third_party/prodigy(prodigyopt 1.1.2, MIT, 원본 그대로 — SOURCE.md)에서 읽는다. 코랩에 pip 로 설치된 prodigyopt 가
있어도 이 폴더가 먼저 잡히도록 sys.path 맨 앞에 넣고, 실제로 불러온 파일이 이 폴더인지 확인한다.

설정(2026-10-22 결정, third_party/prodigy/SOURCE.md 와 같은 내용):
    Prodigy(params, lr=1.0, weight_decay=<config.weight_decay>, decouple=False), 나머지 기본값
    - lr 은 Prodigy 가 추정한 보폭 d 에 곱하는 배수다. 모든 파라미터 그룹이 같은 값이어야 한다(prodigy.py:151).
    - decouple=False: Adam 때와 같은 결합형 L2 weight decay 를 유지한다(Prodigy 기본값은 AdamW 식 분리형).
    - ReduceLROnPlateau 는 그대로 쓴다. d 는 줄지 않으므로(prodigy.py:230-233) 정체되면 스케줄러가 lr 배수를 줄인다.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PRODIGY_DIR = os.path.join(PROJECT_ROOT, "third_party", "prodigy")
PRODIGY_LR = 1.0


def make_prodigy(params, weight_decay):
    """params 는 파라미터 목록이나 param group 목록. group 에 'lr' 이 있으면 모두 PRODIGY_LR 이어야 한다."""
    if PRODIGY_DIR not in sys.path:
        sys.path.insert(0, PRODIGY_DIR)
    import prodigyopt
    path = os.path.abspath(prodigyopt.__file__)
    if not path.startswith(PRODIGY_DIR + os.sep):
        raise RuntimeError(f"[ERROR] prodigyopt 를 third_party/prodigy 가 아닌 곳에서 불러왔습니다: {path}")
    params = list(params)
    for g in params:
        if isinstance(g, dict) and float(g.get("lr", PRODIGY_LR)) != PRODIGY_LR:
            raise ValueError(f"[ERROR] Prodigy 는 그룹마다 다른 lr 을 쓸 수 없습니다(prodigy.py:151): lr={g['lr']}")
    return prodigyopt.Prodigy(params, lr=PRODIGY_LR, weight_decay=float(weight_decay), decouple=False)


def prodigy_d(optimizer):
    """지금 Prodigy 가 쓰는 보폭 추정값 d(모든 그룹이 공유). 실제 보폭은 d x lr 배수."""
    return float(optimizer.param_groups[0]["d"])
