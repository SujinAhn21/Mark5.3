# -*- coding: utf-8 -*-
"""실행 로그를 화면과 logFiles/ 에 동시에 남긴다.

[신설 2026-09-26] train_mark5.py·eval.py 는 그동안 표준출력에만 찍어서, 코랩에서 돌린 뒤 결과를 보려면
화면을 긁어 옮겨야 했다(세션이 끊기면 그대로 사라졌다). select_lr.py·run_experiments.py 가 쓰던 _Tee 를
공용으로 빼서 네 스크립트가 같은 규칙으로 남기게 한다.

파일 이름   logFiles/<run_tag>__<무엇>.log     예) mark5.2_PL_SA_G04_AC_LT_s1__train.log
                                                  mark5.2_PL_SA_G04_AC_LT_s1__eval_val.log
            Mark4.x 비교모델(common/naming.py)이 쓰는 <tag>__train.log 규칙과 같은 모양이다.
이어쓰기   같은 태그를 다시 돌려도 지우지 않고 뒤에 붙인다. 실행마다 구분선·시각·명령줄이 먼저 들어간다.
"""

import os
import sys
import time


class _Tee:
    """여러 스트림에 같이 쓴다. 파일 쪽이 막혀도 화면 출력은 계속되게 예외를 삼킨다."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class RunLogger:
    """with 블록 안의 표준출력·표준에러를 화면과 파일에 같이 남긴다.

        with RunLogger(project_root, run_tag, "train"):
            ...

    블록을 빠져나올 때(예외로 끝나도) 걸린 시간과 종료 상태를 마지막 줄에 적는다.
    """

    def __init__(self, project_root, run_tag, what):
        self.dir = os.path.join(project_root, "logFiles")
        self.path = os.path.join(self.dir, f"{run_tag}__{what}.log")
        self.run_tag = run_tag
        self.what = what
        self._f = None
        self._out = None
        self._err = None
        self._t0 = None

    def __enter__(self):
        # [중복 방지] select_lr.py·run_experiments.py 는 자식 출력을 받아 같은 <run_tag>__<무엇>.log 에 이미 남긴다.
        # 그 실행기가 부를 때는 환경변수가 서 있으므로 여기서는 파일을 열지 않는다(같은 줄이 두 번 들어가지 않게).
        if os.environ.get("MARK5_CHILD_LOG_BY_PARENT") == "1":
            self._skip = True
            return self
        self._skip = False
        os.makedirs(self.dir, exist_ok=True)
        self._t0 = time.time()
        self._f = open(self.path, "a", encoding="utf-8")
        self._f.write("\n" + "=" * 78 + "\n")
        self._f.write(f" {self.run_tag} · {self.what} · 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._f.write(f" $ {' '.join(sys.argv)}\n")
        self._f.write("=" * 78 + "\n")
        self._f.flush()
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._out, self._f)
        sys.stderr = _Tee(self._err, self._f)
        print(f"[로그] {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_skip", False):
            return False
        mins = (time.time() - self._t0) / 60
        state = "정상 종료" if exc_type is None else f"오류로 끝남({exc_type.__name__})"
        try:
            print(f"[로그] {state} · {mins:.1f}분 · {self.path}")
        except Exception:
            pass
        sys.stdout, sys.stderr = self._out, self._err
        try:
            self._f.write(f"=== {state} · {mins:.1f}분 ===\n")
            self._f.close()
        except Exception:
            pass
        return False   # 예외를 삼키지 않는다
