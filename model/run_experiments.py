# run_experiments.py
# [신설 2026-08-30 / Mark5_EXP]
# [확장 2026-09-03] 실행 로그를 logFiles/ 에 파일로 남긴다.
# [수정 2026-09-03] 코드 검수 반영 — H1(단계별 재개) · H2(자식 버퍼링) · M6(stdout 복구) · M7(인터럽트).
#
# 2026-08-24 랩미팅 피드백 1·2·8번을 한 번에 돌리는 실행기.
#
#   1번 seed 를 매 실험마다 바꾼다          -> seed 1~5
#   2번 반복실험으로 평균·표준편차를 낸다   -> 같은 조건을 5 seed 로 반복
#   8번 unlabeled 양에 따른 데이터효율곡선  -> 0/500/1000/1500/2000
#
# 5조건 x 5seed = 25회. 조합마다 [학습 -> val 평가 -> test 평가] 를 돈다.
#
# --------------------------------------------------------------------------
# 재개는 단계 단위다  [H1 수정 2026-09-03]
# --------------------------------------------------------------------------
# 예전에는 완료 판정이 조합 단위였다. val·test CSV 가 둘 다 있어야 "끝난 것"이라,
# 체크포인트가 멀쩡히 있어도 평가에서 죽으면 30~60분짜리 학습을 통째로 다시 돌았다.
# 지금은 세 단계를 따로 본다.
#
#   학습    model/student_model_<run_tag>.pth  +  plots/loss_history_<run_tag>.csv
#   val     plots/performance_summary_<run_tag>_val.csv
#   test    plots/performance_summary_<run_tag>.csv
#
# 요약 CSV 는 존재만 보지 않고 **내용까지 확인한다**. 쓰다가 끊긴 파일을 완성본으로
# 착각하면 잘린 숫자가 논문 표에 들어간다(집계 쪽 M1 과 같은 이유).
#
# ⚠ [재검수 반영 2026-09-03] 학습 완료 판정에 체크포인트만 쓰면 안 된다.
#   train_mark5.py 는 val 이 좋아질 때마다 **에폭 루프 안에서** 체크포인트를 덮어쓴다.
#   80 에폭 중 25 에폭에서 코랩이 끊기거나 OOM 이 나거나 --step_timeout_min 이 자식을
#   죽이면 25 에폭짜리 체크포인트가 디스크에 남는데, 그것만 보면 "학습 끝남"이 되어
#   **덜 배운 모델을 평가한 숫자가 아무 경고 없이 논문 표에 들어간다.**
#   loss_history_<run_tag>.csv 는 에폭 루프가 끝난 뒤에만 쓰이므로 완주 표식으로 쓴다.
#   (H1 을 처음 고칠 때 들어왔던 회귀다. 고치기 전 코드는 조합 단위라 이 경우엔 옳았다.)
#
# --------------------------------------------------------------------------
# 로그를 왜 파일로 남기나
# --------------------------------------------------------------------------
# 코랩 셀 출력은 브라우저 화면에만 있다. 세션이 끊기거나 출력이 길어 잘리면 사라지고,
# 학습 진행 상황을 남에게 보여주려면 화면을 긁어 붙여야 한다.
# 그래서 같은 내용을 화면과 파일에 동시에 쓴다(_Tee). 프로젝트 루트가 드라이브
# 마운트 경로이므로 logFiles/ 에 쓰면 드라이브에 바로 남아 세션이 끊겨도 보존된다.
#
#   logFiles/run_experiments_<mark_version>_<시각>.log   전체 실행 기록(마스터)
#   logFiles/<run_tag>__train.log                        회차별 학습 로그
#   logFiles/<run_tag>__eval_val.log                     회차별 val 평가 로그
#   logFiles/<run_tag>__eval_test.log                    회차별 test 평가 로그
#
# ⚠ [H2 수정 2026-09-03] 자식을 반드시 `-u`(무버퍼)로 띄운다.
#   파이썬은 stdout 이 파이프면 8KB 씩 모았다가 내보낸다. _Tee 의 flush 는 부모가 직접
#   찍는 줄에만 듣고, 30~60분 학습 로그의 대부분인 자식 출력에는 효과가 없다.
#   그 상태로 코랩이 세션을 끊거나 OOM 으로 자식을 죽이면 **버퍼에 남은 마지막 수십 줄이
#   마스터·회차별 로그 양쪽에서 통째로 사라진다.** 왜 죽었는지 알려줄 traceback 이
#   하필 그 구간에 있다. 실측: -u 없으면 4줄이 2.03초에 몰려 나오고, 있으면 0.32/0.72/
#   1.12/1.52초에 실시간으로 나온다.
#
# --------------------------------------------------------------------------
# 실행 순서를 unlabeled 큰 것부터로 잡은 이유
# --------------------------------------------------------------------------
# u2000 이 논문 표 1 의 제안모델 조건(= 최종 모델과 같은 설정)이다.
# 이걸 먼저 5회 끝내두면, 중간에 시간이 모자라 멈추더라도 표 1 은 확보된다.
# 곡선(8번)은 나머지 조건이 채워지면서 완성된다.
#
# --------------------------------------------------------------------------
# 왜 subprocess 로 매번 새 프로세스를 띄우나
# --------------------------------------------------------------------------
# 한 프로세스 안에서 25회를 돌리면 GPU 메모리와 전역 상태(cudnn 플래그, 난수기,
# matplotlib figure)가 회차 사이에 남는다. 회차마다 프로세스를 새로 띄우면
# 매번 완전히 같은 출발선에서 시작하므로 재현성이 보장된다.
#
# --------------------------------------------------------------------------
# 사용법
# --------------------------------------------------------------------------
#   python model/run_experiments.py --mark_version mark5.0
#   python model/run_experiments.py --mark_version mark5.0 --dry_run       # 계획만 출력
#   python model/run_experiments.py --mark_version mark5.0 --seeds 1 2 3   # seed 일부만
#   python model/run_experiments.py --mark_version mark5.0 --unlabeled_ns 2000
#   python model/run_experiments.py --mark_version mark5.0 --force         # 완료분도 다시
#   python model/run_experiments.py --mark_version mark5.0 --no_log        # 로그 파일 안 남김
#   python model/run_experiments.py --mark_version mark5.0 --step_timeout_min 240
#
# [신설 2026-09-06] 천장(upper bound) 조건 — 기존 25회에 여섯 번째 조건으로 이어 붙인다.
#   python model/run_experiments.py --mark_version mark5.0 --ceiling                 # 5 seed
#   python model/run_experiments.py --mark_version mark5.0 --ceiling --dry_run
#   unlabeled 2,000개에 진짜 라벨(ceiling_labels.csv, make_ceiling_labels.py 산출)을 붙여 지도학습.
#   run_tag 가 _ceil_s<k> 라 25회 산출물(_u<n>_s<k>)과 겹치지 않고, 집계는 천장 행으로 따로 모은다.

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "vild"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared_vild"))

LOG_DIR = os.path.join(PROJECT_ROOT, "logFiles")

DEFAULT_SEEDS = [1, 2, 3, 4, 5]
# 큰 것부터. u2000 이 논문 표 1 조건이라 먼저 끝내둔다.
DEFAULT_UNLABELED_NS = [2000, 1500, 1000, 500, 0]

# 요약 CSV 가 온전한지 볼 때 확인하는 지표. aggregate_results.py 의 METRICS 와 같아야 한다.
REQUIRED_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                    "ROC AUC (Macro, OvR)", "Others FPR"]


class _Tee:
    """같은 내용을 화면과 파일에 동시에 쓴다.

    코랩에서는 버퍼링 때문에 출력이 한참 뒤에 몰려 나오는 일이 있어서 매번 flush 한다.
    학습이 몇 시간씩 걸리는데 진행 상황이 안 보이면 멈춘 것인지 알 수 없다.
    """

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass                               # 로그 쓰기 실패가 실험을 멈추면 안 된다
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _summary_is_complete(path):
    """요약 CSV 가 끝까지 쓰였는지 본다.  [H1·M1, 2026-09-03]

    os.path.exists 만 보면 쓰다가 끊긴 파일도 "끝난 것"이 된다. 실제로 중간에서 자른
    파일이 마지막 자리가 잘린 값(0.9871412218852)을 유효한 숫자로 통과시켰다.
    지표 6개가 다 들어와 있어야 완성으로 본다.
    """
    if not os.path.exists(path):
        return False
    try:
        got = set()
        with open(path, encoding="utf-8", newline="") as f:
            for parts in csv.reader(f):
                if not parts:
                    continue
                key = parts[0].strip()
                if key == "Class":
                    break
                if len(parts) >= 2:
                    try:
                        float(parts[1])
                        got.add(key)
                    except ValueError:
                        pass
        return all(m in got for m in REQUIRED_METRICS)
    except Exception:
        return False


def _run_tag(mark_version, seed, unlabeled_n, ceiling=False, labeled_n=None, spill=False, ps=None, lr=None, encoder_lr=None):
    """config 의 run_tag 를 그대로 쓴다(파일명 규칙을 두 곳에 적어두면 어긋난다)."""
    from vild_config import AudioViLDConfig
    c = AudioViLDConfig(mark_version=mark_version)
    c.experiment_seed = seed
    c.ceiling = bool(ceiling)
    c.unlabeled_n = None if ceiling else unlabeled_n
    c.labeled_n = labeled_n          # [신설 2026-09-17 / 8절] _l<n>
    c.spill_labeled = bool(spill)    # _sp
    c.pseudo_source = ps             # _ps<이름>
    c.apply_cli_learning_rates(lr, encoder_lr)   # [신설 2026-09-25 / LR] 기본값과 다를 때만 _lr/_elr (train·eval 과 같은 규칙)
    return c.run_tag


# [신설 2026-09-25 / LR] 학습률 선정(model/select_lr.py) 산출물. 있으면 본실험 전 회차에 자동으로 --lr/--encoder_lr 을 넘긴다.
SELECTED_LR_JSON = os.path.join(PROJECT_ROOT, "plots", "lr_selected.json")


def _resolve_learning_rates(args):
    """(새 층 lr, 백본 lr, 출처 설명). 우선순위: 명령줄 --lr/--encoder_lr(하나라도 주면 json 은 안 읽는다) > plots/lr_selected.json
    (--ignore_selected_lr 이 아니면) > None(config 기본값 = 옛 동작). 선정값이 기본값과 같으면 태그가 안 바뀌어 끝난 회차는 [건너뜀] 이 되고,
    다르면 태그에 _lr/_elr 이 붙어 전 회차를 새로 돈다(옛 산출물은 안 덮인다)."""
    if args.lr is not None or args.encoder_lr is not None:
        return args.lr, args.encoder_lr, "명령줄 --lr/--encoder_lr"
    if args.ignore_selected_lr or not os.path.isfile(SELECTED_LR_JSON):
        return None, None, ("config 기본값 (--ignore_selected_lr)" if args.ignore_selected_lr else "config 기본값 (lr_selected.json 없음)")
    try:
        with open(SELECTED_LR_JSON, encoding="utf-8") as f:
            sel = json.load(f)
        _ = float(sel["lr"])          # 필수 키가 있는지 여기서 확인한다
    except Exception as e:
        raise SystemExit(f"[ERROR] {SELECTED_LR_JSON} 을 읽을 수 없습니다({type(e).__name__}: {e}). "
                         "select_lr.py 를 다시 돌리거나 --ignore_selected_lr 을 주십시오.")
    if sel.get("model") != args.mark_version:
        raise SystemExit(f"[ERROR] {SELECTED_LR_JSON} 은 model={sel.get('model')!r} 로 만든 것인데 지금 --mark_version 은 "
                         f"{args.mark_version!r} 입니다. select_lr.py 를 다시 돌리거나 --ignore_selected_lr 을 주십시오.")
    lr = float(sel["lr"])
    elr = None if sel.get("encoder_lr") is None else float(sel["encoder_lr"])
    src = (f"plots/lr_selected.json (선정 {sel.get('selected_at')} · {sel.get('criterion')} · seed {sel.get('sweep_seed')}"
           + (" · 기본값과 같음" if sel.get("selected_is_default") else " · 기본값과 다름 → run_tag 에 _lr/_elr") + ")")
    return lr, elr, src


# [신설 2026-09-17 / 8절] 논문 회차 계획(체크리스트 8절). (labeled_n, unlabeled_n|"ceil") 조합 x seed.
#   labeled 2,000(전량) 행 : 하한 u0 · 제안 u2000 · 상한 ceil
#   labeled 200·500·1,000  : 지도만(u0) · 우리 방식(u2000)
#   labeled 200 의 unlabeled 곡선 중간점 : u500 · u1000 (u2000 은 위 행과 같은 회차라 중복 제거)
#   -> 조합 11개 x seed 5 = 55회. (체크리스트의 "60회" 는 labeled 200 곡선의 u2000 을 두 번 센 값)
PAPER_PLAN = [
    (None, 0), (None, 2000), (None, "ceil"),
    (200, 0), (200, 2000), (500, 0), (500, 2000), (1000, 0), (1000, 2000),
    (200, 500), (200, 1000),
]
# [신설 2026-09-17 / 8절] pseudo-label 출처 비교. labeled 200 + unlabeled 3,800(dataset_unlabeled 2,000 + 남은 labeled 1,800,
# --spill_labeled) 에서 라벨을 붙이는 주체만 바꾼다. 세 조건 모두 같은 풀이어야 공정하므로 합의 조건도 따로 돈다(3 x seed = 15회).
#   (labeled_n, unlabeled_n, spill, pseudo_source)   pseudo_source None = 전문가 8개 온라인 합의
PLSOURCE_PLAN = [(200, 2000, True, None), (200, 2000, True, "self"), (200, 2000, True, "gen")]
PSEUDO_CSV_FMT = "pseudo_labels_{ps}_l{l}_s{s}.csv"   # model/make_pseudo_labels.py 산출물(프로젝트 루트)


def _artifacts(mark_version, tag):
    """이 조합의 단계별 산출물."""
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    return {
        "ckpt": os.path.join(PROJECT_ROOT, "model", f"student_model_{tag}.pth"),
        # 학습이 끝까지 갔다는 표식. train_mark5.py 가 에폭 루프를 다 돈 뒤에만 쓴다.
        "loss": os.path.join(plot_dir, f"loss_history_{tag}.csv"),
        "val": os.path.join(plot_dir, f"performance_summary_{tag}_val.csv"),
        "test": os.path.join(plot_dir, f"performance_summary_{tag}.csv"),
    }


def _needs(mark_version, tag, force, skip_test):
    """[H1] 어느 단계를 돌려야 하는지 단계별로 판정한다.

    학습은 체크포인트와 완주 표식(loss_history CSV)이 **둘 다** 있어야 끝난 것으로 본다.
    체크포인트만 보면 에폭 도중에 죽어 남은 중간 저장본을 완료로 착각한다
    (파일 머리말의 재검수 반영 항목 참조).
    """
    a = _artifacts(mark_version, tag)
    trained = os.path.exists(a["ckpt"]) and os.path.exists(a["loss"])
    need_train = force or not trained
    need_val = force or not _summary_is_complete(a["val"])
    need_test = (not skip_test) and (force or not _summary_is_complete(a["test"]))
    return need_train, need_val, need_test


def _sh(cmd, log_path=None, timeout_sec=None):
    """한 단계 실행. 실패하면 False 를 돌려 그 단계만 건너뛰고 다음으로 간다.

    자식 프로세스의 출력을 파이프로 받아 한 줄씩 다시 내보낸다. 그래야 부모의
    sys.stdout(=_Tee)을 거쳐 화면과 마스터 로그에 같이 들어간다.
    log_path 를 주면 그 단계만 담은 파일도 따로 남긴다.

    [M7] KeyboardInterrupt 가 와도 자식을 정리하고 파일을 닫는다.
         timeout_sec 을 주면 그 시간을 넘긴 자식을 죽이고 다음으로 넘어간다
         (자식이 데드락에 걸리면 11시간짜리 실행이 그 자리에 무한정 서 있게 된다).

    ⚠ [재검수 반영 2026-09-03] 시간 제한은 출력 루프 안이 아니라 **별도 타이머**로 건다.
      예전에는 `for line in proc.stdout:` 안에서 시계를 봤는데, 자식이 한 줄도 안 찍고
      멈추면 그 for 문이 read 에서 막혀 검사 자체가 돌지 않았다. 막겠다고 적어둔 상황
      (조용한 데드락)에 정확히 안 듣는 구조였다. 타이머는 자식 출력과 무관하게 돈다.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()

    child_log = None
    if log_path:
        try:
            # [낮음] "a" 로 연다. "w" 면 같은 단계가 두 번 실패했을 때 첫 실패의
            # traceback 이 덮여 사라진다(마스터 로그에는 남지만 회차별 로그가 반쪽이 된다).
            child_log = open(log_path, "a", encoding="utf-8", buffering=1)
            child_log.write("\n" + "=" * 70 + "\n")
            child_log.write(f"# {' '.join(cmd)}\n")
            child_log.write(f"# 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"[WARN] 단계 로그 파일을 못 열었습니다({e}). 화면 출력만 남습니다.")
            child_log = None

    proc = None
    killer = None
    timed_out = []                                 # 타이머 스레드가 여기에 표시를 남긴다
    try:
        try:
            # [신설 2026-09-26] 자식(train_mark5·eval)도 스스로 <run_tag>__*.log 를 남기게 됐다. 여기서 이미 같은
            # 파일에 받아 적고 있으므로, 환경변수로 자식 쪽 기록을 끈다(같은 줄이 두 번 들어가지 않게).
            _env = dict(os.environ); _env["MARK5_CHILD_LOG_BY_PARENT"] = "1"
            proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
                # 자식이 한글을 찍는다. 로케일에 끌려가지 않게 UTF-8 을 못박고,
                # 깨진 바이트가 하나 나와도 예외로 죽지 않게 replace 로 흘려보낸다.
                encoding="utf-8", errors="replace", env=_env,
            )
        except Exception as e:
            print(f"[FAIL] 실행 자체가 안 됐습니다: {e}")
            return False

        if timeout_sec:
            def _kill_on_timeout():
                timed_out.append(True)
                try:
                    proc.kill()                    # 죽이면 stdout 이 EOF 라 아래 for 문이 빠져나온다
                except Exception:
                    pass
            killer = threading.Timer(timeout_sec, _kill_on_timeout)
            killer.daemon = True
            killer.start()

        for line in proc.stdout:
            sys.stdout.write(line)                 # _Tee 를 통과 -> 화면 + 마스터 로그
            # [낮음] --no_log 면 sys.stdout 이 _Tee 가 아니라 원래 stdout 이다. 코랩에서
            # `!python ...` 으로 부르면 그쪽도 파이프라 부모에서 다시 블록 버퍼링이 된다.
            # 자식을 -u 로 띄워 놓고 부모가 다시 모아두면 H2 를 고친 의미가 없다.
            sys.stdout.flush()
            if child_log:
                child_log.write(line)
        rc = proc.wait()

    except KeyboardInterrupt:
        # [M7] 인터럽트 시 자식을 정리하고 위로 다시 던진다. finally 에서 파일이 닫힌다.
        print("\n[중단] 사용자 인터럽트 — 실행 중인 자식 프로세스를 정리합니다.", flush=True)
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
        raise
    finally:
        if killer is not None:
            killer.cancel()
        if proc is not None and proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if child_log:
            try:
                child_log.write(f"\n# 종료 {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"· {(time.time()-t0)/60:.1f}분\n")
                child_log.close()
            except Exception:
                pass

    dt = time.time() - t0
    # 종료코드를 먼저 본다. 자식이 스스로 정상 종료한 직후(wait() 반환 뒤, finally 의
    # cancel() 전) 타이머가 발동하는 아주 짧은 창이 있는데, 그때 timed_out 을 먼저 보면
    # 성공한 단계를 "시간 초과 실패"로 적는다. 타이머가 실제로 죽였다면 rc 는 -9 다.
    if rc == 0:
        print(f"[OK] {dt/60:.1f}분", flush=True)
        return True
    if timed_out:
        print(f"[FAIL] {timeout_sec/60:.0f}분을 넘겨 이 단계를 중단했습니다 ({dt/60:.1f}분).",
              flush=True)
        return False
    print(f"[FAIL] 종료코드 {rc} ({dt/60:.1f}분)", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description="Mark5_EXP 반복실험 실행기 (5조건 x 5seed)")
    ap.add_argument("--mark_version", type=str, default="mark5.2")   # [변경 2026-09-17] 기본 mark5.2
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--unlabeled_ns", type=int, nargs="+", default=DEFAULT_UNLABELED_NS)
    # [신설 2026-09-17 / 8절] labeled 축. 0 은 "전량(2,000, 태그 없음)" 을 뜻한다(argparse 로 None 을 받을 수 없어서).
    ap.add_argument("--labeled_ns", type=int, nargs="+", default=[0],
                    help="labeled 개수 목록(200 500 1000 …). 0 = 전량(태그 없음). unlabeled_ns 와 곱해 조합을 만든다.")
    ap.add_argument("--preset", type=str, default=None, choices=["paper", "plsource"],
                    help="paper: 체크리스트 8절 회차 계획(조합 11개 x seed). plsource: pseudo-label 출처 비교(3 x seed). "
                         "--labeled_ns/--unlabeled_ns/--ceiling 를 무시한다.")
    ap.add_argument("--dry_run", action="store_true", help="무엇을 돌릴지만 출력하고 끝낸다")
    ap.add_argument("--force", action="store_true", help="이미 끝난 단계도 다시 돌린다")
    ap.add_argument("--skip_test", action="store_true",
                    help="test 평가를 건너뛴다(val 만). 운영점을 다시 고를 때만 쓸 것")
    ap.add_argument("--no_log", action="store_true", help="로그 파일을 남기지 않는다")
    ap.add_argument("--step_timeout_min", type=int, default=0,
                    help="한 단계가 이 시간을 넘기면 중단하고 다음으로 간다(0=무제한)")
    # [신설 2026-09-06] 천장 조건
    ap.add_argument("--ceiling", action="store_true",
                    help="천장 조건(unlabeled 에 진짜 라벨을 붙인 지도학습)을 seed 별로 돌린다. "
                         "--unlabeled_ns 는 무시된다.")
    ap.add_argument("--ceiling_csv", type=str, default="ceiling_labels.csv",
                    help="천장용 라벨 CSV(프로젝트 루트 기준). model/make_ceiling_labels.py 가 만든다.")
    # [신설 2026-09-25 / LR] 학습률. 안 주면 plots/lr_selected.json(select_lr.py 산출)을 읽고, 그것도 없으면 config 기본값.
    ap.add_argument("--lr", type=float, default=None, help="새 층 학습률을 직접 지정(주면 lr_selected.json 을 안 읽는다)")
    ap.add_argument("--encoder_lr", type=float, default=None, help="백본 학습률을 직접 지정(주면 lr_selected.json 을 안 읽는다)")
    ap.add_argument("--ignore_selected_lr", action="store_true", help="lr_selected.json 이 있어도 무시하고 config 기본값으로 돈다")
    args = ap.parse_args()
    sel_lr, sel_elr, lr_src = _resolve_learning_rates(args)
    _lr_args = ((["--lr", repr(float(sel_lr))] if sel_lr is not None else [])
                + (["--encoder_lr", repr(float(sel_elr))] if sel_elr is not None else []))   # repr: round-trip 보장(:g 는 6자리)

    if args.ceiling:
        _cp = args.ceiling_csv if os.path.isabs(args.ceiling_csv) else os.path.join(PROJECT_ROOT, args.ceiling_csv)
        if not os.path.isfile(_cp):
            print(f"[ERROR] 천장용 라벨 CSV 가 없습니다: {_cp}")
            print("        먼저  python model/make_ceiling_labels.py  를 돌리거나 저장소의 ceiling_labels.csv 를 확인하십시오.")
            sys.exit(1)

    # ---- 로그 파일 준비 --------------------------------------------------
    master = None
    master_path = None
    orig_out, orig_err = sys.stdout, sys.stderr    # [M6] 원래 것을 기억해 뒀다가 복구한다
    # [낮음] --dry_run 은 계획만 찍고 끝나는데 예전에는 그때마다 마스터 로그 파일이
    # 하나씩 쌓였다. 계획 출력은 남길 가치가 없고 logFiles 만 지저분해진다.
    if not args.no_log and not args.dry_run:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            master_path = os.path.join(LOG_DIR, f"run_experiments_{args.mark_version}_{stamp}.log")
            master = open(master_path, "w", encoding="utf-8", buffering=1)
            # [M6] sys.__stdout__ 이 아니라 지금의 sys.stdout 을 감싼다.
            # IPython/코랩에서 sys.__stdout__ 은 인터프리터가 처음 뜰 때의 stdout 이고
            # 셀에 보이는 것은 sys.stdout(ipykernel OutStream)이다. __stdout__ 을 감싸면
            # %run 으로 부를 때 출력이 셀에 안 보이고 파일에만 들어간다.
            sys.stdout = _Tee(orig_out, master)
            sys.stderr = _Tee(orig_err, master)
        except Exception as e:
            print(f"[WARN] 로그 파일을 못 만들었습니다({e}). 화면 출력만 남습니다.")
            master = None

    timeout_sec = args.step_timeout_min * 60 if args.step_timeout_min > 0 else None
    interrupted = False

    try:
        python = sys.executable
        # 천장 모드면 조건 하나("ceil") x seed. 아니면 기존대로 unlabeled x seed.
        # [변경 2026-09-17] 조합 원소가 (labeled_n, unlabeled_n|"ceil", seed) 세 칸이 됐다. labeled_n None = 전량.
        # 조합 원소 = (labeled_n, unlabeled_n|"ceil", seed, spill, pseudo_source)
        if args.preset == "paper":
            combos = [(l, n, s, False, None) for (l, n) in PAPER_PLAN for s in args.seeds]
            if not os.path.isfile(args.ceiling_csv if os.path.isabs(args.ceiling_csv) else os.path.join(PROJECT_ROOT, args.ceiling_csv)):
                print(f"[ERROR] preset paper 에는 상한(ceil) 회차가 있어 천장용 라벨 CSV 가 필요합니다: {args.ceiling_csv}")
                sys.exit(1)
        elif args.preset == "plsource":
            combos = [(l, n, s, sp, ps) for (l, n, sp, ps) in PLSOURCE_PLAN for s in args.seeds]
            _missing = [PSEUDO_CSV_FMT.format(ps=ps, l=l, s=s) for (l, n, sp, ps) in PLSOURCE_PLAN if ps for s in args.seeds
                        if not os.path.isfile(os.path.join(PROJECT_ROOT, PSEUDO_CSV_FMT.format(ps=ps, l=l, s=s)))]
            if _missing:
                print("[ERROR] preset plsource 에 필요한 pseudo-label CSV 가 없습니다(model/make_pseudo_labels.py 로 먼저 만드십시오):")
                for m in _missing:
                    print("        " + m)
                print("        self 는 같은 seed 의 l200_u0 student 체크포인트, gen 은 같은 seed 의 비교모델 sup l200 체크포인트가 재료입니다.")
                sys.exit(1)
        else:
            labeled_ns = [None if l == 0 else int(l) for l in args.labeled_ns]
            if args.ceiling:
                combos = [(l, "ceil", s, False, None) for l in labeled_ns for s in args.seeds]
            else:
                combos = [(l, n, s, False, None) for l in labeled_ns for n in args.unlabeled_ns for s in args.seeds]

        # ---- 계획 출력 ---------------------------------------------------
        print("=" * 78)
        print(f" Mark5_EXP2 반복실험 — {len(combos)}개 조합" + ("  [천장 조건]" if args.ceiling else "") + (f"  [preset {args.preset}]" if args.preset else ""))
        print(f" 시작    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if args.preset == "paper":
            print(f" 조건     : PAPER_PLAN {len(PAPER_PLAN)}개 (labeled 전량·200·500·1000 x u0/u2000/ceil/곡선) — 파일 머리 주석 참조")
        elif args.preset == "plsource":
            print(f" 조건     : PLSOURCE_PLAN {len(PLSOURCE_PLAN)}개 (labeled 200 + unlabeled 3,800 · 합의/self/gen)")
        elif args.ceiling:
            print(f" 조건     : 천장 — unlabeled 2,000 에 진짜 라벨({args.ceiling_csv}) · teacher 미사용 · labeled {args.labeled_ns}")
        else:
            print(f" labeled  : {args.labeled_ns} (0=전량)")
            print(f" unlabeled: {args.unlabeled_ns}")
        print(f" seeds    : {args.seeds}")
        print(" 학습률   : " + (f"새 층 {sel_lr:g} · 백본 {sel_elr:g}" if sel_lr is not None and sel_elr is not None
                             else f"새 층 {'기본' if sel_lr is None else f'{sel_lr:g}'} · 백본 {'기본' if sel_elr is None else f'{sel_elr:g}'}")
              + f"  ← {lr_src}")
        if timeout_sec:
            print(f" 단계 제한: {args.step_timeout_min}분")
        if master_path:
            print(f" 로그    : {master_path}")
        print("=" * 78)

        def _cond_txt(l, n, sp=False, ps=None):
            return (("l" + (f"{l:<4}" if l is not None else "all ")) + " " + ("ceil " if n == "ceil" else f"u{n:<5}")
                    + (" +sp" if sp else "    ") + (f" ps={ps}" if ps else ""))

        todo, done = [], []
        for l, n, s, sp, ps in combos:
            tag = _run_tag(args.mark_version, s, None if n == "ceil" else n, ceiling=(n == "ceil"), labeled_n=l, spill=sp, ps=ps,
                           lr=sel_lr, encoder_lr=sel_elr)
            need = _needs(args.mark_version, tag, args.force, args.skip_test)
            (todo if any(need) else done).append((l, n, s, sp, ps, tag, need))

        for l, n, s, sp, ps, tag, _ in done:
            print(f"  [건너뜀] {_cond_txt(l, n, sp, ps)} seed {s}   {tag}")
        for l, n, s, sp, ps, tag, need in todo:
            # [H1] 무엇을 다시 돌리는지 단계까지 보여준다.
            steps = "·".join(nm for nm, f in
                             zip(["학습", "val", "test"], need) if f)
            print(f"  [실행]   {_cond_txt(l, n, sp, ps)} seed {s}   {tag}   ({steps})")
        print(f"\n 완료 {len(done)}개 / 남은 {len(todo)}개")

        if args.dry_run:
            print("\n--dry_run 이므로 여기서 끝냅니다.")
            return

        if not todo:
            print("\n전부 끝나 있습니다. 다시 돌리려면 --force 를 주십시오.")
            return

        # ---- 실행 --------------------------------------------------------
        t_start = time.time()
        ok_list, fail_list = [], []

        def _log_path(tag, stage):
            if args.no_log:
                return None
            return os.path.join(LOG_DIR, f"{tag}__{stage}.log")

        for i, (l, n, s, sp, ps, tag, need) in enumerate(todo, 1):
            need_train, need_val, need_test = need
            print("\n" + "=" * 78)
            print(f" [{i}/{len(todo)}] labeled={'전량' if l is None else l} · " + ("천장" if n == "ceil" else f"unlabeled={n}")
                  + (" +남은 labeled" if sp else "") + (f" · pseudo={ps}" if ps else "") + f" · seed={s} · run_tag={tag}")
            elapsed = (time.time() - t_start) / 60
            print(f" 지금까지 {elapsed:.0f}분 경과 · 지금 {time.strftime('%H:%M:%S')}")
            skipped = [nm for nm, f in zip(["학습", "val", "test"], need) if not f]
            if skipped:
                print(f" 이미 끝난 단계는 건너뜁니다: {', '.join(skipped)}")
            print("=" * 78, flush=True)

            _lab = ["--labeled_n", str(l)] if l is not None else []      # [2026-09-17] labeled 축
            _sp = ["--spill_labeled"] if sp else []
            _ps = ["--pseudo_csv", PSEUDO_CSV_FMT.format(ps=ps, l=l, s=s), "--pseudo_source", ps] if ps else []
            if n == "ceil":
                # 천장: 학습은 CSV 경로를, 평가는 --ceiling 플래그를 받아 같은 태그(_ceil_s<k>)를 만든다.
                common = ["--mark_version", args.mark_version, "--seed", str(s)] + _lab + _lr_args   # [LR] train·eval 둘 다
                train_args = common + ["--ceiling_csv", args.ceiling_csv]
                eval_args = common + ["--ceiling"]
            else:
                common = ["--mark_version", args.mark_version, "--seed", str(s), "--unlabeled_n", str(n)] + _lab + _lr_args
                train_args = common + _sp + _ps
                eval_args = common + _sp + (["--pseudo_source", ps] if ps else [])
            failed_stage = None

            # [H2] 자식은 반드시 -u (무버퍼). 안 그러면 죽는 순간의 로그가 사라진다.
            if need_train:
                if not _sh([python, "-u", "model/train_mark5.py"] + train_args,
                           _log_path(tag, "train"), timeout_sec):
                    failed_stage = "train"
            if failed_stage is None and need_val:
                if not _sh([python, "-u", "model/eval.py", "--split", "val"] + eval_args,
                           _log_path(tag, "eval_val"), timeout_sec):
                    failed_stage = "eval-val"
            if failed_stage is None and need_test:
                if not _sh([python, "-u", "model/eval.py"] + eval_args,
                           _log_path(tag, "eval_test"), timeout_sec):
                    failed_stage = "eval-test"

            if failed_stage:
                fail_list.append((tag, failed_stage))
            else:
                ok_list.append(tag)

        # ---- 마무리 ------------------------------------------------------
        total = (time.time() - t_start) / 60
        print("\n" + "=" * 78)
        print(f" 끝났습니다. 총 {total:.0f}분 ({total/60:.1f}시간)")
        print(f" 성공 {len(ok_list)}개 / 실패 {len(fail_list)}개")
        if fail_list:
            print("\n 실패한 조합:")
            for tag, stage in fail_list:
                print(f"   {tag}  ({stage} 단계)")
            print(" 다시 실행하면 **끝난 단계는 건너뛰고 실패한 단계부터** 재시도합니다.")
            print(" (학습이 도중에 죽었으면 완주 표식이 없으므로 학습부터 다시 돕니다.)")
        print("\n 다음: python model/aggregate_results.py --mark_version " + args.mark_version)
        if master_path:
            print(f"\n 이 실행의 전체 로그: {master_path}")
            print(f" 단계별 로그       : {LOG_DIR}/<run_tag>__train.log 등")
        print("=" * 78)

    except KeyboardInterrupt:
        # [M7] 인터럽트로 끝나도 어디까지 했는지는 남긴다.
        interrupted = True
        print("\n" + "=" * 78)
        print(" 사용자 인터럽트로 중단했습니다.")
        print(" 다시 실행하면 끝난 단계는 건너뛰고 이어서 돕니다.")
        print(" (학습 도중에 끊었으면 완주 표식이 없으므로 그 조합은 학습부터 다시 돕니다.)")
        print("=" * 78)
    finally:
        # [M6] 어떤 경로로 빠져나가도 stdout 을 되돌리고 파일을 닫는다.
        sys.stdout, sys.stderr = orig_out, orig_err
        if master:
            try:
                master.close()
            except Exception:
                pass

    # [낮음] 중단으로 끝났으면 종료코드로도 알린다. 0 으로 끝내면 코랩 셀이 정상 완료로
    # 보이고, 뒤에 이어 붙인 셀(집계 등)이 그대로 실행된다.
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
