#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""학습률 선정 — 새 층(--lr) x 사전학습 백본(--encoder_lr) 후보 격자를 seed 0 으로 학습해 **val Macro F1** 로 고른다.

[신설 2026-09-25 / LR]  Mark5.2 와 Mark5_EXP2 에 같은 파일을 둔다(저장소는 config 로 자동 감지).

왜 필요한가
  비교모델 4개(Baseline_*/run_experiments.py --phase sweep)는 후보 3개(1e-4·5e-5·2e-5)를 돌려 val Macro F1 이 가장 높은
  학습률을 각자 골랐다. 제안모델은 그 선정을 한 번도 안 했고, 백본 학습률 5e-5 는 EfficientAT 가 자기 구조에서 고른 값을
  빌린 것이었다(vild_config.py encoder_learning_rate 주석). 논문에서 "제안모델만 튜닝 안 했다/했다" 양쪽 시비를 다 막으려면
  **학습률 축마다 후보 3개** 라는 같은 규칙을 제안모델에도 적용해야 한다. 제안모델은 축이 둘(백본 / 새로 만드는 층)이라 3x3=9회.

규칙(비교모델과 같은 것 / 다른 것)
  같다 : 선정 seed 0 을 본실험 seed 1~5 와 분리한다 · 기준은 val Macro F1 · test 는 보지 않는다(--split test 를 부르지 않는다)
         · 학습률 축마다 후보 3개 · 후보별 val 점수를 json 에 남겨 논문 부록 표로 쓴다.
  다르다: (가) 축 개수 — 비교모델은 전체를 한 학습률로 미세조정해 1축 3회, 제안모델은 백본/새 층 2축이라 3x3 = 9회.
         (나) 선정 조건 — 비교모델은 cond=sup(사람 라벨만)에서 고른 값을 pl·ceil 에도 그대로 쓴다. 제안모델은 본실험 조건
              (Mark5.2 는 합의 라벨 포함 PL, EXP2 는 --unlabeled_n 2000)에서 고른다. 제안모델은 지식 증류가 손실의 일부라
              사람 라벨만으로 고른 학습률이 본실험 조건의 최적과 다를 수 있기 때문이다. **논문에 이 두 가지를 그대로 적는다.**
         (다) 동률 처리 — 비교모델은 후보 목록 순서, 이 스크립트는 기본값 조합 우선(산출물 재사용 목적). 동률이면 성능이 같으므로
              표의 숫자는 달라지지 않는다.
  ⚠ cudnn 결정론은 Mark5_EXP2 에만 있다(config.cudnn_deterministic=True). Mark5.2 의 utils/seed_utils.py 는 cudnn 플래그를
    건드리지 않으므로, 같은 seed 로 다시 돌려도 마지막 자리가 흔들릴 수 있다. --force 로 다시 돌릴 때 이 점을 감안한다.
  - 동률이면 config 기본값 조합(새 층 1e-4 · 백본 5e-5)을 우선한다 — 태그가 안 바뀌어 지금까지의 산출물을 그대로 쓸 수 있다.
    그다음은 후보 목록 순서.

산출물
  plots/lr_selected.json       고른 값 + 후보 9개의 val 지표 4종(Accuracy·Macro P·R·F1). run_experiments.py(EXP2)가 읽는다.
  plots/lr_sweep_<mv>.csv      같은 내용을 표로(논문 부록 "후보별 검증 점수" 표의 재료).
  logFiles/select_lr_<mv>_<시각>.log + <run_tag>__train.log / __eval_val.log

run_tag
  학습률이 기본값과 다를 때만 _lr<값>·_elr<값> 이 붙는다(vild_config.lr_tag). 그래서 기본값 조합은 회차 꼬리표로만 구분한다.
    Mark5.2   : <설정>_AC_LT[_lr..][_elr..]_lrsel_s0            (--tag_suffix _lrsel_s0 · AC/LT 는 본 모델 기본이라 언제나 붙는다)
    Mark5_EXP2: <설정>[_lr..][_elr..]_u2000_s0                   (--seed 0 --unlabeled_n 2000 — 본실험 표 1 조건과 같은 조건)
  seed 0 은 본실험(1~5)에 없으므로 어느 산출물과도 안 겹친다.

재개 규칙(단계 단위, EXP2 run_experiments.py 와 같음)
  학습 완료 = 체크포인트(model/student_model_<tag>.pth) + 완주 표식(plots/loss_history_<tag>.csv) 둘 다.
  val 완료   = performance_summary_<tag>_val.csv 에 지표 6개가 다 있고, others 규칙 json 의 run_tag·체크포인트 sha256 이 지금 것과 같음.
  끝난 단계는 건너뛴다. 세션이 끊기면 같은 명령을 다시 돌리면 된다. 학습을 새로 시작하는 회차는 옛 val 산출물을 먼저 지운다.

--extra 로 넘길 수 있는 것(태그 계산에 반영해야 하므로 화이트리스트)
  Mark5.2   : (없음 — 앵커 중심화·학습형 로짓 스케일은 2026-09-26 부터 본 모델 기본이라 인자로 켜고 끄지 않는다)
  Mark5_EXP2: --unlabeled_n N · --labeled_n N          (안 주면 --unlabeled_n 2000 을 넣는다)

예)  python model/select_lr.py --dry_run
     python model/select_lr.py                                             # 3x3 = 9회
     python model/select_lr.py --lrs 2e-4 1e-4 5e-5 --encoder_lrs 1e-4 5e-5 2e-5
     python model/select_lr.py --force                                     # 끝난 회차도 다시
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for _p in ("vild", "utils", "shared_vild"):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, _p))

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logFiles")
SELECTED_JSON = os.path.join(PLOT_DIR, "lr_selected.json")

# 후보. 백본은 비교모델이 쓴 후보 그대로, 새 층은 처음부터 학습하는 층이라 위로 한 칸 넓혔다. 둘 다 config 기본값을 품는다.
DEFAULT_LRS = [2e-4, 1e-4, 5e-5]          # 새 층(투영·branch·background[·로짓 스케일])   기본 1e-4
DEFAULT_ENCODER_LRS = [1e-4, 5e-5, 2e-5]  # 사전학습 백본(mn10)                          기본 5e-5
SWEEP_SEED = 0
METRICS4 = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
REQUIRED_METRICS = METRICS4 + ["ROC AUC (Macro, OvR)", "Others FPR"]   # 요약 CSV 가 온전한지 볼 때(EXP2 run_experiments 와 같음)


# ----------------------------------------------------------------------------------------------------------------------
# 저장소 감지·태그·산출물
# ----------------------------------------------------------------------------------------------------------------------
def _config(mark_version):
    from vild_config import AudioViLDConfig
    return AudioViLDConfig(mark_version=mark_version)


def _repo_kind(mark_version):
    """'exp2' 면 Mark5_EXP2(run_tag 에 _s<seed>·_u<n>), 'mark52' 면 Mark5.2(--tag_suffix)."""
    return "exp2" if hasattr(_config(mark_version), "experiment_seed") else "mark52"


# [2026-09-29 밤 / 묶음] 격차 분석 대응 묶음·ablation 플래그를 --extra 로 받는다(값 개수 = 뒤에 오는 값의 수). 태그는 _tag 가
# vild_config.apply_cli_ablation 에 그대로 넣어 계산하므로 자식(train/eval)의 run_tag 와 반드시 같다.
# [이식 2026-09-29 / Mark5_EXP2] 두 저장소가 같은 플래그를 받는다(EXP2 는 실험 조건 --unlabeled_n/--labeled_n 이 더 있다).
#   "exp2_runner" 는 Mark5_EXP2/model/run_experiments.py 가 쓰는 화이트리스트 — 실험 조건은 실행기가 계획표로 정하므로 플래그만 허용.
_ABLATION_FLAGS = {"--head_pretrained_hidden": 0, "--segment_input": 0, "--clip_input": 0, "--no_feature_kd": 0,
                   "--no_distill_branch": 0, "--linear_head": 0,
                   "--lp_epochs": 1, "--hard_weight": 1, "--others_from_reject": 1, "--pl_min_gap": 1, "--neg_pl": 2}
_WHITELIST = {"mark52": dict(_ABLATION_FLAGS),
              "exp2": {**_ABLATION_FLAGS, "--unlabeled_n": 1, "--labeled_n": 1},
              "exp2_runner": dict(_ABLATION_FLAGS)}
_INT_KEYS = {"--unlabeled_n", "--labeled_n", "--lp_epochs"}
_ABLATION_KW = {"--head_pretrained_hidden": "head_pretrained_hidden", "--segment_input": "segment_input", "--clip_input": "clip_input",
                "--no_feature_kd": "no_feature_kd", "--no_distill_branch": "no_distill_branch", "--linear_head": "linear_head",
                "--lp_epochs": "lp_epochs", "--hard_weight": "hard_weight", "--others_from_reject": "others_from_reject",
                "--pl_min_gap": "pl_min_gap", "--neg_pl": "neg_pl"}


def _parse_extra(extra, kind):
    """--extra 를 (자식에게 그대로 넘길 목록, 태그 계산용 dict) 로. 화이트리스트 밖 인자는 멈춘다 —
    태그에 영향을 주는 인자를 모르고 넘기면 재개 판정이 다른 파일을 본다."""
    allowed = _WHITELIST[kind]
    cfg, i, out = {}, 0, list(extra)
    while i < len(out):
        a = out[i]
        if a not in allowed:
            raise SystemExit(f"[ERROR] --extra 에 허용되지 않은 인자 {a!r}. 이 저장소에서 허용: {sorted(allowed)}")
        n = allowed[a]
        if n == 0:
            cfg[a] = True
            i += 1
        else:
            if i + n >= len(out):
                raise SystemExit(f"[ERROR] --extra 의 {a} 뒤에 값이 {n}개 있어야 합니다.")
            vals = out[i + 1:i + 1 + n]
            if n == 1:
                cfg[a] = int(vals[0]) if a in _INT_KEYS else float(vals[0])
            else:
                cfg[a] = tuple(float(v) for v in vals)
            i += 1 + n
    if kind == "exp2" and "--unlabeled_n" not in cfg:
        # 본실험 표 1(제안 조건)과 같은 조건에서 고른다. 안 주면 넣어 준다(태그 _u2000).
        cfg["--unlabeled_n"] = 2000
        out += ["--unlabeled_n", "2000"]
    return out, cfg


def _tag(mark_version, kind, cfg, lr, encoder_lr, seed):
    c = _config(mark_version)
    if kind == "mark52":
        # [확정 2026-09-26] AC/LT 는 config 기본값으로 언제나 켜져 있다(태그에도 _AC_LT 가 고정으로 붙는다).
        c.run_tag_suffix = f"_lrsel_s{seed}"
    else:
        c.experiment_seed = seed
        c.unlabeled_n = cfg.get("--unlabeled_n")
        c.labeled_n = cfg.get("--labeled_n")
    c.apply_cli_learning_rates(lr, encoder_lr)
    # [2026-09-29 밤] 묶음·ablation 플래그를 자식과 같은 함수로 적용해 태그(_PH·_LP…·_noSEG…)를 맞춘다.
    _kw = {_ABLATION_KW[k]: v for k, v in cfg.items() if k in _ABLATION_KW}
    if _kw:
        c.apply_cli_ablation(**_kw)
    return c.run_tag


def _defaults(mark_version):
    c = _config(mark_version)
    return float(c.learning_rate), (None if getattr(c, "encoder_learning_rate", None) is None else float(c.encoder_learning_rate))


def _lr_tag_of(mark_version, lr, encoder_lr):
    """이 학습률 조합이 본실험 run_tag 에 붙일 꼬리표('' = 기본값이라 안 붙음). vild_config.lr_tag 와 같은 규칙."""
    c = _config(mark_version)
    c.apply_cli_learning_rates(lr, encoder_lr)
    return c.lr_tag


def _artifacts(tag):
    return {
        "ckpt": os.path.join(PROJECT_ROOT, "model", f"student_model_{tag}.pth"),
        "loss": os.path.join(PLOT_DIR, f"loss_history_{tag}.csv"),            # 완주 표식(train_mark5.py 가 에폭 루프 뒤에만 쓴다)
        "val": os.path.join(PLOT_DIR, f"performance_summary_{tag}_val.csv"),
        "rule": os.path.join(PLOT_DIR, f"others_threshold_{tag}.json"),
    }


def _read_summary(path):
    """performance_summary CSV 의 앞 블록(지표,값)을 dict 로. 없으면 None. 숫자가 아닌 행(Others Rule 등)은 건너뛴다."""
    if not os.path.isfile(path):
        return None
    out = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip() or row[0].startswith("#"):
                    continue
                if row[0].strip() == "Class":
                    break
                if len(row) >= 2:
                    try:
                        out[row[0].strip()] = float(row[1])
                    except ValueError:
                        continue
    except Exception:
        return None
    return out


def _summary_complete(path):
    s = _read_summary(path)
    return s is not None and all(m in s for m in REQUIRED_METRICS)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rule_ok(tag, a):
    """others 규칙 json 이 **지금 체크포인트**로 고른 것인가(run_tag·sha256). eval.py --split val 이 쓰는 형식."""
    if not (os.path.isfile(a["rule"]) and os.path.isfile(a["ckpt"])):
        return False
    try:
        with open(a["rule"], encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return False
    return j.get("run_tag") == tag and (j.get("checkpoint") or {}).get("sha256") == _sha256(a["ckpt"])


def _needs(tag, force):
    a = _artifacts(tag)
    trained = os.path.exists(a["ckpt"]) and os.path.exists(a["loss"])
    need_train = force or not trained
    need_val = need_train or not _summary_complete(a["val"]) or not _rule_ok(tag, a)
    return need_train, need_val


def _clear_outputs(tag):
    """학습을 새로 시작하기 전에 그 태그의 옛 체크포인트·완주 표식·val 산출물을 지운다. 안 지우면 학습이 도중에 죽었을 때
    옛 체크포인트가 '학습됨' 으로, 옛 val 요약이 선정 근거로 들어간다."""
    a = _artifacts(tag)
    victims = [a["ckpt"], a["loss"], a["val"], a["rule"]]
    for pat in ("prediction_details_{t}_val.csv", "calibration_details_{t}_val.csv", "confusion_matrix_{t}_val.*",
                "roc_curve_{t}_val.png", "loss_curve_student_{t}.png"):   # [수정 2026-09-25] train_mark5.py 는 loss_curve_student_<tag>.png 로 저장한다
        victims += glob.glob(os.path.join(PLOT_DIR, pat.format(t=tag)))
    n = 0
    for v in victims:
        if os.path.isfile(v):
            os.remove(v)
            n += 1
    for d in glob.glob(os.path.join(PLOT_DIR, f"explanations_{tag}")):    # postprocess_utils 가 만드는 설명 그림 폴더
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    if n:
        print(f"  [INFO] {tag}: 옛 산출물 {n}개 삭제(학습부터 다시 도는 회차)")


# ----------------------------------------------------------------------------------------------------------------------
# 실행(EXP2 run_experiments.py 의 _Tee·_sh 와 같은 방식)
# ----------------------------------------------------------------------------------------------------------------------
class _Tee:
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


def _sh(cmd, log_path=None, timeout_sec=None):
    """한 단계 실행. 자식 출력을 한 줄씩 받아 화면·마스터 로그·단계 로그에 같이 남긴다. 실패하면 False."""
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    child_log = None
    if log_path:
        try:
            child_log = open(log_path, "a", encoding="utf-8", buffering=1)
            child_log.write("\n" + "=" * 70 + f"\n# {' '.join(cmd)}\n# 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"[WARN] 단계 로그 파일을 못 열었습니다({e}). 화면 출력만 남습니다.")
            child_log = None
    proc, killer, timed_out = None, None, []
    try:
        try:
            # [신설 2026-09-26] 자식(train_mark5·eval)도 스스로 <run_tag>__*.log 를 남기게 됐다. 여기서 이미 같은
            # 파일에 받아 적고 있으므로, 환경변수로 자식 쪽 기록을 끈다(같은 줄이 두 번 들어가지 않게).
            _env = dict(os.environ); _env["MARK5_CHILD_LOG_BY_PARENT"] = "1"
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True, bufsize=1, encoding="utf-8", errors="replace", env=_env)
        except Exception as e:
            print(f"[FAIL] 실행 자체가 안 됐습니다: {e}")
            return False
        if timeout_sec:
            def _kill():
                timed_out.append(True)
                try:
                    proc.kill()
                except Exception:
                    pass
            killer = threading.Timer(timeout_sec, _kill)
            killer.daemon = True
            killer.start()
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if child_log:
                child_log.write(line)
        rc = proc.wait()
    except KeyboardInterrupt:
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
                child_log.write(f"\n# 종료 {time.strftime('%Y-%m-%d %H:%M:%S')} · {(time.time() - t0) / 60:.1f}분\n")
                child_log.close()
            except Exception:
                pass
    dt = time.time() - t0
    if rc == 0:
        print(f"[OK] {dt / 60:.1f}분", flush=True)
        return True
    if timed_out:
        print(f"[FAIL] {timeout_sec / 60:.0f}분을 넘겨 이 단계를 중단했습니다 ({dt / 60:.1f}분).", flush=True)
        return False
    print(f"[FAIL] 종료코드 {rc} ({dt / 60:.1f}분)", flush=True)
    return False


# ----------------------------------------------------------------------------------------------------------------------
# 선정
# ----------------------------------------------------------------------------------------------------------------------
def _select(rows):
    """rows: [{lr, encoder_lr, is_default, order, tag, <지표>...}]. val Macro F1 최고. 동률이면 기본값 조합 → 후보 순서."""
    scored = [r for r in rows if r.get("Macro F1") is not None]
    if not scored:
        return None
    best = max(r["Macro F1"] for r in scored)
    tied = [r for r in scored if abs(r["Macro F1"] - best) <= 1e-12]
    tied.sort(key=lambda r: (0 if r["is_default"] else 1, r["order"]))
    return tied[0]


def _fmt(x):
    return "-" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser(description="학습률 선정(새 층 x 백본 격자, seed 0, val Macro F1)")
    ap.add_argument("--mark_version", type=str, default="mark5.2")
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS, help="새 층 학습률 후보")
    ap.add_argument("--encoder_lrs", type=float, nargs="+", default=DEFAULT_ENCODER_LRS, help="백본 학습률 후보")
    ap.add_argument("--sweep_seed", type=int, default=SWEEP_SEED, help="선정용 seed(본실험 1~5 와 겹치지 않게 0)")
    ap.add_argument("--extra", type=str, default="", help='train·eval 에 그대로 넘길 인자(화이트리스트). Mark5_EXP2 예: "--unlabeled_n 2000"')
    ap.add_argument("--force", action="store_true", help="끝난 회차도 다시 돈다")
    ap.add_argument("--dry_run", action="store_true", help="계획만 출력")
    ap.add_argument("--step_timeout_min", type=float, default=None, help="한 단계 시간 제한(분)")
    ap.add_argument("--no_log", action="store_true")
    args = ap.parse_args()

    mv = args.mark_version
    kind = _repo_kind(mv)
    extra, cfg = _parse_extra(shlex.split(args.extra) if args.extra else [], kind)
    d_lr, d_elr = _defaults(mv)
    if d_elr is None:
        raise SystemExit(f"[ERROR] mark_version={mv} 는 백본 학습률 분리(encoder_learning_rate)가 없는 구성입니다. "
                         "이 선정은 mn10 백본(mark5.2)용입니다.")
    if args.sweep_seed in (1, 2, 3, 4, 5):
        print(f"[WARN] sweep_seed {args.sweep_seed} 는 본실험 seed 와 같습니다 — 선정 회차가 본실험에 섞이지 않게 0 을 권합니다.")
    timeout = args.step_timeout_min * 60 if args.step_timeout_min else None

    # 조합: 기본값 조합을 맨 앞에(지금까지의 산출물과 태그가 같아 재사용될 수 있다), 나머지는 격자 순서.
    grid = [(lr, elr) for elr in args.encoder_lrs for lr in args.lrs]
    grid.sort(key=lambda p: 0 if (p[0] == d_lr and p[1] == d_elr) else 1)
    if (d_lr, d_elr) not in grid:
        print(f"[WARN] config 기본값(새 층 {d_lr:g} · 백본 {d_elr:g})이 후보 격자에 없습니다. "
              "지금 쓰는 값이 후보에 포함돼야 '탐색했고 이 값이 최선' 이라고 쓸 수 있습니다.")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    master, master_path = None, None
    orig_out, orig_err = sys.stdout, sys.stderr
    if not args.no_log and not args.dry_run:
        master_path = os.path.join(LOG_DIR, f"select_lr_{mv}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        master = open(master_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(orig_out, master)
        sys.stderr = _Tee(orig_err, master)

    fails = []
    try:
        repo_name = os.path.basename(PROJECT_ROOT)
        print("=" * 78)
        print(f" 학습률 선정 — {repo_name} ({'Mark5_EXP2 방식' if kind == 'exp2' else 'Mark5.2 방식'}) · mark_version {mv} · {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" 새 층 후보 {[f'{x:g}' for x in args.lrs]} x 백본 후보 {[f'{x:g}' for x in args.encoder_lrs]} = {len(grid)}회 · seed {args.sweep_seed}")
        print(f" 기본값   : 새 층 {d_lr:g} · 백본 {d_elr:g} (태그가 안 붙는 조합)")
        print(f" 조건     : {' '.join(extra) if extra else '(config 기본)'}")
        print(" 기준     : val Macro F1 (test 는 보지 않음)")
        if master_path:
            print(f" 로그     : {master_path}")
        print("=" * 78)

        plan = []
        for order, (lr, elr) in enumerate(grid):
            tag = _tag(mv, kind, cfg, lr, elr, args.sweep_seed)
            need_train, need_val = _needs(tag, args.force)
            plan.append({"order": order, "lr": lr, "encoder_lr": elr, "is_default": (lr == d_lr and elr == d_elr),
                         "tag": tag, "need": (need_train, need_val)})
        for p in plan:
            steps = "·".join(nm for nm, f in zip(["학습", "val"], p["need"]) if f)
            print(f"  [{'실행' if steps else '건너뜀'}]   새 층 {p['lr']:<8g} 백본 {p['encoder_lr']:<8g} {'(기본)' if p['is_default'] else '      '}  {p['tag']}"
                  + (f"   ({steps})" if steps else ""))
        print(f"\n 남은 회차 {sum(1 for p in plan if any(p['need']))}개 / {len(plan)}개")
        if args.dry_run:
            print("\n--dry_run 이므로 여기서 끝냅니다.")
            return

        python = sys.executable
        t_start = time.time()
        for i, p in enumerate(plan, 1):
            need_train, need_val = p["need"]
            tag, lr, elr = p["tag"], p["lr"], p["encoder_lr"]
            if not (need_train or need_val):
                continue
            print("\n" + "=" * 78)
            print(f" [{i}/{len(plan)}] 새 층 {lr:g} · 백본 {elr:g} · run_tag {tag} · 경과 {(time.time() - t_start) / 60:.0f}분")
            print("=" * 78, flush=True)
            # 학습률은 repr 로 넘긴다(:g 는 6자리라 round-trip 이 안 될 수 있다). 기본값 조합도 명시적으로 넘긴다(태그는 값으로 비교).
            lr_args = ["--lr", repr(float(lr)), "--encoder_lr", repr(float(elr))]
            if kind == "mark52":
                train_args = ["--mark_version", mv, "--seed", str(args.sweep_seed), "--tag_suffix", f"_lrsel_s{args.sweep_seed}"] + lr_args + extra
                eval_args = ["--mark_version", mv, "--tag_suffix", f"_lrsel_s{args.sweep_seed}"] + lr_args + extra
            else:
                train_args = ["--mark_version", mv, "--seed", str(args.sweep_seed)] + lr_args + extra
                eval_args = ["--mark_version", mv, "--seed", str(args.sweep_seed)] + lr_args + extra
            log = (lambda st: None if args.no_log else os.path.join(LOG_DIR, f"{tag}__{st}.log"))
            ok = True
            if need_train:
                _clear_outputs(tag)
                ok = _sh([python, "-u", "model/train_mark5.py"] + train_args, log("train"), timeout)
                if ok:
                    a = _artifacts(tag)
                    ok = os.path.exists(a["ckpt"]) and os.path.exists(a["loss"])
                    if not ok:
                        print("[FAIL] 학습 프로세스는 끝났지만 체크포인트·완주 표식이 없습니다.")
            if ok and need_val:
                ok = _sh([python, "-u", "model/eval.py", "--split", "val"] + eval_args, log("eval_val"), timeout)
            if not ok:
                fails.append(tag)

        # ---- 선정 --------------------------------------------------------------------------------------------------
        rows = []
        for p in plan:
            s = _read_summary(_artifacts(p["tag"])["val"]) if p["tag"] not in fails else None
            if s is None or not all(m in s for m in METRICS4) or not _rule_ok(p["tag"], _artifacts(p["tag"])):
                rows.append({**p, **{m: None for m in METRICS4}})
                continue
            rows.append({**p, **{m: s[m] for m in METRICS4},
                         "ROC AUC (Macro, OvR)": s.get("ROC AUC (Macro, OvR)"), "Others FPR": s.get("Others FPR")})
        missing = [r["tag"] for r in rows if r["Macro F1"] is None]
        print("\n" + "=" * 78)
        print(f" 후보별 val 결과 (seed {args.sweep_seed})")
        print("=" * 78)
        print(f" {'새 층':>8} {'백본':>8} {'':6} {'Accuracy':>9} {'Macro P':>9} {'Macro R':>9} {'Macro F1':>9}   run_tag")
        for r in rows:
            print(f" {r['lr']:>8g} {r['encoder_lr']:>8g} {'(기본)' if r['is_default'] else '':6} "
                  f"{_fmt(r['Accuracy']):>9} {_fmt(r['Macro Precision']):>9} {_fmt(r['Macro Recall']):>9} {_fmt(r['Macro F1']):>9}   {r['tag']}")
        if missing:
            print(f"\n[WARN] val 결과가 없는 후보 {len(missing)}개: {missing}\n       선정을 보류합니다. 같은 명령을 다시 돌리면 끝난 회차는 건너뛰고 이어서 돕니다.")
            if fails:
                print(f"[WARN] 실패한 회차 {len(fails)}개: {fails}")
            sys.exit(1)      # 코랩 셀이 '정상 완료' 로 보이지 않게(뒤에 이어 붙인 셀이 그대로 돌지 않게)
        best = _select(rows)
        payload = {
            "model": mv, "repo": repo_name,
            "lr": best["lr"], "encoder_lr": best["encoder_lr"],
            "default": {"lr": d_lr, "encoder_lr": d_elr}, "selected_is_default": bool(best["is_default"]),
            "criterion": "val Macro F1", "sweep_seed": args.sweep_seed, "sweep_condition": " ".join(extra),
            "candidates_lrs": [float(x) for x in args.lrs], "candidates_encoder_lrs": [float(x) for x in args.encoder_lrs],
            "candidates": {f"lr={r['lr']:g},elr={r['encoder_lr']:g}": {"tag": r["tag"], "is_default": r["is_default"],
                                                                        **{m: r[m] for m in METRICS4}} for r in rows},
            "candidates_val_macro_f1": {f"lr={r['lr']:g},elr={r['encoder_lr']:g}": r["Macro F1"] for r in rows},
            "selected_tag": best["tag"], "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # [2026-09-25] 임시 파일에 다 쓴 뒤 바꿔치기한다. 바로 쓰다가 세션이 끊기면 잘린 json 이 남고,
        # run_experiments.py 가 그것을 읽어 죽는다(코랩·드라이브에서 실제로 일어날 수 있는 상황).
        _tmp = SELECTED_JSON + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(_tmp, SELECTED_JSON)
        sweep_csv = os.path.join(PLOT_DIR, f"lr_sweep_{mv}.csv")
        with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["lr", "encoder_lr", "is_default", "selected", "run_tag"] + METRICS4 + ["ROC AUC (Macro, OvR)", "Others FPR"])
            for r in rows:
                w.writerow([f"{r['lr']:g}", f"{r['encoder_lr']:g}", int(r["is_default"]), int(r is best), r["tag"]]
                           + [r[m] for m in METRICS4] + [r.get("ROC AUC (Macro, OvR)"), r.get("Others FPR")])
        print(f"\n[INFO] 선정: 새 층 {best['lr']:g} · 백본 {best['encoder_lr']:g} (val Macro F1 {best['Macro F1']:.4f})"
              + ("  = config 기본값 → 본실험 태그·산출물 그대로" if best["is_default"] else
                 f"  ≠ 기본값 → 본실험 run_tag 에 '{_lr_tag_of(mv, best['lr'], best['encoder_lr'])}' 가 붙는다 — 본실험(seed 1~5)을 다시 돌려야 한다"))
        print(f"       -> {SELECTED_JSON}\n       -> {sweep_csv}")
        if kind == "exp2":
            print("       run_experiments.py 는 이 json 을 자동으로 읽어 --lr/--encoder_lr 로 넘긴다(--ignore_selected_lr 로 끌 수 있다).")
        else:
            print("       본실험(seed 1~5)은 train_mark5.py / eval.py 에 --lr·--encoder_lr 로 이 값을 같이 주면 된다.")
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        if master:
            try:
                master.close()
            except Exception:
                pass
    if fails:
        print(f"[WARN] 실패한 회차 {len(fails)}개: {fails}")
        sys.exit(1)


if __name__ == "__main__":
    main()
