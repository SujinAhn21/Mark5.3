#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLAP 제로샷 추론(학습 0회) — LAION-CLAP 으로 같은 두 세트를 문장만으로 판정해 Mark5.2 제로샷 표 옆에 놓는다. (체크리스트 10절)

[신설 2026-09-17 / Mark5_EXP2]

무엇을 재나 — model/zeroshot_eval.py 와 **같은 세트·같은 문장·같은 지표·같은 산출물 형식**. 다른 것은 오디오·문장 인코더뿐이다.
    (1) 제로샷 세트 dataset_eval_zeroshot6.csv (631, 정답 = 새 6종) — 15 앵커 판정
    (2) 정규 test  dataset_test.csv (430, 정답 = 9종)          — 같은 15 앵커 판정
  앵커 문장 = 기존 9종(prompt_bank.json) + 새 6종(zeroshot_prompt_bank.json), 템플릿 4개 x 동의어 평균. Mark5.2 가 쓰는 문장을
  그대로 CLAP 텍스트 인코더(RoBERTa)에 넣는다. 논문 문장: "5M급 분류기 중 softmax 헤드는 클래스를 늘리려면 재학습이 필요한데
  우리 헤드는 문장 하나로 늘릴 수 있고, 같은 능력을 가진 CLAP 은 35배 크다."

모델 — LAION-CLAP 630k-audioset-best (ICASSP 2023, arXiv 2211.06687; HTSAT-tiny 오디오 + RoBERTa 텍스트, 약 158M 파라미터,
  비융합(enable_fusion=False)). pip 패키지 laion-clap 의 CLAP_Module.load_ckpt(model_id=1) 이 이 체크포인트를 받는다
  (hook.py 의 목록: 0 = 630k-best, 1 = 630k-audioset-best, 2 = 630k-fusion-best, 3 = 630k-audioset-fusion-best — 2026-09-17 원문 확인).
  오디오는 CLAP 이 스스로 48 kHz 로 리샘플해 읽는다(get_audio_embedding_from_filelist). 3초 클립은 CLAP 안에서 10초로 반복 패딩된다.

판정 — 코사인 유사도 x CLAP 의 학습된 logit scale(logit_scale_a) 을 softmax 한 15-way 확률.
  others 규칙은 Mark5.2 와 같은 두 후보 가운데 val(dataset_val.csv, 9 앵커)에서 Macro F1 최대를 고른다(--others_rule val_select, 기본):
    argmax(9클래스) · direct(Prob_others >= thr 이면 others, 아니면 others 를 뺀 argmax) 격자 0.02~0.95 간격 0.01,
    동률이면 격자 가운데, argmax 가 최댓값과 동률이면 argmax (vild/postprocess_utils.select_others_threshold 와 같은 규칙).
  Mark5.2 처럼 val 에서 고른 규칙을 15-way 확률 위에 그대로 적용한다. --others_rule argmax 로 고정할 수도 있다.
  CLAP 은 우리 후처리(세그먼트 집계·pair 보정)가 없다 — 클립 하나를 통째로 넣는 모델이라 그 단계가 없는 것이 원래 모습이다.

산출물 (plots/, 태그 = clap_<체크포인트>):
  zeroshot_clap_<ckpt>.json · zeroshot_predictions_clap_<ckpt>_{zeroshot6,test}.csv · zeroshot_confusion_clap_<ckpt>.csv
  + others_threshold_clap_<ckpt>.json (val 에서 고른 규칙·곡선)   ← zeroshot_eval.py 산출물과 이름 규칙이 같다.

로컬(CPU, RAM 7 GB WSL)에서 진짜 패키지·체크포인트(~/.cache/clap/630k-audioset-best.pt, HF sha256 일치)로 동작 확인함(2026-09-17). 실행 전 `pip install laion-clap`.
사용:  python model/clap_zeroshot.py                       # 630k-audioset-best, val 선택 규칙
       python model/clap_zeroshot.py --ckpt_path /content/drive/MyDrive/NewProject/hf_cache/630k-audioset-best.pt   # 받아 둔 파일
       python model/clap_zeroshot.py --limit 3            # 동작 확인(라벨별 3클립, 산출물에 _limit3)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils"), os.path.join(PROJECT_ROOT, "vild"), BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
SHARED_DIR = os.path.join(PROJECT_ROOT, "shared_vild")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from vild_config import AudioViLDConfig                                          # noqa: E402
from prompt_bank import get_prompt_texts_for_class                              # noqa: E402
from zeroshot_eval import compute_metrics, save_and_print, _resolve_csv, PLOT_DIR  # noqa: E402  같은 지표·같은 산출물 형식

# laion_clap hook.py 의 download_names 순서(2026-09-17 원문 확인). 융합(fusion) 판은 amodel 이 달라 enable_fusion=True 가 필요하다.
CKPTS = {"630k-best": (0, False), "630k-audioset-best": (1, False),
         "630k-fusion-best": (2, True), "630k-audioset-fusion-best": (3, True)}


def _load_ckpt_lowmem(model, path):
    """laion_clap hook.py load_ckpt 와 같은 처리(state_dict 꺼내기 · 'module.' 접두사 제거 · transformers>=4.31 이면
    text_branch.embeddings.position_ids 삭제 · strict 로드)를 **mmap** 으로 한다.
    이유: 1.86 GB 체크포인트에는 옵티마이저 상태까지 들어 있어 torch.load 로 통째로 올리면 RSS 7.3 GB 가 되고,
    RAM 7 GB 인 로컬 WSL 에서 OOM 으로 죽었다(2026-09-17 실측, dmesg Out of memory). mmap 은 모델 가중치 페이지만 읽는다.
    mmap 을 못 쓰는 옛 직렬화 파일이면 보통 방식으로 되돌아간다."""
    import transformers
    from packaging import version
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        how = "mmap"
    except (TypeError, RuntimeError) as e:
        print(f"[WARN] mmap 로드 실패({type(e).__name__}) — 보통 방식으로 읽습니다")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        how = "full"
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if next(iter(sd)).startswith("module"):
        sd = {k[7:]: v for k, v in sd.items()}
    if version.parse(transformers.__version__) >= version.parse("4.31.0"):
        sd.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(sd)                       # 패키지와 같이 strict — 키가 하나라도 어긋나면 여기서 멈춘다
    n_loaded = sum(1 for n, _ in model.model.named_parameters() if n in sd)
    n_total = sum(1 for _ in model.model.named_parameters())
    print(f"[INFO] 체크포인트 로드({how}): {os.path.basename(path)} · 파라미터 텐서 {n_loaded}/{n_total} 적재")
    del ck, sd


def load_clap(ckpt_id, ckpt_path, device):
    try:
        import laion_clap
    except ImportError:
        raise SystemExit("[ERROR] laion_clap 이 없습니다. `pip install laion-clap` 뒤에 다시 실행하십시오(코랩 셀 참고).")
    model_id, fusion = CKPTS[ckpt_id]
    model = laion_clap.CLAP_Module(enable_fusion=fusion, amodel="HTSAT-tiny", device=str(device))
    if not ckpt_path:
        # 패키지 load_ckpt 와 같은 자리(패키지 폴더)에 같은 이름으로 받는다. 이미 있으면 그대로 쓴다.
        import wget
        pkg_dir = os.path.dirname(os.path.realpath(laion_clap.__file__))
        name = [k for k, (i, _) in CKPTS.items() if i == model_id][0] + ".pt"
        ckpt_path = os.path.join(pkg_dir, name)
        if not os.path.exists(ckpt_path):
            print(f"[INFO] 체크포인트 다운로드: https://huggingface.co/lukewys/laion_clap/resolve/main/{name} → {pkg_dir}")
            ckpt_path = wget.download("https://huggingface.co/lukewys/laion_clap/resolve/main/" + name, pkg_dir)
            print()
    _load_ckpt_lowmem(model, ckpt_path)
    model.eval()
    return model, ckpt_path


def _mean_anchor(model, prompts):
    # ★ no_grad 필수 — laion_clap 의 get_text_embedding 은 안에서 no_grad 를 걸지 않는다. 안 걸면 호출마다 RoBERTa 역전파
    #   그래프(16문장에 약 800 MB)가 반환 텐서에 매달려 남아 15 클래스에서 7 GB 를 넘겨 OOM 으로 죽었다(2026-09-17 로컬 실측).
    with torch.no_grad():
        emb = model.get_text_embedding(prompts, use_tensor=True).detach()          # [P, D]
    emb = F.normalize(emb.float(), dim=-1).mean(dim=0)
    return F.normalize(emb, dim=0)


def build_anchors(model, config, zeroshot_bank):
    """기존 9 앵커(prompt_bank.json) + 새 6 앵커(zeroshot_prompt_bank.json) — zeroshot_eval.build_anchors 와 같은 문장, 인코더만 CLAP."""
    base_classes = list(config.get_classes_for_evaluation())
    with open(zeroshot_bank, encoding="utf-8") as f:
        new_classes = list(json.load(f)["class_synonyms"].keys())
    prompts_used, embs = {}, []
    for c in base_classes:
        prompts_used[c] = config.get_prompt_texts_for_class(c)
        embs.append(_mean_anchor(model, prompts_used[c]))
    for c in new_classes:
        prompts_used[c] = get_prompt_texts_for_class(c, zeroshot_bank)
        embs.append(_mean_anchor(model, prompts_used[c]))
    return torch.stack(embs, dim=0), base_classes, new_classes, prompts_used


def embed_audio(model, paths, batch_size):
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        with torch.no_grad():                                                      # 위와 같은 이유(오디오 쪽도 그래프를 남긴다)
            e = model.get_audio_embedding_from_filelist(x=chunk, use_tensor=True)  # CLAP 이 48 kHz 로 읽어 임베딩 [B, D]
        out.append(F.normalize(e.float().detach().cpu(), dim=-1))
        if (i + len(chunk)) % 200 < batch_size:
            print(f"  {i + len(chunk)}/{len(paths)}")
    return torch.cat(out, dim=0)


def probs_from(audio_emb, anchors, scale):
    """[N, D] x [C, D] -> softmax([N, C]) (scale = CLAP 학습 logit scale)."""
    logits = scale * (audio_emb @ anchors.cpu().T)
    return torch.softmax(logits, dim=-1).numpy()


def decide(prob, others_idx, rule, thr):
    """others 규칙 — postprocess_utils.apply_others_calibration 의 argmax / direct 와 같은 판정."""
    if rule == "argmax":
        return int(np.argmax(prob))
    if prob[others_idx] >= thr:
        return others_idx
    masked = prob.copy(); masked[others_idx] = -1.0
    return int(np.argmax(masked))


def select_rule(probs9, labels, others_idx, n_classes, grid=(0.02, 0.95, 0.01)):
    """val 에서 규칙 선택 — select_others_threshold 와 같은 후보·기준·동률 처리(우리 후처리는 안 거친다)."""
    idx = list(range(n_classes))
    def _f1(rule, thr):
        preds = [decide(p, others_idx, rule, thr) for p in probs9]
        return float(precision_recall_fscore_support(labels, preds, average="macro", labels=idx, zero_division=0)[2])
    grid = [float(t) for t in np.round(np.arange(grid[0], grid[1] + 1e-9, grid[2]), 4)]
    f1_argmax = _f1("argmax", None)
    f1s = [_f1("direct", t) for t in grid]
    curve = [("argmax", None, f1_argmax)] + [("direct", t, f) for t, f in zip(grid, f1s)]
    best = max(f1s)
    if f1_argmax >= best - 1e-12:
        return "argmax", None, f1_argmax, curve
    top = [i for i, f in enumerate(f1s) if f >= best - 1e-12]
    mid = top[len(top) // 2]
    return "direct", grid[mid], f1s[mid], curve


def _per_label(rows, n):
    seen, out = {}, []
    for r in rows:
        if seen.get(r["label"], 0) < n:
            out.append(r); seen[r["label"]] = seen.get(r["label"], 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description="LAION-CLAP 제로샷 (zeroshot6 + test, 15 앵커)")
    ap.add_argument("--ckpt_id", type=str, default="630k-audioset-best", choices=list(CKPTS))
    ap.add_argument("--ckpt_path", type=str, default=None, help="받아 둔 .pt 파일 경로(있으면 다운로드 안 함). 이름은 --ckpt_id 로 적는다")
    ap.add_argument("--mark_version", type=str, default="mark5.2", help="앵커 9종·프롬프트 뱅크를 가져올 config 버전")
    ap.add_argument("--zeroshot_bank", type=str, default=os.path.join(SHARED_DIR, "resources", "zeroshot_prompt_bank.json"))
    ap.add_argument("--others_rule", type=str, default="val_select", choices=["val_select", "argmax"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="동작 확인용: 세트마다 라벨별 앞 N 클립만(산출물 이름에 _limitN)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AudioViLDConfig(mark_version=args.mark_version)
    tag = f"clap_{args.ckpt_id}"
    suffix = f"_limit{args.limit}" if args.limit else ""
    t0 = time.time()

    model, ckpt_used = load_clap(args.ckpt_id, args.ckpt_path, device)
    n_params = sum(p.numel() for p in model.model.parameters())
    scale = float(model.model.logit_scale_a.exp().detach().cpu())
    anchors, base_classes, new_classes, prompts_used = build_anchors(model, config, args.zeroshot_bank)
    class_names = base_classes + new_classes
    others_idx = class_names.index("others")
    print(f"[INFO] CLAP {args.ckpt_id}: 파라미터 {n_params:,} · logit scale {scale:.2f} · 앵커 {len(class_names)}개 "
          f"({len(base_classes)} + 새 {len(new_classes)}: {new_classes}) · device {device}")

    zs_rows = list(csv.DictReader(open(_resolve_csv("dataset_eval_zeroshot6.csv"), newline="", encoding="utf-8-sig")))
    te_rows = list(csv.DictReader(open(_resolve_csv("dataset_test.csv"), newline="", encoding="utf-8-sig")))
    if args.limit:
        zs_rows, te_rows = _per_label(zs_rows, args.limit), _per_label(te_rows, args.limit)
    bad = sorted({r["label"] for r in zs_rows if r["label"] not in new_classes})
    if bad:
        raise SystemExit(f"[ERROR] dataset_eval_zeroshot6.csv 의 라벨 {bad} 이 zeroshot 뱅크의 클래스 {new_classes} 에 없습니다.")

    # ---- others 규칙 ----
    thr_json = os.path.join(PLOT_DIR, f"others_threshold_{tag}{suffix}.json")
    if args.others_rule == "val_select":
        va_rows = list(csv.DictReader(open(_resolve_csv("dataset_val.csv"), newline="", encoding="utf-8-sig")))
        if args.limit:
            va_rows = _per_label(va_rows, args.limit)
        va_rows = [r for r in va_rows if r["label"] in base_classes]
        print(f"[INFO] (0) val {len(va_rows)}클립으로 others 규칙 선택(9 앵커)")
        va_emb = embed_audio(model, [r["path"] for r in va_rows], args.batch_size)
        probs9 = probs_from(va_emb, anchors[:len(base_classes)], scale)
        labels9 = [base_classes.index(r["label"]) for r in va_rows]
        rule, thr, best_f1, curve = select_rule(probs9, labels9, base_classes.index("others"), len(base_classes),
                                                getattr(config, "others_threshold_grid", (0.02, 0.95, 0.01)))
        os.makedirs(PLOT_DIR, exist_ok=True)
        with open(thr_json, "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "rule": rule, "threshold": thr, "val_macro_f1": best_f1, "criterion": "macro_f1",
                       "n_val": len(va_rows), "anchors_for_selection": base_classes, "logit_scale": scale,
                       "curve": [{"rule": r_, "threshold": t_, "macro_f1": f_} for r_, t_, f_ in curve],
                       "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
        rule_desc = f"val 선택({rule}" + (f", thr {thr:g})" if rule == "direct" else ")") + f" · val Macro F1 {best_f1:.4f}"
    else:
        rule, thr, rule_desc = "argmax", None, "argmax 고정"
    print(f"[INFO] others 규칙: {rule_desc}")

    def run(rows, name):
        print(f"[INFO] {name} {len(rows)}클립")
        emb = embed_audio(model, [r["path"] for r in rows], args.batch_size)
        probs = probs_from(emb, anchors, scale)
        return [(r, decide(p, others_idx, rule, thr), p) for r, p in zip(rows, probs)]

    zs = run(zs_rows, "(1) 제로샷 세트")
    te = run(te_rows, "(2) 정규 test (15 앵커)")
    extra = {"run_tag": tag, "model": {"name": f"LAION-CLAP {args.ckpt_id}", "paper": "ICASSP 2023, arXiv 2211.06687",
                                        "audio_encoder": "HTSAT-tiny", "text_encoder": "RoBERTa", "parameters": int(n_params),
                                        "logit_scale": scale, "ckpt_path": ckpt_used},
             "zeroshot_bank": os.path.relpath(args.zeroshot_bank, PROJECT_ROOT), "prompts_used": prompts_used,
             "others_rule": rule_desc, "others_rule_json": os.path.basename(thr_json) if args.others_rule == "val_select" else None,
             "limit": args.limit}
    metrics = compute_metrics(zs, te, base_classes, new_classes, class_names)
    save_and_print(f"zeroshot_{tag}{suffix}", f"zeroshot_predictions_{tag}", suffix, f"{tag} ({n_params / 1e6:.0f}M)",
                   zs, te, metrics, base_classes, new_classes, class_names, extra, rule_desc, t0)


if __name__ == "__main__":
    main()
