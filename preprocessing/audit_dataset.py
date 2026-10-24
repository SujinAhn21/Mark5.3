"""
audit_dataset.py — 학습 전에 데이터 **구성**을 점검하는 게이트.

validate_dataset.py 가 파일 하나하나의 품질(길이·rms·peak·클리핑)을 본다면, 이 스크립트는
**데이터셋 전체의 짜임새**를 본다. 파일은 전부 멀쩡한데 구성이 잘못돼서 학습이 헛도는 경우를
잡는 것이 목적이다.

■ 왜 만들었나 (2026-08-09)

mark4.5·mark5.0 을 만들면서 같은 성격의 사고가 세 번 났고, 셋 다 **파일 품질은 전부 정상**이었다.
그리고 셋 다 데이터만 보면 학습 전에 알 수 있었는데, 두 번은 수집 직후에 눈으로 발견했고
한 번은 **학습을 끝까지 돌리고 test 까지 본 뒤에야** 알았다.

  (1) test 세션 하나가 타겟 0 / others 55 가 됨 — 그 세션이 곧 클래스가 됨
      원인: 조합별 쿼터가 세션을 고려하지 않는데 타겟은 조합이 하나뿐이라 세션 배분이 안 일어남
  (2) heavy_impact 319개가 어른발걸음 171 / 아이들발걸음 74 / 망치질 74 로 쏠림(균등이면 106씩)
      원인: 이미 쓴 클립을 거르기 전에 카테고리 쿼터를 계산해 예비 목록의 앞 카테고리만 소진됨
  (3) ★train 타겟 250개 중 222개(89%)가 한 세션에서 나옴
      결과: CHiME 쪽 성능이 val 0.9720 -> test 0.7963 (17.6%p 하락), 재학습 필요
      원인: 세션 배정을 val·test 의 클래스 균형만 보고 정하고 train 의 세션 수는 안 봄

(3) 을 학습 전에 잡았다면 재학습 한 번을 아꼈다. 그래서 이 셋을 포함해 지금까지 겪은 구성
결함을 전부 항목으로 만들어 둔다. **수집 -> validate_dataset -> audit_dataset 순으로 돌리고,
audit 이 FAIL 없이 통과해야 학습으로 넘어간다.**

■ 점검 항목

  ① 출처 × 클래스        출처가 클래스를 알려주는가(출처만으로 맞힐 수 있는 비율)
  ② 세션 겹침            train/val/test 가 같은 세션을 공유하는가
  ③ val·test 세션 균형   한 세션이 한 클래스로 쏠렸는가(쏠리면 그 split 에서 지름길을 검출 못 함)
  ④ train 세션 다양성    클래스마다 몇 개 세션에서 왔는가, 최대 세션이 몇 %인가   <- (3) 을 잡는 항목
  ⑤ 카테고리 균등도      클래스 안에서 한 카테고리가 과대 대표되는가              <- (2) 를 잡는 항목
  ⑥ 증강본 격리          val·test 에 증강본이 섞이지 않았는가
  ⑦ 정합성               디스크 wav 수와 provenance active 행 수가 맞는가
  ⑧ val·test 출처 대칭   test 에 있는 (출처×하위라벨)이 train 과 val 에도 있는가  <- [추가 2026-09-21]
                         (09-15 재구성에서 CHiME-Home 'b' 가 test 에만 들어가 construction 전문가의 val 운영점이
                          test 에서 안 맞았다. test 에만 있는 출처는 FAIL, val 에만 있고 test 에 없는 출처는 WARN)
  ⑨ 저장소 교차 녹음 겹침  같은 녹음이 "학습쪽"과 "평가쪽"에 동시에 있는가 — 저장소를 가로질러 본다  <- [추가 2026-09-21 밤]
                         (전문가 train 타겟 276클립이 Mark5.2 val·test 와 같은 AI Hub 녹음이었다. 09-15 규칙 "Mark5.2 val·test
                          원본은 전문가 val·test 에서 제외" 가 others 풀에만 걸려 있었고 전문가 타겟(08-02 분할)은 안 봤다.
                          teacher 가 학생의 test 녹음을 보고 배운 셈이다. ② 는 세션이 있는 출처만, 한 저장소 안에서만 봤다.)
                         녹음 키: AI Hub·ESC-50 = 원본 파일, FSD50K = 파일 id, SINS = ex 세션, CHiME = 녹음 volume.
                         증강본은 aug_source_file 을 따라 원본의 키를 받는다(증강본이 평가쪽에 있으면 ⑥ 이 잡는다).
                           R1 전문가 train ∩ Mark5.2 val·test·data_eval = 0            어기면 FAIL
                           R2 Mark5.2 train·unlabeled ∩ Mark5.2 val·test·data_eval = 0, val ∩ test = 0   FAIL
                           R3 전문가 자기 train ∩ val·test = 0, val ∩ test = 0 (녹음 단위)   FAIL
                           R4 전문가 val·test ∩ Mark5.2 train·unlabeled              허용 — 개수만 보고
                           R5 전문가 train ∩ Mark5.2 train·unlabeled = 0               FAIL  <- [추가 2026-10-23]
                              (교사가 이미 들은 녹음에 가짜 정답을 붙이면 그 가짜 정답이 실제보다 좋아 보인다)
                           R6 전문가 val·test ∩ Mark5.2 val·test·data_eval = 0         FAIL  <- [추가 2026-10-23]
                              (교사를 고르는 val 이 학생 평가 녹음을 보면 안 된다)
                         mark4.k 로 돌리면 그 전문가의 R1·R3~R6, mark5.2 로 돌리면 R2 와 전문가 8개의 R1·R3·R5·R6 요약.

■ 세션이란

  sins        aihub_src_file 의 ex번호(DevNode1_ex227_5.wav -> ex227). 한 세션 = 한 활동의 연속 녹음
  chime_home  aihub_volume(200110_1711). 한 세션 = 한 번의 연속 녹음(TV 를 켰다 껐다 함)
  aihub       세션 개념이 없다(클립이 독립). 세션 항목에서는 제외하고 ⑤ 카테고리로만 본다

■ 사용 예

  python preprocessing/audit_dataset.py --mark_version mark4.5
  python preprocessing/audit_dataset.py --mark_version mark5.0 \
      --data_root . --splits data_source_labeled,data_source_val,data_source_test
"""
import os
import re
import sys
import argparse
from collections import defaultdict, Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 폴더명 -> provenance 의 assigned_split 값(mark5 는 폴더명이 다르다)
DIR2SPLIT = {"data_source_labeled": "train", "data_source_val": "val",
             "data_source_test": "test", "data_source_unlabeled": "unlabeled",
             "train": "train", "val": "val", "test": "test"}

# 임계값 — 넘으면 WARN, 크게 넘으면 FAIL
TH = {
    "source_gap": 0.10,        # ① 두 클래스의 출처 비율 차이
    "sess_balance_warn": 0.15,  # ③ val·test 세션의 클래스 비율이 이 밖이면 WARN
    "sess_share_warn": 0.70,   # ④ train 에서 한 세션이 차지하는 비중
    "sess_share_fail": 0.85,
    "cat_ratio_warn": 3.0,     # ⑤ 클래스 안 카테고리 최대/최소 비
}

results = []


def report(level, item, msg):
    results.append((level, item, msg))
    mark = {"OK": "  ", "WARN": "⚠ ", "FAIL": "✗ "}[level]
    print(f"{mark}[{level:4s}] {item} — {msg}")


def rec_key(row):
    """[추가 2026-09-21] ⑨ 용 녹음 식별자. 같은 녹음에서 잘라낸 클립은 어느 저장소·split 에 있든 같은 키를 받는다."""
    src = str(row.get("source", ""))
    if src in ("aihub", "esc50"):
        f = row.get("aihub_src_file")
        return ("A", str(f)) if pd.notna(f) else None
    if src == "fsd50k":
        f = row.get("fsd50k_fname")
        return ("F", str(int(float(f)))) if pd.notna(f) else None      # [10-23] 236196.0 → 236196 (Clotho 의 Freesound ID 와 맞춘다)
    if src == "clotho":                                                  # [10-23] 외부 출처 — aihub_src_file 에 원본 ID
        f = row.get("aihub_src_file")
        return ("F", str(int(float(f)))) if pd.notna(f) else None      # Clotho 는 Freesound 클립 → FSD50K 와 같은 이름공간
    if src in ("bbc", "bedavin", "snu_b36_50e", "voices", "chime6"):  # voices: rm4 tele 한 녹음(마이크 10개 동시) → 키 하나
        f = row.get("aihub_src_file")                                    # [10-24] chime6: aihub_src_file 에 세션(S02 등) — 배열이 달라도 한 녹음
        return ("X", f"{src}:{f}") if pd.notna(f) else None
    if src == "sins_eval":                                               # [10-24] SINS 평가 세트(DCASE2018 T5 eval) — 개발 세트와 세션 번호 체계가
        m = re.search(r"ex(\d+)", str(row.get("aihub_src_file", "")))   # 달라 키를 따로 둔다(EvalNode*_ex<세션>_<구간>.wav)
        return ("S", "sins_eval:ex" + m.group(1)) if m else None
    if src == "sins":
        m = re.search(r"ex(\d+)", str(row.get("aihub_src_file", "")))
        return ("S", "sins:ex" + m.group(1)) if m else None
    if src == "chime_home":
        v = row.get("aihub_volume")
        return ("S", "chime:" + str(v)) if pd.notna(v) else None
    return None


def sess_of(row):
    """세션 식별자. AI Hub 는 세션 개념이 없어 None."""
    src = str(row.get("source", ""))
    if src == "sins":
        m = re.search(r"ex(\d+)", str(row.get("aihub_src_file", "")))
        return "ex" + m.group(1) if m else None
    if src == "chime_home":
        v = row.get("aihub_volume")
        return str(v) if pd.notna(v) else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark_version", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    args = ap.parse_args()

    prov = pd.read_excel(os.path.abspath(args.provenance_path))
    m = prov[prov["mark_version"] == args.mark_version]
    if "removed_20260715" in m.columns:
        m = m[m["removed_20260715"] == "active"]
    if m.empty:
        print(f"[ERROR] provenance 에 {args.mark_version} active 행이 없습니다.")
        sys.exit(1)
    # [추가 2026-08-09] unlabeled 는 라벨이 없어 ①③④⑤ 검사 대상이 아니다.
    # (target_class 에 'unlabeled:xxx' 로 뽑아온 영역만 추적용으로 기록돼 있다)
    # 그대로 두면 같은 세션에 여러 '클래스'가 있는 것처럼 보여 SINS 가 mixable 로 오판되고,
    # ③ 에서 세션마다 FAIL 이 쏟아진다. 그래서 labeled(train/val/test)만 떼어 검사한다.
    un = m[m["assigned_split"] == "unlabeled"]
    m = m[m["assigned_split"] != "unlabeled"]
    orig = m[m["source_type"] == "original"] if "source_type" in m.columns else m
    aug = m[m["source_type"] != "original"] if "source_type" in m.columns else m.iloc[0:0]
    classes = sorted(m["target_class"].dropna().unique())
    print(f"\n{'='*78}\n[감사] {args.mark_version} — labeled active {len(m)}행 "
          f"(원본 {len(orig)} / 증강 {len(aug)}), 클래스 {len(classes)}개"
          + (f" | unlabeled {len(un)}행" if len(un) else "")
          + f"\n{'='*78}\n")
    if len(un):
        src = dict(un["source"].value_counts()) if "source" in un.columns else {}
        report("OK", "⓪unlabeled",
               f"{len(un)}개 (라벨 없음 — ①③④⑤ 검사 제외). 출처 {src}")

    # ---------- ① 출처 × 클래스 ----------
    print("① 출처가 클래스를 알려주는가")
    if "source" in m.columns:
        for sp in ("train", "val", "test"):
            s = m[m["assigned_split"] == sp]
            if s.empty:
                continue
            tab = s.groupby(["source", "target_class"]).size().unstack(fill_value=0)
            for src in tab.index:
                row = tab.loc[src]
                tot = row.sum()
                top_cls = row.idxmax()
                share = row.max() / tot
                base = 1.0 / len(classes)
                lvl = "OK" if share <= base * 1.5 else ("WARN" if share < 0.9 else "FAIL")
                report(lvl, f"①{sp}/{src}",
                       f"출처만 알 때 최빈 클래스는 {top_cls} {share*100:.1f}% "
                       f"(균등이면 {base*100:.1f}%, n={tot})")
    else:
        report("OK", "①", "source 컬럼 없음 — 단일 출처로 간주")

    # ---------- ② 세션 겹침 ----------
    print("\n② 세션 겹침 (train/val/test 가 같은 세션을 공유하는가)")
    sess_map = defaultdict(lambda: defaultdict(set))
    for _, r in orig.iterrows():
        s = sess_of(r)
        if s:
            sess_map[str(r["source"])][r["assigned_split"]].add(s)
    if not sess_map:
        report("OK", "②", "세션 개념이 있는 출처 없음(AI Hub 단일)")
    for src, d in sorted(sess_map.items()):
        bad = []
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = d.get(a, set()) & d.get(b, set())
            if ov:
                bad.append(f"{a}∩{b}={sorted(ov)}")
        report("FAIL" if bad else "OK", f"②{src}",
               "; ".join(bad) if bad else
               f"겹침 없음 (train {len(d.get('train',()))} / val {len(d.get('val',()))} / "
               f"test {len(d.get('test',()))} 세션)")

    # ---------- ③ val·test 세션 균형 ----------
    # 출처마다 '한 세션에 여러 클래스가 있을 수 있는가'가 다르다. SINS 는 한 세션이 한 활동이라
    # 세션마다 한 클래스만 나오는 것이 **구조적으로 정상**이고 고칠 수 없다. CHiME-Home 은 같은
    # 세션 안에서 TV 가 켜졌다 꺼졌다 하므로 두 클래스가 함께 나올 수 있다. 그래서 출처별로
    # 데이터에서 직접 판정한 뒤, 섞일 수 있는 출처에만 이 검사를 적용한다.
    print("\n③ val·test 세션이 한 클래스로 쏠렸는가")
    mixable = set()
    all_sess = defaultdict(Counter)
    for _, r in orig.iterrows():
        s = sess_of(r)
        if s:
            all_sess[(str(r["source"]), s)][r["target_class"]] += 1
    for (src, _), c in all_sess.items():
        if len(c) > 1:
            mixable.add(src)
    skipped = {src for src, _ in all_sess} - mixable
    for src in sorted(skipped):
        report("OK", f"③{src}",
               "한 세션에 한 클래스만 있는 출처 — 구조적 특성이라 검사 제외"
               " (검출은 섞일 수 있는 출처에서 한다)")
    any3 = False
    for sp in ("val", "test"):
        g = defaultdict(Counter)
        for _, r in orig[orig["assigned_split"] == sp].iterrows():
            s = sess_of(r)
            if s and str(r["source"]) in mixable:
                g[(str(r["source"]), s)][r["target_class"]] += 1
        for k, c in sorted(g.items()):
            any3 = True
            tot = sum(c.values())
            share = max(c.values()) / tot
            hi = 1 - TH["sess_balance_warn"]
            if len(c) == 1:
                lvl = "FAIL"
            elif share > hi:
                lvl = "WARN"
            else:
                lvl = "OK"
            report(lvl, f"③{sp}/{k[0]}/{k[1]}",
                   f"n={tot} " + ", ".join(f"{a} {b}" for a, b in c.most_common())
                   + f"  (최빈 {share*100:.1f}%)")
    if not any3 and not skipped:
        report("OK", "③", "세션 개념이 있는 출처 없음")

    # ---------- ④ train 세션 다양성 ----------
    print("\n④ train 세션 다양성 (한 세션에 몰려 있지 않은가)")
    any4 = False
    tr = orig[orig["assigned_split"] == "train"]
    for src in sorted(set(str(x) for x in tr.get("source", pd.Series(dtype=str)).dropna())):
        for cls in classes:
            sub = tr[(tr["source"] == src) & (tr["target_class"] == cls)]
            if sub.empty:
                continue
            c = Counter(s for s in (sess_of(r) for _, r in sub.iterrows()) if s)
            if not c:
                continue
            any4 = True
            tot = sum(c.values())
            share = max(c.values()) / tot
            lvl = ("FAIL" if share > TH["sess_share_fail"]
                   else "WARN" if share > TH["sess_share_warn"] else "OK")
            report(lvl, f"④{src}/{cls}",
                   f"세션 {len(c)}개, n={tot}, 최대 {c.most_common(1)[0][0]} "
                   f"{max(c.values())}개({share*100:.1f}%)  {dict(c.most_common())}")
    if not any4:
        report("OK", "④", "세션 개념이 있는 출처 없음")

    # ---------- ⑤ 카테고리 균등도 ----------
    print("\n⑤ 클래스 안 카테고리 균등도")
    if "aihub_category" in orig.columns:
        for cls in classes:
            sub = orig[orig["target_class"] == cls]
            c = Counter(str(x) for x in sub["aihub_category"].dropna())
            if len(c) < 2:
                continue
            ratio = max(c.values()) / max(1, min(c.values()))
            top = ", ".join(f"{a.split('_')[-1][:14]} {b}" for a, b in c.most_common(3))
            if cls == "others":
                # others 는 '타겟이 아닌 모든 소리'라 여러 출처·카테고리를 일부러 섞는다.
                # 균등을 요구하면 설계 의도와 어긋나므로 수치만 남기고 판정하지 않는다.
                report("OK", f"⑤{cls}",
                       f"카테고리 {len(c)}종 (다양성이 목적이라 균등 검사 제외)  [{top}]")
                continue
            lvl = "WARN" if ratio > TH["cat_ratio_warn"] else "OK"
            report(lvl, f"⑤{cls}",
                   f"카테고리 {len(c)}종, 최대/최소 {ratio:.1f}배  [{top}]")
    else:
        report("OK", "⑤", "aihub_category 컬럼 없음")

    # ---------- ⑥ 증강본 격리 ----------
    print("\n⑥ 증강본이 val·test 에 섞였는가")
    if len(aug):
        bad = aug[aug["assigned_split"].isin(["val", "test"])]
        report("FAIL" if len(bad) else "OK", "⑥",
               f"증강 {len(aug)}개 중 val·test 에 {len(bad)}개"
               + (f" — {sorted(bad['local_filename'].head(5))}" if len(bad) else ""))
    else:
        report("OK", "⑥", "증강본 없음")

    # ---------- ⑦ 정합성 ----------
    print("\n⑦ 디스크 vs provenance")
    root = os.path.abspath(args.data_root)
    disk = 0
    per_split = Counter()
    for sp in args.splits.split(","):
        d = os.path.join(root, sp.strip())
        if not os.path.isdir(d):
            continue
        for dirpath, _, fns in os.walk(d):
            n = sum(1 for f in fns if f.lower().endswith(".wav"))
            disk += n
            per_split[DIR2SPLIT.get(sp.strip(), sp.strip())] += n
    prov_per = Counter(list(m["assigned_split"]) + list(un["assigned_split"]))
    ok = disk == len(m) + len(un) and all(per_split[k] == prov_per.get(k, 0) for k in per_split)
    report("OK" if ok else "FAIL", "⑦",
           f"디스크 {disk} / provenance active {len(m) + len(un)}  "
           + ", ".join(f"{k} {per_split[k]}vs{prov_per.get(k,0)}" for k in sorted(per_split)))

    # ---------- ⑧ val·test 출처 대칭 [추가 2026-09-21] ----------
    # 하위라벨 = aihub_category (CHiME 는 b/o 같은 라벨 집합, SINS 는 활동, ESC-50 은 카테고리, AI Hub 는 세부 카테고리).
    # 클래스마다 test 의 (출처, 하위라벨) 이 train 에도 있어야 하고(학습에서 본 소리), val 에도 있어야 한다(운영점을 고른 소리).
    # val 에만 있고 test 에 없는 것은 덜 해로우므로 WARN. unlabeled 는 라벨이 없어 제외(위에서 이미 뗐다).
    print("\n⑧ val·test 출처 대칭 (test 에 있는 출처×하위라벨이 train·val 에도 있는가)")
    if "source" in orig.columns and "aihub_category" in orig.columns:
        def sub_key(r):
            c = r.get("aihub_category")
            c = str(c) if pd.notna(c) else ""
            src = str(r.get("source", ""))
            if src == "chime_home":
                # CHiME 라벨 집합(예: cfm·bc·ov)은 화자 조합까지 갈라져 너무 잘다. 배경 종류(b 광대역·o 생활음·v TV)만 남기고
                # 말소리 글자(c·f·m)는 s 하나로 합친다: cfm→s, cf→s, bc→bs, ov→ov, co→os. b 의 유무는 그대로 구분된다.
                c = "".join(sorted(set(c) & set("bov"))) + ("s" if set(c) & set("cfm") else "")
            return (src, c)
        per = {sp: defaultdict(Counter) for sp in ("train", "val", "test")}
        for _, r in orig.iterrows():
            sp = r["assigned_split"]
            if sp in per:
                per[sp][r["target_class"]][sub_key(r)] += 1
        def fmt(items):
            return ", ".join(f"{k[0]}|{k[1][-14:]} {n}" for k, n in items[:6]) + (f" 외 {len(items)-6}" if len(items) > 6 else "")
        for cls in classes:
            tr, va, te = per["train"][cls], per["val"][cls], per["test"][cls]
            if not te and not va:
                continue
            miss_train = sorted(((k, n) for k, n in te.items() if k not in tr), key=lambda x: -x[1])
            miss_val = sorted(((k, n) for k, n in te.items() if k not in va), key=lambda x: -x[1])
            val_only = sorted(((k, n) for k, n in va.items() if k not in te), key=lambda x: -x[1])
            if miss_train or miss_val:
                msg = []
                if miss_train:
                    msg.append(f"test 에만 있고 train 에 없음 {sum(n for _, n in miss_train)}개 [{fmt(miss_train)}]")
                if miss_val:
                    msg.append(f"test 에 있고 val 에 없음 {sum(n for _, n in miss_val)}개 [{fmt(miss_val)}]")
                report("FAIL", f"⑧{cls}", "; ".join(msg))
            elif val_only:
                report("WARN", f"⑧{cls}", f"val 에만 있고 test 에 없음 {sum(n for _, n in val_only)}개 [{fmt(val_only)}] (train·val·test 하위라벨 {len(tr)}/{len(va)}/{len(te)}종)")
            else:
                report("OK", f"⑧{cls}", f"test 의 하위라벨 {len(te)}종 전부 train·val 에 있음 (train {len(tr)} / val {len(va)}종)")
    else:
        report("OK", "⑧", "source·aihub_category 컬럼 없음 — 검사 생략")

    # ---------- ⑨ 저장소 교차 녹음 겹침 [추가 2026-09-21 밤] ----------
    # 이 항목만 provenance 전체(prov)를 본다. 다른 저장소의 split 이 이 저장소의 split 과 같은 녹음을 쓰는지 봐야 하기 때문이다.
    print("\n⑨ 저장소 교차 녹음 겹침 (같은 녹음이 학습쪽과 평가쪽에 동시에 있는가 — 전문가 8 ↔ Mark5.2)")
    need = {"source", "aihub_src_file", "mark_version", "assigned_split", "source_type", "aug_source_file"}
    if need <= set(prov.columns):
        allp = prov[prov["removed_20260715"] == "active"] if "removed_20260715" in prov.columns else prov
        origp = allp[allp["source_type"] == "original"].copy()
        origp["rk9"] = origp.apply(rec_key, axis=1)
        kmap = {(r.mark_version, r.assigned_split, r.local_filename): r.rk9 for r in origp.itertuples()}
        augp = allp[allp["source_type"] != "original"].copy()
        augp["rk9"] = [kmap.get((r.mark_version, r.assigned_split, r.aug_source_file)) for r in augp.itertuples()]
        allk = pd.concat([origp, augp])
        def keys(mv, splits):
            d = allk[(allk["mark_version"] == mv) & (allk["assigned_split"].isin(splits))]
            return set(k for k in d["rk9"] if k is not None)
        def rows_in(mv, splits, keyset):
            d = allk[(allk["mark_version"] == mv) & (allk["assigned_split"].isin(splits))]
            return d[d["rk9"].isin(keyset)]
        m52_eval = keys("mark5.2", ["val", "test"]) | keys("mark5.2_eval", ["eval_zeroshot6", "eval_boundary_B", "eval_far_heldout"])
        m52_train = keys("mark5.2", ["train", "unlabeled"])
        experts = sorted(v for v in allk["mark_version"].unique() if str(v).startswith("mark4."))
        def check_expert(mv, verbose):
            hit = rows_in(mv, ["train"], m52_eval)
            n_r1 = len(hit)
            if n_r1:
                by = Counter(hit["target_class"])
                n_o = int((hit["source_type"] == "original").sum())
                report("FAIL", f"⑨R1 {mv}", f"train 이 Mark5.2 val·test·data_eval 녹음을 {n_r1}클립 씀(원본 {n_o} + 그 증강본 {n_r1 - n_o}) [{', '.join(f'{k} {v}' for k, v in by.most_common(4))}]")
            elif verbose:
                report("OK", f"⑨R1 {mv}", "train ∩ Mark5.2 val·test·data_eval 녹음 0")
            tr, va, te = keys(mv, ["train"]), keys(mv, ["val"]), keys(mv, ["test"])
            n_r3 = len(rows_in(mv, ["train"], va | te)) + len(rows_in(mv, ["val"], te))
            if n_r3:
                report("FAIL", f"⑨R3 {mv}", f"자기 train∩val·test {len(tr & (va | te))} 녹음 / val∩test {len(va & te)} 녹음 ({n_r3}클립)")
            elif verbose:
                report("OK", f"⑨R3 {mv}", f"train∩val·test 0, val∩test 0 (녹음 {len(tr)}/{len(va)}/{len(te)})")
            if verbose:
                n_r4 = len(rows_in(mv, ["val", "test"], m52_train))
                report("OK", f"⑨R4 {mv}", f"val·test 가 Mark5.2 train·unlabeled 녹음을 {n_r4}클립 씀 (허용 — teacher 평가쪽이 학생 학습쪽과 겹치는 것은 통로가 아님)")
            # [추가 2026-10-23] R5·R6
            out = {}
            for tag, splits, keyset, what in (("R5", ["train"], m52_train, "train 이 Mark5.2 train·unlabeled 녹음을"),
                                              ("R6", ["val", "test"], m52_eval, "val·test 가 Mark5.2 val·test·data_eval 녹음을")):
                hit = rows_in(mv, splits, keyset)
                out[tag] = len(hit)
                if len(hit):
                    by = Counter(zip(hit["target_class"], hit["source"]))
                    report("FAIL", f"⑨{tag} {mv}", f"{what} {len(hit)}클립 씀 [{', '.join(f'{c}/{s} {v}' for (c, s), v in by.most_common(5))}]")
                elif verbose:
                    report("OK", f"⑨{tag} {mv}", f"{what} 쓰지 않음")
            return n_r1, n_r3, out["R5"], out["R6"]
        if str(args.mark_version).startswith("mark4."):
            check_expert(args.mark_version, verbose=True)
        elif args.mark_version == "mark5.2":
            n_r2 = len(rows_in("mark5.2", ["train", "unlabeled"], m52_eval)) + len(rows_in("mark5.2", ["val"], keys("mark5.2", ["test"])))
            report("FAIL" if n_r2 else "OK", "⑨R2 mark5.2",
                   (f"train·unlabeled 이 val·test·data_eval 녹음을 {n_r2}클립 씀" if n_r2 else
                    f"train·unlabeled ∩ val·test·data_eval 0, val∩test 0 (녹음 학습쪽 {len(m52_train)} / 평가쪽 {len(m52_eval)})"))
            tot1 = tot3 = tot5 = tot6 = 0
            for mv in experts:
                a1, a3, a5, a6 = check_expert(mv, verbose=False)
                tot1 += a1; tot3 += a3; tot5 += a5; tot6 += a6
            if not (tot1 or tot3 or tot5 or tot6):
                report("OK", "⑨R1·R3·R5·R6 전문가 8", f"전문가 {len(experts)}개 모두 train ∩ Mark5.2 전체 0, val·test ∩ Mark5.2 평가쪽 0, 자기 train∩val·test 0")
            else:
                report("FAIL", "⑨합계", f"R1 {tot1} · R3 {tot3} · R5 {tot5} · R6 {tot6}클립(증강본 포함, 위 저장소별 줄 참조)")
        else:
            report("OK", "⑨", f"{args.mark_version} 은 교차 검사 대상 아님(mark4.k / mark5.2 만)")
    else:
        report("OK", "⑨", f"필요한 컬럼 없음 {sorted(need - set(prov.columns))} — 검사 생략")

    # ---------- 총평 ----------
    n_fail = sum(1 for l, _, _ in results if l == "FAIL")
    n_warn = sum(1 for l, _, _ in results if l == "WARN")
    print(f"\n{'='*78}")
    print(f"판정: OK {len(results)-n_fail-n_warn} / WARN {n_warn} / FAIL {n_fail}")
    if n_fail:
        print("✗ FAIL 이 있습니다. 구성을 고치고 다시 수집한 뒤 학습하십시오.")
        for l, i, msg in results:
            if l == "FAIL":
                print(f"   - {i}: {msg}")
    elif n_warn:
        print("⚠ FAIL 은 없습니다. WARN 은 의도한 것인지 확인한 뒤 진행하십시오.")
    else:
        print("문제 없음 — 학습으로 넘어가도 됩니다.")
    print("=" * 78)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
