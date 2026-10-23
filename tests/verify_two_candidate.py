"""두 후보 라벨(검증 결합기 가짜 정답) 구현 점검. 성능 실험이 아니다.

[신설 2026-10-22 / 검증 결합기]
    python tests/verify_two_candidate.py --source-root . --audio-root ../Mark5.2 --combiner-csv <pseudo_labels_cmb_l200_s1.csv>

1. 손실식: 두 후보 행이 없으면 새 경로 값 = 옛 criterion(가짜 정답 CE, 확신도 가중) 값. 두 후보 행은 −log(p_a + p_b).
2. CSV 읽기: 결합기 CSV 의 한 라벨·두 후보 행 수가 결합기 json 의 개수와 같다. use_two=False 면 두 후보 행이 빠진다.
3. 1에폭 CPU 학습 두 번(두 후보 켬 / --no_two_candidate): 로그의 '두후보' 개수, run_tag _noTWO, 체크포인트 기록.
산출물은 verification/two_candidate/ 아래에만 쓴다(held-out test 클립을 쓰지 않는다 — test 경로는 val 클립으로 채운다).
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import train_mark5 as train


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def two_candidate_loss(logits, lab1, lab2, weights):
    """train_mark5 학습 루프의 두 후보 손실과 같은 식(그 코드를 그대로 옮긴 것)."""
    two = lab2 >= 0
    lp = F.log_softmax(logits, dim=1)
    lp1 = lp.gather(1, lab1[:, None]).squeeze(1)
    lp2 = lp.gather(1, lab2.clamp_min(0)[:, None]).squeeze(1)
    per = torch.where(two, -torch.logaddexp(lp1, lp2), -lp1)
    return (per * weights).sum() / weights.sum().clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audio-root', type=Path, required=True)
    ap.add_argument('--combiner-csv', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    out = ROOT / 'verification' / 'two_candidate'
    out.mkdir(parents=True, exist_ok=True)

    # 1. 손실식
    logits = torch.randn(6, 9)
    lab1 = torch.tensor([0, 1, 2, 3, 4, 5])
    w = torch.tensor([.9, .95, .99, .92, .97, .91])
    crit = train.DistillationLoss(4.0, 0.3, 0.0, use_pseudo_label_ce=True, hard_weight=1.0)
    _, _, old_soft, _ = crit(logits, teacher_logits=F.one_hot(lab1, 9).float(), pseudo_weights=w, pseudo_targets=lab1)
    new_soft = two_candidate_loss(logits, lab1, torch.full_like(lab1, -1), w)
    require(torch.allclose(old_soft, new_soft, atol=1e-6), f'두 후보가 없을 때 옛 값과 다름: {old_soft} vs {new_soft}')
    lab2 = torch.tensor([-1, 7, -1, 8, -1, 0])
    p = torch.softmax(logits, dim=1)
    per = torch.stack([-torch.log(p[i, lab1[i]] + (p[i, lab2[i]] if lab2[i] >= 0 else 0)) for i in range(6)])
    require(torch.allclose(two_candidate_loss(logits, lab1, lab2, w), (per * w).sum() / w.sum(), atol=1e-6), '두 후보 손실식 불일치')

    # 2. CSV 읽기
    classes = list(train.AudioViLDConfig(mark_version='mark5.2').get_classes_for_evaluation())
    rows = list(csv.DictReader(args.combiner_csv.open(encoding='utf-8')))
    n_one = sum(r['mode'] == 'one' for r in rows)
    n_two = sum(r['mode'] == 'two' for r in rows)
    meta_path = args.combiner_csv.parent / 'plots' / 'combiner' / args.combiner_csv.name.replace('pseudo_labels_cmb_', 'combiner_').replace('.csv', '.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    require((n_one, n_two) == (meta['counts']['one'], meta['counts']['two']), f'CSV 행 수와 결합기 기록이 다름: {(n_one, n_two)} vs {meta["counts"]}')
    full = train.load_pseudo_csv(str(args.combiner_csv), classes)
    only_one = train.load_pseudo_csv(str(args.combiner_csv), classes, use_two=False)
    require(len(full) == n_one + n_two and sum(v[2] >= 0 for v in full.values()) == n_two, '두 후보 행을 다 읽지 못함')
    require(len(only_one) == n_one and all(v[2] == -1 for v in only_one.values()), 'use_two=False 인데 두 후보 행이 남음')

    # 3. 1에폭 CPU 학습 두 번
    BaseConfig = train.AudioViLDConfig

    class SmokeConfig(BaseConfig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.num_epochs = 1
            self.batch_size = 16
            self.mn10_pretrained_path = str(args.source_root / 'weights/mn10_as_mAP_471.pt')

    train.AudioViLDConfig = SmokeConfig
    selected = {}
    for split in ('labeled', 'val'):
        records = list(csv.DictReader((args.source_root / f'dataset_{split}.csv').open(encoding='utf-8-sig')))
        selected[split] = []
        for name in classes:
            row = next(r for r in records if r['label'] == name)
            path = args.audio_root / f'data_source_{split}' / name / Path(row['path']).name
            require(path.is_file(), f'Missing fixture: {path}')
            selected[split].append({'path': str(path), 'label': name})
    # 라벨 없는 클립 7개: 한 라벨 3 · 두 후보 3 · CSV 에 없음(침묵) 1. 결합기 CSV 의 행을 그대로 쓴다.
    pick = [r for r in rows if r['mode'] == 'one'][:3] + [r for r in rows if r['mode'] == 'two'][:3]
    selected['unlabeled'], fixture = [], []
    for r in pick + [{'path': next(x['path'] for x in csv.DictReader((args.source_root / 'dataset_unlabeled.csv').open(encoding='utf-8-sig'))
                                    if Path(x['path']).name not in full)}]:
        path = args.audio_root / 'data_source_unlabeled' / Path(r['path']).name
        require(path.is_file(), f'Missing fixture: {path}')
        selected['unlabeled'].append({'path': str(path)})
        if 'mode' in r:
            fixture.append({**r, 'path': str(path)})
    require(len(selected['unlabeled']) == 7 and len(fixture) == 6, '고정 묶음 구성 오류')

    reports = []
    for name, extra in [('two_on', {}), ('two_off', dict(no_two_candidate=True))]:
        run = out / name
        (run / 'model').mkdir(parents=True, exist_ok=True)
        (run / 'plots').mkdir(exist_ok=True)
        for split in ('labeled', 'unlabeled', 'val'):
            write_csv(run / f'dataset_{split}.csv', selected[split], ['path'] if split == 'unlabeled' else ['path', 'label'])
        write_csv(run / 'dataset_test.csv', selected['val'], ['path', 'label'])
        write_csv(run / 'pseudo_labels_cmbtest_l9_s1.csv', fixture, ['path', 'label', 'confidence', 'label2', 'mode'])
        train.PROJECT_ROOT, train.BASE_DIR = str(run), str(run / 'model')
        args_run = dict(mark_version='mark5.2', seed_value=1, unlabeled_n=7, linear_head=True, head_pretrained_hidden=True,
                        hard_weight=1., teacher_seg_mean=True, teacher_clean_view=True, bg_weight=0., student_raw_wave=True,
                        joint_forward=True, pseudo_csv='pseudo_labels_cmbtest_l9_s1.csv', pseudo_source='cmbtest', **extra)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train.train_mark5(**args_run)
        log = buf.getvalue()
        cfg = BaseConfig(mark_version='mark5.2')
        cfg.experiment_seed, cfg.unlabeled_n, cfg.pseudo_source = 1, 7, 'cmbtest'
        cfg.apply_cli_ablation(**{k: v for k, v in args_run.items()
                                  if k not in ('mark_version', 'seed_value', 'unlabeled_n', 'pseudo_csv', 'pseudo_source')})
        ck = torch.load(run / 'model' / f'student_model_{cfg.run_tag}.pth', map_location='cpu', weights_only=False)
        if name == 'two_on':
            require('한 라벨 3 · 두 후보 3' in log and '| 두후보 3' in log, '두 후보 3개가 학습에 들어가지 않음')
            require('_noTWO' not in cfg.run_tag and ck['config_metadata'].get('no_two_candidate') is False, 'run_tag·기록 오류(켬)')
        else:
            require('한 라벨 3 · 두 후보 0' in log and '두후보' not in log, '--no_two_candidate 인데 두 후보가 남음')
            require('_noTWO' in cfg.run_tag and ck['config_metadata'].get('no_two_candidate') is True, 'run_tag·기록 오류(끔)')
        epoch_line = next(line for line in log.splitlines() if 'PL채택' in line)
        reports.append({'run': name, 'run_tag': cfg.run_tag, 'epoch_log': epoch_line.strip()})

    try:
        train.train_mark5(mark_version='mark5.2', seed_value=1, unlabeled_n=7, no_two_candidate=True)
        raise AssertionError('--pseudo_csv 없이 --no_two_candidate 가 통과함')
    except ValueError as e:
        require('--no_two_candidate' in str(e), f'다른 오류: {e}')

    report = {'status': 'passed', 'scope': '구현 점검(1에폭 CPU 두 번), 성능 실험 아님', 'held_out_test_used': False,
              'combiner_csv': args.combiner_csv.name, 'csv_one': n_one, 'csv_two': n_two, 'runs': reports}
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
