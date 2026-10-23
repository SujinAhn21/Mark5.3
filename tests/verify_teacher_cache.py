"""최신 판 확인(label_freshness)과 '옛 교사 경로 복원' 교사 캐시 경로(_TC) 구현 점검. 성능 실험이 아니다.

[신설 2026-10-22]
    python tests/verify_teacher_cache.py --source-root . --audio-root ../Mark5.2 --scores-csv <teacher_scores_*.csv(labeled·unlabeled 포함)>

1. 최신 판 확인: 가짜 교사 파일 8개로 점수 캐시 json 을 만들고 결합기를 돌린 뒤, 그대로면 통과 · 교사 파일 하나를 바꾸면 멈춤 ·
   가짜 정답 CSV 를 바꾸면 멈춤 · 점수 캐시를 바꾸면 멈춤.
2. 교사 캐시 경로 1에폭 CPU 학습: 라벨 없는 7클립의 교사 점수를 손으로 정해 둔다.
     확실 3클립(1등 0.9 · 2등 0.1) → 고정 규칙 채택 3
     전원 낮음 2클립(전부 0.05)     → 전원 거절로 others 2
     애매 2클립(0.6 · 0.5 · 나머지 0.05) → 미채택, 부정 라벨: 0.05 < 0.1 인 6칸씩 = 2샘플 · 12라벨
   기대 로그: 'PL채택 0.714(5/7)' · 'OR→others 2' · '(2샘플·12라벨)' · Feat > 0. 증류 갈래 첫 층 가중치 [960, 384].
   평가(val)도 같은 구조로 체크포인트를 읽는다.
산출물은 verification/teacher_cache/ 아래에만 쓴다(held-out test 를 쓰지 않는다 — test 경로는 val 클립으로 채운다).
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import train_mark5 as train
import eval as evaluate
import label_freshness

SPEC = ['heavy_impact', 'dragging', 'construction', 'machine_noise', 'talking', 'water_toilet', 'water_shower', 'dog_bark']


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fake_release(root):
    """resources/ 에 가짜 mn10 교사 파일 8개. json 의 teacher_files 형식(make_teacher_scores.Mn10Scorer)."""
    res = root / 'resources'
    res.mkdir(parents=True, exist_ok=True)
    files = []
    for i, c in enumerate(SPEC, 1):
        p = res / f'mn10_teacher_mark4.{i}.pt'
        p.write_bytes(f'fake teacher {c}'.encode())
        files.append({'class': c, 'file': p.name, 'sha256': sha(p), 'run_id': 'test', 'saved_epoch': 1})
    return files


def write_scores_meta(scores_csv, files):
    meta = {'teacher_kind': 'mn10', 'teacher_files': files, 'teacher_release_sha256': 'test',
            'scores_sha256': sha(scores_csv)}
    Path(str(scores_csv).replace('.csv', '.json')).write_text(json.dumps(meta), encoding='utf-8')


def expect_stale(fn, what):
    try:
        fn()
    except label_freshness.StaleFileError:
        return
    raise AssertionError(f'옛 판인데 통과함: {what}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audio-root', type=Path, required=True)
    ap.add_argument('--scores-csv', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    out = ROOT / 'verification' / 'teacher_cache'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    # 1. 최신 판 확인 ----------------------------------------------------------------
    fr = out / 'freshness'
    (fr / 'plots').mkdir(parents=True)
    files = fake_release(fr)
    label_freshness.PROJECT_ROOT = str(fr)          # 교사 파일은 이 폴더의 resources/
    scores = fr / 'plots' / 'teacher_scores_mn10.csv'
    shutil.copyfile(args.scores_csv, scores)
    write_scores_meta(scores, files)
    label_freshness.check_scores(scores)
    import make_combiner_labels as mcl
    argv = ['make_combiner_labels.py', '--scores', str(scores), '--labeled_ns', '200', '--seeds', '1', '--out_root', str(fr)]
    with contextlib.redirect_stdout(io.StringIO()), __import__('unittest.mock').mock.patch.object(sys, 'argv', argv):
        mcl.main()
    cmb = fr / 'pseudo_labels_cmb_l200_s1.csv'
    meta = label_freshness.check_pseudo_csv(cmb)
    require(set(meta['outputs']) == {'pseudo_labels_cmb_l200_s1.csv', 'pseudo_labels_cmbone_l200_s1.csv', 'pseudo_labels_fix_l200_s1.csv'},
            '결합기 json 에 CSV 해시 3개가 없음')
    t1 = fr / 'resources' / 'mn10_teacher_mark4.1.pt'
    keep = t1.read_bytes()
    t1.write_bytes(b'retrained teacher')
    expect_stale(lambda: label_freshness.check_pseudo_csv(cmb), '교사를 다시 학습했는데 옛 가짜 정답 CSV')
    t1.write_bytes(keep)
    label_freshness.check_pseudo_csv(cmb)
    raw = cmb.read_bytes()                       # 바이트 그대로 되돌린다(텍스트로 쓰면 줄바꿈 \r\n 이 바뀌어 해시가 달라진다)
    cmb.write_bytes(raw + b'\n')
    expect_stale(lambda: label_freshness.check_pseudo_csv(cmb), '다른 실행의 가짜 정답 CSV')
    cmb.write_bytes(raw)
    s_raw = scores.read_bytes()
    scores.write_bytes(s_raw + b'\n')
    expect_stale(lambda: label_freshness.check_pseudo_csv(cmb), '점수 캐시를 다시 만들었는데 옛 가짜 정답 CSV')
    scores.write_bytes(s_raw)
    label_freshness.check_pseudo_csv(cmb)
    freshness_report = {'pass': True, 'checks': ['그대로 통과', '교사 파일 바뀜 → 멈춤', 'CSV 바뀜 → 멈춤', '점수 캐시 바뀜 → 멈춤']}

    # 2. 교사 캐시 경로 1에폭 ----------------------------------------------------------
    BaseConfig = train.AudioViLDConfig

    class SmokeConfig(BaseConfig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.num_epochs = 1
            self.batch_size = 16
            self.mn10_pretrained_path = str(args.source_root / 'weights/mn10_as_mAP_471.pt')

    train.AudioViLDConfig = evaluate.AudioViLDConfig = SmokeConfig
    classes = list(BaseConfig(mark_version='mark5.2').get_classes_for_evaluation())
    selected = {}
    for split in ('labeled', 'val'):
        records = list(csv.DictReader((args.source_root / f'dataset_{split}.csv').open(encoding='utf-8-sig')))
        selected[split] = []
        for name in classes:
            row = next(r for r in records if r['label'] == name)
            path = args.audio_root / f'data_source_{split}' / name / Path(row['path']).name
            require(path.is_file(), f'Missing fixture: {path}')
            selected[split].append({'path': str(path), 'label': name})
    unl = list(csv.DictReader((args.source_root / 'dataset_unlabeled.csv').open(encoding='utf-8-sig')))[:7]
    selected['unlabeled'] = []
    for r in unl:
        path = args.audio_root / 'data_source_unlabeled' / Path(r['path']).name
        require(path.is_file(), f'Missing fixture: {path}')
        selected['unlabeled'].append({'path': str(path)})
    p_rows = ([[0.9 if j == i else (0.1 if j == i + 1 else 0.05) for j in range(8)] for i in range(3)]
              + [[0.05] * 8] * 2
              + [[0.6, 0.5] + [0.05] * 6] * 2)

    run = out / 'run'
    (run / 'model').mkdir(parents=True)
    (run / 'plots').mkdir()
    for split in ('labeled', 'unlabeled', 'val'):
        write_csv(run / f'dataset_{split}.csv', selected[split], ['path'] if split == 'unlabeled' else ['path', 'label'])
    write_csv(run / 'dataset_test.csv', selected['val'], ['path', 'label'])
    files = fake_release(run)
    label_freshness.PROJECT_ROOT = str(run)
    sc = run / 'plots' / 'teacher_scores_mn10.csv'
    write_csv(sc, [{'path': u['path'], 'split': 'unlabeled', 'label': '', **{f'p_{c}': v for c, v in zip(SPEC, p)}}
                   for u, p in zip(selected['unlabeled'], p_rows)], ['path', 'split', 'label'] + [f'p_{c}' for c in SPEC])
    write_scores_meta(sc, files)
    torch.save({'paths': [u['path'] for u in selected['unlabeled']], 'features': torch.randn(7, 960).half(), 'dim': 960,
                'scores_sha256': sha(sc)}, run / 'plots' / 'teacher_feats_mn10.pt')

    train.PROJECT_ROOT, train.BASE_DIR = str(run), str(run / 'model')
    args_run = dict(mark_version='mark5.2', seed_value=1, unlabeled_n=7, linear_head=True, head_pretrained_hidden=True,
                    hard_weight=1., others_from_reject=.2, neg_pl=(.1, .3), teacher_seg_mean=True, teacher_clean_view=True,
                    bg_weight=0., student_raw_wave=True, joint_forward=True, teacher_cache=str(sc))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train.train_mark5(**args_run)
    log = buf.getvalue()
    epoch_line = next(line for line in log.splitlines() if 'PL채택' in line)
    require('PL채택 0.714(5/7)' in epoch_line, f'채택 수가 기대(5/7)와 다름: {epoch_line}')
    require('OR→others 2' in epoch_line, f'전원 거절 수가 기대(2)와 다름: {epoch_line}')
    require('(2샘플·12라벨)' in epoch_line, f'부정 라벨 수가 기대(2샘플·12라벨)와 다름: {epoch_line}')
    feat = float(re.search(r'Feat ([0-9.]+)', epoch_line).group(1))
    require(feat > 0, f'특징 정렬 손실이 0: {epoch_line}')
    cfg = BaseConfig(mark_version='mark5.2')
    cfg.experiment_seed, cfg.unlabeled_n = 1, 7
    cfg.apply_cli_ablation(**{k: v for k, v in args_run.items() if k not in ('mark_version', 'seed_value', 'unlabeled_n')})
    require('_TC' in cfg.run_tag and '_OR02_NEG' in cfg.run_tag, f'run_tag 오류: {cfg.run_tag}')
    ck = torch.load(run / 'model' / f'student_model_{cfg.run_tag}.pth', map_location='cpu', weights_only=False)
    w = ck['branch_state_dict']['distill_branch.0.weight']
    require(tuple(w.shape) == (960, 384), f'변환 층 모양 오류: {tuple(w.shape)}')
    require(ck['config_metadata'].get('distill_dim') == 960 and ck['config_metadata'].get('teacher_cache') == str(sc), '체크포인트 기록 오류')

    evaluate.PROJECT_ROOT, evaluate.BASE_DIR = str(run), str(run / 'model')
    evaluate._resolve_dataset_csv_path = lambda split: str(run / f'dataset_{split}.csv')
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate.evaluate(split='val', **args_run)
    rows = list(csv.DictReader((run / 'plots' / f'prediction_details_{cfg.run_tag}_val_plain.csv').open()))
    require(len(rows) == 9, '평가가 일부 클립을 건너뜀')

    # 교사를 바꾸면 캐시 경로도 멈춘다
    (run / 'resources' / 'mn10_teacher_mark4.3.pt').write_bytes(b'retrained')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.train_mark5(**args_run)
        raise AssertionError('교사가 바뀌었는데 옛 교사 캐시로 학습이 시작됨')
    except label_freshness.StaleFileError:
        pass

    report = {'status': 'passed', 'scope': '구현 점검(1에폭 CPU), 성능 실험 아님', 'held_out_test_used': False,
              'freshness': freshness_report, 'teacher_cache_run': {'run_tag': cfg.run_tag, 'epoch_log': epoch_line.strip(),
                                                                  'distill_weight_shape': list(w.shape)}}
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
