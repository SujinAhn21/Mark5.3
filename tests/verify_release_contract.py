"""Check removal, inference profiling, the cmb Colab notebooks, and deployment integrity."""
import ast
import csv
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import profile_efficiency as profile
import run_experiments as runner
import train_mark5 as train   # [2026-10-26] 빈 라벨 없는 목록 검사
from verify_release import verify


def main():
    torch.set_num_threads(4)
    checked = verify()
    forbidden = re.compile(r'\bm53_\w*|\b_m53\w*|\bM53_PARTS\b|\bExpertBinaryHead\b|\b_ova_fuse\b|\b_supcon_loss\b|\bdb_total\b|--m53')
    for folder in ('model', 'vild', 'shared_vild', 'utils'):
        for path in (ROOT / folder).glob('*.py'):
            text = path.read_text(encoding='utf-8')
            ast.parse(text)
            assert not forbidden.search(text), path
    try:
        runner._parse_runner_extra('--m53 none')
    except SystemExit:
        pass
    else:
        raise AssertionError('Removed component CLI was accepted')

    # Profile the actual graph; any use of the KD adapter must fail.
    original_branch = profile.DualBranchStudentHead
    class GuardBranch(original_branch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distill_branch.register_forward_pre_hook(self.fail)
        def fail(self, *args):
            raise AssertionError('KD adapter ran in deployment profiling')
    original_config = profile.AudioViLDConfig
    class LocalConfig(original_config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mn10_pretrained_path = str(ROOT / 'weights/mn10_as_mAP_471.pt')
    output = ROOT / 'verification/release_contract'
    output.mkdir(parents=True, exist_ok=True)
    with patch.object(profile, 'DualBranchStudentHead', GuardBranch), patch.object(profile, 'AudioViLDConfig', LocalConfig), \
            patch.object(profile, 'PROJECT_ROOT', str(output)):
        measurement = profile.profile('mark5.2', num_warmup=1, num_runs=3, linear_head=True, head_pretrained_hidden=True)
    assert measurement['total_parameters'] < measurement['training_parameters']
    assert measurement['gmacs_per_segment'] > 0 and measurement['cpu_latency_ms_per_clip'] > 0

    # [2026-10-23] Colab notebooks of the cmb plan. The old single-run notebook and its Adam-era result fixture
    # (verification/joint, run_tag _elr) were replaced, so the checks are static plus the real run-tag builder;
    # nothing trains here. select_lr is not checked any more: since 2026-10-25 (A안) the learning rates are fixed to the
    # 2026-10-08 selection (Adam, new layers and backbone 1e-4), so no run selects a learning rate.
    nb_dir = ROOT / 'notebooks'
    names = ['1 교사 점수·채택 관문.ipynb', '2 Mark5.3 본실험 40회.ipynb', '3 비교모델 60회.ipynb',
             '4 Mark5.3 라벨 없음 0 5회·결과물 4개.ipynb',   # [2026-10-26] 논문 결과물 4개용 · 라벨 없음 0 5회(노트북 4)
             '5 비교모델 평가만 60회(argmax).ipynb']        # [2026-10-26] 비교모델 판정 규칙 = 보정 없는 argmax(평가만 다시)
    assert sorted(p.name for p in nb_dir.glob('*.ipynb')) == sorted(names), sorted(p.name for p in nb_dir.glob('*.ipynb'))
    sources = {}
    for name in names:
        notebook = json.loads((nb_dir / name).read_text(encoding='utf-8'))
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        cells = [''.join(c['source']) for c in code_cells]
        assert all(not c['outputs'] and c['execution_count'] is None for c in code_cells), name
        for src in cells:   # IPython '!' lines are not Python; replace them before parsing
            ast.parse('\n'.join(('pass  # ' + line) if line.lstrip().startswith('!') else line for line in src.splitlines()))
        assert 'Blackwell' in cells[0] and 'raise SystemExit' in cells[0], name             # L4 rule: stop only on sm_100/120
        text = '\n'.join(cells)
        assert '--step_timeout_min' not in text and 'WAIT_MAX' not in text, name           # no time limits (CPU runtime allowed)
        assert '--m53' not in text and '"--lr"' not in text and '--encoder_lr' not in text, name
        sources[name] = text
    gate_nb, main_nb, base_nb, u0_nb, reeval_nb = (sources[n] for n in names)
    assert '"--kind", kind' in gate_nb and '("mn10", "segment")' in gate_nb and 'teacher_gate_report.py' in gate_nb
    assert 'teacher_gate_mn10_vs_segment.json' in gate_nb and 'teacher_gate_mn10_vs_segment.json' in main_nb
    # [2026-10-26] 본실험은 8조합(--cmb_paper) 40회, 마지막 셀이 비교모델 60회를 기다려 make_four_results.py. 비교모델은 정답 라벨만(plc 없음) × seed 1~5.
    assert '--preset cmb --cmb_paper --seeds 1 2 3 4 5' in main_nb and 'make_four_results.py' in main_nb and 'label_freshness' in main_nb
    assert 'plc' not in main_nb and 'make_paper_tables' not in main_nb
    assert '--conds sup --labeled_ns 0 1000 500 200 --seeds 1 2 3 4 5' in base_nb and '--preset cmb' not in base_nb and 'plc' not in base_nb
    # [2026-10-26] 노트북 2 는 끝났다 — 라벨 없음 0 수정 판에서는 [셀 7] 이 멈춘다. 노트북 4 는 u0 5회만 돌리고 결과물 4개를 만든다.
    assert 'if is_labeled and len(self.samples) == 0:' in main_nb and '4 Mark5.3 라벨 없음 0 5회' in main_nb
    assert '--preset cmb --cmb_paper --cmb_unlabeled_n 0 --seeds 1 2 3 4 5' in u0_nb and 'make_four_results.py' in u0_nb
    assert 'matches_plain_policy' in u0_nb and '_summary_is_complete' in u0_nb and 'plc' not in u0_nb
    # [2026-10-26] 노트북 5: 비교모델 60회 평가만(학습이 끼면 멈춤) · 판정 규칙 argmax_v1 · 성능 요약 = argmax 값
    assert '--conds sup --labeled_ns 0 1000 500 200 --seeds 1 2 3 4 5' in reeval_nb and '"학습" in r' in reeval_nb
    assert 'rule_json_is_current' in reeval_nb and 'argmax_v1' in reeval_nb and 'make_argmax_summary.py' in reeval_nb and 'plc' not in reeval_nb
    assert reeval_nb.count('--eval_only') >= 2 and '"--eval_only"' in reeval_nb and '5 비교모델 평가만 60회' in u0_nb   # [셀 4]·[셀 5] 모두 평가만 · 노트북 4 가 노트북 5 를 기다림
    # [2026-10-26] 정답 라벨 쪽만 0개면 멈추고 라벨 없는 쪽 0개(u0·천장)는 통과 · --cmb_unlabeled_n 0 은 --cmb_paper 에서 u0 한 조합
    _src = (ROOT / 'model/train_mark5.py').read_text(encoding='utf-8')
    assert 'if is_labeled and len(self.samples) == 0:' in _src and "'labeled' if is_labeled else 'unlabeled'" not in _src
    assert runner.select_cmb_plan('cmb', True, 0) == [(200, 0, None, '')]                    # 노트북 4 의 5회
    assert runner.select_cmb_plan('cmb', True, None) == runner.CMB_PAPER_PLAN
    assert len(runner.select_cmb_plan('cmb', False, 0)) == 4                                    # --cmb_paper 없이: 라벨 없음 0 네 조합
    for _bad in (('cmb', True, 7), (None, True, 0), ('paper', False, 0)):
        try:
            runner.select_cmb_plan(*_bad)
            raise AssertionError(f'select_cmb_plan{_bad} 가 멈추지 않음')
        except SystemExit as _e:
            assert '[ERROR]' in str(_e), _e
    # 빈 목록: 라벨 없는 쪽은 통과(0개), 정답 라벨 쪽은 멈춤
    _cfg = train.AudioViLDConfig(mark_version='mark5.2')
    assert len(train.SemiSupervisedDataset([], None, _cfg, is_labeled=False)) == 0
    try:
        train.SemiSupervisedDataset([], None, _cfg, is_labeled=True)
        raise AssertionError('빈 정답 라벨 목록이 통과함')
    except ValueError as _e:
        assert 'labeled 샘플이 0개' in str(_e), _e
    assert '_seeds != [1, 2, 3, 4, 5]' in u0_nb and 'pred_missing' in u0_nb and 'def baseline_argmax' in u0_nb
    # [2026-10-26] 결과물 스크립트: 비교모델도 보정 없는 argmax(PreBias_Prob_*) · 빠진 회차면 멈춤
    _mfr = (ROOT / 'model/make_four_results.py').read_text(encoding='utf-8')
    assert 'PreBias_Prob_' in _mfr and 'def baseline_argmax' in _mfr and '--allow_missing' in _mfr
    import tempfile
    import make_four_results as mfr
    with tempfile.TemporaryDirectory() as _td:
        _td = Path(_td)
        _cls = list(mfr.CLASSES)
        _rows = []   # 정답 heavy_impact 2개 · dragging 1개. Predicted Label(보정 뒤)은 셋 다 맞고, 보정 전 argmax 는 셋째만 틀림
        for _i, (_y, _top) in enumerate([('heavy_impact', 'heavy_impact'), ('heavy_impact', 'heavy_impact'), ('dragging', 'others')]):
            _p = {c: 0.01 for c in _cls}; _p[_top] = 0.9
            _rows.append({'Filename': f'f{_i}.wav', 'True Label': _y, 'Predicted Label': _y,
                          **{f'PreBias_Prob_{c}': _p[c] for c in _cls}})
        with (_td / 'pred.csv').open('w', newline='', encoding='utf-8') as _h:
            _w = csv.DictWriter(_h, fieldnames=list(_rows[0])); _w.writeheader(); _w.writerows(_rows)
        _summ = mfr.four_metrics([r['True Label'] for r in _rows], [r['Predicted Label'] for r in _rows])
        _arg = mfr.baseline_argmax(str(_td / 'pred.csv'), _summ, 't')
        assert abs(_summ['Accuracy'] - 1.0) < 1e-12 and abs(_arg['Accuracy'] - 2 / 3) < 1e-12, (_summ, _arg)
        try:
            mfr.baseline_argmax(str(_td / 'pred.csv'), dict(_summ, Accuracy=0.5), 't')
            raise AssertionError('요약과 예측 CSV 가 어긋나도 통과함')
        except SystemExit as _e:
            assert '같은 회차의 파일이 아닙니다' in str(_e), _e
        (_td / 'Mark5.3' / 'plots').mkdir(parents=True)
        import subprocess as _sp
        _r = _sp.run([sys.executable, str(ROOT / 'model/make_four_results.py'), '--root', str(_td / 'Mark5.3'),
                      '--baseline_root', str(_td), '--no_plot'], capture_output=True, text=True)
        assert _r.returncode == 1 and '아직 없는 회차' in (_r.stdout + _r.stderr) and not any((_td / 'Mark5.3' / 'plots').iterdir()), _r.stdout[-500:]
    # The 80-run plan the notebook launches: 16 combos x seeds 1-5, all Adam with the fixed selected rates (_ADAM), no duplicate run tags.
    # [2026-10-25] supervised-only (u0) at every labeled level of table 4 (full, 1000, 500, 200), like the baselines' sup runs.
    tags = []
    for (labeled, unlabeled, pseudo, extra) in runner.CMB_PLAN:
        kw = runner._parse_runner_extra((runner.CMB_BASE_EXTRA + ' ' + extra).strip())[1]
        tags += [runner._run_tag('mark5.2', s, unlabeled, labeled_n=labeled, ps=pseudo, ablation_kw=kw) for s in (1, 2, 3, 4, 5)]
    assert len(runner.CMB_PLAN) == 16 and len(tags) == 80 and len(set(tags)) == 80, (len(runner.CMB_PLAN), len(set(tags)))
    sup_levels = sorted((l or 0) for (l, n, ps, ex) in runner.CMB_PLAN if n == 0 and not ex)
    assert sup_levels == [0, 200, 500, 1000], sup_levels
    assert all('_ADAM' in t and '_PD' not in t for t in tags)
    # [2026-10-26] 논문 결과물 4개용 8조합: CMB_PLAN 의 부분집합이라 80회 판에서 끝난 회차와 run_tag 가 같다.
    paper_tags = []
    for (labeled, unlabeled, pseudo, extra) in runner.CMB_PAPER_PLAN:
        kw = runner._parse_runner_extra((runner.CMB_BASE_EXTRA + ' ' + extra).strip())[1]
        paper_tags += [runner._run_tag('mark5.2', s, unlabeled, labeled_n=labeled, ps=pseudo, ablation_kw=kw) for s in (1, 2, 3, 4, 5)]
    assert len(runner.CMB_PAPER_PLAN) == 8 and len(set(paper_tags)) == 40 and set(paper_tags) <= set(tags)
    report = {'status': 'passed', 'release_files_checked': checked,
              'six_component_code_absent': True, 'removed_cli_rejected': True,
              'profile_adapter_unused': True, 'cmb_notebooks_checked': names, 'cmb_plan_tags': len(tags), 'cmb_paper_tags': len(paper_tags),
              'deployment_parameters': measurement['total_parameters'],
              'training_parameters': measurement['training_parameters']}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
