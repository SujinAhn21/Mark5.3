"""Check removal, inference profiling, the cmb Colab notebooks, and deployment integrity."""
import ast
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
    names = ['1 교사 점수·채택 관문.ipynb', '2 Mark5.3 본실험 80회.ipynb', '3 비교모델 결합기 조건 24회.ipynb']
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
    gate_nb, main_nb, base_nb = (sources[n] for n in names)
    assert '"--kind", kind' in gate_nb and '("mn10", "segment")' in gate_nb and 'teacher_gate_report.py' in gate_nb
    assert 'teacher_gate_mn10_vs_segment.json' in gate_nb and 'teacher_gate_mn10_vs_segment.json' in main_nb
    assert '--preset cmb --seeds 1 2 3 4 5' in main_nb and '"--main_pseudo_source", "cmb"' in main_nb
    assert '"--baseline_cond", "plc"' in main_nb and 'label_freshness' in main_nb
    assert '--preset cmb --data_root' in base_nb and '--conds sup --labeled_ns 0 1000 500 200 --seeds 1' in base_nb
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
    report = {'status': 'passed', 'release_files_checked': checked,
              'six_component_code_absent': True, 'removed_cli_rejected': True,
              'profile_adapter_unused': True, 'cmb_notebooks_checked': names, 'cmb_plan_tags': len(tags),
              'deployment_parameters': measurement['total_parameters'],
              'training_parameters': measurement['training_parameters']}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
