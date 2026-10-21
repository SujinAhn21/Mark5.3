"""Check removal, inference profiling, notebook execution, and deployment integrity."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import profile_efficiency as profile
import run_experiments as runner
import select_lr
import aggregate_results as aggregate
import output_policy
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

    # Execute every default cell, keeping verification and the dry-run real.
    # Capture the training launch so this contract test cannot run a full experiment.
    notebook = json.loads((ROOT / 'notebooks/Mark5.3 제안모델 실험.ipynb').read_text(encoding='utf-8'))
    scope = {'display': lambda obj: None}
    previous_cwd = Path.cwd()
    previous_root = os.environ.get('MARK_ROOT')
    os.environ['MARK_ROOT'] = str(ROOT)
    real_run = subprocess.run
    training_commands = []
    def notebook_run(command, *args, **kwargs):
        if 'model/run_experiments.py' in command and '--dry_run' not in command:
            assert '--skip_test' in command
            assert kwargs.get('check') is True
            training_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)
    try:
        with patch.object(subprocess, 'run', side_effect=notebook_run):
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    assert not cell['outputs'] and cell['execution_count'] is None
                    exec(compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec'), scope)
    finally:
        os.chdir(previous_cwd)
        if previous_root is None:
            os.environ.pop('MARK_ROOT', None)
        else:
            os.environ['MARK_ROOT'] = previous_root
    assert len(scope['SELECTED_TAGS']) == 1
    assert scope['RUN_TRAINING'] and not scope['EVALUATE_TEST'] and not scope['RUN_PROFILE']
    assert training_commands == [scope['COMMAND'] + ['--skip_test']]
    assert '--m53' not in scope['BUNDLE'] and '--plain_output' not in scope['BUNDLE']
    # Exercise the notebook result cell with real smoke outputs, not just the empty-results path.
    fixture = ROOT / 'verification/joint'
    displayed = []
    fixture_scope = dict(scope, ROOT=fixture, LABELED=[0], UNLABELED=[7], SEEDS=[1], display=displayed.append)
    with patch.object(aggregate, 'PROJECT_ROOT', str(fixture)), patch.object(aggregate, 'PLOT_DIR', str(fixture / 'plots')), \
            patch.object(aggregate, '_FULL_LABELED', None), patch.object(output_policy, 'dataset_identity', return_value=None):
        exec(compile(''.join(notebook['cells'][6]['source']), 'notebook-results-fixture', 'exec'), fixture_scope)
    assert len(displayed) == 1 and len(displayed[0]) == 1
    # Ensure retained LR tooling can find the new val outputs if explicitly invoked.
    reports = json.loads((ROOT / 'verification/verification.json').read_text())
    for run in reports['runs']:
        run_root = ROOT / 'verification' / run['path']
        with patch.object(select_lr, 'PROJECT_ROOT', str(run_root)), patch.object(select_lr, 'PLOT_DIR', str(run_root / 'plots')), \
                patch.object(output_policy, 'dataset_identity', return_value=None):
            assert select_lr._needs(run['run_tag'], False) == (False, False)
    report = {'status': 'passed', 'release_files_checked': checked,
              'six_component_code_absent': True, 'removed_cli_rejected': True,
              'profile_adapter_unused': True, 'notebook_default_cells_executed': True,
              'notebook_training_default': True, 'notebook_training_launch_captured': True,
              'lr_tool_new_output_compatible': True, 'deployment_parameters': measurement['total_parameters'],
              'training_parameters': measurement['training_parameters']}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
