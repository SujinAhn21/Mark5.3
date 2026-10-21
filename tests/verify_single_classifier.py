"""Bounded implementation checks. This is not a performance experiment.

Runs on the existing CPU environment with the original teachers and audio.
All reduced-data training artifacts stay under verification/.
"""
import argparse
import ast
import contextlib
import csv
import io
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import train_mark5 as train
import eval as evaluate
import run_experiments as runner
import aggregate_results as aggregate
from output_policy import checkpoint_sha256, matches_plain_policy


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audio-root', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    out = ROOT / 'verification'
    out.mkdir(exist_ok=True)

    # Test loss connectivity, including the exact production DistillationLoss.
    enc = torch.nn.Linear(12, 384)
    branches = train.DualBranchStudentHead(384, supervised_identity=True)
    cls = torch.nn.Linear(384, 9)
    x = torch.randn(5, 12)
    h = enc(x)
    sup, kd = branches(h)
    criterion = train.DistillationLoss(4.0, 0.3, 0.3, use_pseudo_label_ce=True, hard_weight=1.0)
    labels = torch.arange(5)
    logits = cls(sup)
    teacher_logits = torch.randn(5, 9)
    teacher_features = torch.randn(5, 384)
    gt = criterion(logits, hard_targets=labels)[0]
    pl, _, pl_raw, feat_raw = criterion(logits, teacher_logits=teacher_logits,
        student_features=kd, teacher_features=teacher_features, pseudo_targets=labels)
    expected_feature = 0.5 * F.l1_loss(F.normalize(kd, dim=1), F.normalize(teacher_features, dim=1))
    expected_feature += 0.5 * (1 - F.cosine_similarity(F.normalize(kd, dim=1), F.normalize(teacher_features, dim=1)).mean())
    require(torch.allclose(pl, .3 * F.cross_entropy(logits, labels) + .3 * expected_feature), 'Unexpected auxiliary classification loss')
    adapter = tuple(branches.distill_branch.parameters())
    ce_grads = torch.autograd.grad(gt + .3 * pl_raw, adapter, allow_unused=True, retain_graph=True)
    require(all(g is None for g in ce_grads), 'Classification gradient reached KD adapter')
    feature_grads = torch.autograd.grad(.3 * feat_raw, (enc.weight,) + adapter, retain_graph=True)
    require(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in feature_grads), 'Feature KD gradient missing')
    before = cls(branches.forward_supervised(h)).detach()
    with torch.no_grad():
        for p in branches.distill_branch.parameters():
            p.normal_(100, 30)
    require(torch.equal(before, cls(branches.forward_supervised(h)).detach()), 'KD adapter changed inference')

    source_text = (ROOT / 'model/train_mark5.py').read_text(encoding='utf-8')
    require('db_total' not in source_text and '_distill_classify' not in source_text, 'Distill classification code remains')
    ast.parse(source_text)

    BaseConfig = train.AudioViLDConfig
    class SmokeConfig(BaseConfig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.num_epochs = 1
            self.batch_size = 16
            self.mn10_pretrained_path = str(args.source_root / 'weights/mn10_as_mAP_471.pt')
        def get_class_text_embeddings(self, *a, **kw):
            if self.mark_version.startswith('mark5'):
                raise AssertionError('Unused student text encoder was requested')
            return super().get_class_text_embeddings(*a, **kw)

    train.AudioViLDConfig = evaluate.AudioViLDConfig = SmokeConfig
    train.RESOURCES_DIR = str(args.source_root / 'resources')
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
    consensus = list(csv.DictReader((args.source_root / 'consensus_labels.csv').open(encoding='utf-8-sig')))
    selected['unlabeled'] = []
    for name in classes[:7]:
        row = next(r for r in consensus if r['label'] == name)
        path = args.audio_root / 'data_source_unlabeled' / Path(row['path']).name
        require(path.is_file(), f'Missing fixture: {path}')
        selected['unlabeled'].append({'path': str(path)})

    observations = {'classifier_calls': 0, 'adapter_training_calls': 0, 'adapter_grad_nonzero': 0,
                    'encoder_grad_nonzero': 0, 'validation_batches': 0}
    sup_ptrs = set()
    heads = []
    classifiers = []
    teachers = []
    BaseBranch = train.DualBranchStudentHead
    class CheckedBranch(BaseBranch):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            heads.append(self)
            def check_adapter(module, inputs):
                require(self.training, 'Feature KD adapter executed in validation/inference')
                observations['adapter_training_calls'] += 1
            self.distill_branch.register_forward_pre_hook(check_adapter)
            next(self.distill_branch.parameters()).register_hook(
                lambda g: observations.__setitem__('adapter_grad_nonzero', observations['adapter_grad_nonzero'] + int(bool(g.abs().sum() > 0))))
        def forward_supervised(self, features):
            result = super().forward_supervised(features)
            sup_ptrs.add(result.data_ptr())
            if not self.training:
                observations['validation_batches'] += 1
            return result
    train.DualBranchStudentHead = evaluate.DualBranchStudentHead = CheckedBranch

    original_build_classifier = train.build_student_head
    def build_classifier(config):
        model = original_build_classifier(config)
        classifiers.append(model)
        def check(module, inputs):
            require(inputs[0].data_ptr() in sup_ptrs, 'Classifier consumed a non-supervised feature')
            observations['classifier_calls'] += 1
        model.register_forward_pre_hook(check)
        return model
    train.build_student_head = evaluate.build_student_head = build_classifier
    original_build_encoder = train.build_audio_encoder
    def build_encoder(config):
        model = original_build_encoder(config)
        next(model.parameters()).register_hook(
            lambda g: observations.__setitem__('encoder_grad_nonzero', observations['encoder_grad_nonzero'] + int(bool(g.abs().sum() > 0))))
        return model
    train.build_audio_encoder = build_encoder
    OriginalTeacher = train.EnsembleTeacher
    class CheckedTeacher(OriginalTeacher):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            teachers.append(self)
    train.EnsembleTeacher = CheckedTeacher

    # [2026-10-22 / Prodigy] 자동 학습률이라 lr/encoder_lr 을 넘기지 않는다(vild_config 가 받으면 멈춘다).
    common = dict(mark_version='mark5.2', seed_value=1, unlabeled_n=7,
                  linear_head=True, head_pretrained_hidden=True, hard_weight=1., others_from_reject=.2,
                  neg_pl=(.1, .3), teacher_seg_mean=True, teacher_clean_view=True, bg_weight=0., student_raw_wave=True)
    reports = []
    # Verify the joint and separate encoder-forward modes of the unified loss path.
    for path_name, extra in [('joint', dict(joint_forward=True)), ('standard', {})]:
        run = out / path_name
        (run / 'model').mkdir(parents=True, exist_ok=True)
        (run / 'plots').mkdir(exist_ok=True)
        for split in ('labeled', 'unlabeled', 'val'):
            fields = ['path'] if split == 'unlabeled' else ['path', 'label']
            write_csv(run / f'dataset_{split}.csv', selected[split], fields)
        # Test-route fixture uses validation clips; no held-out test data are touched.
        write_csv(run / 'dataset_test.csv', selected['val'], ['path', 'label'])
        train.PROJECT_ROOT, train.BASE_DIR = str(run), str(run / 'model')
        args_run = dict(common, **extra)
        train.train_mark5(**args_run)
        feature_cache = torch.load(train.get_feature_cache_path(str(run), 'mark5.2', 'unlabeled_train'), map_location='cpu', weights_only=False)
        metadata_cache = torch.load(train.get_metadata_cache_path(str(run), 'mark5.2', 'unlabeled_train'), map_location='cpu', weights_only=False)
        require(len(feature_cache) > 0 and len(metadata_cache) > 0, 'Teacher cache was not saved')
        cfg = BaseConfig(mark_version='mark5.2')
        cfg.experiment_seed = 1
        cfg.unlabeled_n = 7
        cfg.apply_cli_ablation(**{k: v for k, v in args_run.items() if k not in ('mark_version', 'seed_value', 'unlabeled_n', 'lr', 'encoder_lr')})
        require(not cfg.distill_branch_classify and cfg.use_feature_kd and '_SINGLE' in cfg.run_tag, 'Invalid new model configuration')
        require(cfg.run_tag.endswith('_PD_u7_s1'), f'Prodigy run_tag missing: {cfg.run_tag}')
        checkpoint_path = run / 'model' / f'student_model_{cfg.run_tag}.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        require(checkpoint['config_metadata']['classification_policy'] == 'supervised_argmax_v1', 'Checkpoint policy missing')
        require(not checkpoint['config_metadata']['distill_branch_classify'], 'Wrong classification provenance')
        require(checkpoint['config_metadata'].get('optimizer_name') == 'prodigy' and checkpoint['config_metadata'].get('learning_rate') == 1.0,
                'Checkpoint optimizer provenance missing')
        require(not any(k.startswith('m53_') for k in checkpoint['config_metadata']), 'Removed component metadata remains')
        history = list(csv.DictReader((run / 'plots' / f'loss_history_{cfg.run_tag}.csv').open()))
        require(len(history) == 1, 'Smoke run exceeded one epoch')

        evaluate.PROJECT_ROOT, evaluate.BASE_DIR = str(run), str(run / 'model')
        evaluate._resolve_dataset_csv_path = lambda split: str(run / f'dataset_{split}.csv')
        sentinel = run / 'plots' / f'others_threshold_{cfg.run_tag}.json'
        sentinel.write_text('INVALID OLD CALIBRATION JSON: MUST NEVER BE READ', encoding='utf-8')
        sentinel_hash = checkpoint_sha256(sentinel)
        for split in ('val', 'test'):
            evaluate.evaluate(split=split, **args_run)
        require(checkpoint_sha256(sentinel) == sentinel_hash, 'Old calibration was changed')

        outputs = {}
        for split, suffix in [('val', '_val_plain'), ('test', '_plain')]:
            rows = list(csv.DictReader((run / 'plots' / f'prediction_details_{cfg.run_tag}{suffix}.csv').open()))
            require(len(rows) == 9, 'Evaluation fixture was partially skipped')
            for row in rows:
                probabilities = [float(row['Prob_' + name]) for name in classes]
                expected = classes[int(np.argmax(probabilities))]
                require(row['Predicted Label'] == row['Raw Predicted Label'] == expected, 'Prediction differs from supervised argmax')
                require(row['Forced To Others'] == 'False' and row['Abstained'] == 'False', 'Postprocessing changed prediction')
            outputs[split] = rows
        require(outputs['val'] == outputs['test'], 'Validation and test routes disagree on identical input')

        runner.PROJECT_ROOT = str(run)
        require(runner._needs('mark5.2', cfg.run_tag, False, False) == (False, False, False), 'Completed new run not recognized')
        artifacts = runner._artifacts('mark5.2', cfg.run_tag)
        test_policy = Path(artifacts['test_policy'])
        good = test_policy.read_bytes()
        bad = json.loads(good)
        bad['checkpoint']['sha256'] = 'stale'
        test_policy.write_text(json.dumps(bad))
        require(runner._needs('mark5.2', cfg.run_tag, False, False) == (False, False, True), 'Stale checkpoint evaluation reused')
        test_policy.write_bytes(good)
        aggregate.PROJECT_ROOT, aggregate.PLOT_DIR = str(run), str(run / 'plots')
        aggregate._FULL_LABELED = None
        df = aggregate.collect('mark5.2', 'test')
        require(len(df) == 1 and df.iloc[0]['run_tag'] == cfg.run_tag, 'Collector did not select new result')
        require(len(aggregate.collect('mark5.2', 'val')) == 1, 'Validation collector mismatch')
        # Legacy output with a complete CSV must never enter the new result table.
        legacy_tag = cfg.run_tag.replace('_SINGLE', '')
        shutil.copyfile(artifacts['test'], run / 'plots' / f'performance_summary_{legacy_tag}_plain.csv')
        require(len(aggregate.collect('mark5.2', 'test')) == 1, 'Legacy output entered new table')
        cw, cm, _, n = aggregate.classwise_and_confusion(df, 'test')
        require(n == 1 and int(cm.to_numpy().sum()) == 9 and len(cw) == 9, 'Classwise collector reads wrong filenames')
        reports.append({'path': path_name, 'run_tag': cfg.run_tag, 'training_epochs': 1, 'labeled_clips': 9,
                        'unlabeled_clips': 7, 'evaluation_fixture_clips': 9, 'checkpoint_sha256': checkpoint_sha256(checkpoint_path)})

    # Inspect real command construction without launching another training process.
    commands = []
    argv = ['run_experiments.py', '--mark_version', 'mark5.2', '--seeds', '1', '--labeled_ns', '0',
            '--unlabeled_ns', '2000', '--no_log',
            '--extra', '--linear_head --head_pretrained_hidden --student_raw_wave --joint_forward']
    with patch.object(sys, 'argv', argv), patch.object(runner, '_needs', return_value=(True, True, True)), \
            patch.object(runner, '_sh', side_effect=lambda cmd, *a: commands.append(cmd) or True), \
            contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    require(len(commands) == 3, 'Runner did not build train/val/test commands')
    require('model/train_mark5.py' in commands[0] and '--plain_output' not in commands[0], 'Evaluation flag leaked into training')
    require(all('model/eval.py' in cmd and '--plain_output' in cmd for cmd in commands[1:]), 'Runner omitted plain evaluation')
    require(observations['adapter_grad_nonzero'] >= 2 and observations['encoder_grad_nonzero'] >= 2, 'Training gradient path did not execute twice')
    require(observations['validation_batches'] > 0, 'Validation path not checked')
    for teacher in teachers:
        for entry in teacher.specialists.values():
            for value in entry.values():
                if isinstance(value, torch.nn.Module):
                    require(all(p.grad is None for p in value.parameters()), 'Teacher received gradients')
    report = {'status': 'passed', 'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
              'scope': 'implementation verification only; two single-epoch CPU smoke runs, not accuracy evaluation',
              'held_out_test_used': False, 'loss_formula_and_gradient_checks': True, 'inference_adapter_unused': True,
              'classifier_only_supervised_features': True, 'calibration_sentinel_unread': True,
              'validation_test_route_identity': True, 'stale_evaluation_rejected': True,
              'legacy_results_excluded': True, 'runner_command_routing': True, 'teacher_cache_saved': True,
              'observations': observations, 'runs': reports}
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
