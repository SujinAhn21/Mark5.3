import sys,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
import feedback_core as f
ROOT=Path(os.environ.get('KAMPA_PROJECT_ROOT',BASE.parents[1]))

@pytest.fixture(scope='module')
def inputs():return f.load_inputs(ROOT,legacy_scores=True)

def test_full_probabilities_reproduce_twenty_original_label_files(inputs):
    sets,scores,_=inputs
    for n in [200,500,1000,2000]:
        for seed in range(1,6):
            lab=f.select_labeled(sets['train'],n,seed)
            q,_,_=f.get_q(ROOT,scores,lab,sets['unlabeled'],n,seed)
            targets=f.make_targets(q,'kampa');active=targets['weight']>0
            old=pd.read_csv(ROOT/f'pseudo_labels_cmb_l{"all" if n==2000 else n}_s{seed}.csv',keep_default_na=False)
            old['filename']=old.path.map(lambda p:Path(p).name);old=old.set_index('filename')
            names=sets['unlabeled'].filename[active].tolist()
            assert set(names)==set(old.index)
            assert [f.CLASSES[i] for i in targets['first'][active]]==old.loc[names,'label'].tolist()
            assert [f.CLASSES[i] if i>=0 else '' for i in targets['second'][active]]==old.loc[names,'label2'].tolist()
            np.testing.assert_allclose(targets['weight'][active],old.loc[names,'confidence'],atol=1e-6)

def test_factorial_and_soft_controls_preserve_intended_factors():
    q=np.array([[.92,.01,.01,.01,.01,.01,.01,.01,.01],[.6,.36,.01,.01,.005,.005,.005,.005,0],[.4,.2,.1,.1,.05,.05,.05,.05,0]])
    a=f.make_targets(q,'kampa');b=f.make_targets(q,'soft_selected');c=f.make_targets(q,'meta_top1_matched');d=f.make_targets(q,'meta_single')
    np.testing.assert_array_equal(a['weight'],b['weight']);np.testing.assert_array_equal(a['weight'],c['weight'])
    assert a['second'].tolist()==[-1,1,-1]
    assert (d['weight']>0).tolist()==[True,False,False]
    assert (f.make_targets(q,'soft_all')['weight']==1).all()
    fq=f.fixed_distribution(np.array([[.99,.001,.001,.001,.001,.001,.001,.001],[.001]*8]))
    assert fq.argmax(1).tolist()==[0,8]
    np.testing.assert_allclose(fq.sum(1),1)

def test_losses_match_closed_form_and_silent_gradients():
    logits=torch.tensor([[2.,1.,0.],[0.,1.,2.]],requires_grad=True)
    q=torch.tensor([[.6,.3,.1],[.1,.2,.7]])
    a=torch.tensor([0,2]);b=torch.tensor([1,-1]);w=torch.tensor([.9,.7])
    loss=f.target_loss(logits,q,a,b,w)
    prob=logits.softmax(1)
    expected=(-.9*torch.log(prob[0,0]+prob[0,1])-.7*torch.log(prob[1,2]))/1.6
    torch.testing.assert_close(loss,expected);loss.backward();assert torch.isfinite(logits.grad).all()
    soft=f.target_loss(logits,q,a,b,w,True)
    torch.testing.assert_close(soft,(-(q*logits.log_softmax(1)).sum(1)*w).sum()/w.sum())
    logits.grad=None;f.target_loss(logits,q,a,b,w*0).backward();assert (logits.grad==0).all()

def test_oracle_does_not_borrow_student_hidden_labels(inputs):
    sets,_,prov=inputs;o,a=f.oracle_pool(sets,prov,ROOT.parent)
    assert len(a)==16000 and len(o)==12477 and o.sha256.is_unique
    assert set(o.label)==set(f.CLASSES)
    assert set(o.group).isdisjoint(set(pd.concat([sets['val'],sets['test']]).group))
    assert not o.evidence.str.contains('student').any()
    assert (o.split=='train').all()

@pytest.mark.parametrize('seed',[101,102,103])
def test_new_partitions_exclude_teacher_and_u_exposures(inputs,seed):
    sets,_,prov=inputs;s=f.grouped_resplit(sets,prov,seed,2000)
    assert not any(f.source_audit(s).values())
    protected=set(prov[(prov.version!='mark5.2')&prov.split.isin(['train','val'])].group)|set(sets['unlabeled'].group)
    for split in ['val','test']:assert set(s[split].group).isdisjoint(protected)
    assert len(f.select_labeled(s['train'],2000,3))==2000
    assert len(s['unlabeled'])==2000
    assert sum(len(s[x]) for x in ['train','val','test'])==2860

def test_models_can_forward_and_backward_without_training_audio():
    torch.set_num_threads(2)
    for kind in ['kampa','efficientat']:
        model=f.build_model(ROOT,kind).eval()
        wave=torch.randn(2,48000)*.01
        z=model(wave);assert z.shape==(2,9) and torch.isfinite(z).all()
        torch.nn.functional.cross_entropy(z,torch.tensor([0,1])).backward()
        assert torch.isfinite(model.classifier.linear.weight.grad if kind=='kampa' else model.classifier.weight.grad).all()
        # No optimizer step; these are implementation checks, not research runs.

def test_plan_counts_and_input_ground_truth_hidden(inputs):
    sets,_,_=inputs
    assert (sets['unlabeled'].label=='').all()
    assert len(f.recipe_plan('1_3',[200],range(1,6)))==45
    assert len(f.recipe_plan('4',[200],range(1,6)))==30
