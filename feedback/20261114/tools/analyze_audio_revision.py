"""Read-only inference sensitivity audit of the one historical U clip revision.

No training, optimizer step, checkpoint overwrite or original score mutation.
Pass --drive-input with the downloaded original data_unlabeled/1358 WAV.
"""
from pathlib import Path
import argparse,hashlib,io,json,sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
BASE=Path(__file__).resolve().parents[1]
CLASSES=['heavy_impact','dragging','construction','machine_noise','talking','water_toilet','water_shower','dog_bark','others']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def decision(q):
    order=np.argsort(-q,kind='stable');a,b=order[:2];one=q[a]>=.9;two=not one and q[a]+q[b]>=.95
    return dict(first=CLASSES[a],second=CLASSES[b] if two else '',weight=float(q[a] if one else q[a]+q[b] if two else 0),accepted=bool(one or two))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BASE.parents[1]);p.add_argument('--drive-input',type=Path,required=True);p.add_argument('--corrected-source',type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.root));sys.path.insert(0,str(a.root/'model'));sys.path.insert(0,str(BASE/'src'))
    from make_teacher_scores import Mn10Scorer
    from feedback_core import meta_distribution
    torch.set_num_threads(2)
    old_hash='667f1910ac25365ea4c5f6383dec129eb594a98e4545d34ee342515df57eaa69'
    assert sha(a.drive_input)==old_hash
    source=a.corrected_source or a.root.parent/'dataset_releases/unlabeled_1358_restore_20261114/unlabeled_1358_crop8-11_before_restore.wav'
    assert sha(source)=='e51b1f160623cc39d86bbafdbc66bebcae228170a1b74305290a71ac5e1d6f3a'
    before=a.root.parent/'dataset_releases/mark53_test_independent_20261022/silence_review_737/originals_before_replacement/unlabeled_1358_unlabeled_1358.wav.before.wav'
    wave,sr=sf.read(before,dtype='float32');b=io.BytesIO();sf.write(b,wave,sr,format='WAV',subtype='PCM_16')
    assert hashlib.sha256(b.getvalue()).hexdigest()==old_hash
    wave,sr=sf.read(source,dtype='float32');b=io.BytesIO();sf.write(b,wave,sr,format='WAV',subtype='PCM_16')
    corrected=b.getvalue();new_hash=hashlib.sha256(corrected).hexdigest()
    import tempfile
    with tempfile.TemporaryDirectory(prefix='kampa-clip-inference-') as d:
        new=Path(d)/'corrected_pcm16.wav';new.write_bytes(corrected)
        scorer=Mn10Scorer(CLASSES[:8],str(a.root/'resources'),torch.device('cpu'))
        probs=scorer([str(a.drive_input),str(new)]).numpy()
    frame=pd.read_csv(a.root/'plots/teacher_scores_mn10.csv',keep_default_na=False)
    cached=frame[frame.path.str.endswith('/unlabeled_1358.wav')][['p_'+c for c in CLASSES[:8]]].to_numpy()[0]
    comparisons=[]
    for n in [200,500,1000,2000]:
        for seed in range(1,6):
            tag='all' if n==2000 else str(n)
            meta=json.loads((a.root/f'plots/combiner/combiner_l{tag}_s{seed}.json').read_text())
            q=meta_distribution(np.stack([cached,probs[0],probs[1]]),meta['coef'],meta['intercept'])
            old=decision(q[0]);cpuold=decision(q[1]);new=decision(q[2])
            csv=pd.read_csv(a.root/f'pseudo_labels_cmb_l{tag}_s{seed}.csv',keep_default_na=False)
            selected=csv[csv.path.str.endswith('/unlabeled_1358.wav')]
            assert old['accepted']==bool(len(selected))
            if len(selected):assert old['first']==selected.iloc[0].label
            comparisons.append(dict(n=n,seed=seed,historical=old,historical_cpu_replay=cpuold,corrected_inference=new,
                                    target_changed=(old['first'],old['second'],old['accepted'])!=(new['first'],new['second'],new['accepted']),
                                    q_historical=q[0].tolist(),q_corrected=q[2].tolist()))
    report=dict(filename='unlabeled_1358.wav',scope='Frozen teacher inference and existing pseudo-label inspection only; no student retraining',
        source_group='aihub:N-10_221006_A_5_a_16484.wav',historical_input_sha256=old_hash,corrected_pcm16_sha256=new_hash,
        before_source_sha256=sha(before),revised_source_sha256=sha(source),old_source_conversion_matches_drive=True,
        teacher_files=scorer.files,classes=CLASSES,teacher_cached=cached.tolist(),teacher_old_cpu=probs[0].tolist(),teacher_corrected_cpu=probs[1].tolist(),
        max_old_replay_probability_error=float(np.abs(probs[0]-cached).max()),historical_selected_runs=sum(r['historical']['accepted'] for r in comparisons),
        corrected_selected_runs=sum(r['corrected_inference']['accepted'] for r in comparisons),changed_targets=sum(r['target_changed'] for r in comparisons),
        comparisons=comparisons,limitations=['CPU re-inference may differ numerically from historical GPU inference.',
        'All U clips pass through joint-forward BatchNorm, including clips whose pseudo-label loss is zero.',
        'Final test-metric counterfactual cannot be established without controlled student retraining.',
        'The 430 test files and already saved predictions are not modified by this audit.'])
    (BASE/'review/unlabeled_1358_impact.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['comparisons','teacher_files']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
