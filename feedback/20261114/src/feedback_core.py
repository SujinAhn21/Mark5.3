"""KAMPA professor-feedback experiments, frozen protocol 2026-11-14.

Only reads the original Mark5.3 project. All new artifacts go to a separate
versioned output directory. New controls use this same runner; historical
results are not silently substituted for these controls.
"""
from pathlib import Path
from collections import Counter, defaultdict
from types import SimpleNamespace
import csv, hashlib, io, json, os, random, re, sys, time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CLASSES=['heavy_impact','dragging','construction','machine_noise','talking','water_toilet','water_shower','dog_bark','others']
TAU1=.9
TAU2=.95
VARIANTS=['fixed_single','fixed_set','meta_single','kampa','meta_top1_matched','soft_selected','soft_all']
VERSION='20261114-v2'
COMPLETE_FILES=['best.pt','metrics.json','test_predictions.csv','val_predictions.csv','history.csv','protocol.json',
                'train_manifest.csv','val_manifest.csv','test_manifest.csv','pseudo_targets.csv','meta_classifier.json']

def runtime_metadata():
    """CPU is supported; record and separate hardware/library configurations."""
    import importlib.metadata,platform,subprocess
    import torch
    info=dict(device='cpu',device_name=platform.processor() or 'CPU',capability=None,
              python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(),driver=None)
    for dist in ['torchaudio','torchvision','numpy','pandas','scipy','scikit-learn','soundfile']:
        info[dist]=importlib.metadata.version(dist)
    if torch.cuda.is_available():
        cap=torch.cuda.get_device_capability(0)
        if cap[0] in (10,12):
            raise RuntimeError('The recorded project protocol excludes Blackwell runtimes (sm_100/sm_120 families). Select L4 or CPU.')
        info.update(device='cuda',device_name=torch.cuda.get_device_name(0),capability=list(cap))
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True)
            info['driver']=r.stdout.strip().splitlines()[0]
        except (OSError,subprocess.CalledProcessError,IndexError):info['driver']='unavailable'
    return info

def validate_complete(folder,identity=None):
    folder=Path(folder);p=folder/'complete.json'
    if not p.is_file():return None
    complete=json.loads(p.read_text())
    if identity is not None and complete.get('identity')!=identity:raise RuntimeError('Completion identity mismatch')
    files=complete.get('files',{})
    if set(files)!=set(COMPLETE_FILES):raise RuntimeError(f'Incomplete artifact manifest: {folder}')
    for name,h in files.items():
        if not (folder/name).is_file() or digest(folder/name)!=h:raise RuntimeError(f'Completed artifact damaged: {folder/name}')
    spec=json.loads((folder/'protocol.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    if spec['identity']!=complete['identity'] or metrics['identity']!=complete['identity']:
        raise RuntimeError(f'Completed artifacts belong to different runs: {folder}')
    return complete

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def write_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False,encoding='utf-8-sig');os.replace(tmp,path)

def select_labeled(frame,n,seed):
    if n==0 or n==len(frame):return frame.copy().reset_index(drop=True)
    if not 9<=n<=len(frame):raise ValueError(f'Invalid labeled budget: {n}/{len(frame)}')
    rng=random.Random(seed);parts=[]
    for ci,c in enumerate(CLASSES):
        ids=frame.index[frame.label==c].tolist();rng.shuffle(ids)
        take=n//9+int(ci<n%9)
        if len(ids)<take:raise ValueError(f'{c}: {len(ids)} labels available, {take} required')
        parts+=ids[:take]
    return frame.loc[parts].reset_index(drop=True)

def fixed_distribution(p):
    """Nonlearned 9-class closure: q_k = odds(p_k)/(1+sum odds), q_other=1/Z.

    This is the independent-binary likelihood conditional on zero or one
    foreground class. It is a specified baseline assumption, not a calibrated
    posterior. No test-tuned thresholds or fitted parameters are introduced.
    """
    p=np.clip(np.asarray(p,float),1e-4,1-1e-4);odds=p/(1-p)
    q=np.column_stack([odds,np.ones(len(p))]);return q/q.sum(1,keepdims=True)

def meta_distribution(p,coef,intercept):
    p=np.clip(np.asarray(p,float),1e-4,1-1e-4)
    z=np.log(p/(1-p))@np.asarray(coef,float).T+np.asarray(intercept,float)
    z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)

def make_targets(q,variant):
    order=np.argsort(-q,axis=1,kind='stable');a=order[:,0];b=order[:,1];ii=np.arange(len(q))
    q1=q[ii,a];mass=q1+q[ii,b];one=q1>=TAU1;two=(~one)&(mass>=TAU2)
    active=one|two;weights=np.where(one,q1,mass)
    if variant.endswith('_single'):active=one;two=np.zeros(len(q),bool)
    if variant=='soft_all':active=np.ones(len(q),bool);weights=np.ones(len(q))
    # meta_top1_matched preserves accepted rows and confidence weights.
    if variant in ('meta_top1_matched','soft_selected','soft_all'):two=np.zeros(len(q),bool)
    weights=np.where(active,weights,0.)
    return dict(q=q.astype('float32'),first=a,second=np.where(two,b,-1),weight=weights.astype('float32'))

def target_loss(logits,q,first,second,weight,soft=False):
    import torch
    import torch.nn.functional as F
    keep=weight>0
    if not keep.any():return logits.sum()*0
    lp=F.log_softmax(logits[keep],dim=1);w=weight[keep]
    if soft:per=-(q[keep]*lp).sum(1)
    else:
        a=first[keep];b=second[keep];idx=torch.arange(len(a),device=lp.device)
        l1=lp[idx,a];l2=lp[idx,b.clamp_min(0)]
        per=torch.where(b>=0,-torch.logaddexp(l1,l2),-l1)
    return (per*w).sum()/w.sum().clamp_min(1e-6)

def score_metrics(y,pred):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
    pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=list(range(9)),average='macro',zero_division=0)
    cm=confusion_matrix(y,pred,labels=list(range(9)))
    return dict(accuracy=float(accuracy_score(y,pred)),macro_precision=float(pr),macro_recall=float(re),macro_f1=float(f1),
                confusion=cm.tolist(),n=len(y))

def resolve_path(value,exp):
    p=Path(value)
    if p.is_file():return p
    tail=value.replace('\\','/').split('/experiments/')[-1]
    m=re.search(r'(Mark[45]\.\d/.+)$',tail)
    if m:tail=m.group(1)
    candidates=[Path(exp)/tail]
    for old,new in [('data_labeled/','data_source_labeled/'),('data_val/','data_source_val/'),('data_test/','data_source_test/'),('data_unlabeled/','data_source_unlabeled/')]:
        if old in tail:candidates.append(Path(exp)/tail.replace(old,new,1))
    if tail.startswith('Mark4.'):
        candidates+=[Path(exp)/'Mark4_series'/tail,Path(exp).parent/'Mark4_series'/tail]
    for c in candidates:
        if c.is_file():return c
    raise FileNotFoundError(f'Audio not found: {value}; searched {candidates}')

def load_inputs(root,legacy_scores=False):
    root=Path(root);expected=json.loads((HERE/'input_hashes.json').read_text())
    for rel,sha in expected.items():
        if rel.startswith('plots/') and not legacy_scores:continue
        if not (root/rel).is_file() or digest(root/rel)!=sha:
            raise RuntimeError(f'Frozen input changed or missing: {root/rel}. Rebuild the protocol against the new data version.')
    prov=pd.read_csv(HERE/'provenance_active.csv',keep_default_na=False)
    own=prov[prov.version=='mark5.2'].set_index('filename')
    prepared=pd.read_csv(HERE/'audio_inputs.csv',keep_default_na=False).set_index('filename')
    if not prepared.index.is_unique:raise ValueError('Input audio filenames are not unique')
    sets={}
    for split,file in [('train','dataset_labeled.csv'),('val','dataset_val.csv'),('test','dataset_test.csv'),('unlabeled','dataset_unlabeled.csv')]:
        frame=pd.read_csv(root/file,keep_default_na=False)
        frame['filename']=frame.path.map(lambda x:Path(x).name)
        assert frame.filename.is_unique
        rows=own.loc[frame.filename]
        assert (rows.split==split).all()
        if split!='unlabeled':assert frame.label.tolist()==rows.label.tolist()
        else:frame['label']=''  # Do not expose the hidden U ground truth to training.
        for col in ['group','recording']:frame[col]=rows[col].to_numpy()
        frame['source_sha256']=rows.sha256.to_numpy()
        actual=prepared.loc[frame.filename]
        if actual.path.tolist()!=frame.path.tolist() or actual.source_sha256.tolist()!=frame.source_sha256.tolist():
            raise ValueError('Prepared audio manifest does not match the source/index snapshot')
        frame['sha256']=actual.input_sha256.to_numpy()
        frame['input_encoding']='PCM_16'
        for col in ['conversion_source_sha256','conversion_source_path']:frame[col]=actual[col].to_numpy()
        frame['origin_split']=split
        sets[split]=frame.reset_index(drop=True)
    scores=None
    if legacy_scores:
        scores=pd.read_csv(root/'plots/teacher_scores_mn10.csv',keep_default_na=False)
        scores['filename']=scores.path.map(lambda x:Path(x).name)
        assert scores.filename.is_unique
        scores=scores.set_index('filename')
    return sets,scores,prov

def source_audit(sets):
    return {a+'__'+b:len(set(sets[a].group)&set(sets[b].group))
            for a,b in [('train','val'),('train','test'),('unlabeled','val'),('unlabeled','test'),('val','test')]}

def oracle_pool(sets,prov,exp):
    """Teacher TRAIN only, mapped by original provenance, deduplicated by hash.

    Never use binary 'others' as a nine-class label without provenance evidence.
    Exclude all original student evaluation recording/session groups.
    """
    t=prov[(prov.version!='mark5.2')&(prov.split=='train')].copy()
    if (~t.label.isin(CLASSES)).any():
        raise ValueError('Some teacher TRAIN labels remain unresolved. See provenance_active.csv.')
    forbidden=set(pd.concat([sets['val'],sets['test']]).group)
    forbidden_hash=set(pd.concat([sets['val'],sets['test']]).source_sha256)
    t['exclude_reason']=''
    t.loc[t.group.isin(forbidden),'exclude_reason']='student_eval_source_group'
    t.loc[t.sha256.isin(forbidden_hash),'exclude_reason']='student_eval_file_hash'
    eligible=t[t.exclude_reason==''].copy()
    conflict=eligible.groupby('sha256').label.nunique()
    if (conflict>1).any():raise ValueError('Teacher labels conflict for identical file content')
    duplicate=eligible.duplicated('sha256',keep='first')
    t.loc[eligible.index[duplicate],'exclude_reason']='duplicate_teacher_file_hash'
    eligible=eligible[~duplicate].copy()
    eligible['path']=[str(Path(exp)/('Mark'+r.version[4:])/'data/train'/r.filename) for r in eligible.itertuples()]
    eligible['source_sha256']=eligible.sha256
    eligible['input_encoding']='original'
    eligible['origin_split']='teacher_train'
    return eligible.reset_index(drop=True),t

def grouped_resplit(sets,prov,split_seed,max_budget=2000):
    """New train/val/test partitions; teachers and original U remain frozen.

    Only groups unseen in any teacher train/val and absent from the original U
    are eligible for held-out data. Group sizes vary; actual counts are saved.
    Approximately 35 clips/class/split are sought, without using model scores.
    """
    pool=pd.concat([sets[s] for s in ('train','val','test')],ignore_index=True)
    teacher=prov[(prov.version!='mark5.2')&prov.split.isin(['train','val'])]
    protected=set(teacher.group)|set(sets['unlabeled'].group)
    groups={g:f for g,f in pool.groupby('group',sort=True) if g not in protected}
    total=Counter(pool.label);remaining=total.copy();chosen={'val':set(),'test':set()};counts={s:Counter() for s in chosen}
    rng=random.Random(split_seed);keys=list(groups);rng.shuffle(keys)
    required={c:max_budget//9+int(i<max_budget%9) for i,c in enumerate(CLASSES)}
    for _ in range(2):
        for g in keys:
            if any(g in v for v in chosen.values()):continue
            cc=Counter(groups[g].label)
            if any(remaining[c]-n<required[c] for c,n in cc.items()):continue
            scores={s:sum(min(n,max(0,35-counts[s][c])) for c,n in cc.items()) for s in chosen}
            best=max(chosen,key=lambda s:(scores[s],-sum(counts[s].values()),s))
            if scores[best]<=0:continue
            # Avoid assigning a large mixed session when most of its clips are surplus.
            if len(groups[g])>2*scores[best]:continue
            chosen[best].add(g);counts[best].update(cc);remaining.subtract(cc)
    used=chosen['val']|chosen['test']
    result={'train':pool[~pool.group.isin(used)].copy(),'val':pool[pool.group.isin(chosen['val'])].copy(),
            'test':pool[pool.group.isin(chosen['test'])].copy(),'unlabeled':sets['unlabeled'].copy()}
    for s in ('val','test'):
        if any(counts[s][c]<10 for c in CLASSES):raise ValueError(f'Insufficient safe holdout groups: split={split_seed}, {s}, {counts[s]}')
        assert not set(result[s].group)&protected
    assert not any(source_audit(result).values()),source_audit(result)
    select_labeled(result['train'],max_budget,1)  # Enforce required per-class availability.
    return {s:f.reset_index(drop=True) for s,f in result.items()}

def get_q(root,scores,labeled,unlabeled,n,seed,refit=False):
    cols=['p_'+c for c in CLASSES[:8]]
    x=scores.loc[labeled.filename,cols].to_numpy(float)
    u=scores.loc[unlabeled.filename,cols].to_numpy(float)
    if refit:
        from sklearn.linear_model import LogisticRegression
        z=np.log(np.clip(x,1e-4,1-1e-4)/(1-np.clip(x,1e-4,1-1e-4)))
        y=np.array([CLASSES.index(c) for c in labeled.label])
        model=LogisticRegression(C=1.,max_iter=5000).fit(z,y)
        if model.classes_.tolist()!=list(range(9)):raise ValueError('Meta class order mismatch')
        coef,intercept=model.coef_,model.intercept_
    else:
        meta=json.loads((Path(root)/'plots/combiner'/f'combiner_l{"all" if n==2000 else n}_s{seed}.json').read_text())
        assert meta['classes']==CLASSES and meta['experts']==CLASSES[:8]
        assert meta['scores_sha256']==digest(Path(root)/'plots/teacher_scores_mn10.csv')
        assert meta['labeled_paths_sha256']==hashlib.sha256('\n'.join(sorted(labeled.path)).encode()).hexdigest(), 'Meta classifier uses a different labeled subset'
        coef,intercept=meta['coef'],meta['intercept']
    return meta_distribution(u,coef,intercept),fixed_distribution(u),dict(coef=np.asarray(coef).tolist(),intercept=np.asarray(intercept).tolist())

def recipe_plan(stage,budgets,seeds,split_seeds=(101,102,103)):
    plan=[]
    if stage=='1_3':
        for n in budgets:
            for seed in seeds:
                for variant in VARIANTS+['supervised','oracle']:
                    plan.append(dict(stage=stage,n=n,seed=seed,variant=variant,split_seed=0))
    elif stage=='4':
        for sp in split_seeds:
            for n in budgets:
                for seed in seeds:
                    for variant in ['kampa','efficientat']:
                        plan.append(dict(stage=stage,n=n,seed=seed,variant=variant,split_seed=sp))
    else:raise ValueError(stage)
    return plan

def prepare(config):
    if config.get('stage') not in ('1_3','4'):raise ValueError('stage must be 1_3 or 4')
    if not config.get('budgets') or any(n not in (200,500,1000,2000) for n in config['budgets']):
        raise ValueError('budgets must contain 200, 500, 1000 and/or 2000')
    if not config.get('seeds') or any(type(s) is not int or not 1<=s<=5 for s in config['seeds']):
        raise ValueError('This protocol supports training seeds 1 through 5')
    if type(config.get('epochs',80)) is not int or config.get('epochs',80)<1:raise ValueError('epochs must be positive')
    if config['stage']=='4' and not config.get('split_seeds'):raise ValueError('stage 4 requires split_seeds')
    root=Path(config['root']);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    sets,scores,prov=load_inputs(root)
    audit=dict(original_source_overlap=source_audit(sets),version=VERSION)
    if any(audit['original_source_overlap'].values()):
        raise RuntimeError(f'Original student split source overlap detected: {audit}')
    oracle,excluded=oracle_pool(sets,prov,root.parent)
    write_csv(out/'audit/teacher_oracle_pool.csv',oracle)
    write_csv(out/'audit/teacher_oracle_audit.csv',excluded)
    audit['teacher_rows']=len(excluded);audit['oracle_unique_rows']=len(oracle)
    audit['oracle_original_files']=int((~oracle.augmented).sum())
    audit['oracle_augmented_files']=int(oracle.augmented.sum())
    audit['oracle_source_groups']=int(oracle.group.nunique())
    audit['oracle_exclusions']=excluded.exclude_reason.value_counts().to_dict()
    audit['oracle_class_counts']=oracle.label.value_counts().to_dict()
    audit['original_val_teacher_exposed_groups']=len(set(sets['val'].group)&set(prov[(prov.version!='mark5.2')&prov.split.isin(['train','val'])].group))
    partitions={0:sets}
    for sp in config.get('split_seeds',[]):
        partitions[sp]=grouped_resplit(sets,prov,sp,max(2000,max(config['budgets'])))
        audit[f'split_{sp}']={s:dict(n=len(f),groups=f.group.nunique(),classes=f.label.value_counts().to_dict()) for s,f in partitions[sp].items() if s!='unlabeled'}
        for s,f in partitions[sp].items():write_csv(out/f'splits/{sp}/{s}.csv',f)
    dump(out/'audit/preflight.json',audit)
    plan=recipe_plan(config['stage'],config['budgets'],config['seeds'],config.get('split_seeds',[]))
    write_csv(out/'run_plan.csv',pd.DataFrame(plan))
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(f'PLANNED RUNS: {len(plan)}',flush=True)
    return partitions,scores,oracle,plan

def audit_audio(config):
    """Check Drive audio versions before committing time to any training run."""
    from concurrent.futures import ThreadPoolExecutor
    root=Path(config['root']);sets,_,prov=load_inputs(root)
    frames=list(sets.values())
    if config['stage']=='1_3':frames.append(oracle_pool(sets,prov,root.parent)[0])
    frame=pd.concat(frames,ignore_index=True).drop_duplicates('path')
    def check(r):
        try:
            audio,p=verified_audio(r,root.parent)
            return dict(filename=r.filename,path=str(p),matches=True,input_sha256=r.sha256,source_sha256=r.source_sha256)
        except Exception as e:return dict(filename=r.filename,path=r.path,matches=False,error=str(e))
    checked=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(check,frame.itertuples(index=False))):
            checked.append(r)
            if (i+1)%1000==0:print(f'[audio audit] {i+1}/{len(frame)}',flush=True)
    result=pd.DataFrame(checked);write_csv(Path(config['output'])/'audit/audio_files.csv',result)
    if not result.matches.all():raise RuntimeError(f'{int((~result.matches).sum())} audio files differ or are absent. See audit/audio_files.csv.')
    print(f'Audio verified: {len(result)} files',flush=True)
    return result

def build_model(root,kind='kampa',pretrained=True):
    import torch
    from torch import nn
    from vendor.vild_model import Mn10AudioEncoder,LinearClassHead
    from vendor.vild_head import DualBranchStudentHead
    cfg=SimpleNamespace(embedding_dim=384,mn10_pretrained_path=str(Path(root)/'weights/mn10_as_mAP_471.pt'),
        mn10_load_pretrained=pretrained,head_pretrained_hidden=True,sample_rate=16000,
        spec_augment_freq_mask_param=8,n_mels=64,spec_augment_time_mask_param=16,segment_length=101,
        spec_augment_num_freq_masks=2,spec_augment_num_time_masks=2)
    class Student(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=Mn10AudioEncoder(cfg)
            if kind=='efficientat':
                hidden=self.encoder.proj[0].in_features;self.encoder.proj=nn.Identity()
                self.branch=nn.Identity();self.classifier=nn.Linear(hidden,9)
            else:
                self.branch=DualBranchStudentHead(384,supervised_identity=True)
                self.classifier=LinearClassHead(384,9)
        def forward(self,wave,augment=False):
            z=self.encoder(wave,augment=augment)
            z=self.branch(z) if kind=='efficientat' else self.branch.forward_supervised(z)
            return self.classifier(z)
    return Student()

def verified_audio(row,experiments):
    """Return verified model-input bytes, never silently substitute source bytes.

    USB holds source FLOAT WAVs. Colab uses prepared PCM_16 WAVs. Only the
    explicitly frozen conversion is accepted, and its resulting byte SHA must
    equal the actual Drive input SHA. Oracle teacher files use original bytes.
    """
    import soundfile as sf
    path=resolve_path(row.path,experiments);data=path.read_bytes()
    actual=hashlib.sha256(data).hexdigest()
    if actual==row.sha256:return io.BytesIO(data),path
    if getattr(row,'input_encoding','')!='PCM_16' or actual!=row.source_sha256:
        raise RuntimeError(f'Audio differs from frozen input/source: {path}')
    revision_path=getattr(row,'conversion_source_path','')
    if revision_path:
        relative=Path(revision_path)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid historical source path')
        path=Path(experiments)/relative
        if not path.is_file():raise FileNotFoundError(f'Historical source backup unavailable; supply the frozen prepared Drive input: {row.path}')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row.conversion_source_sha256:raise RuntimeError(f'Historical source changed: {path}')
    wave,sr=sf.read(io.BytesIO(data),dtype='float32')
    if sr!=16000 or wave.ndim!=1:raise ValueError(f'Unexpected source format: {path}')
    if len(wave)<48000:wave=np.pad(wave,(0,48000-len(wave)))
    converted=io.BytesIO();sf.write(converted,wave,16000,format='WAV',subtype='PCM_16')
    data=converted.getvalue()
    if hashlib.sha256(data).hexdigest()!=row.sha256:
        raise RuntimeError(f'Source conversion differs from frozen Drive input: {path}')
    return io.BytesIO(data),path

def cached_waves(frame,root,cache_dir):
    """Verify actual input bytes before caching mono 16-kHz 3-second waves."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True);waves=[]
    for i,r in enumerate(frame.itertuples()):
        p=cache/(r.sha256+'.npy')
        stamp=p.with_suffix('.json');valid=False
        if p.exists() and stamp.exists():
            w=np.load(p,allow_pickle=False)
            if w.shape!=(48000,) or not np.isfinite(w).all():raise ValueError(f'Corrupt waveform cache: {p}')
            meta=json.loads(stamp.read_text())
            valid=(meta.get('source_sha256')==r.sha256 and meta.get('preprocessing')=='raw16k3s-v1' and
                   meta.get('wave_sha256')==hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest())
            if not valid:raise ValueError(f'Waveform cache checksum mismatch: {p}; remove that cache entry before retrying')
        if not valid:
            audio,src=verified_audio(r,Path(root).parent)
            w,sr=sf.read(audio,dtype='float32')
            if w.ndim==2:w=w.mean(1 if w.shape[0]>=w.shape[1] else 0)
            if not len(w) or not np.isfinite(w).all():raise ValueError(f'Invalid audio: {src}')
            # Same normalize-then-restore calculation as the existing raw-wave parser.
            t=torch.from_numpy(np.ascontiguousarray(w));peak=float(t.abs().max()) or 1.;t=t/peak
            if sr!=16000:t=AF.resample(t,sr,16000)
            t=t[:48000]*peak
            if t.numel()<48000:t=torch.nn.functional.pad(t,(0,48000-t.numel()))
            w=t.numpy().astype('float32')
            temp=p.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,w)
            os.replace(temp,p)
            dump(stamp,dict(source_sha256=r.sha256,preprocessing='raw16k3s-v1',
                            wave_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
        waves.append(w)
        if i%1000==0:print(f'[audio] {i+1}/{len(frame)}',flush=True)
    return torch.from_numpy(np.stack(waves))

def _atomic_torch_save(path,obj):
    import torch
    p=Path(path);tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def verified_teacher_scores(root,sets):
    """Reuse the frozen score table only after teacher, index and input checks.

    The source FLOAT and actual PCM_16 files have different byte hashes.
    The identity below uses actual prepared-input hashes, in the historical
    split/row order. No new teacher inference or teacher retraining is needed.
    """
    root=Path(root);_,scores,_=load_inputs(root,legacy_scores=True)
    meta=json.loads((root/'plots/teacher_scores_mn10.json').read_text())
    expected=json.loads((HERE/'teacher_hashes.json').read_text())
    got={item['file']:item['sha256'] for item in meta['teacher_files']}
    if got!=expected:raise RuntimeError('Teacher-score metadata refers to a different teacher release')
    for filename,sha in expected.items():
        if not (root/'resources'/filename).is_file() or digest(root/'resources'/filename)!=sha:
            raise RuntimeError(f'Frozen teacher differs or is missing: {filename}')
    if meta.get('limit')!=0 or set(meta['split_counts'])!={'labeled','unlabeled','val','test'}:
        raise RuntimeError('Teacher scores must cover all four full input partitions')
    entries=[];filenames=[]
    for split,count in meta['split_counts'].items():
        frame=sets['train' if split=='labeled' else split]
        original=pd.read_csv(root/('dataset_'+split+'.csv'),keep_default_na=False)
        if len(frame)!=count or len(original)!=count:raise RuntimeError('Teacher-score row count differs')
        for row,source in zip(frame.itertuples(index=False),original.itertuples(index=False)):
            entries.append([split,row.path,getattr(source,'label',''),row.sha256]);filenames.append(row.filename)
    identity=hashlib.sha256(json.dumps(entries,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if identity!=meta.get('mark53_audio_sha256'):raise RuntimeError('Teacher score input identity differs from the frozen prepared audio')
    if meta.get('scores_sha256')!=digest(root/'plots/teacher_scores_mn10.csv') or set(scores.index)!=set(filenames):
        raise RuntimeError('Teacher score table differs from its metadata')
    values=scores[['p_'+c for c in CLASSES[:8]]].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Invalid teacher probabilities')
    print('[teacher scores verified]',identity,flush=True)
    return scores

def train_one(root,out,train,val,test,targets,recipe,config,meta):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    import torch.nn.functional as F
    d=Path(out);d.mkdir(parents=True,exist_ok=True);runtime=runtime_metadata()
    fingerprint=dict(recipe=recipe,protocol=VERSION,epochs=config.get('epochs',80),batch_size=16,
        lr=1e-4,weight_decay=1e-4,alpha=.3,patience=10,temperature=1.,
        sources={s:hashlib.sha256(f[['path','label','sha256']].to_csv(index=False).encode()).hexdigest() for s,f in [('train',train),('val',val),('test',test)]},
        target_hash=hashlib.sha256(b''.join(np.ascontiguousarray(targets[k]).tobytes() for k in sorted(targets))).hexdigest(),
        source_code={p.relative_to(HERE).as_posix():digest(p) for p in sorted(HERE.rglob('*.py'))},
        input_metadata=digest(HERE/'provenance_active.csv'),
        weights=digest(Path(root)/'weights/mn10_as_mAP_471.pt'),runtime=runtime)
    identity=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    spec=d/'protocol.json'
    if spec.exists() and json.loads(spec.read_text())['identity']!=identity:
        raise RuntimeError(f'Run settings changed; choose a NEW output directory: {d}')
    if validate_complete(d,identity):
        print('[skip completed]',d.name,flush=True);return
    dump(spec,dict(identity=identity,**fingerprint))
    print('[runtime]',json.dumps(runtime,ensure_ascii=False),flush=True)
    seed=recipe['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if runtime['device']=='cuda':torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(runtime['device']);kind='efficientat' if recipe['variant']=='efficientat' else 'kampa'
    model=build_model(root,kind=kind).to(device)
    # All layers have lr=1e-4; Adam coupled weight decay matches existing training.
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3)
    cache=config.get('cache','/content/kampa_feedback_audio_cache')
    tw=cached_waves(train,root,cache);vw=cached_waves(val,root,cache);ew=cached_waves(test,root,cache)
    y=torch.tensor([CLASSES.index(c) if c in CLASSES else -1 for c in train.label],dtype=torch.long)
    vy=torch.tensor([CLASSES.index(c) for c in val.label],dtype=torch.long)
    data=TensorDataset(tw,y,torch.from_numpy(targets['q']),torch.from_numpy(targets['first']),torch.from_numpy(targets['second']),torch.from_numpy(targets['weight']))
    loader=DataLoader(data,batch_size=16,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    vloader=DataLoader(TensorDataset(vw,vy),batch_size=16,shuffle=False,num_workers=0)
    start=0;best=float('inf');bad=0;best_epoch=0;history=[];best_state=None
    last=d/'last.pt'
    if last.exists():
        cp=torch.load(last,map_location=device,weights_only=False)
        assert cp['identity']==identity
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        start=cp['epoch']+1;best=cp['best'];bad=cp['bad'];best_epoch=cp['best_epoch'];history=cp['history']
        best_state={k:v.cpu() for k,v in cp['best_model'].items()}
        random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']])
        print(f'[resume] {d.name}, completed epochs={start}, patience={bad}/10',flush=True)
    dump(d/'meta_classifier.json',meta)
    write_csv(d/'train_manifest.csv',train);write_csv(d/'val_manifest.csv',val);write_csv(d/'test_manifest.csv',test)
    pseudo=pd.DataFrame({'filename':train.filename,'weight':targets['weight'],'first':targets['first'],'second':targets['second']})
    for i,c in enumerate(CLASSES):pseudo['q_'+c]=targets['q'][:,i]
    write_csv(d/'pseudo_targets.csv',pseudo)
    soft=recipe['variant'] in ('soft_selected','soft_all')
    for epoch in range(start,config.get('epochs',80)):
        if bad>=10:break
        begun=time.time();model.train();train_loss=0.
        for wave,yy,q,aa,bb,ww in loader:
            # Same joint L/U forward, with all U retained even when its loss is silent.
            idx=torch.cat([torch.where(yy>=0)[0],torch.where(yy<0)[0]])
            wave,yy,q,aa,bb,ww=[z[idx].to(device,non_blocking=True) for z in (wave,yy,q,aa,bb,ww)]
            optimizer.zero_grad(set_to_none=True);logits=model(wave,augment=True)
            hard=yy>=0;loss=logits.sum()*0
            if hard.any():loss=loss+F.cross_entropy(logits[hard],yy[hard])
            if (~hard).any():loss=loss+.3*target_loss(logits[~hard],q[~hard],aa[~hard],bb[~hard],ww[~hard],soft)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss in {d}')
            if loss.detach().item()>0:loss.backward();optimizer.step()
            train_loss+=float(loss.detach())
        model.eval();vl=[]
        with torch.no_grad():
            for wave,yy in vloader:vl.append(float(F.cross_entropy(model(wave.to(device)),yy.to(device))))
        value=float(np.mean(vl))
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
        scheduler.step(value)
        if value<best:
            best=value;bad=0;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        record=dict(epoch=epoch+1,train_loss=train_loss/len(loader),val_loss=value,best_epoch=best_epoch,seconds=time.time()-begun,lr=optimizer.param_groups[0]['lr'])
        history.append(record);print(d.name,json.dumps(record),flush=True)
        # One atomic checkpoint binds current weights, best weights, history and RNG.
        _atomic_torch_save(last,dict(identity=identity,model=model.state_dict(),best_model=best_state,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,bad=bad,
            best_epoch=best_epoch,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        write_csv(d/'history.csv',pd.DataFrame(history))
    # Re-export from the committed checkpoint even when no further epochs run.
    if best_state is None:raise RuntimeError('No completed training epoch / best checkpoint')
    write_csv(d/'history.csv',pd.DataFrame(history))
    _atomic_torch_save(d/'best.pt',best_state)
    model.load_state_dict(torch.load(d/'best.pt',map_location=device,weights_only=True));model.eval()
    summary=dict(recipe=recipe,best_epoch=best_epoch,best_val_loss=best,identity=identity)
    for split,frame,waves in [('val',val,vw),('test',test,ew)]:
        probs=[]
        with torch.no_grad():
            for (wave,) in DataLoader(TensorDataset(waves),batch_size=16,shuffle=False):probs.append(model(wave.to(device)).softmax(1).cpu().numpy())
        pr=np.concatenate(probs);pred=pr.argmax(1);truth=np.array([CLASSES.index(c) for c in frame.label])
        result=frame[['filename','label','group']].copy();result['prediction']=[CLASSES[i] for i in pred]
        for i,c in enumerate(CLASSES):result['prob_'+c]=pr[:,i]
        write_csv(d/f'{split}_predictions.csv',result);summary[split]=score_metrics(truth,pred)
    dump(d/'metrics.json',summary)
    dump(d/'complete.json',dict(identity=identity,files={name:digest(d/name) for name in COMPLETE_FILES}))
    print('[completed]',d.name,'macro-F1',summary['test']['macro_f1'],flush=True)

def run_suite(config):
    partitions,scores,oracle,plan=prepare(config);root=Path(config['root']);out=Path(config['output'])
    audit_audio(config)
    original_sets,_,_=load_inputs(root)
    scores=verified_teacher_scores(root,original_sets)
    dump(out/'configuration.json',config)
    for recipe in plan:
        sets=partitions[recipe['split_seed']];n=recipe['n'];seed=recipe['seed'];variant=recipe['variant']
        labeled=select_labeled(sets['train'],n,seed);meta={}
        if variant=='oracle':
            extra=oracle[~oracle.source_sha256.isin(set(labeled.source_sha256))]
            train=pd.concat([labeled,extra],ignore_index=True)
        elif variant in ('supervised','efficientat'):train=labeled
        else:train=pd.concat([labeled,sets['unlabeled']],ignore_index=True)
        targets=dict(q=np.zeros((len(train),9),dtype='float32'),first=np.zeros(len(train),dtype=np.int64),second=np.full(len(train),-1,dtype=np.int64),weight=np.zeros(len(train),dtype='float32'))
        if variant in VARIANTS:
            qm,qf,meta=get_q(root,scores,labeled,sets['unlabeled'],n,seed,refit=True)
            target=make_targets(qf if variant.startswith('fixed') else qm,variant)
            for k in targets:targets[k][len(labeled):]=target[k]
            meta.update(dict(labeled_filenames=labeled.filename.tolist(),tau1=TAU1,tau2=TAU2,variant=variant,
                             accepted=int((target['weight']>0).sum()),two=int((target['second']>=0).sum())))
        name=f"split{recipe['split_seed']}_n{n}_s{seed}_{variant}"
        train_one(root,out/'runs'/name,train,sets['val'],sets['test'],targets,recipe,config,meta)
        aggregate(out)
    aggregate(out)

def aggregate(out):
    rows=[];runtimes=set()
    for p in sorted((Path(out)/'runs').glob('*/complete.json')):
        validate_complete(p.parent)
        spec=json.loads((p.parent/'protocol.json').read_text());runtimes.add(json.dumps(spec['runtime'],sort_keys=True))
        m=json.loads((p.parent/'metrics.json').read_text());rows.append({**m['recipe'],**{k:v for k,v in m['test'].items() if k!='confusion'},'best_epoch':m['best_epoch']})
    if len(runtimes)>1:raise RuntimeError('Results from different hardware/library configurations must not be pooled')
    if not rows:return
    frame=pd.DataFrame(rows);write_csv(Path(out)/'results_all_runs.csv',frame)
    group=['stage','n','split_seed','variant']
    summary=frame.groupby(group)[['accuracy','macro_f1']].agg(['count','mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    write_csv(Path(out)/'results_summary.csv',summary)
    contrasts=[]
    for keys,part in frame.groupby(['stage','n','split_seed']):
        reference='kampa' if keys[0]=='1_3' else 'efficientat'
        ref=part[part.variant==reference].set_index('seed')
        for variant,other in part.groupby('variant'):
            paired=other.set_index('seed').join(ref[['accuracy','macro_f1']],rsuffix='_reference',how='inner')
            for seed,r in paired.iterrows():
                contrasts.append(dict(stage=keys[0],n=keys[1],split_seed=keys[2],variant=variant,reference=reference,seed=seed,
                    accuracy_delta=r.accuracy-r.accuracy_reference,macro_f1_delta=r.macro_f1-r.macro_f1_reference))
    write_csv(Path(out)/'paired_seed_differences.csv',pd.DataFrame(contrasts))
    effects=[]
    for keys,part in frame[frame.stage=='1_3'].groupby(['n','seed']):
        p=part.set_index('variant')
        needed={'fixed_single','fixed_set','meta_single','kampa'}
        if needed.issubset(p.index):
            r=dict(n=keys[0],seed=keys[1])
            for metric in ['accuracy','macro_f1']:
                fs,ft,ms,mt=[float(p.loc[v,metric]) for v in ['fixed_single','fixed_set','meta_single','kampa']]
                r.update({metric+'_meta_effect_single':ms-fs,metric+'_meta_effect_set':mt-ft,
                          metric+'_set_effect_fixed':ft-fs,metric+'_set_effect_meta':mt-ms,
                          metric+'_interaction':(mt-ms)-(ft-fs)})
            effects.append(r)
    if effects:write_csv(Path(out)/'ablation_factorial_effects.csv',pd.DataFrame(effects))
    privileged=[]
    for keys,part in frame[frame.stage=='1_3'].groupby(['n','seed']):
        p=part.set_index('variant')
        if {'supervised','oracle'}.issubset(p.index):
            privileged.append(dict(n=keys[0],seed=keys[1],accuracy_delta=float(p.loc['oracle','accuracy']-p.loc['supervised','accuracy']),
                                   macro_f1_delta=float(p.loc['oracle','macro_f1']-p.loc['supervised','macro_f1'])))
    if privileged:write_csv(Path(out)/'oracle_vs_supervised.csv',pd.DataFrame(privileged))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    if a.prepare_only:prepare(config)
    else:run_suite(config)

if __name__=='__main__':main()
