"""Freeze source metadata from the active ledger; never edit research inputs."""
import argparse, csv, json, re
from pathlib import Path
from collections import Counter, defaultdict

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ledger',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
CLASSES = ['heavy_impact','dragging','construction','machine_noise','talking','water_toilet','water_shower','dog_bark','others']
from openpyxl import load_workbook
book=load_workbook(args.ledger,read_only=True,data_only=True)
values=iter(book['Sheet1'].values);header=next(values)
raw=[dict(zip(header,row)) for row in values]
book.close()
rows = [{k:('' if v is None else v) for k,v in r.items()} for r in raw
        if r['removed_20260715']=='active' and r['mark_version'] in ['mark5.2']+[f'mark4.{i}' for i in range(1,9)]]
by = {(r['mark_version'],r['local_filename']):r for r in rows}
assert len(by)==len(rows)

def origin(r, seen=()):
    key=(r['mark_version'],r['local_filename'])
    assert key not in seen, key
    if r['aug_source_file']:
        return origin(by[(r['mark_version'],r['aug_source_file'])],seen+(key,))
    return r

def identities(r):
    r=origin(r); src=r['source']; sid=str(r['aihub_src_file'])
    if src=='fsd50k': sid=re.sub(r'\.0$|\.wav$','',str(r['fsd50k_fname']))
    assert sid,(src,r['local_filename'])
    recording=src+':'+sid
    group=recording
    if src in ('sins','sins_eval'):
        m=re.search(r'ex(\d+)',sid);assert m, sid
        group=src+':ex'+m.group(1)
    elif src=='chime_home': group=src+':'+str(r['aihub_volume'])
    elif src=='esc50':
        # ESC-50 clips from the same Freesound recording must stay together.
        group='freesound:'+sid.split('-')[1]
    elif src in ('fsd50k','clotho'): group='freesound:'+sid
    return recording,group

positive=defaultdict(set);negative=defaultdict(set);studenttruth=defaultdict(set)
for r in rows:
    rec,g=identities(r)
    if r['mark_version']=='mark5.2': studenttruth[rec].add(r['target_class'].removeprefix('unlabeled:'))
    elif r['target_class']!='others': positive[rec].add(r['target_class'])
    else: negative[rec].add(r['mark_version'])

out=[]
for r in rows:
    rec,g=identities(r);label=r['target_class'].removeprefix('unlabeled:');evidence='student_annotation'
    if r['mark_version']!='mark5.2' and label=='others':
        org=origin(r);note=str(org['mark52_note']);cat=str(org['original_labels'])
        restored=re.search(r'(?:hardneg|cross_target)[^\(]*\(from=([^:]+):',note)
        candidates=positive[rec]
        if restored and restored.group(1) in CLASSES:
            label=restored.group(1);evidence='explicit_original_class_in_provenance'
        elif note.startswith('bghouse_'):
            label='others';evidence='documented_house_background_build'
        elif r['mark_version']=='mark4.5' and org['source']=='fsd50k' and note.startswith('talk_20261024('):
            label='others';evidence='documented_non_speech_human_sounds_build'
        elif org['source'] in ('sins','sins_eval') and cat in ('cooking','working','eating'):
            label='others';evidence='documented_sins_background_classes'
        elif org['source']=='aihub' and (cat.startswith('D.교통소음') or cat.startswith('C.사업장_1.상가건물_a')):
            label='others';evidence='documented_aihub_far_background'
        elif org['source']=='esc50' and cat in ('crow','crackling_fire'):
            label='others';evidence='documented_esc_background_classes'
        elif len(candidates)==1: label=next(iter(candidates));evidence='same_recording_teacher_positive'
        elif len(candidates)>1: label='';evidence='conflicting_recording_labels'
        elif len(negative[rec])==8: label='others';evidence='negative_in_all_eight_specialists'
        else: label='';evidence='unresolved_binary_negative'
    elif r['mark_version']!='mark5.2': evidence='teacher_positive_annotation'
    # Oracle must never borrow annotations from student validation/test or hidden U labels.
    out.append(dict(version=r['mark_version'],filename=r['local_filename'],split=r['assigned_split'],
                    label=label,binary_label=r['target_class'],recording=rec,group=g,sha256=r['sha256'],
                    evidence=evidence,source=r['source'],augmented=bool(r['aug_source_file'])))

def save(name, rr):
    p=args.output/name
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
save('provenance_active.csv',out)
students=[r for r in out if r['version']=='mark5.2']
teachers=[r for r in out if r['version']!='mark5.2']
exposed={r['group'] for r in teachers if r['split'] in ('train','val')}
u={r['group'] for r in students if r['split']=='unlabeled'}
safe=[r for r in students if r['split']!='unlabeled' and r['group'] not in exposed|u]
print('safe_for_new_holdout',len(safe),Counter(r['label'] for r in safe),Counter(r['split'] for r in safe))
print('original_eval_overlap',Counter(r['split'] for r in students if r['split'] in ('val','test') and r['group'] in exposed))
print('mapped_teacher_train',Counter(r['evidence'] for r in teachers if r['split']=='train'))
print('student_groups',len({r['group'] for r in students}), 'test_groups',len({r['group'] for r in students if r['split']=='test'}))
