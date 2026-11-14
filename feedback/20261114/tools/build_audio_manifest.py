"""Freeze actual Drive inputs and prove their derivation from local source WAVs.

First collect `rclone lsjson REMOTE/Mark5.2/data_SPLIT --recursive --hash`
for labeled, unlabeled, val and test as SPLIT.json in --drive-hashes.
This command checks all 4,860 files; it never edits or uploads audio.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,sys
import numpy as np
import pandas as pd
import soundfile as sf
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from feedback_core import resolve_path,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=BASE.parents[1]);ap.add_argument('--drive-hashes',type=Path,required=True)
    args=ap.parse_args();prov=pd.read_csv(BASE/'src/provenance_active.csv',keep_default_na=False)
    own=prov[prov.version=='mark5.2'].set_index('filename');records=[];summary={};identities={}
    revisions=json.loads((BASE/'src/audio_source_revisions.json').read_text())
    for split in ['labeled','unlabeled','val','test']:
        remote=json.loads((args.drive_hashes/(split+'.json')).read_text())
        mapping={Path(r['Path']).name:r['Hashes']['sha256'] for r in remote if not r['IsDir']}
        frame=pd.read_csv(args.root/('dataset_'+split+'.csv'),keep_default_na=False)
        assert len(mapping)==len(frame),(split,len(mapping),len(frame))
        lengths={}
        for i,row in enumerate(frame.itertuples(index=False)):
            name=Path(row.path).name;source=row.path.replace('/data_'+split+'/', '/data_source_'+split+'/')
            p=resolve_path(source,args.root.parent);raw=p.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
            assert source_sha==own.loc[name,'sha256'],name
            conversion_sha=source_sha;conversion_path=''
            if name in revisions:
                revision=revisions[name]
                assert source_sha==revision['current_source_sha256'] and mapping[name]==revision['input_sha256']
                conversion_path=revision['conversion_source_path'];raw=(args.root.parent/conversion_path).read_bytes()
                conversion_sha=hashlib.sha256(raw).hexdigest();assert conversion_sha==revision['conversion_source_sha256']
            wave,sr=sf.read(io.BytesIO(raw),dtype='float32');assert sr==16000 and wave.ndim==1,name
            if len(wave)<48000:wave=np.pad(wave,(0,48000-len(wave)))
            buf=io.BytesIO();sf.write(buf,wave,16000,format='WAV',subtype='PCM_16')
            sha=hashlib.sha256(buf.getvalue()).hexdigest();assert sha==mapping[name],name
            lengths[str(len(wave))]=lengths.get(str(len(wave)),0)+1
            records.append(dict(path=row.path,filename=name,split=split,source_sha256=source_sha,input_sha256=sha,samples=len(wave),input_encoding='PCM_16',conversion_source_sha256=conversion_sha,conversion_source_path=conversion_path))
            if (i+1)%500==0:print(split,i+1,flush=True)
        summary[split]=dict(verified=len(frame),mismatches=0,sample_counts=lengths,provider_manifest_sha256=digest(args.drive_hashes/(split+'.json')))
        payload=dict(csv_sha256=digest(args.root/('dataset_'+split+'.csv')),rows=[dict(path=r.path,label=getattr(r,'label',''),audio_sha256=mapping[Path(r.path).name]) for r in frame.itertuples(index=False)])
        identities[split]=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pd.DataFrame(records).to_csv(BASE/'src/audio_inputs.csv',index=False)
    scoremeta=json.loads((args.root/'plots/teacher_scores_mn10.json').read_text())
    lookup={r['filename']:r['input_sha256'] for r in records};entries=[]
    for split in scoremeta['split_counts']:
        frame=pd.read_csv(args.root/('dataset_'+split+'.csv'),keep_default_na=False)
        entries.extend([[split,r.path,getattr(r,'label',''),lookup[Path(r.path).name]] for r in frame.itertuples(index=False)])
    audio_id=hashlib.sha256(json.dumps(entries,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert audio_id==scoremeta['mark53_audio_sha256'],(audio_id,scoremeta['mark53_audio_sha256'])
    report=dict(verified_files=len(records),current_source_conversions=len(records)-len(revisions),historical_source_conversions=len(revisions),source_revisions=revisions,splits=summary,dataset_identities=identities,teacher_score_audio_identity=audio_id,
                score_metadata_matches=True,transformation='mono 16kHz source -> minimum48000 zero pad, no truncation -> WAV PCM_16',
                input_manifest_sha256=digest(BASE/'src/audio_inputs.csv'),provenance_sha256=digest(BASE/'src/provenance_active.csv'))
    (BASE/'review/audio_manifest_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
