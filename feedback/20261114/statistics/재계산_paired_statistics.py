"""Paired recording/session cluster bootstrap; five fitted seeds remain fixed.

This CI describes test-group sampling uncertainty, not uncertainty across new
training splits. Seed-based t intervals are reported separately, with n=5.
"""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

BASE = Path(__file__).resolve().parent
CODE = BASE/'원자료/KAMPA'
CLASSES = ['heavy_impact','dragging','construction','machine_noise','talking','water_toilet','water_shower','dog_bark','others']

def metrics(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(-1)+cm.sum(-2)
    f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)
    acc=tp.sum(-1)/cm.sum(axis=(-2,-1))
    return np.stack([acc,f1],-1)

def main():
    prov=pd.read_csv(BASE/'원자료/시험_출처_그룹.csv',keep_default_na=False)
    test=prov[(prov.version=='mark5.2')&(prov.split=='test')].set_index('filename')
    names=sorted(test.index); y=np.array([CLASSES.index(test.loc[n,'label']) for n in names])
    groups,gi=np.unique(test.loc[names,'group'],return_inverse=True);ng=len(groups)
    outputs=[];seed_rows=[];audit=[]
    prefix='prediction_details_mark5.2_PL_SA_G04_AC_LT_noDB_noSEG_LIN_PH_HW1_SM_CT_BG0_RW_JF_SINGLE_ADAM'
    for n in (200,500,1000,2000):
        cms=np.zeros((ng,2,5,9,9),dtype=float)
        for method in range(2):
            for seed in range(1,6):
                ln='' if n==2000 else f'_l{n}'
                p=(CODE/f'{prefix}{ln}_u2000_pscmb_s{seed}_plain.csv') if method==0 else (BASE/'원자료/EfficientAT'/f'prediction_details_efficientat_mn10_sup_adam{ln}_s{seed}.csv')
                frame=pd.read_csv(p).set_index('Filename')
                assert frame.index.is_unique and set(frame.index)==set(names),p
                frame=frame.loc[names]
                assert frame['True Label'].tolist()==test.loc[names,'label'].tolist(),p
                probs=frame[['PreBias_Prob_'+c for c in CLASSES]].to_numpy(float)
                assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-5)
                pred=probs.argmax(1)
                np.add.at(cms[:,method,seed-1],(gi,y,pred),1)
                audit.append(dict(file=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=n,seed=seed,method=['KAMPA','EfficientAT'][method],rows=len(frame)))
        point=metrics(cms.sum(0))  # method, seed, metric
        for seed in range(5):
            seed_rows.append(dict(labeled_n=n,seed=seed+1,KAMPA_accuracy=point[0,seed,0],EfficientAT_accuracy=point[1,seed,0],KAMPA_macro_f1=point[0,seed,1],EfficientAT_macro_f1=point[1,seed,1]))
        rng=np.random.default_rng(20261114);draws=[]
        for start in range(0,10000,250):
            counts=rng.multinomial(ng,np.full(ng,1/ng),size=250)
            sampled=(counts@cms.reshape(ng,-1)).reshape(250,2,5,9,9)
            draws.append(metrics(sampled).mean(2))
        boot=np.concatenate(draws,axis=0)  # draw, method, metric
        avg=point.mean(1)
        for m,name in enumerate(['accuracy','macro_f1']):
            diff=point[0,:,m]-point[1,:,m]
            half=t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5)
            lo,hi=np.quantile(boot[:,0,m]-boot[:,1,m],[.025,.975])
            outputs.append(dict(labeled_n=n,metric=name,kampa_mean=avg[0,m],efficientat_mean=avg[1,m],delta=diff.mean(),
                cluster_ci_low=lo,cluster_ci_high=hi,seed_t_ci_low=diff.mean()-half,seed_t_ci_high=diff.mean()+half,
                n_test=430,n_groups=ng,n_seeds=5,bootstrap_reps=10000))
    pd.DataFrame(outputs).to_csv(BASE/'paired_ci_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_rows).to_csv(BASE/'paired_seed_metrics.csv',index=False,encoding='utf-8-sig')
    (BASE/'input_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    lines=['5번 통계 검토 및 계산 결과 (2026-11-14)','',
        '기존 예측만 재분석했다. 새 학습이나 새 시험 예측을 수행하지 않았다.',
        'KAMPA와 EfficientAT: N=200,500,1000,2000 각각 seed 1~5, 각 430개 시험 클립.',
        f'모든 파일의 파일명·정답·9개 클래스 확률을 확인했다. 시험 자료의 원본 녹음/세션 그룹은 {ng}개다.',
        'PreBias_Prob_*의 argmax를 사용했다. 운영점 보정 예측은 사용하지 않았다.',
        '',
        '주 분석: 동일 그룹을 두 모델·다섯 seed에 함께 적용하는 paired cluster bootstrap 10,000회.',
        '각 bootstrap에서 seed별 지표를 계산한 후 5회 평균의 차이를 구하고, 2.5/97.5 백분위로 95% CI를 구했다.',
        '이는 현재 학습된 5개 모델씩을 고정한 시험 그룹 표집 불확실성이다. 새 학습/새 분할 전체의 CI가 아니다.',
        'seed 5개를 2,150개의 독립 시험 표본처럼 합치지 않았다. 앙상블 성능도 아니다.',
        '별도 seed t CI는 seed별 차이 5개의 기술적 보조 결과다. 작은 표본·공통 시험셋이라는 한계를 갖는다.',
        '8개 비교의 CI는 개별 95% CI이며 다중비교 동시 신뢰구간은 아니다. p값/전 조건 유의성 주장을 추가하지 않는다.',
        '공통 시험 표본 대응을 검증했으며 각 모델의 학습 부분집합이 완전히 동일하다는 추가 검증을 뜻하지 않는다.',
        'AIHub는 원본 파일, SINS·CHiME는 세션, ESC-50은 원본 Freesound id 기준 그룹이다. 집/기기 독립성까지 보증하지 않는다.','']
    for r in outputs:
        lines.append(f"N={r['labeled_n']}, {r['metric']}: KAMPA {100*r['kampa_mean']:.3f}%, EfficientAT {100*r['efficientat_mean']:.3f}%, 차이 {100*r['delta']:+.3f}%p, paired cluster 95% CI [{100*r['cluster_ci_low']:+.3f}, {100*r['cluster_ci_high']:+.3f}]%p; seed t CI [{100*r['seed_t_ci_low']:+.3f}, {100*r['seed_t_ci_high']:+.3f}]%p")
    lines+=['','상태: 5번의 계산 산출물은 확보했다. 원고 반영은 아직 수행하지 않았다.',
             '4번의 새 분할 결과는 이 고정 시험셋 통계와 별도로 보고해야 한다.']
    (BASE/'5번_통계검토결과.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(pd.DataFrame(outputs)[['labeled_n','metric','delta','cluster_ci_low','cluster_ci_high']].to_string(index=False))

if __name__=='__main__': main()
