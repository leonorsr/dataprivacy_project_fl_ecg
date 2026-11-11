"""Avaliação binária: classe positiva = anomalia; versões macro apenas auxiliares."""
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score

def mean_valid(values):
    a=np.asarray(values,float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')

def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan,dtype=float),where=np.asarray(b)>0)

def evaluate(y,p,dataset):
    y=np.asarray(y);p=np.asarray(p)
    assert set(np.unique(y))<={0,1} and p.shape==(len(y),2) and np.isfinite(p).all()
    assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    both=np.unique(y).size==2
    div=lambda a,b:float(a/b) if b else float('nan')
    precision=div(tp,tp+fp) if tp+fp else (0. if tp+fn else np.nan)
    recall=div(tp,tp+fn);miss=div(fn,tp+fn);far=div(fp,fp+tn)
    f1=div(2*tp,2*tp+fp+fn) if tp+fn else np.nan
    auc=roc_auc_score(y,p[:,1]) if both else np.nan
    ap=average_precision_score(y,p[:,1]) if both else np.nan
    tpv=cm.diagonal().astype(float);fnv=cm.sum(1)-tpv;fpv=cm.sum(0)-tpv;tnv=cm.sum()-tpv-fnv-fpv
    recalls=ratio(tpv,tpv+fnv);precisions=ratio(tpv,tpv+fpv);precisions[(tpv+fnv>0)&(tpv+fpv==0)]=0
    f1s=ratio(2*tpv,2*tpv+fpv+fnv);f1s[tpv+fnv==0]=np.nan
    macro_ap=float(np.mean([average_precision_score(y==k,p[:,k]) for k in [0,1]])) if both else np.nan
    result=dict(accuracy=float((y==pred).mean()),f1=f1,precision=precision,recall=recall,roc_auc=auc,pr_auc_ap=ap,miss_rate=miss,far=far,
        f1_macro=mean_valid(f1s),precision_macro=mean_valid(precisions),recall_macro=mean_valid(recalls),
        miss_rate_macro=mean_valid(1-recalls),far_macro=mean_valid(ratio(fpv,fpv+tnv)),
        roc_auc_macro_ovr=auc,pr_auc_ap_macro_ovr=macro_ap,auc_valid_classes=2 if both else 0,n_examples=len(y),
        anomaly_roc_auc=auc,anomaly_pr_auc_ap=ap,anomaly_recall=recall,anomaly_miss_rate=miss,anomaly_far=far,
        anomaly_precision=precision,anomaly_f1=f1,anomaly_n=len(y))
    return result,cm
