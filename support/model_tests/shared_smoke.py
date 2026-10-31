"""Smoke tests centralizados sobre a mesma seleção train/dev utilizada no Flower."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'FedAvg tests'))
from common.data import prepare,ROOT,DATA_ROOT
from common.learning import MODELS,LOCAL_ONLY,DISPLAY_NAMES,factory,fit,predict
from common.metrics import evaluate

SEED=2026

def load_dataset(dataset):
    clients,cfg,registry,transforms=prepare(dataset,80,64,SEED,False)
    def merge(split):
        blocks=[(c,c['fit_indices'] if split=='train' else c['hold_indices']) for c in clients.values() if c['split']==split]
        out={key:np.concatenate([c[key][ix] for c,ix in blocks]) for key in ['X','F','Z','y','sample_ids','valid_lengths']}
        out['fit_indices']=np.arange(len(out['y'])) if split=='train' else np.array([],dtype=int)
        out['hold_indices']=np.arange(len(out['y']))
        out['split']=split;out['client_id']='CENTRAL'
        return out
    train,dev=merge('train'),merge('dev')
    assert set(train['sample_ids']).isdisjoint(dev['sample_ids'])
    assert not any(c['split']=='test' for c in clients.values())
    return train,dev,cfg,clients

def run_model(dataset,name,train,dev,cfg):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],SEED)
    start=time.perf_counter()
    if name=='LogReg':
        model=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=SEED)
    if name=='SVM':
        model=SVC(C=1,kernel='rbf',class_weight='balanced',probability=True,random_state=SEED)
    if name in LOCAL_ONLY or name in {'LogReg','SVM'}:
        model.fit(train['Z'],train['y']);p=np.zeros((len(dev['y']),2))
        p[:,model.classes_.astype(int)]=model.predict_proba(dev['Z'])
    else:
        fit(model,name,train,epochs=3,seed=SEED+1,ae_epochs=2 if name=='Autoencoder_CNN_GradCAM' else 0)
        p=predict(model,name,dev,dev['hold_indices'])
    elapsed=time.perf_counter()-start
    metrics,cm=evaluate(dev['y'],p,dataset)
    folder=ROOT/'artifacts' / 'test_models' / 'model_smoke_tests'/'binary_smoke_v3'/dataset
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{name}_dev_predictions.npz',p=p,y=dev['y'],sample_ids=dev['sample_ids'],cm=cm)
    if isinstance(model,torch.nn.Module):torch.save(model.state_dict(),folder/f'{name}.pt')
    row=dict(model=DISPLAY_NAMES[name],model_id=name,fit_seconds=elapsed,epochs=3 if name not in LOCAL_ONLY|{'LogReg','SVM'} else 0,**metrics)
    print(f"{row['model']}: PR-AUC={metrics['anomaly_pr_auc_ap']:.4f}; {elapsed:.1f}s",flush=True)
    return row

def sample_percentages(dataset,clients):
    folder='physionet2027_prepared_featured' if dataset=='physionet2017' else 'icentia11k_prepared_featured'
    manifest=pd.read_csv(ROOT/'DataPreparation'/'FeatureExtraction'/folder/'manifest.csv')
    totals=manifest.groupby('split').n_examples.sum()
    rows=[]
    for split in ['train','dev']:
        cs=[c for c in clients.values() if c['split']==split]
        selected=sum(len(c['fit_indices'])+len(c['hold_indices']) for c in cs)
        used=sum(len(c['fit_indices']) if split=='train' else len(c['hold_indices']) for c in cs)
        rows.append(dict(dataset=dataset,split=split,clients=len(cs),available=int(totals[split]),selected=selected,
                         fit_or_dev=used,local_holdout=selected-used if split=='train' else 0,excluded_selected=sum(len(c['y']) for c in cs)-selected,
                         selected_percent=100*selected/int(totals[split]),used_percent=100*used/int(totals[split])))
    return rows


def save_gradcam(dataset,dev,cfg):
    """Explicação posterior do modelo treinado, sem ajustar pesos ou escolher exemplos por acerto."""
    folder=ROOT/'artifacts' / 'test_models' / 'model_smoke_tests'/'binary_smoke_v3'/dataset
    model=factory('Autoencoder_CNN_GradCAM',len(cfg['feature_names']),cfg['signal_vector_samples'],SEED)
    model.load_state_dict(torch.load(folder/'Autoencoder_CNN_GradCAM.pt',map_location='cpu',weights_only=True))
    model.eval();cache={}
    handle=model.cam_target().register_forward_hook(lambda module,inputs,out:cache.update(activation=out))
    indices=[int(np.flatnonzero(dev['y']==k)[0]) for k in range(2) if np.any(dev['y']==k)]
    maps=[];predictions=[]
    try:
        for i in indices:
            model.zero_grad(set_to_none=True)
            out=model(torch.from_numpy(dev['X'][i:i+1,None,:]),torch.from_numpy(dev['F'][i:i+1]),torch.from_numpy(dev['valid_lengths'][i:i+1]))
            target=int(out.argmax(1)[0]);a=cache['activation']
            grad=torch.autograd.grad(out[0,target],a)[0]
            heat=(grad.mean(-1,keepdim=True)*a).sum(1,keepdim=True).relu()
            heat=torch.nn.functional.interpolate(heat,size=dev['X'].shape[1],mode='linear',align_corners=False)
            heat=heat/heat.amax(-1,keepdim=True).clamp_min(1e-8)
            maps.append(heat.detach().numpy()[0,0]);predictions.append(target)
    finally:handle.remove()
    np.savez_compressed(folder/'central_gradcam_dev.npz',heatmaps=np.stack(maps),indices=indices,predictions=predictions,sample_ids=dev['sample_ids'][indices])
    return np.stack(maps),indices,predictions


def save_confusion_matrices(dataset,dev,comparison):
    """PNGs individuais e grelha de todos os modelos, exclusivamente no dev binário."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
    folder=ROOT/'artifacts/test_models/model_smoke_tests/binary_smoke_v3'/dataset
    output=folder/'confusion_matrices/dev';output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for row in comparison.to_dict('records'):
        with np.load(folder/f"{row['model_id']}_dev_predictions.npz") as z:
            assert np.array_equal(z['y'],dev['y']) and np.array_equal(z['sample_ids'],dev['sample_ids'])
            cm=confusion_matrix(z['y'],z['p'].argmax(1),labels=[0,1])
            assert cm.shape==(2,2) and cm.sum()==len(dev['y'])
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm,display_labels=['Normal','Anomalia']).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
        ax.set_title(row['model']+' — dev');ax.set_xlabel('Classe prevista');ax.set_ylabel('Classe real')
        fig.tight_layout();fig.savefig(output/f"{row['model_id']}_dev.png",dpi=160,bbox_inches='tight');plt.close(fig)
        entries.append((row['model'],cm))
    fig,axes=plt.subplots(4,4,figsize=(18,17),constrained_layout=True)
    for ax,(name,cm) in zip(axes.ravel(),entries):
        ConfusionMatrixDisplay(cm,display_labels=['Normal','Anomalia']).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
        ax.set_title(name,fontsize=10);ax.set_xlabel('Prevista');ax.set_ylabel('Real')
    for ax in axes.ravel()[len(entries):]:ax.set_visible(False)
    fig.suptitle(f'{dataset} — matrizes de confusão em dev ({len(dev["y"])} exemplos)',fontsize=16)
    joint=folder/'confusion_matrices_dev.png';fig.savefig(joint,dpi=150,bbox_inches='tight')
    return fig,output,joint
