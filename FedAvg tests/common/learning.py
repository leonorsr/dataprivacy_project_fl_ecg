import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from . import models
MODELS=['LogReg','SVM','KNN','RandomForest','CNN1D','CNN_BiLSTM','CNN_BiLSTM_Attention','CNN_BiLSTM_CBAM','Autoencoder_CNN_GradCAM','ResNet34','MobileNetV2','CNN1D_Optimized','GRU5','GRU_Simple']
LOCAL_ONLY={'KNN','RandomForest'}

def _architecture_factory(name,feature_dim,signal_length,seed):
    torch.manual_seed(seed);models.SIGNAL_LENGTH=signal_length
    if name in {'LogReg','SVM'}:return nn.Linear(32+feature_dim,4)
    if name=='KNN':return KNeighborsClassifier(n_neighbors=5,weights='distance',n_jobs=1)
    if name=='RandomForest':return RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=1,random_state=seed)
    if name in {'CNN1D','CNN1D_Optimized'}:return models.CNN1D(feature_dim)
    if name=='CNN_BiLSTM':return models.CNNBiLSTM(feature_dim)
    if name=='CNN_BiLSTM_Attention':return models.CNNBiLSTM(feature_dim,attention=True)
    if name=='CNN_BiLSTM_CBAM':return models.CNNBiLSTM(feature_dim,cbam=True)
    if name=='ResNet34':return models.ResNet34ECG(feature_dim)
    if name=='MobileNetV2':return models.MobileNetV2ECG(feature_dim)
    if name=='Autoencoder_CNN_GradCAM':return models.AutoencoderCNN(feature_dim,models.ConvAutoencoder().encoder)
    if name in {'GRU5','GRU_Simple'}:return models.GRUECG(feature_dim,5 if name=='GRU5' else 1,signal_length)
    raise ValueError(name)

def factory(name,feature_dim,signal_length,seed):
    """Todos os modelos ativos têm duas saídas: normal (0), anomalia (1)."""
    m=_architecture_factory(name,feature_dim,signal_length,seed)
    if isinstance(m,nn.Linear):return nn.Linear(m.in_features,2)
    if isinstance(m,nn.Module):
        old=m.head.classifier[-1]
        m.head.classifier[-1]=nn.Linear(old.in_features,2)
    return m

def get_parameters(m):return [v.detach().cpu().numpy().copy() for v in m.state_dict().values()]
def set_parameters(m,parameters):
    state=m.state_dict();assert len(state)==len(parameters)
    replacement={}
    for (k,v),a in zip(state.items(),parameters):
        assert tuple(a.shape)==tuple(v.shape) and np.isfinite(a).all()
        replacement[k]=torch.as_tensor(a.copy(),dtype=v.dtype)
    m.load_state_dict(replacement,strict=True)

def logits(m,name,c,ix):
    if name in {'LogReg','SVM'}:return m(torch.from_numpy(c['Z'][ix]))
    return m(torch.from_numpy(c['X'][ix,None,:]),torch.from_numpy(c['F'][ix]),torch.from_numpy(c['valid_lengths'][ix]))

def fit(m,name,c,epochs,seed,ae_epochs=0):
    ix=c['fit_indices'];y=c['y'][ix]
    if name in LOCAL_ONLY:
        m.fit(c['Z'][ix],y);return
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if name=='Autoencoder_CNN_GradCAM' and ae_epochs:
        ae=models.ConvAutoencoder();ae.encoder=copy.deepcopy(m.encoder)
        opt=torch.optim.AdamW(ae.parameters(),lr=.001)
        for _ in range(ae_epochs):
            for start in range(0,len(ix),32):
                x=torch.from_numpy(c['X'][ix[start:start+32],None,:]);opt.zero_grad()
                loss=F.mse_loss(ae(x),x);loss.backward();opt.step()
        m.encoder.load_state_dict(ae.encoder.state_dict())
    counts=np.bincount(y,minlength=2);weights=np.zeros(2,dtype=np.float32)
    present=counts>0;weights[present]=len(y)/(present.sum()*counts[present])
    opt=torch.optim.AdamW(m.parameters(),lr=.0003 if name=='CNN1D_Optimized' else .001,weight_decay=.0001)
    m.train()
    for _ in range(epochs):
        order=rng.permutation(ix)
        for start in range(0,len(order),32):
            batch=order[start:start+32]
            if len(batch)<2:continue  # BatchNorm exige pelo menos dois exemplos nos mapas de 1 ponto.
            target=torch.from_numpy(c['y'][batch]).long();opt.zero_grad()
            out=logits(m,name,c,batch)
            loss=F.multi_margin_loss(out,target,weight=torch.from_numpy(weights)) if name=='SVM' else F.cross_entropy(out,target,weight=torch.from_numpy(weights))
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()

def predict(m,name,c,ix):
    if name in LOCAL_ONLY:
        p=np.zeros((len(ix),2));p[:,m.classes_.astype(int)]=m.predict_proba(c['Z'][ix]);return p
    m.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(ix),32):out.append(logits(m,name,c,ix[start:start+32]).softmax(1).numpy())
    return np.concatenate(out)


DISPLAY_NAMES = {
    'LogReg': 'Log Reg', 'SVM': 'SVM', 'KNN': 'KNN', 'RandomForest': 'Random Forest',
    'CNN1D': 'CNN 1D', 'CNN1D_Optimized': 'CNN 1D otimizada',
    'CNN_BiLSTM': 'CNN 1D + BiLSTM',
    'CNN_BiLSTM_Attention': 'CNN 1D + BiLSTM + Attention',
    'CNN_BiLSTM_CBAM': 'CNN 1D + BiLSTM + CBAM',
    'Autoencoder_CNN_GradCAM': 'Autoencoder + CNN 1D + Grad-CAM',
    'ResNet34': 'ResNet-34 1D', 'MobileNetV2': 'MobileNetV2 1D',
    'GRU5': 'GRU de cinco camadas', 'GRU_Simple': 'GRU simplificada',
}
