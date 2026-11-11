"""Joint-vector Laplace LDP: clipping before any client-to-server release."""
import math,secrets,threading,uuid
from pathlib import Path
import numpy as np
from diffprivlib.mechanisms import Laplace
from support.federated_global.shared import dump

class PrivacyLedger:
    """Conservative sequential composition, including interrupted attempts."""
    def __init__(self,path):
        import json
        self.path=Path(path);self.lock=threading.Lock()
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'version':1,'clients':{},'attempts':{}}
    def begin(self,label,epsilon):
        key=uuid.uuid4().hex
        with self.lock:
            self.state['attempts'][key]={'label':label,'epsilon_per_complete_client_cycle':epsilon,'releases':0,'status':'running'}
            dump(self.path,self.state)
        return key
    def charge(self,attempt,client,epsilon):
        with self.lock:
            item=self.state['clients'].setdefault(client,{'epsilon_spent_private_releases':0.,'private_releases':0,'non_private_release':False})
            if epsilon is None:item['non_private_release']=True
            else:
                item['epsilon_spent_private_releases']+=float(epsilon);item['private_releases']+=1
            self.state['attempts'][attempt]['releases']+=1
            dump(self.path,self.state)
    def finish(self,attempt):
        with self.lock:self.state['attempts'][attempt]['status']='complete';dump(self.path,self.state)


def clip_logits(logits,bound=32.):
    z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(32,2) or not np.isfinite(z).all():raise ValueError('Expected 32 finite binary anchor logits.')
    if not math.isfinite(bound) or bound<=0:raise ValueError('L1 bound must be positive and finite.')
    # Public numerical guard, then remove the irrelevant per-anchor common offset.
    z=np.clip(z,-1e6,1e6);z=z-z.mean(axis=1,keepdims=True)
    norm=float(np.abs(z).sum());result=z*(bound/norm if norm>bound else 1.)
    assert np.abs(result).sum()<=bound+1e-9
    return result

class LocalLogitMechanism:
    def __init__(self,mode,epsilon=None,bound=32.,rounds=3,ledger=None,attempt=None,client=None):
        if mode not in {'raw','clipped','ldp'}:raise ValueError('Unknown mode')
        if rounds!=3:raise ValueError('The current training loop has exactly three releases.')
        self.mode,self.epsilon,self.bound,self.rounds=mode,epsilon,float(bound),rounds
        if not math.isfinite(self.bound) or self.bound<=0:raise ValueError('Invalid clipping bound')
        self.ledger,self.attempt,self.client=ledger,attempt,client
        self.releases=0
        if mode=='ldp':
            if epsilon is None or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('epsilon must be finite and positive')
            self.epsilon_round=float(epsilon)/rounds
            self.sensitivity=2*self.bound
            self.noise_scale=self.sensitivity/self.epsilon_round
            self.laplace=Laplace(epsilon=self.epsilon_round,delta=0,sensitivity=self.sensitivity,random_state=None)
            assert isinstance(self.laplace._rng,secrets.SystemRandom),'Do not use public/reproducible noise seeds.'
    def release(self,logits):
        if self.releases>=self.rounds:raise RuntimeError('Client privacy budget exhausted for this cycle.')
        z=np.asarray(logits,dtype=np.float64)
        if z.shape!=(32,2) or not np.isfinite(z).all():raise ValueError('Invalid logits')
        clipped=clip_logits(z,self.bound) if self.mode!='raw' else z
        # Charge before randomisation; a crash may overcount, never undercount a release.
        if self.ledger is not None:self.ledger.charge(self.attempt,self.client,self.epsilon_round if self.mode=='ldp' else None)
        self.releases+=1
        if self.mode=='ldp':
            out=np.array([self.laplace.randomise(float(v)) for v in clipped.ravel()],dtype=np.float64).reshape(z.shape)
        else:out=clipped.copy()
        assert np.isfinite(out).all()
        return out
    def certificate(self):
        # Public parameters only. Never release private norms, clipping flags or noise seeds.
        return dict(mode=self.mode,privacy_unit='whole prepared client training dataset',
            adjacency='replacement of a known fixed-size client; participation is public',
            epsilon_total=self.epsilon if self.mode=='ldp' else None,
            epsilon_per_round=self.epsilon_round if self.mode=='ldp' else None,delta=0 if self.mode=='ldp' else None,
            rounds=self.rounds,releases=self.releases,vector_shape=[32,2],joint_l1_bound=self.bound,
            joint_l1_sensitivity=2*self.bound if self.mode=='ldp' else None,
            laplace_scale=self.noise_scale if self.mode=='ldp' else None,
            secure_randomness=self.mode=='ldp',guarantee='one-cycle message channel only; other releases require composition')
