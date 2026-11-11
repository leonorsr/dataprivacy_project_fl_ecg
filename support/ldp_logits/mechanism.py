"""Joint-client clipping and secure-random diffprivlib noise for research training."""
import json,secrets,threading,uuid
from pathlib import Path
import numpy as np
from diffprivlib.mechanisms import Laplace,GaussianAnalytic
from .ldp_logits import representar,recortar_por_cliente,para_logits
from support.federated_global.shared import dump

class Ledger:
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.Lock()
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {'clients':{},'attempts':{},'scope':'prepared message channel under fixed transforms; not end-to-end pipeline'}
    def begin(self,label):
        key=uuid.uuid4().hex
        with self.lock:self.data['attempts'][key]={'label':label,'status':'running','releases':0};dump(self.path,self.data)
        return key
    def charge(self,attempt,uid,epsilon,delta):
        with self.lock:
            c=self.data['clients'].setdefault(uid,{'epsilon_private_sum':0.,'delta_private_sum':0.,'private_releases':0,'non_private_releases':0})
            if epsilon is None:c['non_private_releases']+=1
            else:c['epsilon_private_sum']+=epsilon;c['delta_private_sum']+=delta;c['private_releases']+=1
            self.data['attempts'][attempt]['releases']+=1;dump(self.path,self.data)
    def finish(self,attempt):
        with self.lock:self.data['attempts'][attempt]['status']='complete';dump(self.path,self.data)

class ClientMechanism:
    def __init__(self,condition,rounds,ledger=None,attempt=None,uid=None):
        self.condition=condition;self.rounds=rounds;self.releases=0;self.ledger=ledger;self.attempt=attempt;self.uid=uid
        self.epsilon=condition['epsilon_total']/rounds if condition['mode']=='ldp' else None
        self.delta=condition['delta_total']/rounds if condition['mode']=='ldp' else 0.
        self.norm=1 if condition['mechanism']=='laplace' else 2
        self.noise=None
        if condition['mode']=='ldp':
            cls=Laplace if condition['mechanism']=='laplace' else GaussianAnalytic
            self.noise=cls(epsilon=self.epsilon,delta=self.delta,sensitivity=2*condition['clip_C'],random_state=None)
            assert isinstance(self.noise._rng,secrets.SystemRandom),'Privacy noise must not use public training seeds'
    def release(self,logits):
        if self.releases>=self.rounds:raise RuntimeError('Client-cycle release budget exhausted')
        values=representar(np.asarray(logits)[None,:,:],self.condition['representation'])
        clipped,_=recortar_por_cliente(values,self.condition['clip_C'],self.norm)
        if self.ledger:self.ledger.charge(self.attempt,self.uid,self.epsilon,self.delta)
        self.releases+=1
        if self.noise is not None:
            private=np.array([self.noise.randomise(float(v)) for v in clipped.ravel()]).reshape(clipped.shape)
        else:private=clipped
        out=para_logits(private,self.condition['representation'])[0]
        assert out.shape==(32,2) and np.isfinite(out).all()
        return out
    def certificate(self):
        return dict(mode=self.condition['mode'],mechanism=self.condition['mechanism'],representation=self.condition['representation'],
            clip_C=self.condition['clip_C'],norm=self.norm,joint_sensitivity=2*self.condition['clip_C'],
            epsilon_cycle=self.condition['epsilon_total'] if self.noise else None,delta_cycle=self.condition['delta_total'] if self.noise else None,
            epsilon_round=self.epsilon,delta_round=self.delta,rounds=self.rounds,releases=self.releases,
            secure_randomness=self.noise is not None,scope='one prepared-client message channel conditional on fixed transforms; other releases require composition')
