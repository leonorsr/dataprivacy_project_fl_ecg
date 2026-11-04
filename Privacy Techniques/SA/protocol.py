"""Secure Aggregation (Bonawitz et al., 2017) para o consenso das soft labels.

Servidor semi-honesto. Cada cliente mascara as probabilidades quantizadas com
máscaras de pares (troca de chaves X25519) e uma máscara própria; os dois
segredos são partilhados com Shamir para o servidor remover as máscaras de quem
desiste, desde que pelo menos t clientes respondam no desmascaramento.
O servidor só fica a saber a soma. Apenas biblioteca padrão + numpy.
"""
import hashlib,hmac,secrets,time
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- troca de chaves X25519 (RFC 7748)
_P=2**255-19;_A24=121665

def _scalar(k):
    k=bytearray(k);k[0]&=248;k[31]&=127;k[31]|=64;return int.from_bytes(k,'little')

def x25519(k,u):
    k=_scalar(k);b=bytearray(u);b[31]&=127;x1=int.from_bytes(b,'little')%_P
    x2,z2,x3,z3,swap=1,0,x1,1,0
    for t in reversed(range(255)):
        bit=(k>>t)&1;swap^=bit
        if swap:x2,x3,z2,z3=x3,x2,z3,z2
        swap=bit
        a=(x2+z2)%_P;aa=a*a%_P;b_=(x2-z2)%_P;bb=b_*b_%_P;e=(aa-bb)%_P
        c=(x3+z3)%_P;d=(x3-z3)%_P;da=d*a%_P;cb=c*b_%_P
        x3=(da+cb)**2%_P;z3=x1*(da-cb)**2%_P;x2=aa*bb%_P;z2=e*(aa+_A24*e)%_P
    if swap:x2,x3,z2,z3=x3,x2,z3,z2
    out=(x2*pow(z2,_P-2,_P)%_P).to_bytes(32,'little')
    if out==bytes(32):raise ValueError('Low-order point')
    return out

BASE=(9).to_bytes(32,'little')

def keypair():
    sk=secrets.token_bytes(32);return sk,x25519(sk,BASE)

def agree(sk,pk,label):
    # segredo partilhado -> chave de 32 bytes; label separa os usos (cifrar / mascarar)
    return hashlib.sha256(x25519(sk,pk)+label).digest()

# ---------------------------------------------------------------- Shamir em GF(2^521-1)
Q=2**521-1;SHARE_BYTES=66

def share(secret,t,xs):
    coef=[secret]+[secrets.randbelow(Q) for _ in range(t-1)]  # polinómio de grau t-1
    def ev(x):
        y=0
        for c in reversed(coef):y=(y*x+c)%Q
        return y
    return {x:ev(x) for x in xs}

def reconstruct(points):
    # interpolação de Lagrange em 0
    s=0
    for i in points:
        num=den=1
        for j in points:
            if j!=i:num=num*(-j)%Q;den=den*(i-j)%Q
        s=(s+points[i]*num*pow(den,Q-2,Q))%Q
    return s

# ---------------------------------------------------------------- cifragem dos pedaços (SHAKE-256 + HMAC)
def seal(key,data):
    nonce=secrets.token_bytes(12);ct=bytes(a^b for a,b in zip(data,hashlib.shake_256(key+nonce).digest(len(data))))
    return nonce+ct+hmac.new(key,nonce+ct,'sha256').digest()[:16]

def unseal(key,box):
    nonce,ct,tag=box[:12],box[12:-16],box[-16:]
    if not hmac.compare_digest(tag,hmac.new(key,nonce+ct,'sha256').digest()[:16]):raise ValueError('Bad tag')
    return bytes(a^b for a,b in zip(ct,hashlib.shake_256(key+nonce).digest(len(ct))))

# ---------------------------------------------------------------- máscaras e quantização (módulo 2^32)
MOD_BITS=32;MOD=1<<MOD_BITS

def prg(seed,length):
    return np.frombuffer(hashlib.shake_256(b'mask'+seed).digest(4*length),dtype='<u4').astype(np.uint64)

def quantize(x,bits):
    return np.rint(np.asarray(x,dtype=np.float64)*(1<<bits)).astype(np.uint64)%MOD

def dequantize(q,bits):
    return np.asarray(q,dtype=np.float64)/(1<<bits)

def max_bits(n_summed):
    # maior escala sem transbordar ao somar n_summed valores em [0,1]
    return MOD_BITS-int(np.ceil(np.log2(n_summed+1)))

# ---------------------------------------------------------------- cliente e servidor
class Client:
    def __init__(self,uid,vector):
        self.uid,self.x=uid,np.asarray(vector,dtype=np.uint64)
        self.c_sk,self.c_pk=keypair()  # cifrar pedaços
        self.s_sk,self.s_pk=keypair()  # máscaras de pares
        self.time=0.;self.up=0;self.down=0
    def advertise(self):  # etapa 0: chaves públicas
        self.up+=64;return self.c_pk,self.s_pk
    def share_keys(self,keys,t):  # etapa 1: pedaços cifrados de s_sk e da semente b
        start=time.perf_counter();self.keys=keys;self.t=t
        self.down+=64*(len(keys)-1)
        self.b=secrets.token_bytes(32)
        xs=sorted(keys);sh_s=share(int.from_bytes(self.s_sk,'big'),t,xs);sh_b=share(int.from_bytes(self.b,'big'),t,xs)
        boxes={}
        for v in xs:
            if v==self.uid:self.own=(sh_s[v],sh_b[v]);continue
            msg=self.uid.to_bytes(4,'big')+v.to_bytes(4,'big')+sh_s[v].to_bytes(SHARE_BYTES,'big')+sh_b[v].to_bytes(SHARE_BYTES,'big')
            boxes[v]=seal(agree(self.c_sk,keys[v][0],b'enc'),msg)
        self.up+=sum(len(b) for b in boxes.values());self.time+=time.perf_counter()-start
        return boxes
    def masked_input(self,boxes):  # etapa 2: x + máscara própria ± máscaras de pares
        start=time.perf_counter();self.received={}
        for u,box in boxes.items():
            msg=unseal(agree(self.c_sk,self.keys[u][0],b'enc'),box)
            assert int.from_bytes(msg[:4],'big')==u and int.from_bytes(msg[4:8],'big')==self.uid
            self.received[u]=(int.from_bytes(msg[8:8+SHARE_BYTES],'big'),int.from_bytes(msg[8+SHARE_BYTES:],'big'))
        self.received[self.uid]=self.own
        self.down+=sum(len(b) for b in boxes.values())
        y=(self.x+prg(self.b,len(self.x)))%MOD
        for v in self.keys:
            if v==self.uid:continue
            m=prg(agree(self.s_sk,self.keys[v][1],b'pair'),len(self.x))
            y=(y+m)%MOD if self.uid<v else (y+MOD-m)%MOD
        self.up+=4*len(y);self.time+=time.perf_counter()-start
        return y
    def unmask(self,survivors,dropped):  # etapa 3: nunca revela os dois pedaços do mesmo cliente
        if set(survivors)&set(dropped):raise ValueError('A client cannot be both survivor and dropped')
        if len(survivors)<self.t:raise ValueError('Fewer survivors than t: refuse to unmask')
        self.down+=4*(len(survivors)+len(dropped))
        out=dict(b={u:self.received[u][1] for u in survivors if u in self.received},
                 s={v:self.received[v][0] for v in dropped if v in self.received})
        self.up+=SHARE_BYTES*(len(out['b'])+len(out['s']))
        return out

class Server:
    def __init__(self,t,length):
        self.t,self.length=t,length;self.time=0.;self.view={}
    def collect_keys(self,adverts):
        if len(adverts)<self.t:raise RuntimeError('Not enough clients')
        self.keys=adverts;return adverts
    def route(self,all_boxes):
        routed={v:{} for v in all_boxes}
        for u,boxes in all_boxes.items():
            for v,box in boxes.items():
                if v in routed:routed[v][u]=box
        return routed
    def aggregate(self,masked,responses):
        start=time.perf_counter();survivors=sorted(masked);dropped=sorted(set(self.keys)-set(masked))
        if len(responses)<self.t:raise RuntimeError(f'Only {len(responses)} responses, t={self.t}')
        self.view=dict(masked)  # tudo o que o servidor vê de cada cliente
        first_t=list(responses.items())[:self.t]
        total=np.zeros(self.length,dtype=np.uint64)
        for y in masked.values():total=(total+y)%MOD
        for u in survivors:
            b=reconstruct({r:resp['b'][u] for r,resp in first_t})
            total=(total+MOD-prg(b.to_bytes(32,'big'),self.length))%MOD
        for v in dropped:
            s_sk=reconstruct({r:resp['s'][v] for r,resp in first_t}).to_bytes(32,'big')
            for u in survivors:
                m=prg(agree(s_sk,self.keys[u][1],b'pair'),self.length)
                total=(total+MOD-m)%MOD if u<v else (total+m)%MOD
        self.time+=time.perf_counter()-start
        return total

def secure_sum(vectors,t,drop_masked=(),drop_unmask=()):
    """Uma ronda de SA. vectors: {uid (1..n): array uint64}.
    drop_masked: desistem antes de enviar os dados; drop_unmask: desistem antes de desmascarar."""
    clients={u:Client(u,x) for u,x in vectors.items()};length=len(next(iter(vectors.values())))
    server=Server(t,length);start=time.perf_counter()
    keys=server.collect_keys({u:c.advertise() for u,c in clients.items()})
    routed=server.route({u:c.share_keys(keys,t) for u,c in clients.items()})
    masked={u:c.masked_input(routed[u]) for u,c in clients.items() if u not in drop_masked}
    if len(masked)<t:raise RuntimeError(f'Only {len(masked)} masked inputs, t={t}')
    dropped=sorted(set(clients)-set(masked))
    responses={u:clients[u].unmask(sorted(masked),dropped) for u in masked if u not in drop_unmask}
    total=server.aggregate(masked,responses)
    active=[clients[u] for u in masked]
    report=dict(n=len(clients),t=t,survivors=len(masked),responders=len(responses),
        client_seconds_mean=float(np.mean([c.time for c in active])),server_seconds=server.time,
        wall_seconds=time.perf_counter()-start,
        upload_bytes_mean=float(np.mean([c.up for c in active])),download_bytes_mean=float(np.mean([c.down for c in active])),
        plain_upload_bytes=4*length,survivor_ids=sorted(masked),server_view=server.view)
    return total,report

# ---------------------------------------------------------------- soft labels do projeto
ROOT=Path(__file__).resolve().parents[2]
# Validated fit-weighted training exports; provenance in sa_inputs/fit_weighted_v2/manifest.json.
RUNS=ROOT/'SoftLabels tests'/'final candidates'/'sa_inputs'/'fit_weighted_v2'/'confirmation'

def load_rounds(candidate='A_AE_CNN__MobileNetV2',seed=2031):
    rows=[]
    for r in (1,2,3):
        d=np.load(RUNS/candidate/str(seed)/f'teacher_round_{r}.npz',allow_pickle=False)
        rows.append(dict(round=r,logits=d['logits'],consensus=d['consensus'],weights=d['weights'],
            client_ids=[str(c) for c in d['client_ids']]))
    return rows

def softmax(z):
    z=np.asarray(z,dtype=np.float64);z=z-z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)

def origin(client_id):return client_id.split(':')[0]

def normalized_weights(ids,planned_weights):
    """Public fit-size weights, normalized before fixed-point encoding."""
    w=np.asarray(planned_weights,dtype=np.float64)
    if w.shape!=(len(ids),) or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Provide one positive finite fit weight per client')
    return w/w.sum()

def plain_consensus(probs,ids,survivors=None,planned_weights=None):
    """Fit-weighted reference, renormalized over clients that sent masked inputs."""
    w=normalized_weights(ids,planned_weights)
    keep=np.arange(len(ids)) if survivors is None else np.asarray(survivors,dtype=int)
    if not len(keep):raise ValueError('No surviving clients')
    return np.average(np.asarray(probs)[keep],axis=0,weights=w[keep])

def sa_consensus(probs,ids,t,bits,mode='weighted',drop_masked=(),drop_unmask=(),planned_weights=None):
    """Fit-weighted consensus from secure sums, with survivor renormalization.

    weighted: a single sum of w_i*p_i (only the total is revealed).
    per_origin: the same weighted contributions in separate source blocks;
    source totals are added, never averaged with a 50/50 rule.
    """
    probs=np.asarray(probs,dtype=np.float64)
    n=len(ids);shape=probs.shape[1:];size=int(np.prod(shape))
    if probs.shape[0]!=n or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():
        raise ValueError('Expected finite probabilities in [0,1] for every client')
    w=normalized_weights(ids,planned_weights)
    if not isinstance(bits,(int,np.integer)) or not 0<=bits<=max_bits(1):raise ValueError('bits overflow the modulus')
    origins=sorted({origin(c) for c in ids})
    if mode=='per_origin':
        vectors={}
        for i,c in enumerate(ids):
            v=np.zeros(len(origins)*size);k=origins.index(origin(c))
            v[k*size:(k+1)*size]=w[i]*probs[i].ravel()
            vectors[i+1]=quantize(v,bits)
    elif mode=='weighted':
        vectors={i+1:quantize(w[i]*probs[i].ravel(),bits) for i in range(n)}
    else:raise ValueError(mode)
    total,report=secure_sum(vectors,t,drop_masked,drop_unmask)
    total=dequantize(total,bits);survivors=[u-1 for u in report['survivor_ids']]
    if mode=='per_origin':total=total.reshape(len(origins),size).sum(axis=0)
    result=(total/w[survivors].sum()).reshape(shape)
    report['surviving_weight']=float(w[survivors].sum())
    report['source_weights']={o:float(sum(w[i] for i in survivors if origin(ids[i])==o)/w[survivors].sum()) for o in origins}
    return result,report,survivors

def compare(new,ref):
    """Erro máximo, erro médio absoluto e decisões normal/anomalia alteradas."""
    return dict(erro_max=float(np.abs(new-ref).max()),mae=float(np.abs(new-ref).mean()),
        decisoes_alteradas=int((new.argmax(-1)!=ref.argmax(-1)).sum()))
