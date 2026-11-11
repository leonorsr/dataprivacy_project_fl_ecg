import os
os.environ.setdefault('FLWR_TELEMETRY_ENABLED','0')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icentia-mpl')
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
DOMAINS=['icentia11k','physionet2017']
SEEDS=[2026,2027,2028]
FED=ROOT/'FedAvg tests/artifacts/global_model_v1'
SOFT=ROOT/'SoftLabels tests/artifacts/global_student_v1'

def package(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
package('global_fed',ROOT/'FedAvg tests/common')
package('global_soft',ROOT/'SoftLabels tests/common')
from global_fed import data as raw,learning as learning,experiment as original_fed
from global_soft import data as old_soft
ELIGIBLE=[m for m in learning.MODELS if m not in {'RandomForest','KNN','GRU5','GRU_Simple'}]

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def signature():
    files=sorted(Path(__file__).parent.glob('*.py'))+[ROOT/'FedAvg tests/common'/n for n in ['models.py','learning.py','experiment.py','data.py','metrics.py']]+[ROOT/'SoftLabels tests/common'/n for n in ['data.py','exp4_data.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()

def arrays_hash(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def model_hash(model):return arrays_hash(learning.get_parameters(model))

def load_smoke():
    old_soft.SEED=2026
    return {ds:old_soft.load_domain(ds,'binary') for ds in DOMAINS}

def sample_registry(domains):
    return {c['uid']:dict(dataset=ds,split=c['split'],fit=c['sample_ids'][c['fit_indices']].tolist(),holdout=c['sample_ids'][c['hold_indices']].tolist()) for ds,d in domains.items() for c in d['clients'].values()}

def save_model(path,model):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(model.state_dict(),temp);temp.replace(path)
