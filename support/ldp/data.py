"""Public synthetic-only feature normalisation, without pooled private fit statistics."""
import ast,importlib.util,inspect,shutil,sys
from pathlib import Path
import numpy as np
from support.federated_global import shared
from support.federated_global.shared import ROOT,DOMAINS,dump,old_soft


def public_anchor_generator():
    tree=ast.parse(inspect.getsource(old_soft.synthetic_queries));found=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Return) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict':
            node.value.keywords.append(ast.keyword(arg='raw_features',value=ast.Name(id='raw',ctx=ast.Load())))
            found+=1
    assert found==1
    ast.fix_missing_locations(tree);namespace=dict(old_soft.__dict__);namespace['SEED']=2026
    exec(compile(tree,'<public synthetic features>','exec'),namespace)
    return namespace['synthetic_queries']


def prepare(base,calibration_anchors=128):
    spec=importlib.util.spec_from_file_location('global_soft.ldp_grouped_data',ROOT/'SoftLabels tests/common/exp4_data.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.BASE=Path(base)/'grouped_data'
    source=ROOT/'SoftLabels tests/artifacts/exp4/data'
    for p in source.glob('*/*.npz'):
        target=module.BASE/'data'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
    domains=module.prepare();generator=public_anchor_generator();public={}
    for ds,d in domains.items():
        dims=len(d['config']['feature_names']);projection=d['projection']
        calibration=generator(d['config'],dict(mean=np.zeros(dims),scale=np.ones(dims),projection=projection),n=calibration_anchors)
        raw=calibration['raw_features'];mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-6)
        for c in d['clients'].values():
            c['F']=np.clip((c['features']-mean)/scale,-8,8).astype(np.float32)
            proj=np.einsum('ij,jk->ik',c['X'].astype(float),projection.astype(float),optimize=False)
            c['Z']=np.column_stack([proj,c['F']]).astype(np.float32)
            if c['split']=='train':c['fit_indices']=c['final_fit_indices']
        anchor=generator(d['config'],dict(mean=mean,scale=scale,projection=projection),n=32)
        anchor.pop('raw_features')
        d['anchor']=anchor
        public[ds]=dict(feature_names=d['config']['feature_names'],mean=mean.tolist(),scale=scale.tolist(),
            calibration=f'{calibration_anchors} public synthetic examples; no private feature statistics',
            source='fixed synthetic generation at public seed 2026',anchor_ids=anchor['sample_ids'].tolist())
        fit={v for c in d['clients'].values() for v in c['sample_ids'][c['fit_indices']]}
        hold={v for c in d['clients'].values() for v in c['sample_ids'][c['hold_indices']]}
        assert not fit&hold and not fit&set(anchor['sample_ids'])
    dump(Path(base)/'public_preprocessing.json',public)
    dump(Path(base)/'research_private'/'sample_registry.json',shared.sample_registry(domains))
    return domains
