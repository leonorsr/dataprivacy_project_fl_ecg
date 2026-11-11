"""Functional checks of the combined client LDP / SA chain; no test data."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from support.federated_global.shared import dump
from support.ldp_logits import closed_loop as previous
from support.ldp_logits.mechanism import Ledger
from .experiment import BASE,CONDITION,aggregate
from . import protocol as SA

def main():
    results={}
    rng=np.random.default_rng(42)
    probabilities=rng.dirichlet([2,3],size=(22,32))
    counts=np.arange(1,23,dtype=float)
    actual,report=aggregate(probabilities.reshape(22,-1),counts,True,'Softlabels')
    assert np.allclose(actual,np.average(probabilities.reshape(22,-1),axis=0,weights=counts),atol=22*.5/2**24)
    results['weighted_sa_sum_matches_reference']=report['secure_integer_sum_verified']
    results['sa_quantization_max_error']=report['aggregate_max_error']
    # Client mechanism is exercised through the same generated training engine.
    with tempfile.TemporaryDirectory(prefix='ldp-sa-smoke-') as temp:
        path=Path(temp);ledger=Ledger(path/'ledger.json');attempt=ledger.begin('smoke/synthetic')
        module,mechanisms=previous.engine(CONDITION,ledger,attempt,path)
        fake_logits=np.linspace(-1,1,64).reshape(32,2)
        with patch.object(previous.core.soft,'predictions',return_value=fake_logits):
            for r in range(1,4):
                p=module.private_prediction(None,'unused','icentia11k',None,'synthetic_client',r)
                assert p.shape==(32,2) and np.allclose(p.sum(-1),1)
        cert=json.loads((path/'certificates/synthetic_client.json').read_text())
        assert cert['releases']==3 and cert['secure_randomness']
        budget=ledger.data['clients']['synthetic_client']
        assert np.isclose(budget['epsilon_private_sum'],4) and np.isclose(budget['delta_private_sum'],1e-5)
        try:mechanisms['synthetic_client'].release(fake_logits)
        except RuntimeError:pass
        else:raise AssertionError('Fourth release allowed')
    results['ldp_before_aggregation_and_cycle_budget']=True
    try:aggregate(np.full((22,64),np.nan),np.ones(22),True,'Softlabels')
    except ValueError:pass
    else:raise AssertionError('Invalid message accepted')
    results['invalid_messages_rejected_in_simulator']=True
    vectors={i:np.array([i,2*i],dtype=np.uint64) for i in range(1,5)}
    total,report=SA.secure_sum(vectors,3,drop_masked=(4,))
    assert np.array_equal(total,sum(vectors[i] for i in [1,2,3])%SA.MOD)
    try:SA.secure_sum(vectors,3,drop_masked=(3,4))
    except RuntimeError:pass
    else:raise AssertionError('Sub-threshold unmasking accepted')
    results['dropout_sum_and_below_threshold_refusal']=True
    key=b'a'*32;sealed=bytearray(SA.seal(key,b'synthetic share'));sealed[13]^=1
    try:SA.unseal(key,bytes(sealed))
    except ValueError:pass
    else:raise AssertionError('Tampered ciphertext accepted')
    results['tampered_share_ciphertext_rejected']=True
    results['scope']='functional checks only, no adversarial training or malicious-security proof'
    dump(BASE/'smoke_checks.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
