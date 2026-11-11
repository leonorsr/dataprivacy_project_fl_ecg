"""Write only the active protocol notebooks, with current results and explicit status."""
import base64,contextlib,io
import nbformat
import IPython.display
from .shared import *

def refresh():
    import matplotlib.pyplot as plt
    from .report import render_fedavg,render_soft
    targets=[(ROOT/'FedAvg tests/comparation_fedavg.ipynb',render_fedavg,'render_fedavg'),
             (ROOT/'SoftLabels tests/comparation_softlabels.ipynb',render_soft,'render_soft')]
    for path,render,name in targets:
        output=[]
        def display(value,**kwargs):
            if hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
            elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False,na_rep='N/D'),'text/plain':repr(value)}
            else:data={'text/plain':repr(value)}
            output.append(nbformat.v4.new_output('display_data',data=data))
        original=IPython.display.display;IPython.display.display=display
        try:
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):render()
            if stream.getvalue():output.append(nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
        finally:IPython.display.display=original
        setup="""from pathlib import Path
import sys
PROJECT_ROOT=Path.cwd().resolve()
while not (PROJECT_ROOT/'requirements.txt').exists() and PROJECT_ROOT!=PROJECT_ROOT.parent:
    PROJECT_ROOT=PROJECT_ROOT.parent
sys.path.insert(0,str(PROJECT_ROOT))
"""
        code=f'from support.federated_global.report import {name}\n{name}()'
        cell=nbformat.v4.new_code_cell(code,execution_count=1,outputs=output)
        n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Protocolo ativo: modelo global treinado/agregado e redistribuído. Resultados do fluxo anterior estão arquivados e não sustentam a decisão atual.'),nbformat.v4.new_code_cell(setup),cell],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
        nbformat.validate(n);nbformat.write(n,path)
        print('NOTEBOOK UPDATED',path,flush=True)
