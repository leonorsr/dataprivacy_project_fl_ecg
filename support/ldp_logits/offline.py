"""Run only clipping/noise experiments on frozen logits; never start FL training."""
import argparse
from .experiencias_logits import executar,FONTE_PADRAO,SAIDA_PADRAO
from . import graficos
from .offline_notebooks import refresh
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeticoes',type=int,default=200);args=p.parse_args()
    executar(FONTE_PADRAO,SAIDA_PADRAO,args.repeticoes,2031,1e-5)
    frame=pd.read_csv(SAIDA_PADRAO/'resumo.csv');chosen=graficos.escolher_melhor_c(frame)
    graficos.SAIDA.mkdir(parents=True,exist_ok=True);graficos.configurar_estilo()
    graficos.graficos_epsilon(chosen,150);graficos.grafico_efeito_c(frame,chosen,4.,150);refresh()

if __name__=='__main__':main()
