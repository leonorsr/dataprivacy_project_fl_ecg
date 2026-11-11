"""Run/update notebooks for offline LDP screening and closed-loop utility evaluation."""
from .experiencias_logits import executar,FONTE_PADRAO,SAIDA_PADRAO
from .closed_loop import run
from . import graficos
import pandas as pd

def main():
    if not (SAIDA_PADRAO/'resumo.csv').exists():executar(FONTE_PADRAO,SAIDA_PADRAO,200,2031,1e-5)
    f=pd.read_csv(SAIDA_PADRAO/'resumo.csv');chosen=graficos.escolher_melhor_c(f)
    graficos.SAIDA.mkdir(parents=True,exist_ok=True);graficos.configurar_estilo()
    graficos.graficos_epsilon(chosen,150);graficos.grafico_efeito_c(f,chosen,4.,150)
    run()

if __name__=='__main__':main()
