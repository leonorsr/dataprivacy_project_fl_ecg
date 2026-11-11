#!/usr/bin/env python3
"""Gerar gráficos offline a partir de resultados/resumo.csv.

Exemplos (executar a partir da pasta LDP ou de qualquer outra pasta)::

    python3 graficos.py
    python3 graficos.py --epsilon 1.0

A entrada é sempre relativa a este ficheiro; não se procuram outros dados.
Os PNG são escritos em resultados/graficos/ e substituídos em execuções seguintes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib

    # Permite gerar ficheiros mesmo num servidor sem ecrã ou ambiente gráfico.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, PercentFormatter
except ImportError as exc:
    raise SystemExit(
        f"Dependência em falta: {exc}. Instale pandas e matplotlib "
        "no ambiente Python antes de executar este script."
    ) from exc


PASTA = Path(__file__).resolve().parents[2] / 'artifacts/ldp_logits'
CSV = PASTA / "offline" / "resumo.csv"
SAIDA = PASTA / "offline" / "graficos"

COLUNAS = (
    "mechanism", "representation", "epsilon_total", "clip_C",
    "prob_mae_mean", "consensus_mae_mean", "consensus_mae_p025",
    "consensus_mae_p975", "consensus_mse_mean", "decision_flips_mean",
    "decision_flips_p025", "decision_flips_p975", "clip_fraction_mean",
)
NUMERICAS = COLUNAS[2:]
MECANISMOS = {"laplace": "Laplace", "gaussiano": "Gaussiano"}
REPRESENTACOES = {
    "original": ("Original", "#2563A3", "o", "-"),
    "centrado": ("Centrado", "#B45309", "s", "--"),
    "margem": ("Margem", "#087F75", "^", "-."),
}
CHAVE = ["mechanism", "representation", "epsilon_total"]
AVISO_SELECAO = (
    "Seleção exploratória de C nos mesmos dados usados na avaliação; "
    "não constitui validação independente."
)
AVISO_QUANTIS = (
    "Bandas: quantis empíricos p2,5–p97,5 fornecidos no CSV; "
    "não são intervalos de confiança."
)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera quatro gráficos de LDP sem treinar modelos. "
            "Lê apenas resultados/resumo.csv, junto deste script."
        ),
        epilog=(
            "Sem --epsilon, o gráfico do efeito de C usa o epsilon central "
            "entre os valores disponíveis. Os PNG existentes são substituídos."
        ),
    )
    parser.add_argument(
        "--epsilon", type=float, default=None, metavar="VALOR",
        help="epsilon_total existente no CSV para o gráfico do efeito de C",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, metavar="DPI",
        help="resolução dos PNG, entre 72 e 600 (predefinição: 200)",
    )
    args = parser.parse_args()
    if args.epsilon is not None and (not math.isfinite(args.epsilon) or args.epsilon <= 0):
        parser.error("--epsilon tem de ser um número finito e positivo")
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi tem de estar entre 72 e 600")
    return args


def dentro_da_pasta(path: Path) -> bool:
    """Evita que uma ligação simbólica desvie a entrada ou saída para fora de LDP."""
    return path.resolve().is_relative_to(PASTA)


def ler_csv() -> pd.DataFrame:
    if not dentro_da_pasta(CSV):
        raise ValueError("O caminho do CSV aponta para fora da pasta LDP.")
    if not CSV.is_file():
        raise ValueError(
            f"CSV não encontrado: {CSV}\n"
            "Crie primeiro resultados/resumo.csv com as colunas documentadas "
            "na especificação da experiência. Nenhum gráfico foi gerado."
        )
    try:
        dados = pd.read_csv(CSV, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Não foi possível ler o CSV {CSV}: {exc}") from exc

    em_falta = [coluna for coluna in COLUNAS if coluna not in dados.columns]
    if em_falta:
        raise ValueError("Faltam colunas obrigatórias no CSV: " + ", ".join(em_falta))
    if dados.empty:
        raise ValueError("O CSV não contém linhas de resultados; nenhum gráfico foi gerado.")

    # Aceitam-se espaços exteriores e maiúsculas, mas não categorias desconhecidas.
    for coluna, validos in (("mechanism", MECANISMOS), ("representation", REPRESENTACOES)):
        if dados[coluna].isna().any():
            raise ValueError(f"Existem valores vazios na coluna {coluna}.")
        dados[coluna] = dados[coluna].astype(str).str.strip().str.lower()
        invalidos = sorted(set(dados[coluna]) - set(validos))
        if invalidos:
            raise ValueError(
                f"Valores desconhecidos em {coluna}: {', '.join(invalidos)}. "
                f"Valores aceites: {', '.join(validos)}."
            )

    for coluna in NUMERICAS:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
        # Verifica igualmente NaN e infinitos; falhar é preferível a omitir linhas.
        invalidos = dados[coluna].isna() | ~dados[coluna].map(math.isfinite)
        if invalidos.any():
            exemplos = ", ".join(str(i + 1) for i in dados.index[invalidos][:5])
            raise ValueError(
                f"A coluna {coluna} contém valores não numéricos ou não finitos "
                f"(registos: {exemplos}; cabeçalho excluído)."
            )

    for coluna in ("epsilon_total", "clip_C"):
        if (dados[coluna] <= 0).any():
            raise ValueError(f"Todos os valores de {coluna} têm de ser positivos.")
    for coluna in ("prob_mae_mean", "consensus_mae_mean", "consensus_mae_p025",
                   "consensus_mae_p975", "consensus_mse_mean"):
        if (dados[coluna] < 0).any():
            raise ValueError(f"A coluna {coluna} não pode conter valores negativos.")
    for coluna in ("decision_flips_mean", "decision_flips_p025", "decision_flips_p975"):
        if ((dados[coluna] < 0) | (dados[coluna] > 32)).any():
            raise ValueError(f"{coluna} tem de indicar um número entre 0 e 32 âncoras.")
    if ((dados["clip_fraction_mean"] < 0) | (dados["clip_fraction_mean"] > 1)).any():
        raise ValueError("clip_fraction_mean tem de ser uma fração entre 0 e 1.")
    for inferior, superior in (("consensus_mae_p025", "consensus_mae_p975"),
                               ("decision_flips_p025", "decision_flips_p975")):
        if (dados[inferior] > dados[superior]).any():
            raise ValueError(f"Há quantis invertidos: {inferior} > {superior}.")

    # Uma repetição da mesma chave deixaria ambíguo qual a linha a escolher.
    chave_completa = CHAVE + ["clip_C"]
    if dados.duplicated(chave_completa).any():
        raise ValueError(
            "Há linhas duplicadas para o mesmo mecanismo, representação, epsilon e C; "
            "o CSV deve conter uma única linha agregada por combinação."
        )
    return dados


def escolher_melhor_c(dados: pd.DataFrame) -> pd.DataFrame:
    """Mínimo MAE por mecanismo/representação/epsilon; empate: menor C."""
    ordenados = dados.sort_values(CHAVE + ["consensus_mae_mean", "clip_C"], kind="stable")
    return ordenados.drop_duplicates(CHAVE, keep="first").copy()


def escolher_epsilon(dados: pd.DataFrame, pedido: float | None) -> float:
    valores = sorted(float(valor) for valor in dados["epsilon_total"].unique())
    if pedido is None:
        return valores[len(valores) // 2]
    # Igualdade exata primeiro; tolerância apenas para erros de representação decimal.
    if pedido in valores:
        return pedido
    candidatos = [v for v in valores if math.isclose(v, pedido, rel_tol=1e-9, abs_tol=1e-12)]
    if len(candidatos) == 1:
        return candidatos[0]
    disponiveis = ", ".join(format(v, ".12g") for v in valores)
    if len(candidatos) > 1:
        raise ValueError(f"--epsilon é ambíguo. Valores disponíveis: {disponiveis}")
    raise ValueError(f"--epsilon {pedido:g} não existe no CSV. Valores disponíveis: {disponiveis}")


def configurar_estilo() -> None:
    # Matplotlib e DejaVu Sans são suficientes; não depende de seaborn nem da rede.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#94A3B8", "axes.labelcolor": "#263449",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "text.color": "#17243A", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "grid.color": "#CBD5E1", "grid.alpha": 0.65, "grid.linewidth": 0.7,
        "lines.linewidth": 2.0,
    })


def escala_x(ax: plt.Axes, valores: pd.Series) -> None:
    """Usa escala logarítmica apenas quando os valores cobrem pelo menos uma década."""
    minimo, maximo = float(valores.min()), float(valores.max())
    if maximo / minimo >= 10:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}"))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="both", which="major")
    ax.set_axisbelow(True)


def criar_paineis(dados: pd.DataFrame, titulo: str, subtitulo: str,
                  quantis: bool = False, marcador_selecao: bool = False):
    mecanismos = [m for m in MECANISMOS if m in set(dados["mechanism"])]
    fig, axes = plt.subplots(1, len(mecanismos), figsize=(6.4 * len(mecanismos), 6.3),
                             squeeze=False, sharey=True)
    axes = axes[0]
    fig.suptitle(titulo, x=0.5, y=0.97, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.915, subtitulo, ha="center", fontsize=10, color="#475569")
    handles = []
    for rep, (nome, cor, marcador, estilo) in REPRESENTACOES.items():
        if rep in set(dados["representation"]):
            handles.append(Line2D([0], [0], color=cor, marker=marcador,
                                  linestyle=estilo, markersize=6, label=nome))
    if marcador_selecao:
        handles.append(Line2D([0], [0], color="#17243A", marker="*", linestyle="None",
                              markerfacecolor="white", markersize=11, label="C selecionado"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.87),
               ncol=min(len(handles), 4), frameon=False, fontsize=9)
    fig.text(0.5, 0.105, AVISO_SELECAO, ha="center", fontsize=8.4, color="#475569")
    if quantis:
        fig.text(0.5, 0.065, AVISO_QUANTIS, ha="center", fontsize=8.4, color="#475569")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.77, bottom=0.22, wspace=0.22)
    return fig, axes, mecanismos


def desenhar_series(ax: plt.Axes, dados: pd.DataFrame, eixo_x: str, media: str,
                     inferior: str | None = None, superior: str | None = None) -> None:
    for rep, (_, cor, marcador, estilo) in REPRESENTACOES.items():
        serie = dados.loc[dados["representation"] == rep].sort_values(eixo_x)
        if serie.empty:
            continue
        x = serie[eixo_x].to_numpy()
        y = serie[media].to_numpy()
        ax.plot(x, y, color=cor, marker=marcador, linestyle=estilo, markersize=5.5)
        if inferior is not None and superior is not None:
            baixo, alto = serie[inferior].to_numpy(), serie[superior].to_numpy()
            if len(serie) == 1:
                # Uma banda com apenas um x não tem área; mostrar o intervalo como segmento.
                ax.vlines(x, baixo, alto, color=cor, alpha=0.55, linewidth=3)
            else:
                ax.fill_between(x, baixo, alto, color=cor, alpha=0.13, linewidth=0)


def guardar(fig: plt.Figure, nome: str, dpi: int) -> Path:
    destino = SAIDA / nome
    # Não seguir um PNG preexistente que seja uma ligação simbólica para outro ficheiro.
    if destino.is_symlink() or not dentro_da_pasta(destino):
        raise ValueError(f"Destino inseguro (ligação simbólica ou fora de LDP): {destino}")
    fig.savefig(destino, dpi=dpi)
    return destino


def graficos_epsilon(escolhidos: pd.DataFrame, dpi: int) -> list[Path]:
    delta = escolhidos.loc[escolhidos["mechanism"] == "gaussiano", "delta_total"].unique() if "delta_total" in escolhidos.columns else []
    nota_delta = f" | δ gaussiano = {float(delta[0]):.1g}" if len(delta) == 1 else ""
    configuracoes = [
        ("consensus_mae_vs_epsilon.png", "Erro médio de consenso vs. privacidade",
         "consensus_mae_mean", "consensus_mae_p025", "consensus_mae_p975",
         "MAE de consenso"),
        ("mudancas_decisao_vs_epsilon.png", "Mudanças de decisão vs. privacidade",
         "decision_flips_mean", "decision_flips_p025", "decision_flips_p975",
         "Mudanças médias (em 32 âncoras)"),
        ("fracao_clipping_vs_epsilon.png", "Fração de clipping vs. privacidade",
         "clip_fraction_mean", None, None, "Fração média de clipping"),
    ]
    caminhos = []
    for nome, titulo, media, inferior, superior, ylabel in configuracoes:
        fig, axes, mecanismos = criar_paineis(
            escolhidos, titulo, "Menor MAE por C | ε total em 3 rondas" + nota_delta,
            quantis=inferior is not None,
        )
        try:
            for ax, mecanismo in zip(axes, mecanismos):
                grupo = escolhidos.loc[escolhidos["mechanism"] == mecanismo]
                desenhar_series(ax, grupo, "epsilon_total", media, inferior, superior)
                escala_x(ax, escolhidos["epsilon_total"])
                ax.set_title(MECANISMOS[mecanismo], pad=12)
                ax.set_xlabel("Orçamento total de privacidade (ε)")
            # Com eixos y partilhados, limitar só depois de desenhar todos os
            # painéis evita que a escala do primeiro corte os pontos do segundo.
            if media == "decision_flips_mean":
                axes[0].set_ylim(0, 32)
            elif media == "clip_fraction_mean":
                axes[0].set_ylim(0, 1)
                axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
            else:
                axes[0].set_ylim(bottom=0)
            axes[0].set_ylabel(ylabel)
            caminhos.append(guardar(fig, nome, dpi))
        finally:
            plt.close(fig)
    return caminhos


def grafico_efeito_c(dados: pd.DataFrame, escolhidos: pd.DataFrame,
                     epsilon: float, dpi: int) -> Path:
    amostra = dados.loc[dados["epsilon_total"] == epsilon]
    pontos = escolhidos.loc[escolhidos["epsilon_total"] == epsilon]
    delta = amostra.loc[amostra["mechanism"] == "gaussiano", "delta_total"].unique() if "delta_total" in amostra.columns else []
    nota_delta = f"  |  δ gaussiano = {float(delta[0]):.1g}" if len(delta) == 1 else ""
    fig, axes, mecanismos = criar_paineis(
        amostra, "Efeito de C no erro de consenso",
        f"ε total = {epsilon:g} em 3 rondas  |  Todos os valores C" + nota_delta, marcador_selecao=True,
    )
    try:
        for ax, mecanismo in zip(axes, mecanismos):
            grupo = amostra.loc[amostra["mechanism"] == mecanismo]
            desenhar_series(ax, grupo, "clip_C", "consensus_mae_mean")
            selecionados = pontos.loc[pontos["mechanism"] == mecanismo]
            ax.scatter(selecionados["clip_C"], selecionados["consensus_mae_mean"],
                       marker="*", s=160, facecolor="white", edgecolor="#17243A",
                       linewidth=1.2, zorder=5)
            escala_x(ax, amostra["clip_C"])
            ax.set_title(MECANISMOS[mecanismo], pad=12)
            ax.set_xlabel("Limiar de clipping C")
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel("MAE de consenso")
        return guardar(fig, "efeito_de_C.png", dpi)
    finally:
        plt.close(fig)


def main() -> int:
    args = argumentos()
    try:
        dados = ler_csv()
        epsilon = escolher_epsilon(dados, args.epsilon)
        escolhidos = escolher_melhor_c(dados)
        if not dentro_da_pasta(SAIDA):
            raise ValueError("A pasta de saída aponta para fora da pasta LDP.")
        SAIDA.mkdir(parents=True, exist_ok=True)
        configurar_estilo()
        caminhos = graficos_epsilon(escolhidos, args.dpi)
        caminhos.append(grafico_efeito_c(dados, escolhidos, epsilon, args.dpi))
    except (ValueError, OSError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"Entrada: {CSV}")
    print(f"Epsilon usado no efeito de C: {epsilon:g}")
    print("Seleção: menor consensus_mae_mean por mecanismo, representação e epsilon; "
          "em caso de empate, menor C.")
    print(AVISO_SELECAO)
    print(AVISO_QUANTIS)
    print("decision_flips é interpretado como número de mudanças em 32 âncoras; "
          "clip_fraction_mean como fração entre 0 e 1.")
    print("Gráficos gerados:")
    for caminho in caminhos:
        print(f"  {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
