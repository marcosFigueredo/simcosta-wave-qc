from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_3classes" / "figures"

BACKBONE = [
    ("Dados das boias SiMCosta\n(BA-1: 11 variaveis; 6 boias: Hsig)", "#2f5f8a"),
    ("Pre-processamento e\nselecao de variaveis", "#2f5f8a"),
]

# Ramo dos testes estatisticos classicos, aplicado as onze variaveis
# principais sobre a serie real completa, sem anomalia sintetica -
# independente do ramo de IA a direita, os dois so compartilham o
# pre-processamento. Inalterado pela mudanca de blueprint da IA.
CLASSIC_BRANCH = [
    ("Indice de Qualidade\n(QI, 7 testes combinados)", "#8a5a2f"),
    ("Dixon 4-sigma", "#8a5a2f"),
    ("Q-Dixon", "#8a5a2f"),
]
CLASSIC_CONSOLIDATION = ("Consolidacao dos flags\n(script 05)", "#8a5a2f")
CLASSIC_LABEL_STEP = ("Rotulo QC classico\n(use / review / remove)", "#c9711f")
CLASSIC_DB = ("Base QC-ready\n(11 variaveis, dado real)", "#0d3b66")

GROUP_CLASSIC_LABEL = "Testes estatisticos classicos (11 variaveis, dado real, sem anomalia sintetica)"
GROUP_IA_LABEL = "Classificador QC causal e ordinal (Hsig, 6 boias combinadas, com anomalia sintetica)"

# Blueprint atual (causal e ordinal, treino geral multi-boia): a LSTM
# preditora univariada (janela de 168h, passo unico causal) fornece a
# previsao Hsig_t; a injecao balanceada de 7 familias de anomalia sintetica,
# os indicadores causais s_t (residuo instantaneo) e os acumuladores de
# drift a_t (EWMA/CUSUM/inclinacao/persistencia) sao combinados num unico
# vetor de features v_t, consumido por UMA cabeca ordinal (duas saidas
# sigmoid q1,q2) que decide Q_t em 3 classes por dois limiares (tau_A,
# tau_B) - substitui a cabeca softmax e o vetor sem acumuladores da versao
# anterior da figura.
LSTM_BOX = ("LSTM preditora causal univariada\n(janela 168h, previsao Hsig_t)", "#5b4b9a")
INJECT_BOX = ("Injecao balanceada de anomalias\n(7 familias, orcamento por familia)", "#b3261e")
STATS_BOX = ("Indicadores causais s_t\n(acc, level, rate, range, MAD, GPD)", "#0f8a5f")
ACCUM_BOX = ("Acumuladores de drift a_t\n(EWMA, CUSUM, inclinacao, persistencia)", "#1f7a6a")
VECTOR_BOX = ("Vetor de features v_t\n(residuo, s_t, a_t, mascara m_t)", "#7a5a9a")
CLASSIFIER_BOX = ("Cabeca ordinal\n(Dense 64-32, saidas sigmoid q1, q2)", "#c9711f")
LABEL_STEP = ("Rotulo QC (Q_t)\nGOOD / SUSPECT / BAD via tau_A, tau_B", "#c9711f")

DB = ("Banco de dados\n(dado original + Q_t + Hsig reconciliado)", "#0d3b66")
RECON_BOX = ("Reconciliacao de Hsig\n(mais severo entre R e Q_t)", "#4f6a2f")

EVAL = (
    "Avaliacao vs. baselines binarios\n(LSTM-Peak, robusto mediana/MAD, GPD-POT,\n"
    "spike tradicional, Isolation Forest, VAE-LSTM)\n"
    "+ desempenho por boia e estabilidade multi-semente (15 execucoes)\n"
    "+ simulador de fluxo em tempo real (deteccao t+1, 10 sementes)\n"
    "valida o metodo, nao alimenta o banco",
    "#1c1c1c",
)


def add_shadow(ax, cx: float, cy: float, w: float, h: float):
    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.07, cy - h / 2 - 0.09),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=0,
        facecolor="#000000",
        alpha=0.06,
        zorder=1,
    )
    ax.add_patch(shadow)


def add_box(ax, cx: float, cy: float, w: float, h: float, label: str, color: str, fontsize: float = 9.3):
    add_shadow(ax, cx, cy, w, h)
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color,
        alpha=0.13,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, color="#1c1c1c",
             linespacing=1.35, zorder=3, fontweight="medium")


def add_arrow(ax, p_from: tuple[float, float], p_to: tuple[float, float], color: str = "#555555", style: str = "-", lw: float = 1.3):
    arrow = FancyArrowPatch(
        p_from,
        p_to,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=style,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)


def add_split(ax, x_top: float, y_top: float, y_mid: float, targets_x: list[float], y_bottom: float, color: str = "#999999"):
    span = targets_x + [x_top]
    ax.plot([x_top, x_top], [y_top, y_mid], color=color, linewidth=1.1, zorder=1)
    ax.plot([min(span), max(span)], [y_mid, y_mid], color=color, linewidth=1.1, zorder=1)
    for tx in targets_x:
        add_arrow(ax, (tx, y_mid), (tx, y_bottom), color="#555555")


def add_merge(ax, sources_x: list[float], y_top: float, y_mid: float, x_bottom: float, y_bottom: float, color: str = "#999999"):
    for sx in sources_x:
        ax.plot([sx, sx], [y_top, y_mid], color=color, linewidth=1.1, zorder=1)
    ax.plot([min(sources_x), max(sources_x)], [y_mid, y_mid], color=color, linewidth=1.1, zorder=1)
    add_arrow(ax, (x_bottom, y_mid), (x_bottom, y_bottom), color="#555555")


def draw_diagram() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(19.5, 11.5))

    box_w, box_h = 4.1, 1.15
    y_row1, y_row2, y_row3, y_row4, y_row5, y_row6 = 19.8, 17.1, 14.4, 11.2, 8.0, 4.6

    # Linha 1: backbone (dados -> preprocessamento), lado a lado
    x_dados, x_prep = -2.35, 2.35
    add_box(ax, x_dados, y_row1, box_w, box_h, BACKBONE[0][0], BACKBONE[0][1])
    add_box(ax, x_prep, y_row1, box_w, box_h, BACKBONE[1][0], BACKBONE[1][1])
    add_arrow(ax, (x_dados + box_w / 2, y_row1), (x_prep - box_w / 2, y_row1))

    # Linha 2: tres testes classicos (esquerda) e LSTM preditora causal
    # (direita), ramos que so compartilham o pre-processamento em comum.
    x_lstm = 3.4
    lstm_w = 5.2
    add_box(ax, x_lstm, y_row2, lstm_w, box_h, LSTM_BOX[0], LSTM_BOX[1], fontsize=8.8)

    classic_w, classic_gap = 3.2, 0.35
    classic_step = classic_w + classic_gap
    classic_group_right = -1.0 - classic_w / 2
    classic_centers = [classic_group_right - i * classic_step for i in range(3)][::-1]
    for cx, (label, color) in zip(classic_centers, CLASSIC_BRANCH):
        add_box(ax, cx, y_row2, classic_w, box_h, label, color, fontsize=8.6)

    add_split(
        ax, x_prep, y_row1 - box_h / 2, (y_row1 + y_row2) / 2,
        classic_centers + [x_lstm], y_row2 + box_h / 2,
    )

    ax.text(
        (min(classic_centers) + max(classic_centers)) / 2, y_row2 + box_h / 2 + 0.75,
        GROUP_CLASSIC_LABEL, ha="center", va="bottom", fontsize=8.6, color="#8a5a2f", fontweight="bold",
    )
    ax.text(
        x_lstm, y_row2 + box_h / 2 + 0.35,
        GROUP_IA_LABEL, ha="center", va="bottom", fontsize=8.6, color="#5b4b9a", fontweight="bold",
    )

    # Linha 3: Consolidacao classica (esquerda) e, a direita, os dois
    # ingredientes que alimentam o vetor de features v_t - a injecao de
    # anomalias sinteticas (atua sobre o valor observado, independente da
    # LSTM) e os indicadores estatisticos s_t (calculados a partir do
    # residuo observado-previsto).
    x_classic = (min(classic_centers) + max(classic_centers)) / 2
    add_box(ax, x_classic, y_row3, classic_w, box_h, CLASSIC_CONSOLIDATION[0], CLASSIC_CONSOLIDATION[1], fontsize=8.6)
    add_merge(ax, classic_centers, y_row2 - box_h / 2, (y_row2 + y_row3) / 2 + 0.35, x_classic, y_row3 + box_h / 2, color="#c9a877")

    stat_w, stat_gap = 3.5, 0.3
    stat_step = stat_w + stat_gap
    x_inject, x_stats, x_accum = x_lstm - stat_step, x_lstm, x_lstm + stat_step
    add_box(ax, x_inject, y_row3, stat_w, box_h, INJECT_BOX[0], INJECT_BOX[1], fontsize=8.0)
    add_box(ax, x_stats, y_row3, stat_w, box_h, STATS_BOX[0], STATS_BOX[1], fontsize=8.0)
    add_box(ax, x_accum, y_row3, stat_w, box_h, ACCUM_BOX[0], ACCUM_BOX[1], fontsize=8.0)
    add_arrow(ax, (x_lstm - lstm_w / 4, y_row2 - box_h / 2), (x_inject, y_row3 + box_h / 2), color="#999999")
    add_arrow(ax, (x_lstm, y_row2 - box_h / 2), (x_stats, y_row3 + box_h / 2), color="#5b4b9a")
    add_arrow(ax, (x_lstm + lstm_w / 4, y_row2 - box_h / 2), (x_accum, y_row3 + box_h / 2), color="#5b4b9a")

    # Linha 4: rotulo classico classico (esquerda) e vetor v_t (direita),
    # ponto onde a previsao/estado oculto da LSTM, a injecao sintetica e os
    # indicadores estatisticos convergem antes da cabeca classificadora.
    classic_label_w, classic_label_h = classic_w, box_h
    add_box(ax, x_classic, y_row4, classic_label_w, classic_label_h, CLASSIC_LABEL_STEP[0], CLASSIC_LABEL_STEP[1], fontsize=8.2)
    add_arrow(ax, (x_classic, y_row3 - box_h / 2), (x_classic, y_row4 + classic_label_h / 2), color="#8a5a2f")

    vector_w = 6.4
    add_box(ax, x_lstm, y_row4, vector_w, box_h, VECTOR_BOX[0], VECTOR_BOX[1], fontsize=8.8)
    add_merge(ax, [x_inject, x_stats, x_accum], y_row3 - box_h / 2, (y_row3 + y_row4) / 2 + 0.35, x_lstm, y_row4 + box_h / 2, color="#7a5a9a")

    # Linha 5: banco classico (esquerda) e cabeca classificadora (direita).
    classic_db_w, classic_db_h = 3.6, 1.3
    add_box(ax, x_classic, y_row5, classic_db_w, classic_db_h, CLASSIC_DB[0], CLASSIC_DB[1], fontsize=8.6)
    add_arrow(ax, (x_classic, y_row4 - classic_label_h / 2), (x_classic, y_row5 + classic_db_h / 2), color="#8a5a2f")

    classifier_w = 5.0
    add_box(ax, x_lstm, y_row5, classifier_w, box_h, CLASSIFIER_BOX[0], CLASSIFIER_BOX[1], fontsize=8.8)
    add_arrow(ax, (x_lstm, y_row4 - box_h / 2), (x_lstm, y_row5 + box_h / 2), color="#7a5a9a")

    # Linha 6: Rotulo QC de 3 classes -> Banco de dados compartilhado.
    label_w, label_h = 6.0, 1.15
    add_box(ax, x_lstm, y_row6, label_w, label_h, LABEL_STEP[0], LABEL_STEP[1], fontsize=8.8)
    add_arrow(ax, (x_lstm, y_row5 - box_h / 2), (x_lstm, y_row6 + label_h / 2), color="#c9711f")

    db_w, db_h = 6.8, 1.35
    y_db = y_row6 - 2.6
    add_box(ax, x_lstm, y_db, db_w, db_h, DB[0], DB[1], fontsize=9.6)
    add_arrow(ax, (x_lstm, y_row6 - label_h / 2), (x_lstm, y_db + db_h / 2), color="#0d3b66", lw=1.6)

    # Reconciliacao de Hsig entre os dois ramos: so para essa variavel, R
    # (ramo classico) e Q_t (ramo IA) sao combinados pelo mais severo dos
    # dois num rotulo adicional, gravado junto do Q_t no mesmo banco, sem
    # substituir nenhum dos dois rotulos originais.
    x_recon = (x_classic + x_lstm) / 2 - 0.3
    y_recon = (y_row6 + y_db) / 2 + 0.55
    recon_w, recon_h = 4.4, 1.15
    add_box(ax, x_recon, y_recon, recon_w, recon_h, RECON_BOX[0], RECON_BOX[1], fontsize=8.0)
    add_arrow(ax, (x_classic, y_row5 - classic_db_h / 2), (x_recon - recon_w / 3, y_recon + recon_h / 2),
              color="#4f6a2f", style=(0, (4, 3)))
    add_arrow(ax, (x_lstm, y_row6 - label_h / 2), (x_recon + recon_w / 3, y_recon + recon_h / 2),
              color="#4f6a2f", style=(0, (4, 3)))
    add_arrow(ax, (x_recon, y_recon - recon_h / 2), (x_lstm - db_w / 2 + 0.6, y_db + db_h / 2),
              color="#4f6a2f", style=(0, (4, 3)))

    # Caminho 2 (dado bruto, sempre salvo): Dados do sensor -> Banco de
    # dados, contornando toda a etapa de rotulacao. Roteado pela margem
    # esquerda para nao cruzar as outras caixas.
    x_raw = min(classic_centers) - classic_w / 2 - 1.5
    y_raw_start = y_row1 - box_h / 2
    ax.plot([x_dados, x_dados], [y_raw_start, y_raw_start - 0.4], color="#2f5f8a", linewidth=1.3, linestyle=(0, (6, 3)), zorder=1)
    ax.plot([x_dados, x_raw], [y_raw_start - 0.4, y_raw_start - 0.4], color="#2f5f8a", linewidth=1.3, linestyle=(0, (6, 3)), zorder=1)
    ax.plot([x_raw, x_raw], [y_raw_start - 0.4, y_db], color="#2f5f8a", linewidth=1.3, linestyle=(0, (6, 3)), zorder=1)
    add_arrow(ax, (x_raw, y_db), (x_lstm - db_w / 2, y_db), color="#2f5f8a", style=(0, (6, 3)))
    ax.text(
        x_raw - 0.25, (y_raw_start - 0.4 + y_db) / 2, "dado bruto\n(sempre salvo,\nindependente do QC)",
        ha="right", va="center", fontsize=7.6, color="#2f5f8a", style="italic",
    )

    # Ramo lateral de validacao: Rotulo QC -> Avaliacao (nao alimenta o
    # banco, existe so porque este benchmark tem gabarito sintetico e
    # baselines binarios de comparacao).
    eval_w, eval_h = 7.8, 2.9
    x_eval = x_lstm + classifier_w / 2 + eval_w / 2 + 1.0
    y_eval = y_row6
    add_box(ax, x_eval, y_eval, eval_w, eval_h, EVAL[0], EVAL[1], fontsize=7.8)
    add_arrow(
        ax,
        (x_lstm + label_w / 2, y_row6),
        (x_eval - eval_w / 2, y_eval),
        color="#8a8a8a",
        style=(0, (4, 3)),
    )

    ax.set_xlim(x_raw - 2.1, x_eval + eval_w / 2 + 0.6)
    ax.set_ylim(y_db - db_h / 2 - 0.5, y_row1 + box_h / 2 + 0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_fluxo_geral.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    draw_diagram()
    print("Figura salva em", FIG_DIR / "fig08_fluxo_geral.png")
