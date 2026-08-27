from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_3classes" / "figures"

ENCODER_BLOCKS = [
    ("Entrada\n24h x n variaveis", "#3f6f8f"),
    ("LSTM\n128 unidades", "#5b4b8a"),
    ("LSTM\n64 unidades", "#5b4b8a"),
    ("LSTM\n32 unidades\n(estado oculto h_t)", "#5b4b8a"),
]

PRED_BLOCKS = [
    ("Dropout 0,2", "#8a8a8a"),
    ("Dense linear\n1 saida", "#d17a22"),
    ("Previsao\nHsig_t", "#16875d"),
]

CLASS_BLOCKS = [
    ("Vetor v_t\nh_t, residual, s_t, z_t, m_t", "#8a5a2f"),
    ("Dense 64\nReLU + Dropout 0,2", "#c9711f"),
    ("Dense 32\nReLU + Dropout 0,1", "#c9711f"),
    ("Dense 3\nSoftmax", "#b3261e"),
    ("Q_t in\n{GOOD, SUSPECT, BAD}", "#7b1f1a"),
]


def add_box(ax, x, y, w, h, label, color, fontsize=9.0):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.18,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#222222")


def add_arrow(ax, p_from, p_to, color="#444444"):
    arrow = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color=color)
    ax.add_patch(arrow)


def draw_diagram() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7.5))

    box_w, box_h = 1.9, 1.05
    gap = 0.55
    n = len(ENCODER_BLOCKS)
    total_w = n * box_w + (n - 1) * gap
    x0 = -total_w / 2
    y_encoder = 2.6

    centers = []
    for i, (label, color) in enumerate(ENCODER_BLOCKS):
        x = x0 + i * (box_w + gap) + box_w / 2
        add_box(ax, x, y_encoder, box_w, box_h, label, color)
        centers.append(x)
    for c_from, c_to in zip(centers[:-1], centers[1:]):
        add_arrow(ax, (c_from + box_w / 2, y_encoder), (c_to - box_w / 2, y_encoder))

    h_center = centers[-1]

    # bifurcacao: cabeca de previsao (esquerda) e cabeca classificadora (direita)
    x_pred = h_center - 2.6
    x_class = h_center + 2.6

    add_arrow(ax, (h_center, y_encoder - box_h / 2), (x_pred, y_encoder - 1.1))
    add_arrow(ax, (h_center, y_encoder - box_h / 2), (x_class, y_encoder - 1.1))

    y = y_encoder - 1.9
    prev_center = None
    for label, color in PRED_BLOCKS:
        add_box(ax, x_pred, y, 1.9, box_h, label, color)
        if prev_center is not None:
            add_arrow(ax, (x_pred, prev_center + box_h / 2), (x_pred, y + box_h / 2))
        prev_center = y
        y -= 1.35

    y = y_encoder - 1.9
    prev_center = None
    for i, (label, color) in enumerate(CLASS_BLOCKS):
        fontsize = 8.3 if i == 0 else 9.0
        add_box(ax, x_class, y, 2.5, box_h, label, color, fontsize=fontsize)
        if prev_center is not None:
            add_arrow(ax, (x_class, prev_center + box_h / 2), (x_class, y + box_h / 2))
        prev_center = y
        y -= 1.35

    # ligacao explicita: previsao e o proprio h_t tambem alimentam o vetor v_t da cabeca classificadora
    add_arrow(ax, (x_pred, y_encoder - 1.9 - 1.35 * 2 + box_h / 2), (x_class - 1.1, y_encoder - 1.9 + box_h / 2), color="#999999")
    ax.annotate(
        "residual = Hsig_t - previsao",
        xy=((x_pred + x_class) / 2, y_encoder - 1.9 + 0.55),
        ha="center", fontsize=7.5, color="#666666",
    )

    ax.annotate(
        "LSTM preditora estritamente causal\n(janela termina em t-1)",
        xy=(h_center, y_encoder + box_h / 2), xytext=(h_center, y_encoder + box_h / 2 + 0.85),
        ha="center", fontsize=8.3, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8, shrinkA=0, shrinkB=2),
    )

    ax.set_xlim(x0 - 0.4, x_class + 1.6)
    ax.set_ylim(y - 0.8, y_encoder + box_h / 2 + 1.4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_arquitetura_lstm.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    draw_diagram()
    print("Figura salva em", FIG_DIR / "fig07_arquitetura_lstm.png")
