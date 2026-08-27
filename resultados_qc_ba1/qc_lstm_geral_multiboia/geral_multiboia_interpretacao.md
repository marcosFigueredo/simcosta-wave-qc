# Modelo geral multi-boia (so Hsig, sem variaveis auxiliares)

Treinado com dados combinados de 6 boias SiMCosta (BA-1, ES-1, PR-1, RJ-1, RJ-2, RJ-4),
divisao 60/20/20 cronologica dentro de cada boia (nunca misturando tempo entre boias),
depois agrupadas num unico conjunto de treino/validacao/teste.

| boia                   |    n |   n_weak |   tau_b |   binary_f1_all |   macro_f1 |   f1_good |   f1_suspect |   f1_bad |   binary_f1 |   binary_precision |   binary_recall |   auprc_bad |   auroc_bad |
|:-----------------------|-----:|---------:|--------:|----------------:|-----------:|----------:|-------------:|---------:|------------:|-------------------:|----------------:|------------:|------------:|
| TODAS (rótulos fortes) | 6456 |     1899 |   0.700 |           0.746 |      0.742 |     0.977 |        0.634 |    0.614 |       0.853 |              0.890 |           0.819 |       0.647 |       0.943 |
| BA-1                   | 1555 |       29 |   0.700 |           0.717 |      0.697 |     0.981 |        0.552 |    0.557 |       0.689 |              0.762 |           0.629 |       0.594 |       0.947 |
| ES-1                   |  251 |      184 |   0.700 |           0.819 |      0.751 |     0.977 |        0.848 |    0.429 |       0.920 |              0.912 |           0.929 |       0.560 |       0.922 |
| PR-1                   |  584 |     1000 |   0.700 |           0.513 |      0.540 |     0.907 |        0.398 |    0.315 |       0.755 |              0.992 |           0.610 |       0.443 |       0.827 |
| RJ-1                   | 1293 |      291 |   0.700 |           0.934 |      0.880 |     0.989 |        0.850 |    0.800 |       0.913 |              0.926 |           0.901 |       0.841 |       0.960 |
| RJ-2                   | 1537 |       47 |   0.700 |           0.935 |      0.703 |     0.990 |        0.379 |    0.738 |       0.945 |              0.955 |           0.934 |       0.938 |       0.982 |
| RJ-4                   | 1236 |      348 |   0.700 |           0.839 |      0.711 |     0.969 |        0.727 |    0.438 |       0.836 |              0.779 |           0.902 |       0.544 |       0.893 |

As métricas principais usam somente pontos com rótulo forte; n_weak registra os pontos
com rótulo fraco, e binary_f1_all conserva a análise de sensibilidade no conjunto completo.