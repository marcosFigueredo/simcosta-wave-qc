# Validacao cruzada entre boias, modelo univariado treinado na BA-1

Modelo treinado uma unica vez na BA-1 (so a variavel Hsig, sem nenhuma variavel auxiliar) e
aplicado sem retreino nem recalibracao de limiar as demais boias SiMCosta que relatam Hsig,
com anomalias sinteticas injetadas na propria serie real de cada boia para permitir avaliacao
quantitativa (nao ha rotulo manual de qualidade em nenhuma delas).

| boia                  |   n_pontos |   macro_f1 |   weighted_f1 |   balanced_accuracy |   f1_good |   f1_suspect |   f1_bad |   binary_f1 |   binary_precision |   binary_recall |   auprc_bad |   auroc_bad |   n_bad_real |
|:----------------------|-----------:|-----------:|--------------:|--------------------:|----------:|-------------:|---------:|------------:|-------------------:|----------------:|------------:|------------:|-------------:|
| BA-1 (origem, treino) |       1728 |      0.648 |         0.853 |               0.654 |     0.906 |        0.844 |    0.193 |       0.898 |              0.948 |           0.853 |       0.184 |       0.818 |           59 |
| ES-1                  |       2991 |      0.283 |         0.261 |               0.403 |     0.248 |        0.037 |    0.563 |       0.165 |              0.091 |           0.876 |       0.347 |       0.796 |          184 |
| PR-1                  |       8736 |      0.237 |         0.338 |               0.362 |     0.351 |        0.042 |    0.318 |       0.111 |              0.060 |           0.867 |       0.151 |       0.860 |          135 |
| RJ-1                  |       8736 |      0.182 |         0.213 |               0.331 |     0.219 |        0.038 |    0.289 |       0.106 |              0.056 |           0.914 |       0.155 |       0.888 |          135 |
| RJ-2                  |       8736 |      0.438 |         0.802 |               0.556 |     0.837 |        0.124 |    0.354 |       0.255 |              0.150 |           0.844 |       0.193 |       0.872 |          135 |
| RJ-4                  |       8736 |      0.159 |         0.195 |               0.290 |     0.201 |        0.031 |    0.245 |       0.103 |              0.055 |           0.897 |       0.125 |       0.866 |          135 |

BA-1 e a boia de origem (onde o modelo foi treinado), as demais linhas sao transferencia pura,
sem nenhum ajuste especifico aplicado aos dados dessa boia.