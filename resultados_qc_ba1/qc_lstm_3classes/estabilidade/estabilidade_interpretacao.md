# Estabilidade multi-semente - QC-LSTM 3 classes (secao 10.5 do protocolo)

3 sementes de modelo (LSTM preditora + cabeca classificadora retreinadas do zero)
x 3 replicas de injecao sintetica = 9 execucoes completas (reduzido do 3x5=15 sugerido pelo protocolo por custo computacional nesta rodada).

## Resultado agregado

| metrica                      |   mean |      std |
|:-----------------------------|-------:|---------:|
| macro_f1                     | 0.7202 |   0.0252 |
| weighted_f1                  | 0.8657 |   0.0228 |
| balanced_accuracy            | 0.7234 |   0.0204 |
| mcc                          | 0.7538 |   0.0427 |
| f1_good                      | 0.9161 |   0.0133 |
| f1_suspect                   | 0.8643 |   0.0360 |
| f1_bad                       | 0.3802 |   0.0686 |
| auprc_bad                    | 0.3250 |   0.0884 |
| macro_f1_bootstrap_ci95_low  | 0.7057 | nan      |
| macro_f1_bootstrap_ci95_high | 0.7366 | nan      |

Desvio-padrao do macro-F1 entre execucoes, 0.0252 (meta do protocolo, < 0.05, atingida: True).

## Ranking dos modelos por semente (media do macro-F1 sobre as replicas de injecao)

|   model_seed |   mean |    std |
|-------------:|-------:|-------:|
|      42.0000 | 0.7326 | 0.0286 |
|     123.0000 | 0.7214 | 0.0121 |
|       7.0000 | 0.7068 | 0.0329 |

## Execucoes individuais

|   model_seed |   anomaly_replica |   macro_f1 |   weighted_f1 |   balanced_accuracy |   mcc |   f1_good |   f1_suspect |   f1_bad |   auprc_bad |
|-------------:|------------------:|-----------:|--------------:|--------------------:|------:|----------:|-------------:|---------:|------------:|
|       42.000 |             0.000 |      0.710 |         0.891 |               0.713 | 0.802 |     0.918 |        0.905 |    0.308 |       0.225 |
|       42.000 |             1.000 |      0.723 |         0.864 |               0.734 | 0.750 |     0.907 |        0.894 |    0.366 |       0.333 |
|       42.000 |             2.000 |      0.765 |         0.884 |               0.758 | 0.800 |     0.934 |        0.899 |    0.462 |       0.425 |
|        7.000 |             0.000 |      0.683 |         0.865 |               0.689 | 0.740 |     0.908 |        0.855 |    0.286 |       0.287 |
|        7.000 |             1.000 |      0.693 |         0.829 |               0.714 | 0.687 |     0.900 |        0.816 |    0.364 |       0.244 |
|        7.000 |             2.000 |      0.744 |         0.870 |               0.746 | 0.765 |     0.924 |        0.872 |    0.437 |       0.405 |
|      123.000 |             0.000 |      0.719 |         0.898 |               0.713 | 0.805 |     0.938 |        0.889 |    0.330 |       0.240 |
|      123.000 |             1.000 |      0.711 |         0.847 |               0.715 | 0.718 |     0.909 |        0.836 |    0.387 |       0.300 |
|      123.000 |             2.000 |      0.734 |         0.845 |               0.729 | 0.717 |     0.907 |        0.812 |    0.483 |       0.467 |