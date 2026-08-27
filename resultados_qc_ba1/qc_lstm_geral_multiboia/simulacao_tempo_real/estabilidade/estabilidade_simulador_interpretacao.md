# Estabilidade do simulador em tempo real, multiplas sementes

Mesmo modelo geral ja treinado e salvo (sem retreino), mesma serie real da BA-1, 10 sementes de simulador diferentes (cada uma sorteia uma sequencia distinta
de episodios de anomalia, quando comecam, familia, gravidade e duracao). Metrica principal,
deteccao no instante t+1 (inicio de cada episodio).

Taxa geral t+1, media 0.529, desvio padrao entre sementes 0.082 (n=10 sementes).

## Por familia (eventos de todas as sementes agrupados, IC 95% via bootstrap sobre as taxas por-semente)

| familia   |   n_sementes |   n_eventos_total |   taxa_deteccao_t1_ponderada |   media_das_taxas_por_semente |   desvio_padrao_entre_sementes |   ic95_inferior |   ic95_superior |
|:----------|-------------:|------------------:|-----------------------------:|------------------------------:|-------------------------------:|----------------:|----------------:|
| A         |           10 |                76 |                        0.579 |                         0.576 |                          0.216 |           0.447 |           0.708 |
| B         |           10 |                99 |                        0.747 |                         0.755 |                          0.133 |           0.678 |           0.831 |
| C         |           10 |                80 |                        0.838 |                         0.837 |                          0.207 |           0.704 |           0.941 |
| D         |           10 |                88 |                        0.011 |                         0.009 |                          0.029 |           0.000 |           0.027 |
| E         |           10 |                84 |                        0.048 |                         0.044 |                          0.079 |           0.000 |           0.095 |
| F         |           10 |                98 |                        0.510 |                         0.490 |                          0.223 |           0.348 |           0.609 |
| G         |           10 |                71 |                        1.000 |                         1.000 |                          0.000 |           1.000 |           1.000 |

## Taxa geral por semente

|   sim_seed |   taxa_deteccao_t1_geral |   n_eventos_distintos |
|-----------:|-------------------------:|----------------------:|
|   2026.000 |                    0.667 |                45.000 |
|      7.000 |                    0.525 |                59.000 |
|    123.000 |                    0.424 |                85.000 |
|    501.000 |                    0.607 |                56.000 |
|    909.000 |                    0.566 |                53.000 |
|   3141.000 |                    0.436 |                55.000 |
|      8.000 |                    0.478 |                46.000 |
|     55.000 |                    0.472 |                72.000 |
|   2024.000 |                    0.619 |                63.000 |
|     77.000 |                    0.500 |                62.000 |