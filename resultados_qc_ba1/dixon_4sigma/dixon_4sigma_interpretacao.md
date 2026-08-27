# Dixon 4σ - SIMCOSTA_BA-1

## Metodologia

Para cada variavel foi calculada a media global `μ` e o desvio padrao amostral `σ`. Em seguida, cada observacao foi padronizada por `z_i = (x_i - μ) / σ`.

Classificacao usada:

- GOOD: `|z_i| <= 3`
- SUSPECT: `3 < |z_i| <= 4`
- BAD: `|z_i| > 4`

O Dixon 4σ e simples e transparente, mas deve ser interpretado com cautela em series oceanograficas costeiras, porque a distribuicao pode ser assimetrica, sazonal e influenciada por eventos fisicos reais.

## Resumo estatistico

| Variavel   |   Número total de observações |   Média (μ) |   Desvio padrão (σ) |   Quantidade de GOOD |   Quantidade de SUSPECT |   Quantidade de BAD |   Percentual de observações BAD |   Valor mínimo observado |   Valor máximo observado |   Limite inferior Dixon 4σ |   Limite superior Dixon 4σ | Interpretação                      |
|:-----------|------------------------------:|------------:|--------------------:|---------------------:|------------------------:|--------------------:|--------------------------------:|-------------------------:|-------------------------:|---------------------------:|---------------------------:|:-----------------------------------|
| Hsig       |                         97233 |      0.4719 |              0.2767 |                95279 |                    1348 |                 606 |                          0.6232 |                   0.0900 |                   2.4100 |                    -0.6349 |                     1.5787 | Boa qualidade com poucos outliers. |
| Tp         |                         97233 |      8.0799 |              2.9885 |                97212 |                      16 |                   5 |                          0.0051 |                   1.7000 |                  28.6000 |                    -3.8741 |                    20.0338 | Excelente qualidade dos dados.     |
| Hmax       |                         97233 |      0.7837 |              0.4538 |                95229 |                    1368 |                 636 |                          0.6541 |                   0.1400 |                   4.7100 |                    -1.0315 |                     2.5989 | Boa qualidade com poucos outliers. |
| HM0        |                         97233 |      0.5061 |              0.2895 |                95307 |                    1359 |                 567 |                          0.5831 |                   0.0900 |                   2.5100 |                    -0.6520 |                     1.6642 | Boa qualidade com poucos outliers. |
| Avg_Sal    |                        114535 |     35.9666 |              1.0586 |               114185 |                     279 |                  71 |                          0.0620 |                   0.4100 |                  37.6000 |                    31.7321 |                    40.2012 | Excelente qualidade dos dados.     |
| Avg_DO     |                        114120 |      4.1388 |              0.3725 |               112116 |                     792 |                1212 |                          1.0620 |                   0.0400 |                   5.6300 |                     2.6488 |                     5.6288 | Serie requer investigacao.         |
| Avg_W_Tmp1 |                        114228 |     27.1813 |              1.1384 |               114127 |                     101 |                   0 |                          0.0000 |                  23.3100 |                  31.4900 |                    22.6275 |                    31.7351 | Excelente qualidade dos dados.     |
| Avg_W_Tmp2 |                        111704 |     27.2866 |              1.2456 |               111256 |                     438 |                  10 |                          0.0090 |                  23.6600 |                  32.4600 |                    22.3041 |                    32.2691 | Excelente qualidade dos dados.     |
| Avg_Turb   |                        111227 |      2.8255 |             16.8903 |               109961 |                     275 |                 991 |                          0.8910 |                   0.0000 |                 346.8000 |                   -64.7358 |                    70.3867 | Boa qualidade com poucos outliers. |
| Avg_Chl    |                        111227 |      0.5911 |              0.5067 |               110791 |                     202 |                 234 |                          0.2104 |                   0.0300 |                  58.1800 |                    -1.4357 |                     2.6178 | Boa qualidade com poucos outliers. |
| Avg_CDOM   |                        111227 |      0.9397 |              1.8715 |               110672 |                     127 |                 428 |                          0.3848 |                  -0.2000 |                 106.4700 |                    -6.5463 |                     8.4257 | Boa qualidade com poucos outliers. |

## Interpretacao geral

- `Hsig`: 0.6232% BAD. Boa qualidade com poucos outliers.
- `Tp`: 0.0051% BAD. Excelente qualidade dos dados.
- `Hmax`: 0.6541% BAD. Boa qualidade com poucos outliers.
- `HM0`: 0.5831% BAD. Boa qualidade com poucos outliers.
- `Avg_Sal`: 0.0620% BAD. Excelente qualidade dos dados.
- `Avg_DO`: 1.0620% BAD. Serie requer investigacao.
- `Avg_W_Tmp1`: 0.0000% BAD. Excelente qualidade dos dados.
- `Avg_W_Tmp2`: 0.0090% BAD. Excelente qualidade dos dados.
- `Avg_Turb`: 0.8910% BAD. Boa qualidade com poucos outliers.
- `Avg_Chl`: 0.2104% BAD. Boa qualidade com poucos outliers.
- `Avg_CDOM`: 0.3848% BAD. Boa qualidade com poucos outliers.

## Eventos consecutivos BAD

Eventos com `n_points >= 2` sao candidatos a eventos fisicos reais ou periodos persistentes de anomalia. Eventos com `n_points = 1` sao candidatos mais fortes a spikes instrumentais isolados, especialmente se nao forem confirmados por outras variaveis.

| Variavel   | start                     | end                       |   n_points |   min_value |   max_value |   max_abs_z | event_type                  |
|:-----------|:--------------------------|:--------------------------|-----------:|------------:|------------:|------------:|:----------------------------|
| Hsig       | 2019-08-16 06:21:40+00:00 | 2019-08-16 07:51:40+00:00 |          4 |      1.6100 |      1.7600 |      4.6552 | possible_physical_event     |
| Hsig       | 2019-08-16 19:21:40+00:00 | 2019-08-16 19:21:40+00:00 |          1 |      1.6500 |      1.6500 |      4.2577 | possible_instrumental_spike |
| Hsig       | 2019-08-17 00:21:40+00:00 | 2019-08-17 00:51:40+00:00 |          2 |      1.6100 |      1.6200 |      4.1492 | possible_physical_event     |
| Hsig       | 2019-08-28 15:51:40+00:00 | 2019-08-28 15:51:40+00:00 |          1 |      1.6200 |      1.6200 |      4.1492 | possible_instrumental_spike |
| Hsig       | 2019-08-28 18:21:40+00:00 | 2019-08-28 18:21:40+00:00 |          1 |      1.7000 |      1.7000 |      4.4384 | possible_instrumental_spike |
| Hsig       | 2019-08-28 22:51:40+00:00 | 2019-08-28 22:51:40+00:00 |          1 |      1.5900 |      1.5900 |      4.0408 | possible_instrumental_spike |
| Hsig       | 2019-08-29 06:21:40+00:00 | 2019-08-29 06:21:40+00:00 |          1 |      1.6000 |      1.6000 |      4.0770 | possible_instrumental_spike |
| Hsig       | 2019-11-25 22:51:40+00:00 | 2019-11-25 22:51:40+00:00 |          1 |      1.6100 |      1.6100 |      4.1131 | possible_instrumental_spike |
| Hsig       | 2019-11-26 00:21:40+00:00 | 2019-11-26 00:21:40+00:00 |          1 |      1.5900 |      1.5900 |      4.0408 | possible_instrumental_spike |
| Hsig       | 2019-11-26 12:21:40+00:00 | 2019-11-26 12:21:40+00:00 |          1 |      1.8400 |      1.8400 |      4.9443 | possible_instrumental_spike |
| Hsig       | 2020-03-23 23:51:40+00:00 | 2020-03-24 01:21:40+00:00 |          3 |      1.6500 |      1.6600 |      4.2938 | possible_physical_event     |
| Hsig       | 2020-03-24 04:21:40+00:00 | 2020-03-24 07:51:40+00:00 |          8 |      1.6100 |      1.9200 |      5.2334 | possible_physical_event     |
| Hsig       | 2020-03-24 09:21:40+00:00 | 2020-03-24 09:21:40+00:00 |          1 |      1.6400 |      1.6400 |      4.2215 | possible_instrumental_spike |
| Hsig       | 2020-03-24 11:51:40+00:00 | 2020-03-24 13:21:40+00:00 |          4 |      1.6000 |      1.9200 |      5.2334 | possible_physical_event     |
| Hsig       | 2020-03-24 22:21:40+00:00 | 2020-03-24 22:21:40+00:00 |          1 |      1.5900 |      1.5900 |      4.0408 | possible_instrumental_spike |
| Hsig       | 2020-03-24 23:51:40+00:00 | 2020-03-25 02:21:40+00:00 |          5 |      1.5800 |      2.0900 |      5.8478 | possible_physical_event     |
| Hsig       | 2020-03-25 05:21:40+00:00 | 2020-03-25 05:21:40+00:00 |          1 |      1.6000 |      1.6000 |      4.0770 | possible_instrumental_spike |
| Hsig       | 2020-03-25 07:51:40+00:00 | 2020-03-25 07:51:40+00:00 |          1 |      1.6100 |      1.6100 |      4.1131 | possible_instrumental_spike |
| Hsig       | 2020-04-19 10:21:40+00:00 | 2020-04-19 12:51:40+00:00 |          6 |      1.6100 |      1.9200 |      5.2334 | possible_physical_event     |
| Hsig       | 2020-04-19 13:51:40+00:00 | 2020-04-19 14:21:40+00:00 |          2 |      1.5900 |      1.6800 |      4.3661 | possible_physical_event     |
| Hsig       | 2020-04-24 07:51:40+00:00 | 2020-04-24 07:51:40+00:00 |          1 |      1.6000 |      1.6000 |      4.0770 | possible_instrumental_spike |
| Hsig       | 2020-04-24 19:21:40+00:00 | 2020-04-24 19:21:40+00:00 |          1 |      1.6000 |      1.6000 |      4.0770 | possible_instrumental_spike |
| Hsig       | 2020-04-27 10:51:40+00:00 | 2020-04-27 10:51:40+00:00 |          1 |      1.6200 |      1.6200 |      4.1492 | possible_instrumental_spike |
| Hsig       | 2020-04-27 14:21:40+00:00 | 2020-04-27 14:21:40+00:00 |          1 |      1.6400 |      1.6400 |      4.2215 | possible_instrumental_spike |
| Hsig       | 2020-05-08 21:21:40+00:00 | 2020-05-09 10:21:40+00:00 |         26 |      1.6000 |      2.2400 |      6.3899 | possible_physical_event     |
| Hsig       | 2020-05-09 11:21:40+00:00 | 2020-05-09 16:21:40+00:00 |         10 |      1.6400 |      2.1400 |      6.0285 | possible_physical_event     |
| Hsig       | 2020-05-09 17:51:40+00:00 | 2020-05-09 17:51:40+00:00 |          1 |      1.6900 |      1.6900 |      4.4022 | possible_instrumental_spike |
| Hsig       | 2020-05-09 19:21:40+00:00 | 2020-05-10 03:21:40+00:00 |         12 |      1.6900 |      2.4100 |      7.0043 | possible_physical_event     |
| Hsig       | 2020-05-10 08:21:40+00:00 | 2020-05-10 10:51:40+00:00 |          6 |      1.5900 |      1.9200 |      5.2334 | possible_physical_event     |
| Hsig       | 2020-05-10 12:51:40+00:00 | 2020-05-10 15:51:40+00:00 |          7 |      1.5800 |      1.8000 |      4.7997 | possible_physical_event     |
| Hsig       | 2020-05-10 21:51:40+00:00 | 2020-05-10 22:51:40+00:00 |          3 |      1.6200 |      1.7700 |      4.6913 | possible_physical_event     |
| Hsig       | 2020-05-11 00:21:40+00:00 | 2020-05-11 00:51:40+00:00 |          2 |      1.6600 |      1.7300 |      4.5468 | possible_physical_event     |
| Hsig       | 2020-05-19 13:51:40+00:00 | 2020-05-19 13:51:40+00:00 |          1 |      1.6600 |      1.6600 |      4.2938 | possible_instrumental_spike |
| Hsig       | 2020-05-19 16:21:40+00:00 | 2020-05-19 20:21:40+00:00 |          9 |      1.5800 |      1.8000 |      4.7997 | possible_physical_event     |
| Hsig       | 2020-05-20 04:51:40+00:00 | 2020-05-20 05:51:40+00:00 |          3 |      1.6000 |      1.6800 |      4.3661 | possible_physical_event     |
| Hsig       | 2020-05-29 04:51:40+00:00 | 2020-05-29 04:51:40+00:00 |          1 |      1.5800 |      1.5800 |      4.0047 | possible_instrumental_spike |
| Hsig       | 2020-06-04 21:51:40+00:00 | 2020-06-04 22:51:40+00:00 |          3 |      1.6600 |      1.9900 |      5.4864 | possible_physical_event     |
| Hsig       | 2020-06-04 23:51:40+00:00 | 2020-06-05 00:21:40+00:00 |          2 |      1.7900 |      1.9200 |      5.2334 | possible_physical_event     |
| Hsig       | 2020-06-18 05:51:40+00:00 | 2020-06-18 05:51:40+00:00 |          1 |      1.6700 |      1.6700 |      4.3299 | possible_instrumental_spike |
| Hsig       | 2020-08-03 10:51:40+00:00 | 2020-08-03 10:51:40+00:00 |          1 |      1.6400 |      1.6400 |      4.2215 | possible_instrumental_spike |
| Hsig       | 2020-08-04 07:51:40+00:00 | 2020-08-04 07:51:40+00:00 |          1 |      1.6000 |      1.6000 |      4.0770 | possible_instrumental_spike |
| Hsig       | 2020-08-23 16:51:40+00:00 | 2020-08-23 17:21:40+00:00 |          2 |      1.5900 |      1.6200 |      4.1492 | possible_physical_event     |
| Hsig       | 2020-08-24 01:21:40+00:00 | 2020-08-24 20:51:40+00:00 |         40 |      1.5800 |      2.3700 |      6.8597 | possible_physical_event     |
| Hsig       | 2020-08-24 23:21:40+00:00 | 2020-08-24 23:51:40+00:00 |          2 |      1.5900 |      1.6900 |      4.4022 | possible_physical_event     |
| Hsig       | 2020-08-25 00:51:40+00:00 | 2020-08-25 05:51:40+00:00 |         11 |      1.6200 |      1.8200 |      4.8720 | possible_physical_event     |
| Hsig       | 2020-11-02 12:21:40+00:00 | 2020-11-02 13:21:40+00:00 |          3 |      1.6000 |      1.6900 |      4.4022 | possible_physical_event     |
| Hsig       | 2020-11-02 14:21:40+00:00 | 2020-11-02 14:21:40+00:00 |          1 |      1.6800 |      1.6800 |      4.3661 | possible_instrumental_spike |
| Hsig       | 2020-11-03 01:21:40+00:00 | 2020-11-03 01:21:40+00:00 |          1 |      1.7200 |      1.7200 |      4.5106 | possible_instrumental_spike |
| Hsig       | 2020-11-03 07:51:40+00:00 | 2020-11-03 08:21:40+00:00 |          2 |      1.6000 |      1.7100 |      4.4745 | possible_physical_event     |
| Hsig       | 2020-11-03 09:51:40+00:00 | 2020-11-03 11:21:40+00:00 |          3 |      1.6100 |      1.7300 |      4.5468 | possible_physical_event     |

## Conclusao cientifica resumida

O Dixon 4σ fornece uma primeira avaliacao global da compatibilidade das series com sua variabilidade estatistica central. Percentuais baixos de BAD indicam que os extremos sao raros em relacao a media e ao desvio padrao da serie; percentuais elevados indicam necessidade de investigacao, podendo refletir assimetria natural, eventos costeiros extremos, falhas instrumentais ou processamento inadequado. A classificacao final deve ser combinada com Spike test, range fisico, rate-of-change, lacunas temporais e coerencia multivariada.

## Arquivos gerados

- `dixon_4sigma_resumo.csv`
- `dixon_4sigma_eventos_bad.csv`
- `dixon_4sigma_top_positivos.csv`
- `dixon_4sigma_top_negativos.csv`
- `dixon_4sigma_<variavel>.csv` para a tabela Timestamp/Valor/z-score/Flag de cada variavel.