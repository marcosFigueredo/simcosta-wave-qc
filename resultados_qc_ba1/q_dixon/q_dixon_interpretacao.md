# Q-Dixon - SIMCOSTA_BA-1

## Objetivo

Aplicar o Q-Dixon como teste complementar para identificar valores extremos isolados em janelas locais da serie temporal. Diferente do Dixon 4σ, que compara cada observacao com a media e o desvio-padrao globais, o Q-Dixon pergunta se o ponto central de uma janela e um extremo isolado em relacao aos demais valores daquela janela.

## Metodologia

Foi usada janela movel centrada de `11` observacoes. Para cada janela, o ponto central so recebe Q-Dixon diferente de zero se for o maior ou menor valor da janela. O score e calculado como `Q = gap / range`, onde `gap` e a distancia entre o extremo e seu vizinho mais proximo na serie ordenada, e `range` e a amplitude da janela.

Classificacao usada:

- GOOD: `Q < 0.412`
- SUSPECT: `0.412 <= Q <= 0.517`
- BAD: `Q > 0.517`

Esses limiares sao uma adaptacao operacional para janela local. O resultado deve ser interpretado como triagem, nao como remocao automatica.

## Resumo estatistico

| Variavel   |      N |   Janela |   Q_suspect |   Q_bad |   GOOD |   SUSPECT |   BAD |   BAD (%) |   Valor mínimo |   Valor máximo |   Q mediano |   Q P95 |   Q máximo | Interpretação                                  |
|:-----------|-------:|---------:|------------:|--------:|-------:|----------:|------:|----------:|---------------:|---------------:|------------:|--------:|-----------:|:-----------------------------------------------|
| Hsig       |  97233 |       11 |      0.4120 |  0.5170 |  96859 |       304 |    70 |    0.0720 |         0.0900 |         2.4100 |      0.0000 |  0.1667 |     0.7500 | Excelente qualidade; poucos extremos isolados. |
| Tp         |  97233 |       11 |      0.4120 |  0.5170 |  95739 |       614 |   880 |    0.9050 |         1.7000 |        28.6000 |      0.0000 |  0.1538 |     1.0000 | Boa qualidade com poucos extremos isolados.    |
| Hmax       |  97233 |       11 |      0.4120 |  0.5170 |  96349 |       558 |   326 |    0.3353 |         0.1400 |         4.7100 |      0.0000 |  0.1905 |     0.8833 | Boa qualidade com poucos extremos isolados.    |
| HM0        |  97233 |       11 |      0.4120 |  0.5170 |  96867 |       308 |    58 |    0.0597 |         0.0900 |         2.5100 |      0.0000 |  0.1630 |     0.7083 | Excelente qualidade; poucos extremos isolados. |
| Avg_Sal    | 114535 |       11 |      0.4120 |  0.5170 | 114241 |       150 |   144 |    0.1257 |         0.4100 |        37.6000 |      0.0000 |  0.0492 |     1.0000 | Boa qualidade com poucos extremos isolados.    |
| Avg_DO     | 114120 |       11 |      0.4120 |  0.5170 | 113662 |       301 |   157 |    0.1376 |         0.0400 |         5.6300 |      0.0000 |  0.1176 |     0.9043 | Boa qualidade com poucos extremos isolados.    |
| Avg_W_Tmp1 | 114228 |       11 |      0.4120 |  0.5170 | 114036 |       127 |    65 |    0.0569 |        23.3100 |        31.4900 |      0.0000 |  0.0625 |     0.9520 | Excelente qualidade; poucos extremos isolados. |
| Avg_W_Tmp2 | 111704 |       11 |      0.4120 |  0.5170 | 111594 |        80 |    30 |    0.0269 |        23.6600 |        32.4600 |      0.0000 |  0.0541 |     0.8827 | Excelente qualidade; poucos extremos isolados. |
| Avg_Turb   | 111227 |       11 |      0.4120 |  0.5170 | 108062 |       727 |  2438 |    2.1919 |         0.0000 |       346.8000 |      0.0000 |  0.2000 |     1.0000 | Serie requer investigacao local.               |
| Avg_Chl    | 111227 |       11 |      0.4120 |  0.5170 | 110383 |       378 |   466 |    0.4190 |         0.0300 |        58.1800 |      0.0000 |  0.1304 |     0.9912 | Boa qualidade com poucos extremos isolados.    |
| Avg_CDOM   | 111227 |       11 |      0.4120 |  0.5170 | 110155 |       340 |   732 |    0.6581 |        -0.2000 |       106.4700 |      0.0000 |  0.0833 |     1.0000 | Boa qualidade com poucos extremos isolados.    |

## Interpretacao geral

- `Hsig`: 0.0720% BAD. Excelente qualidade; poucos extremos isolados.
- `Tp`: 0.9050% BAD. Boa qualidade com poucos extremos isolados.
- `Hmax`: 0.3353% BAD. Boa qualidade com poucos extremos isolados.
- `HM0`: 0.0597% BAD. Excelente qualidade; poucos extremos isolados.
- `Avg_Sal`: 0.1257% BAD. Boa qualidade com poucos extremos isolados.
- `Avg_DO`: 0.1376% BAD. Boa qualidade com poucos extremos isolados.
- `Avg_W_Tmp1`: 0.0569% BAD. Excelente qualidade; poucos extremos isolados.
- `Avg_W_Tmp2`: 0.0269% BAD. Excelente qualidade; poucos extremos isolados.
- `Avg_Turb`: 2.1919% BAD. Serie requer investigacao local.
- `Avg_Chl`: 0.4190% BAD. Boa qualidade com poucos extremos isolados.
- `Avg_CDOM`: 0.6581% BAD. Boa qualidade com poucos extremos isolados.

## Eventos consecutivos BAD

Eventos com `n_points = 1` sao candidatos a spikes instrumentais isolados. Eventos com `n_points >= 2` podem representar extremos fisicos persistentes, mas tambem podem indicar trecho instrumental problematico.

| Variavel   | start                     | end                       |   n_points |   min_value |   max_value |   max_q | event_type                  |
|:-----------|:--------------------------|:--------------------------|-----------:|------------:|------------:|--------:|:----------------------------|
| Hsig       | 2019-08-05 18:51:40+00:00 | 2019-08-05 18:51:40+00:00 |          1 |      0.4800 |      0.4800 |  0.6364 | possible_instrumental_spike |
| Hsig       | 2019-10-22 07:51:40+00:00 | 2019-10-22 07:51:40+00:00 |          1 |      0.2100 |      0.2100 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2019-10-29 19:21:40+00:00 | 2019-10-29 19:21:40+00:00 |          1 |      0.3300 |      0.3300 |  0.6250 | possible_instrumental_spike |
| Hsig       | 2019-11-28 10:21:40+00:00 | 2019-11-28 10:21:40+00:00 |          1 |      0.4500 |      0.4500 |  0.6154 | possible_instrumental_spike |
| Hsig       | 2019-12-18 13:51:40+00:00 | 2019-12-18 13:51:40+00:00 |          1 |      0.3100 |      0.3100 |  0.5556 | possible_instrumental_spike |
| Hsig       | 2020-02-01 13:51:40+00:00 | 2020-02-01 13:51:40+00:00 |          1 |      0.3300 |      0.3300 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2020-02-21 09:51:40+00:00 | 2020-02-21 09:51:40+00:00 |          1 |      0.4400 |      0.4400 |  0.5294 | possible_instrumental_spike |
| Hsig       | 2020-03-19 08:21:40+00:00 | 2020-03-19 08:21:40+00:00 |          1 |      0.5400 |      0.5400 |  0.6250 | possible_instrumental_spike |
| Hsig       | 2020-05-07 05:51:40+00:00 | 2020-05-07 05:51:40+00:00 |          1 |      0.8100 |      0.8100 |  0.5200 | possible_instrumental_spike |
| Hsig       | 2020-05-22 22:21:40+00:00 | 2020-05-22 22:21:40+00:00 |          1 |      0.7500 |      0.7500 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2020-10-17 15:21:40+00:00 | 2020-10-17 15:21:40+00:00 |          1 |      0.2900 |      0.2900 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2020-12-07 09:21:40+00:00 | 2020-12-07 09:21:40+00:00 |          1 |      0.2600 |      0.2600 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2020-12-12 21:51:40+00:00 | 2020-12-12 21:51:40+00:00 |          1 |      0.3200 |      0.3200 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2020-12-20 05:51:40+00:00 | 2020-12-20 05:51:40+00:00 |          1 |      0.2700 |      0.2700 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2020-12-29 01:51:40+00:00 | 2020-12-29 01:51:40+00:00 |          1 |      0.3600 |      0.3600 |  0.6250 | possible_instrumental_spike |
| Hsig       | 2021-01-05 07:21:40+00:00 | 2021-01-05 07:21:40+00:00 |          1 |      0.3800 |      0.3800 |  0.5833 | possible_instrumental_spike |
| Hsig       | 2021-01-26 09:21:40+00:00 | 2021-01-26 09:21:40+00:00 |          1 |      0.6200 |      0.6200 |  0.5385 | possible_instrumental_spike |
| Hsig       | 2021-02-11 02:21:40+00:00 | 2021-02-11 02:21:40+00:00 |          1 |      0.3900 |      0.3900 |  0.5333 | possible_instrumental_spike |
| Hsig       | 2021-03-24 12:21:40+00:00 | 2021-03-24 12:21:40+00:00 |          1 |      0.2300 |      0.2300 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2021-04-09 01:21:40+00:00 | 2021-04-09 01:21:40+00:00 |          1 |      0.9000 |      0.9000 |  0.5400 | possible_instrumental_spike |
| Hsig       | 2021-04-14 11:21:40+00:00 | 2021-04-14 11:21:40+00:00 |          1 |      0.3500 |      0.3500 |  0.6429 | possible_instrumental_spike |
| Hsig       | 2021-04-25 01:21:40+00:00 | 2021-04-25 01:21:40+00:00 |          1 |      0.3000 |      0.3000 |  0.5600 | possible_instrumental_spike |
| Hsig       | 2021-05-09 17:21:40+00:00 | 2021-05-09 17:21:40+00:00 |          1 |      1.1200 |      1.1200 |  0.5417 | possible_instrumental_spike |
| Hsig       | 2021-06-29 01:21:40+00:00 | 2021-06-29 01:21:40+00:00 |          1 |      1.3400 |      1.3400 |  0.5714 | possible_instrumental_spike |
| Hsig       | 2021-07-14 21:21:40+00:00 | 2021-07-14 21:21:40+00:00 |          1 |      0.6500 |      0.6500 |  0.6800 | possible_instrumental_spike |
| Hsig       | 2021-09-09 16:51:40+00:00 | 2021-09-09 16:51:40+00:00 |          1 |      0.3700 |      0.3700 |  0.5714 | possible_instrumental_spike |
| Hsig       | 2021-09-11 23:51:40+00:00 | 2021-09-11 23:51:40+00:00 |          1 |      0.6500 |      0.6500 |  0.5294 | possible_instrumental_spike |
| Hsig       | 2021-12-19 09:21:40+00:00 | 2021-12-19 09:21:40+00:00 |          1 |      0.2000 |      0.2000 |  0.7500 | possible_instrumental_spike |
| Hsig       | 2022-01-02 14:51:40+00:00 | 2022-01-02 14:51:40+00:00 |          1 |      0.3200 |      0.3200 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2022-06-17 09:21:40+00:00 | 2022-06-17 09:21:40+00:00 |          1 |      0.8900 |      0.8900 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2022-07-04 04:51:40+00:00 | 2022-07-04 04:51:40+00:00 |          1 |      0.6900 |      0.6900 |  0.5217 | possible_instrumental_spike |
| Hsig       | 2022-07-12 08:21:40+00:00 | 2022-07-12 08:21:40+00:00 |          1 |      0.8200 |      0.8200 |  0.6364 | possible_instrumental_spike |
| Hsig       | 2022-07-22 13:21:40+00:00 | 2022-07-22 13:21:40+00:00 |          1 |      0.5200 |      0.5200 |  0.5833 | possible_instrumental_spike |
| Hsig       | 2022-08-04 10:51:40+00:00 | 2022-08-04 10:51:40+00:00 |          1 |      0.4800 |      0.4800 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2022-08-20 17:21:40+00:00 | 2022-08-20 17:21:40+00:00 |          1 |      1.0800 |      1.0800 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2022-10-04 17:51:40+00:00 | 2022-10-04 17:51:40+00:00 |          1 |      0.3800 |      0.3800 |  0.5714 | possible_instrumental_spike |
| Hsig       | 2022-10-11 20:51:40+00:00 | 2022-10-11 20:51:40+00:00 |          1 |      0.7300 |      0.7300 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2022-10-21 03:21:40+00:00 | 2022-10-21 03:21:40+00:00 |          1 |      0.5100 |      0.5100 |  0.5294 | possible_instrumental_spike |
| Hsig       | 2022-11-06 18:21:40+00:00 | 2022-11-06 18:21:40+00:00 |          1 |      1.3000 |      1.3000 |  0.5172 | possible_instrumental_spike |
| Hsig       | 2022-11-10 15:51:40+00:00 | 2022-11-10 15:51:40+00:00 |          1 |      0.2200 |      0.2200 |  0.5714 | possible_instrumental_spike |
| Hsig       | 2022-11-11 04:21:40+00:00 | 2022-11-11 04:21:40+00:00 |          1 |      0.2300 |      0.2300 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2022-11-24 08:51:40+00:00 | 2022-11-24 08:51:40+00:00 |          1 |      0.4800 |      0.4800 |  0.5556 | possible_instrumental_spike |
| Hsig       | 2023-03-17 04:21:40+00:00 | 2023-03-17 04:21:40+00:00 |          1 |      0.7300 |      0.7300 |  0.5294 | possible_instrumental_spike |
| Hsig       | 2023-03-23 03:51:40+00:00 | 2023-03-23 03:51:40+00:00 |          1 |      0.3400 |      0.3400 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2023-04-11 01:21:40+00:00 | 2023-04-11 01:21:40+00:00 |          1 |      0.4200 |      0.4200 |  0.5333 | possible_instrumental_spike |
| Hsig       | 2023-04-14 05:21:40+00:00 | 2023-04-14 05:21:40+00:00 |          1 |      0.4400 |      0.4400 |  0.5455 | possible_instrumental_spike |
| Hsig       | 2023-04-15 17:51:40+00:00 | 2023-04-15 17:51:40+00:00 |          1 |      0.2900 |      0.2900 |  0.6000 | possible_instrumental_spike |
| Hsig       | 2023-04-20 03:21:40+00:00 | 2023-04-20 03:21:40+00:00 |          1 |      0.2000 |      0.2000 |  0.5556 | possible_instrumental_spike |
| Hsig       | 2023-05-21 07:51:40+00:00 | 2023-05-21 07:51:40+00:00 |          1 |      1.2200 |      1.2200 |  0.5357 | possible_instrumental_spike |
| Hsig       | 2023-06-24 07:51:40+00:00 | 2023-06-24 07:51:40+00:00 |          1 |      0.5200 |      0.5200 |  0.5714 | possible_instrumental_spike |

## Relevancia cientifica

O Q-Dixon e especialmente util para separar extremos isolados de eventos persistentes. Quando o Dixon 4σ marca um valor como BAD, mas o Q-Dixon nao marca, isso sugere que o valor extremo pode estar dentro de um bloco coerente de variabilidade. Quando ambos marcam o mesmo ponto, a evidencia de spike instrumental ou extremo local isolado fica mais forte.

## Arquivos gerados

- `q_dixon_resumo.csv`
- `q_dixon_eventos_bad.csv`
- `q_dixon_top_positivos.csv`
- `q_dixon_top_negativos.csv`
- `q_dixon_por_ano.csv`
- `q_dixon_por_mes.csv`
- `q_dixon_por_estacao.csv`
- `q_dixon_<variavel>.csv` com Timestamp, Valor, Q-Dixon e Flag.