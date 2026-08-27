# Spike Test e QC Robusto - SIMCOSTA_BA-1

## Objetivo

Avaliar valores isolados incompatíveis com a evolução temporal dos dados da boia SIMCOSTA_BA-1. A formula principal usada foi $$S_i = |x_i - (x_{i-1}+x_{i+1})/2|$$. O teste classico responde: existe um valor isolado incompatível com os vizinhos?

A rotina tambem aplica testes em ordem crescente/complementar de robustez: Mediana + MAD, segunda diferenca, QARTOD spike, gradiente fisico maximo, range test e desvio-padrao local. A decisao final nao remove automaticamente todo alerta: `removed` e verdadeiro quando o valor viola range fisico ou quando tres ou mais testes concordam.

## Parametros

- Janela robusta MAD: 49 registros, aproximadamente 24,5 h para intervalo de 30 min.
- Spike classico: limiar regional P99.5 de `S_i` por variavel.
- MAD: flag quando `Score_MAD > 6`.
- QARTOD spike: flag mais severa quando `Score_MAD > 8`.
- Segunda diferenca: limiar P99.5 da curvatura temporal por variavel.
- Gradiente fisico maximo: limites por hora definidos por variavel no script.
- Range: limites fisicos amplos definidos por variavel no script.
- Desvio-padrao local: flag quando o score local excede 4 desvios-padrao.
- QI: `QI = 100 * (1 - Score/Scoremax)`, com classes Excellent, Good, Suspect e Bad.

## Tabela principal do artigo

| Variavel   |      N |   Removed (%) |   Spike (%) |   MAD (%) |   SecondDiff (%) |   QARTOD (%) |   Range (%) |   ROC (%) |   LocalSTD (%) |   QI median |
|:-----------|-------:|--------------:|------------:|----------:|-----------------:|-------------:|------------:|----------:|---------------:|------------:|
| Avg_CDOM   | 111227 |        0.9539 |      0.4972 |    5.5229 |           0.4972 |       3.8893 |      0.0018 |    0.0468 |         0.6572 |    100.0000 |
| Avg_Chl    | 111227 |        0.8622 |      0.4963 |    3.5522 |           0.4963 |       1.9905 |      0.0000 |    0.0279 |         0.6824 |     97.9741 |
| Avg_DO     | 114120 |        0.6774 |      0.4968 |    3.1230 |           0.4968 |       1.4713 |      0.0000 |    0.0719 |         0.5275 |     94.6080 |
| Avg_Sal    | 114535 |        0.6723 |      0.4933 |    4.6798 |           0.4994 |       2.3303 |      0.0000 |    0.1144 |         0.4095 |    100.0000 |
| Avg_Turb   | 111227 |        1.4664 |      0.4972 |   10.0659 |           0.4972 |       8.1032 |      0.0000 |    0.3479 |         1.1409 |     99.9521 |
| Avg_W_Tmp1 | 114228 |        0.8947 |      0.4894 |    3.7408 |           0.4937 |       1.6152 |      0.0000 |    1.2011 |         0.4237 |     90.8907 |
| Avg_W_Tmp2 | 111704 |        0.8612 |      0.4977 |    3.7761 |           0.4977 |       1.7349 |      0.0000 |    0.8782 |         0.4306 |     90.6150 |
| HM0        |  97233 |        0.1563 |      0.4916 |    0.9801 |           0.4895 |       0.2314 |      0.0000 |    0.0010 |         0.1008 |     84.8154 |
| Hmax       |  97233 |        0.2221 |      0.4926 |    0.9184 |           0.4947 |       0.2540 |      0.0000 |    0.0041 |         0.2365 |     85.1251 |
| Hsig       |  97233 |        0.1440 |      0.4854 |    0.9925 |           0.4854 |       0.2427 |      0.0000 |    0.0010 |         0.0956 |     84.6016 |
| Tp         |  97233 |        1.4645 |      0.4906 |    3.1512 |           0.4875 |       1.5931 |      0.0000 |    9.6798 |         0.2098 |     92.4267 |

## Tabela de performance dos testes

| Teste               |   Flags unicas |   Flags compartilhadas |   Flags totais |
|:--------------------|---------------:|-----------------------:|---------------:|
| Range               |              0 |                      2 |              2 |
| Spike classico      |             13 |                   5798 |           5811 |
| MAD                 |          16013 |                  28567 |          44580 |
| Segunda diferenca   |             13 |                   5807 |           5820 |
| QARTOD spike        |              0 |                  25941 |          25941 |
| Rate of Change      |           8355 |                   4099 |          12454 |
| Desvio-padrao local |            134 |                   5286 |           5420 |

## Interpretacao das saidas

A Figura 1 mostra a sobrevivencia dos dados ao longo do fluxo de QC. Quedas pequenas indicam que a serie e majoritariamente consistente; quedas concentradas em um teste indicam o tipo dominante de problema.

As Figuras 2 mostram a serie temporal com flags para cada variavel principal. Pontos isolados em vermelho sugerem ruido eletronico, erro de transmissao ou medicao incompatível com a dinamica local. Blocos coerentes em mais de uma variavel podem representar evento oceanografico real.

As figuras anuais empilhadas colocam cada ano em uma linha, de janeiro a dezembro, com cores de fundo por estacao. Elas servem para comparar recorrencia sazonal, anos problematicos e trechos de possivel instabilidade instrumental.

A Figura 3 mostra a distribuicao dos scores MAD. Caudas longas indicam presenca de valores extremos; concentracao abaixo do limiar sugere boa estabilidade local.

A Figura 4 e um heatmap de QC por mes e ano. Padroes verticais sugerem meses/estacoes recorrentes; padroes horizontais sugerem anos com maior problema de qualidade.

Os diagramas `fig_scatter_xi_madscore_*.png` mostram valor observado versus Score MAD. Essa figura e forte para publicacao porque separa visualmente valores extremos reais de valores com anomalia local.

## Interpretacao por tipo de problema

- Ruido eletronico: spike isolado, sem recorrencia sazonal e sem concordancia com outros testes.
- Falha do sensor: concentracao em uma unica variavel, em periodo continuo ou em um ano especifico.
- Erro de transmissao: flags proximas a lacunas ou simultaneas em variaveis sem relacao fisica direta.
- Valor fisicamente impossivel: flag de `Range`; deve ser removido ou tratado como `bad`.
- Medicao isolada incompatível: falha no spike classico/MAD sem suporte das variaveis relacionadas.
- Evento oceanografico real: flags simultaneas e coerentes entre onda, turbidez, salinidade, temperatura ou variaveis opticas.

## Comparacao anual

| variable   |   year |     N |   removed |   suspect |   qi_median |   removed_pct |   suspect_pct |
|:-----------|-------:|------:|----------:|----------:|------------:|--------------:|--------------:|
| Avg_Turb   |   2021 | 16770 |       424 |      1707 |     99.9491 |        2.5283 |       10.1789 |
| Avg_CDOM   |   2024 | 16203 |       403 |      1861 |    100.0000 |        2.4872 |       11.4855 |
| Avg_Turb   |   2024 | 16203 |       353 |      2246 |     99.9554 |        2.1786 |       13.8616 |
| Tp         |   2024 | 16350 |       279 |      2035 |     92.3378 |        1.7064 |       12.4465 |
| Tp         |   2023 | 16312 |       274 |      1976 |     92.4267 |        1.6797 |       12.1138 |
| Avg_Chl    |   2020 | 17050 |       285 |       752 |     97.9885 |        1.6716 |        4.4106 |
| Tp         |   2025 | 11011 |       174 |      1168 |     92.4267 |        1.5802 |       10.6076 |
| Avg_DO     |   2022 | 15063 |       235 |       692 |     94.5774 |        1.5601 |        4.5940 |
| Avg_CDOM   |   2023 | 16278 |       248 |      1271 |    100.0000 |        1.5235 |        7.8081 |
| Tp         |   2022 | 12958 |       187 |      1353 |     92.4267 |        1.4431 |       10.4414 |
| Avg_Turb   |   2025 | 13707 |       193 |      1739 |     99.9521 |        1.4080 |       12.6869 |
| Tp         |   2020 | 17102 |       237 |      1672 |     92.5183 |        1.3858 |        9.7766 |
| Avg_W_Tmp2 |   2024 | 13724 |       174 |       680 |     90.4663 |        1.2679 |        4.9548 |
| Avg_Chl    |   2024 | 16203 |       204 |       798 |     98.0584 |        1.2590 |        4.9250 |
| Avg_W_Tmp1 |   2024 | 16247 |       201 |       773 |     90.7553 |        1.2372 |        4.7578 |
| Avg_W_Tmp2 |   2026 |  9512 |       117 |       463 |     90.5270 |        1.2300 |        4.8675 |
| Avg_Turb   |   2023 | 16278 |       193 |      1313 |     99.9491 |        1.1856 |        8.0661 |
| Tp         |   2021 | 16796 |       197 |      1689 |     92.3965 |        1.1729 |       10.0560 |
| Avg_Turb   |   2026 |  9509 |       111 |      1704 |     99.9531 |        1.1673 |       17.9199 |
| Tp         |   2019 |  6704 |        76 |       682 |     92.3378 |        1.1337 |       10.1730 |

## Comparacao por estacao

| variable   | season    |     N |   removed |   suspect |   qi_median |   removed_pct |   suspect_pct |
|:-----------|:----------|------:|----------:|----------:|------------:|--------------:|--------------:|
| Avg_W_Tmp1 | verao     | 28802 |       497 |      1786 |     90.4091 |        1.7256 |        6.2010 |
| Avg_W_Tmp2 | verao     | 28804 |       480 |      1580 |     90.1581 |        1.6664 |        5.4853 |
| Avg_Turb   | outono    | 28305 |       464 |      3474 |     99.9521 |        1.6393 |       12.2734 |
| Tp         | inverno   | 26223 |       429 |      3047 |     92.3794 |        1.6360 |       11.6196 |
| Avg_Turb   | inverno   | 28768 |       467 |      2923 |     99.9498 |        1.6233 |       10.1606 |
| Tp         | outono    | 21994 |       336 |      2196 |     92.4749 |        1.5277 |        9.9845 |
| Tp         | verao     | 24743 |       340 |      2898 |     92.4267 |        1.3741 |       11.7124 |
| Avg_Turb   | primavera | 25815 |       342 |      2202 |     99.9502 |        1.3248 |        8.5299 |
| Tp         | primavera | 24273 |       319 |      2434 |     92.4267 |        1.3142 |       10.0276 |
| Avg_Turb   | verao     | 28339 |       358 |      2857 |     99.9521 |        1.2633 |       10.0815 |
| Avg_CDOM   | verao     | 28339 |       338 |      1724 |    100.0000 |        1.1927 |        6.0835 |
| Avg_CDOM   | outono    | 28305 |       307 |      1517 |    100.0000 |        1.0846 |        5.3595 |
| Avg_Chl    | primavera | 25815 |       248 |       990 |     97.9421 |        0.9607 |        3.8350 |
| Avg_Chl    | verao     | 28339 |       269 |      1065 |     97.9830 |        0.9492 |        3.7581 |
| Avg_CDOM   | primavera | 25815 |       244 |      1693 |    100.0000 |        0.9452 |        6.5582 |
| Avg_DO     | verao     | 28778 |       255 |      1065 |     94.6406 |        0.8861 |        3.7007 |
| Avg_Sal    | inverno   | 28874 |       243 |      1241 |    100.0000 |        0.8416 |        4.2980 |
| Avg_Chl    | outono    | 28305 |       234 |      1024 |     97.9741 |        0.8267 |        3.6177 |
| Avg_Sal    | outono    | 28434 |       233 |      1461 |    100.0000 |        0.8194 |        5.1382 |
| Avg_W_Tmp1 | primavera | 28272 |       231 |      1285 |     91.1139 |        0.8171 |        4.5451 |