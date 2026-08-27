# Bateria de testes comportamentais - QC-LSTM 3 classes (secao 8 do protocolo)

Todos os testes usam o modelo `E_full` treinado por `codigos/10_qc_lstm_3classes_ba1.py`
(uma unica semente de modelo/injecao) aplicado sobre o conjunto de teste real da BA-1.

## 8.3 Teste de monotonicidade

- Taxa de monotonicidade, 0.999 (meta do protocolo, >= 0.90, atingida: True).
- 200 janelas reais, spikes aditivos k in [0.5, 1, 2, 3, 4, 6], ambas as direcoes.

## 8.4 Teste de coerencia fisica contextual

- Proporcao de janelas em que o mesmo aumento de Hsig produz p(BAD) menor (ou p(SUSPECT) maior que p(BAD)) quando ha suporte coerente das variaveis auxiliares, 0.960.
- p(BAD) medio sem suporte fisico (cenario A), 0.717; com suporte fisico (cenario B), 0.679.

## 8.5 Teste de causalidade

- Taxa de violacao de causalidade (rotulo em t muda quando Hsig_t+1 e alterado), 0.420.
- Violacao esperada: o indicador stat_spike usa Hsig_{t+1} por definicao (secao 5.3 do protocolo, documentado como adequado para QC atrasado, nao para operacao causal em tempo real). O estado oculto e a previsao da LSTM em si permanecem causais (dependem so de x_{t-L..t-1}).

## 8.6 Teste de sensibilidade a entrada auxiliar

- Taxa de mudanca de rotulo para BAD por perturbacao pequena e isolada em uma unica variavel auxiliar, 0.0000 (meta implicita, proximo de 0).
- Variacao media absoluta em p(GOOD) por essas perturbacoes, 0.0083.

## 8.7 Teste de recuperacao

Anomalia isolada (spike +4 desvios-padrao) realimentada nas janelas seguintes; dois modos de
recuperacao comparados.

| mode                    |   t_rec_mean |   t_rec_median |   censor_rate |   n_events |
|:------------------------|-------------:|---------------:|--------------:|-----------:|
| realimentacao_observado |         4.00 |           3.00 |          0.00 |      12.00 |
| substituicao_predicao   |         0.00 |           0.00 |          0.00 |      12.00 |

## 8.8 Teste de falha persistente

Episodios de 12 horas por familia (sensor travado, drift, mudanca de nivel),
tempo ate a primeira sinalizacao SUSPECT/BAD e proporcao do episodio detectada.

| family      |   first_suspect_mean |   first_bad_mean |   detection_proportion_mean |   n_events |
|:------------|---------------------:|-----------------:|----------------------------:|-----------:|
| drift       |                12.00 |           nan    |                        0.04 |          6 |
| level_shift |                 0.00 |             0.00 |                        0.60 |          6 |
| stuck       |                 6.00 |           nan    |                        0.19 |          6 |

## Leitura geral

O teste de causalidade confirma, de forma isolada e mensuravel, a ressalva ja documentada no
protocolo (secao 5.3) de que o indicador estatistico de spike usa informacao futura (Hsig_t+1) e
portanto so e apropriado para QC atrasado, nao para operacao estritamente causal em tempo real;
a previsao e o estado oculto da LSTM continuam causais. Os demais testes comportamentais indicam
se o classificador reage de forma qualitativamente correta (monotonicidade a magnitude do desvio,
reducao de falso alarme quando ha suporte fisico coerente, estabilidade a ruido irrelevante nas
variaveis auxiliares, recuperacao apos anomalia isolada) e nao apenas um bom desempenho agregado.