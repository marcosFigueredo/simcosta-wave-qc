# QC-LSTM de tres classes para Hsig (BA-1)

Reproducao do protocolo `codigos/protocolo_teste_qc_lstm_3classes.md`. A LSTM deixa de ser apenas
um preditor auxiliar de um detector de picos com regra fixa; sua previsao, seu estado oculto e
indicadores estatisticos alimentam uma cabeca classificadora treinada que decide diretamente
Q_t in {GOOD, SUSPECT, BAD}.

## Divisao dos dados

- Treino 60%, validacao 20%, teste 20%, estritamente cronologica.
- Taxa de injecao sintetica, treino 4.0%, validacao/teste 1.5%.

## Desempenho preditivo da LSTM (fase A, previsao direta de passo unico)

| Metrica | Valor |
|---|---:|
| MAE | 0.0502 |
| RMSE | 0.0701 |
| MAPE (%) | 8.65 |

## Distribuicao das classes no teste

| classe   |   n |
|:---------|----:|
| SUSPECT  | 900 |
| GOOD     | 769 |
| BAD      |  59 |

## Modelo completo (E_full) no conjunto de teste

| Metrica | Valor |
|---|---:|
| Macro-F1 | 0.713 |
| Weighted-F1 | 0.901 |
| Balanced accuracy | 0.722 |
| MCC | 0.813 |
| F1 GOOD | 0.929 |
| F1 SUSPECT | 0.916 |
| F1 BAD | 0.292 |
| AUPRC BAD | 0.192 |
| ECE | 0.023 |

Matriz de confusao (linhas = real, colunas = predito, ordem GOOD/SUSPECT/BAD),

```
[[739  15  15]
 [ 68 795  37]
 [ 15  25  19]]
```

## Ablacao (Tabela 4 do protocolo, modelos B a F)

| config            |   macro_f1 |   f1_bad |   auprc_bad |   ece |   balanced_accuracy |
|:------------------|-----------:|---------:|------------:|------:|--------------------:|
| B_residual        |      0.714 |    0.326 |       0.262 | 0.054 |               0.729 |
| C_residual_stats  |      0.721 |    0.329 |       0.253 | 0.028 |               0.754 |
| D_hidden_residual |      0.684 |    0.295 |       0.231 | 0.032 |               0.691 |
| E_full            |      0.713 |    0.292 |       0.192 | 0.023 |               0.722 |
| F_no_lstm         |      0.366 |    0.202 |       0.296 | 0.165 |               0.501 |

Modelo B usa so residual, C acrescenta o vetor estatistico s_t, D troca s_t pelo estado oculto
h_t da LSTM, E_full e o modelo completo (h_t + residual + s_t + variaveis auxiliares z_t + mascara
m_t), F_no_lstm remove a LSTM inteiramente e usa apenas um residual de persistencia e as variaveis
auxiliares observadas, para medir se a representacao temporal aprendida agrega informacao.

## Comparacao com baselines binarios (deteccao de BAD)

Os seis detectores originais (LSTM-Peak, robusto mediana/MAD, GPD-POT, spike tradicional,
Isolation Forest) so produzem uma decisao binaria; aqui sao comparados apenas na deteccao da
classe BAD do teste (VAE-LSTM omitido nesta rodada por custo computacional).

| metodo                             |   precision |   recall |    f1 |
|:-----------------------------------|------------:|---------:|------:|
| LSTM-Peak (H/T/d fixos)            |       0.538 |    0.119 | 0.194 |
| Robusto mediana/MAD                |       0.044 |    0.610 | 0.083 |
| GPD-POT                            |       0.714 |    0.085 | 0.152 |
| Spike tradicional                  |       0.176 |    0.153 | 0.164 |
| Isolation Forest                   |       0.051 |    0.051 | 0.051 |
| QC-LSTM 3 classes (E_full, so BAD) |     nan     |  nan     | 0.292 |

## Limitacoes desta rodada

Esta primeira execucao cobre as fases A a C do protocolo (preditor, geracao de rotulos,
cabeca classificadora com LSTM congelada) com uma unica semente de modelo e uma unica semente
de injecao por particao. Ainda faltam a bateria de testes comportamentais (secao 8, monotonicidade,
coerencia fisica contextual, causalidade, recuperacao, falha persistente), o teste de estabilidade
multi-semente (secao 10.5) e o fine-tuning conjunto opcional (fase D).

## Figuras geradas

- `fig_matriz_confusao.png`: matriz de confusao do modelo completo no teste.
- `fig_probabilidades_tempo.png`: probabilidades GOOD/SUSPECT/BAD ao longo do tempo (ultimos pontos do teste).