# Modelo QC-LSTM causal e ordinal (config C), especificacao de entrada e saida

Gerado por `codigos/15_salvar_modelo_final_ba1.py`. Todos os numeros de referencia abaixo (faixa fisica,
media/desvio do z-score, tau_B) sao os salvos em `metadata.json` desta mesma pasta, sempre derivados dos
dados de treino, nao valores fixos de codigo. Para rodar noutra boia, retreine com `codigos/13_qc_lstm_causal_ordinal_ba1.py`
e `codigos/15_salvar_modelo_final_ba1.py` apontando para os dados dessa boia, os arquivos aqui salvos sao
especificos da BA-1.

## Arquivos

- `predictor_lstm.keras`, rede LSTM (128-64-32 unidades) que preve `Hsig` um passo a frente, causal
  (janela de entrada vai de `t-24h` ate `t-1h`, nunca ve `t`).
- `classificador_ordinal.keras`, cabeca classificadora (Dense 64-32, duas saidas sigmoid `q1`/`q2`) que
  decide a classe de qualidade a partir do vetor de 25 features descrito abaixo.
- `input_scaler_lstm.pkl`, `MinMaxScaler` do scikit-learn, ajustado no treino, normaliza as variaveis de
  entrada da LSTM antes da previsao.
- `feature_scaler_classificador.pkl`, `ClippedRobustScaler` (classe definida em
  `codigos/13_qc_lstm_causal_ordinal_ba1.py`, mediana/IQR com winsorizacao nos percentis 0,5 e 99,5 do
  treino), normaliza o vetor de 25 features antes do classificador. Precisa do modulo `qc13` registrado em
  `sys.modules` para ser carregado com `pickle.load` (replicar o padrao usado em `codigos/15_salvar_modelo_final_ba1.py`).
- `metadata.json`, todos os parametros numericos e a ordem exata das features.

## Entrada

Para pontuar um novo instante `t`, e necessario ter a serie horaria de `Hsig` e das variaveis auxiliares
listadas em `metadata.json -> variaveis_auxiliares_ordem` disponivel de forma continua desde pelo menos
`t-48h` (a maior janela usada pelos indicadores de acumulacao e 48 horas).

1. Reamostrar para frequencia horaria (mediana da hora) e interpolar lacunas curtas (ate 3 horas), o
   mesmo pre-processamento usado no treino.
2. Normalizar as variaveis de entrada da LSTM com `input_scaler_lstm.pkl`.
3. Montar a janela de 24 horas terminando em `t-1` e chamar `predictor_lstm.predict` para obter
   `Hsig_previsto_t`.
4. Calcular o residuo `e_t = Hsig_t - Hsig_previsto_t`.
5. Calcular os 6 indicadores estatisticos causais (`S_acc`, `S_level`, `S_rate`, `S_range`, `S_MAD_score`,
   `S_GPD_score`), todos usando so dados ate `t`, nunca `t+1`. `S_MAD_score` e `S_GPD_score` precisam da
   escala robusta e dos parametros da GPD ajustados no periodo de treino (reproduzir
   `causal_stat_vector`/`fit_robust_scale`/`fit_gpd_tail` de `codigos/13_qc_lstm_causal_ordinal_ba1.py`
   e `codigos/04_lstm_peak_qc_ba1.py`).
6. Calcular os 10 indicadores de acumulacao (EWMA nas 3 escalas, CUSUM positivo e negativo com reinicio em
   20 desvios-padrao, inclinacao local nas 4 janelas, persistencia do sinal), todos recursivos sobre o
   historico do residuo, precisam do residuo de todos os instantes anteriores, nao so do instante atual.
7. Montar o vetor de 25 features na ordem exata de `metadata.json -> vetor_features_classificador_ordem`,
   `Hsig_observado`, `Hsig_previsto`, `residuo`, `residuo_absoluto`, os 6 indicadores estatisticos, os 10
   de acumulacao e por fim as 5 variaveis auxiliares atuais (nesta ordem).
8. Se o valor observado em `t` for ausente ou um codigo sentinela conhecido (`NaN`, valor negativo
   impossivel), imputar pela mediana do treino antes de montar o vetor (o residuo tambem deve ser zerado
   nesse caso), e marcar esse ponto para o override deterministico descrito abaixo, o classificador nao
   precisa decidir esses casos.
9. Normalizar o vetor de 25 features com `feature_scaler_classificador.pkl`.

## Saida

O classificador retorna `q1` (probabilidade de nao ser GOOD) e `q2` (probabilidade de ser BAD dado que
nao e GOOD). As probabilidades finais das tres classes sao

```
P(GOOD)    = 1 - q1
P(SUSPECT) = q1 * (1 - q2)
P(BAD)     = q1 * q2
```

A decisao final usa o limiar `tau_B` salvo em `metadata.json`,

```
Q_t = BAD                                          se P(BAD) >= tau_B
Q_t = SUSPECT                                       se P(BAD) <  tau_B  e  P(SUSPECT) >= P(GOOD)
Q_t = GOOD                                          caso contrario
```

com um override deterministico por cima dessa regra, qualquer ponto identificado no passo 8 (ausencia ou
codigo sentinela) e classificado diretamente como BAD, independente da saida da rede.

## Desempenho de referencia (BA-1, teste, uma execucao representativa)

Ver `metadata.json -> metricas_teste_referencia`. Os numeros completos, incluindo estabilidade entre 15
execucoes (3 sementes de modelo x 5 replicas de injecao), estao em
`resultados_qc_ba1/qc_lstm_causal_ordinal/estabilidade/estabilidade_interpretacao.md`.
