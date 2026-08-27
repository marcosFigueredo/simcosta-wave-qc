# Modelo geral multi-boia, especificacao de entrada e saida

Treinado com dados combinados de BA-1, ES-1, PR-1, RJ-1, RJ-2, RJ-4 (so Hsig, sem variaveis
auxiliares). Mesma logica de entrada/saida do modelo especialista da BA-1
(`resultados_qc_ba1/qc_lstm_causal_ordinal/modelo_final/README_entrada_saida.md`), a
diferenca e so o vetor de features aqui tem 21 posicoes (sem as 5 variaveis auxiliares que
so a BA-1 tinha disponiveis) em vez de 25, e os parametros de calibracao (`tau_b`, faixa
fisica, media/desvio de referencia) vem do treino combinado, nao so da BA-1.

Arquivos, `predictor_lstm.keras` (preve Hsig um passo a frente, entrada (24h, 1)),
`classificador_ordinal.keras` (duas saidas sigmoid q1/q2, entrada com 21 features na ordem
de `metadata.json -> vetor_features_classificador_ordem`), `input_scaler_lstm.pkl`,
`feature_scaler_classificador.pkl`, `metadata.json`.