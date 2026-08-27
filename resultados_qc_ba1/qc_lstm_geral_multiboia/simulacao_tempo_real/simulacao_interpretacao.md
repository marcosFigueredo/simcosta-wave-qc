# Simulacao em tempo real, modelo geral multi-boia na BA-1

Fluxo horario real da BA-1 tratado como dado OK, com um simulador que insere anomalias de
sete familias em momentos e intensidades aleatorios ao longo da execucao (nao um cronograma
fixo). O modelo processa o fluxo ponto a ponto, causal (so ve o passado e o instante atual),
incluindo a realimentacao real, um valor corrompido de um evento em andamento entra nas
janelas de entrada dos instantes seguintes, exatamente como aconteceria num sensor de verdade.

- Horas simuladas, 8592
- Horas dentro de algum evento de anomalia, 751 (8.7%)
- Macro-F1, 0.453
- Deteccao geral (GOOD vs nao-GOOD), precision 0.527, recall 0.236, F1 0.326

## Deteccao no instante t+1 (metrica principal)

Taxa de deteccao exatamente no primeiro instante em que cada episodio de anomalia aparece
(a transicao de GOOD para anomalo), o foco da simulacao, e nao a cobertura ao longo de todo
um episodio longo. Taxa geral, 0.667, sobre 45 eventos distintos.

| familia_letra   |   taxa_deteccao_t1 |   n_eventos |
|:----------------|-------------------:|------------:|
| A               |              0.500 |           2 |
| B               |              0.727 |          11 |
| C               |              0.909 |          11 |
| D               |              0.000 |           3 |
| E               |              0.000 |           5 |
| F               |              0.800 |          10 |
| G               |              1.000 |           3 |

## Cobertura ao longo de toda a duracao do episodio (metrica secundaria)

Fracao das horas dentro de cada familia que continuam sinalizadas do inicio ao fim do
episodio. Cai para familias longas (mudanca de nivel, drift, sensor travado) porque o valor
corrompido realimenta a propria janela de entrada da LSTM e o modelo passa a prever perto do
novo nivel depois de algumas horas, efeito de adaptacao do preditor, nao de falha em
detectar o inicio do evento.

| familia   |   n_horas |   acuracia_exata |   fracao_sinalizada |
|:----------|----------:|-----------------:|--------------------:|
| A         |         2 |            0.500 |               0.500 |
| B         |        11 |            0.455 |               0.727 |
| C         |       444 |            0.128 |               0.189 |
| D         |        96 |            0.146 |               0.146 |
| E         |        87 |            0.000 |               0.000 |
| F         |       108 |            0.296 |               0.620 |
| G         |         3 |            1.000 |               1.000 |

## Figura

`fig_simulacao_tempo_real.png`, trecho recente da simulacao, serie observada e prevista,
pontos detectados como SUSPECT/BAD e os eventos realmente injetados (circulo preto).