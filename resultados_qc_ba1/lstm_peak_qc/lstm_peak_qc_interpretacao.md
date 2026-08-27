# Teste LSTM-Peak inspirado em Xie et al. (2023) para BA-1

## Objetivo

Testar, de forma controlada, a proposta do artigo `QC.pdf`: usar uma LSTM multivariada para prever a altura significativa de onda e aplicar detecção estatística de picos sobre a diferença entre valor observado e valor previsto.

## Adaptação para a boia BA-1

O artigo usa altura significativa de onda e velocidade do vento. O arquivo OCEAN da BA-1 não contém vento medido pela boia. Para aproximar a variável física usada no artigo, este teste incorpora vento horário de reanálise ERA5 (via API Open-Meteo) no ponto de grade mais próximo das coordenadas da boia — ver `codigos/00_fetch_wind_era5_ba1.py` e `dadosSimcosta/ERA5_BA-1_WIND_2019-08-02_2026-07-25.csv`. Isso NÃO é vento medido in situ: é uma estimativa de modelo em grade de ~0.25°, a 10 m de altura (o anemômetro da SiMCosta, quando presente, mede a 3 m), portanto tratada como uma fragilidade metodológica documentada, não como equivalente ao vento do artigo original. Para evitar um segundo problema — selecionar automaticamente `HM0`, `Hmax`, `Havg` ou `H10` como variável auxiliar, que são apenas outras estatísticas de ordem do mesmo registro de onda que gera `Hsig` (r>0.98) e tornariam a previsão quase tautológica — essas variáveis são excluídas do conjunto candidato; a seleção das 5 variáveis auxiliares ocorre entre variáveis fisicamente distintas (vento ERA5, período, direção/espalhamento de onda e variáveis da coluna d'água).

- Variável-alvo: `Hsig`
- Variáveis usadas no modelo: `Hsig`, `Avg_Wv_Spread_N`, `wind_speed_10m`, `wind_gusts_10m`, `Tavg`, `Tsig`
- Janela temporal de entrada: 24 horas
- Fração de teste: 30%
- Taxa de anomalias artificiais: aproximadamente 1:130

## Correlações com Hsig

| Variável | r de Pearson |
|---|---:|
| H10 | 0.999 |
| HM0 | 0.999 |
| Havg | 0.999 |
| Hmax | 0.989 |
| Avg_Wv_Spread_N | -0.626 |
| wind_speed_10m | 0.478 |
| wind_gusts_10m | 0.453 |
| Tavg | 0.382 |
| Tsig | 0.259 |
| Avg_W_Tmp2 | -0.231 |
| Avg_W_Tmp1 | -0.228 |
| Avg_Sal | -0.183 |
| T10 | 0.171 |
| Tp | -0.077 |
| Avg_DO | 0.068 |
| Avg_Chl | 0.057 |
| Tp5 | -0.039 |
| Avg_Wv_Dir_N | 0.028 |
| Avg_Turb | 0.025 |
| Avg_CDOM | -0.010 |

## Desempenho preditivo da LSTM (média ± desvio-padrão, 3 modelos retreinados)

| Métrica | Valor |
|---|---:|
| MAE | 0.0745 ± 0.0023 |
| MSE | 0.0125 |
| RMSE | 0.1117 |
| MAPE (%) | 11.40 ± 0.21 |

## Desempenho do QC com anomalias artificiais (média ± desvio-padrão, 3 modelos × 5 réplicas = 15 execuções)

Vai além do protocolo do artigo (Seção 3.2/4.3, que só varia as anomalias sintéticas): aqui a LSTM também é retreinada do zero em cada modelo (init de pesos e split de EarlyStopping diferentes), para checar se a vantagem de um método sobre outro depende de qual modelo treinado calhou de ser usado, e não só de qual conjunto de anomalias foi sorteado.

| Método | Precision | Recall | F1 | Flags detectadas |
|---|---:|---:|---:|---:|
| GPD-POT (cauda) | 0.416 ± 0.012 | 1.000 ± 0.000 | 0.588 ± 0.012 | 48.1 ± 1.3 |
| Isolation Forest | 0.016 ± 0.006 | 0.067 ± 0.024 | 0.026 ± 0.009 | 81.1 ± 4.8 |
| LSTM-Peak (artigo, H/T/d fixos) | 1.000 ± 0.000 | 0.603 ± 0.069 | 0.750 ± 0.054 | 12.1 ± 1.4 |
| Robusto mediana/MAD (Student-t) | 0.847 ± 0.047 | 1.000 ± 0.000 | 0.917 ± 0.027 | 23.7 ± 1.3 |
| Spike tradicional robusto | 0.286 ± 0.008 | 1.000 ± 0.000 | 0.445 ± 0.010 | 70.0 ± 2.0 |
| VAE-LSTM | 0.031 ± 0.001 | 0.997 ± 0.013 | 0.061 ± 0.003 | 636.6 ± 27.2 |

### Consistência por modelo retreinado (F1 médio das 5 réplicas de anomalia, por modelo)

| Método | Modelo seed=42 | Modelo seed=7 | Modelo seed=123 |
|---|---:|---:|---:|
| GPD-POT (cauda) | 0.599 | 0.576 | 0.588 |
| Isolation Forest | 0.030 | 0.025 | 0.024 |
| LSTM-Peak (artigo, H/T/d fixos) | 0.748 | 0.748 | 0.756 |
| Robusto mediana/MAD (Student-t) | 0.889 | 0.909 | 0.952 |
| Spike tradicional robusto | 0.445 | 0.445 | 0.445 |
| VAE-LSTM | 0.061 | 0.063 | 0.058 |

## Teste de erro probabilístico (extensão além do artigo)

O artigo decide `Q_t=0`/`Q_t=1` com constantes fixas (`H=1, T=1, d=50`), iguais para as 4 estações chinesas, sem calibração por site nem controle formal de taxa de falso positivo. Aqui isso é substituído/comparado por dois testes estatísticos calibrados na própria distribuição de DiffMean da BA-1, ajustados apenas no período *limpo* (antes da injeção de anomalias artificiais) para não contaminar a referência do que é 'erro normal' com as próprias anomalias que o teste precisa detectar:

1. **Robusto mediana/MAD (Student-t dobrada)**: como o DiffMean já é `|erro|/média local` (não-negativo), ele é tratado como uma Student-t dobrada; a escala é obtida a partir da mediana do DiffMean limpo (robusta a outliers) e `dof` fixo (mais cauda pesada que Gaussiana). Cada ponto recebe uma probabilidade de cauda; sinaliza-se `Q_t=0` se p < α.
2. **GPD-POT (peaks-over-threshold)**: a cauda superior do DiffMean limpo (acima do percentil 90) é ajustada com uma Distribuição de Pareto Generalizada — a abordagem padrão da teoria de valores extremos para estimar probabilidade de cauda sem supor a forma da distribuição inteira. Mais apropriado se o erro tiver cauda mais pesada do que a Student-t consegue capturar.

- α (nível de significância): 0.01
- Robusto mediana/MAD: dof=4, escala ajustada=0.1195
- GPD-POT: limiar (percentil 90%)=0.2368, forma (ξ)=-0.0741, escala=0.0837, P(DiffMean>limiar)=0.1003

(Parâmetros de calibração acima são do modelo de referência, seed=42; cada um dos 3 modelos retreinados recalibra os dois testes nos seus próprios resíduos.)

## Baselines nativos do artigo: Isolation Forest e VAE-LSTM

O artigo (Seção 3.3) compara o LSTM-Peak com dois outros modelos, que estavam ausentes desta reprodução até agora e foram adicionados:

3. **Isolation Forest** (Liu et al., 2009): ajustado apenas no período de treino limpo (nunca vê as anomalias injetadas), com `contamination` igual à mesma proporção 1:130 que o próprio artigo usa para a regra de injeção (Seção 3.2). Aplicado sobre o vetor de features do timestamp de teste (alvo + variáveis auxiliares), reajustado a cada um dos modelos retreinados.
4. **VAE-LSTM** (Lin et al., 2020, citado pelo artigo): um VAE reconstrói janelas curtas (features locais) e uma LSTM modela a correlação de longo prazo entre os códigos latentes do VAE; o sinal de anomalia combina o erro de reconstrução do alvo com o erro de predição no espaço latente. Em vez de um limiar arbitrário, esse escore combinado é calibrado com o mesmo teste Student-t robusto (mediana/MAD) usado acima, ajustado só no período limpo — o mesmo tratamento estatístico aplicado de forma uniforme onde há um escore contínuo de anomalia.

- VAE-LSTM: escala de reconstrução=0.000505, escala do preditor latente=0.000823, escala robusta do escore combinado=3.2513

## Sensibilidade a α

α=0.01 foi escolhido, não derivado. `fig05_sensibilidade_alpha.png` e `sensibilidade_alpha.csv` mostram como Precision/Recall/F1 dos três testes calibrados por α (robusto mediana/MAD, GPD-POT, VAE-LSTM) respondem a α em [0.001, 0.005, 0.01, 0.025, 0.05, 0.1], para checar se a conclusão é sensível à escolha específica de α ou se é estável num intervalo razoável.

## Interpretação geral

**Ressalva de leitura importante**: as anomalias sintéticas injetadas são grandes (3-6× o desvio-padrão de Hsig), então quase todo método consegue recall alto - o que de fato diferencia os métodos aqui é a taxa de falso positivo em cima de dados reais (precision), não a capacidade de detectar um desvio grande. Uma comparação com anomalias sutis (mais próximas do limiar de detecção) tende a separar melhor os métodos pela capacidade de detecção, não só pela robustez a ruído normal.

Este teste não prova ainda um método final para a BA-1; ele avalia se a lógica do artigo é transferível para os dados disponíveis. A principal diferença metodológica é a ausência de vento no arquivo OCEAN. Como vento é a variável física usada no artigo para auxiliar a previsão de altura de onda, a BA-1 depende aqui de relações internas entre variáveis de onda e da coluna d'água. Isso ainda pode reduzir a capacidade de distinguir evento físico real de falha instrumental, mesmo com as estatísticas de ordem redundantes com `Hsig` (`HM0`, `Hmax`, `Havg`, `H10`) já excluídas da seleção de variáveis auxiliares.

Se a LSTM apresentar erro preditivo baixo, mas o LSTM-Peak tiver baixo recall, isso indica que o modelo reproduz a série, mas o critério de pico no resíduo não está capturando bem as anomalias artificiais. Se o recall for alto e a precision baixa, o método é sensível, mas gera muitos falsos positivos. Se ambos forem baixos, a proposta do artigo não se transfere diretamente para a BA-1 sem vento, calibração regional ou mudança de modelo.

## Figuras geradas

- `fig01_correlacao_hsig_variaveis.png`: seleção das variáveis auxiliares.
- `fig02_hsig_observado_previsto_flags.png`: comparação observado, previsto, anomalias artificiais e flags do LSTM-Peak.
- `fig03_diffmean_peak_detection.png`: razão de diferença local com flags dos 3 métodos baseados em DiffMean (LSTM-Peak, Robusto mediana/MAD, GPD-POT).
- `fig04_comparacao_metodos_qc.png`: Precision/Recall/F1 dos 6 métodos lado a lado (média ± desvio-padrão sobre modelos × réplicas).
- `fig05_sensibilidade_alpha.png`: Precision/Recall/F1 dos 3 testes calibrados por α (Robusto mediana/MAD, GPD-POT, VAE-LSTM).