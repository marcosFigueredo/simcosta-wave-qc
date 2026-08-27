# Protocolo de desenvolvimento e teste do modelo QC-LSTM de três classes

## 1. Objetivo

Desenvolver e testar um método de controle de qualidade para a variável **altura significativa de onda (`Hsig`)** no qual a inteligência artificial participe diretamente da produção do rótulo de qualidade.

O produto final para cada observação será:

\[
Q_t^{Hsig}\in\{\mathrm{GOOD},\mathrm{SUSPECT},\mathrm{BAD}\}.
\]

A proposta mantém a LSTM como núcleo do método, mas amplia sua função. A rede não será usada apenas para prever `Hsig`; sua representação temporal, sua previsão e os resíduos associados alimentarão uma cabeça classificadora responsável por produzir o rótulo QC.

O método deverá responder à seguinte pergunta:

> Considerando o histórico recente de `Hsig`, as variáveis oceanográficas e meteorológicas auxiliares e os indicadores estatísticos de consistência, qual é a classe de qualidade mais provável para a observação `Hsig_t`?

---

## 2. Diferença em relação ao método de Xie et al.

No método LSTM-Peak de Xie et al., o fluxo é:

\[
\text{LSTM}\rightarrow \widehat{Hsig}_t
\rightarrow \mathrm{DiffMean}_t
\rightarrow \text{detecção de picos}
\rightarrow Q_t\in\{0,1\}.
\]

A LSTM é utilizada apenas para previsão. O rótulo é produzido posteriormente por uma regra de detecção de picos com parâmetros fixos.

No método proposto, o fluxo será:

\[
\text{LSTM}\rightarrow
\begin{cases}
\widehat{Hsig}_t,\\
\mathbf h_t
\end{cases}
\rightarrow
\text{resíduos + indicadores estatísticos + contexto}
\rightarrow
\text{classificador neural QC}
\rightarrow
Q_t\in\{\mathrm{GOOD},\mathrm{SUSPECT},\mathrm{BAD}\}.
\]

A principal diferença metodológica é que a decisão QC não será obtida por uma regra fixa aplicada ao erro da previsão. A própria IA aprenderá a combinar:

- a previsão de `Hsig`;
- a representação temporal interna da LSTM;
- o erro entre observado e previsto;
- a dinâmica local da série;
- indicadores estatísticos robustos;
- coerência com as variáveis auxiliares.

---

## 3. Escopo da primeira versão

### 3.1 Variável-alvo

A primeira versão avaliará apenas:

\[
y_t = Hsig_t.
\]

As demais variáveis serão utilizadas como contexto e não receberão rótulo QC pela mesma saída.

### 3.2 Entrada multivariada

Para uma janela temporal de comprimento \(L\), a entrada será:

\[
\mathbf X_t=
[\mathbf x_{t-L},\ldots,\mathbf x_{t-1}],
\]

com:

\[
\mathbf x_i=
[Hsig_i,z_i^{(1)},z_i^{(2)},\ldots,z_i^{(p)}],
\]

onde \(z_i^{(j)}\) representa variáveis auxiliares, como:

- velocidade e rajada do vento ERA5;
- período de onda;
- direção ou espalhamento direcional;
- outras variáveis selecionadas sem redundância quase tautológica com `Hsig`.

A janela inicial recomendada é de **24 horas**, para manter comparabilidade com a implementação atual.

### 3.3 Saídas

O modelo terá duas saídas principais:

1. **saída de regressão**

\[
\widehat{Hsig}_t;
\]

2. **saída de classificação**

\[
\mathbf p_t=
[p_t^{G},p_t^{S},p_t^{B}],
\]

com:

\[
p_t^{G}+p_t^{S}+p_t^{B}=1.
\]

O rótulo final será:

\[
Q_t^{Hsig}=\arg\max_{c\in\{G,S,B\}}p_t^c.
\]

---

## 4. Definição operacional das classes

### 4.1 GOOD

Uma observação será considerada `GOOD` quando for consistente com o comportamento esperado e não houver evidência relevante de falha.

Características típicas:

- dentro do intervalo físico e instrumental;
- consistente com a evolução temporal recente;
- coerente com vento, período e demais variáveis auxiliares;
- residual de previsão compatível com o regime local;
- ausência de sinais fortes de spike, drift ou travamento;
- não localizada em região fortemente afetada por lacuna ou interpolação.

### 4.2 SUSPECT

A classe `SUSPECT` representa incerteza operacional. É necessária para evitar que todo desvio inesperado seja tratado como falha.

Características típicas:

- residual elevado, mas valor fisicamente plausível;
- mudança rápida que pode ser um evento oceanográfico real;
- discordância entre a LSTM e os indicadores estatísticos;
- observação próxima a lacuna ou trecho interpolado;
- evidência moderada de anomalia;
- baixa confiança entre as duas classes extremas.

### 4.3 BAD

Uma observação será considerada `BAD` quando houver forte evidência de corrupção, falha ou incompatibilidade com o processo físico e instrumental.

Características típicas:

- valor impossível ou fora do intervalo instrumental;
- código de erro ou valor sentinela;
- spike isolado severo sem suporte nas demais variáveis;
- sensor travado;
- deslocamento abrupto incompatível com o contexto;
- drift persistente;
- alta probabilidade neural de anomalia acompanhada de evidência estatística ou multivariada.

---

## 5. Arquitetura proposta

## 5.1 Codificador LSTM

A LSTM processará a janela histórica:

\[
\mathbf h_t=\operatorname{LSTM}_{\theta}(\mathbf X_t),
\]

onde \(\mathbf h_t\) é o vetor de estado que resume o comportamento temporal recente.

A arquitetura inicial pode preservar a configuração já implementada:

- primeira camada LSTM: 128 unidades;
- segunda camada LSTM: 64 unidades;
- terceira camada LSTM: 32 unidades;
- dropout: 0,2.

## 5.2 Cabeça de previsão

A previsão será direta para o instante avaliado:

\[
\widehat{Hsig}_t=g_{\mathrm{pred}}(\mathbf h_t).
\]

Não será usada a média de previsões associadas a instantes diferentes.

O residual será:

\[
e_t=Hsig_t-\widehat{Hsig}_t.
\]

Também serão calculados:

\[
|e_t|,
\qquad
\Delta Hsig_t=Hsig_t-Hsig_{t-1}.
\]

## 5.3 Indicadores estatísticos

Os indicadores estatísticos serão entradas da IA, e não decisões finais independentes.

Um vetor inicial pode conter:

\[
\mathbf s_t=
[S_t^{\mathrm{spike}},
S_t^{\mathrm{MAD}},
S_t^{\mathrm{GPD}},
S_t^{\mathrm{rate}},
S_t^{\mathrm{range}}].
\]

Exemplos:

### Spike local

\[
S_t^{\mathrm{spike}}=
\left|
Hsig_t-\frac{Hsig_{t-1}+Hsig_{t+1}}{2}
\right|.
\]

Esse indicador utiliza informação futura e, portanto, é apropriado para QC atrasado. Para operação em tempo real deverá ser substituído por uma versão causal.

### Escore robusto

\[
S_t^{\mathrm{MAD}}=
\frac{|e_t-\operatorname{mediana}(e)|}
{1{,}4826\operatorname{MAD}(e)+\varepsilon}.
\]

### Escore de cauda

\[
S_t^{\mathrm{GPD}}=-\log(p_t^{\mathrm{GPD}}+\varepsilon).
\]

### Taxa de variação

\[
S_t^{\mathrm{rate}}=|Hsig_t-Hsig_{t-1}|.
\]

### Indicador de range

\[
S_t^{\mathrm{range}}=
\mathbf 1(Hsig_t\notin[a,b]).
\]

## 5.4 Cabeça classificadora QC

O vetor de classificação será:

\[
\mathbf v_t=
[\mathbf h_t,
Hsig_t,
\widehat{Hsig}_t,
e_t,
|e_t|,
\Delta Hsig_t,
\mathbf s_t,
\mathbf z_t,
\mathbf m_t],
\]

onde:

- \(\mathbf z_t\): variáveis auxiliares atuais;
- \(\mathbf m_t\): máscaras de ausência, interpolação e confiabilidade das entradas.

A cabeça classificadora será:

\[
\mathbf p_t=
\operatorname{softmax}
(g_{\mathrm{QC}}(\mathbf v_t)).
\]

Arquitetura inicial sugerida:

```text
Dense(64) → ReLU → Dropout(0.2)
Dense(32) → ReLU → Dropout(0.1)
Dense(3)  → Softmax
```

---

## 6. Estratégia de treinamento

A separação entre treinamento, validação e teste deve ser estritamente cronológica.

## 6.1 Divisão recomendada

Proposta:

- 60% inicial: treinamento;
- 20% seguinte: validação;
- 20% final: teste.

Alternativamente, pode-se usar 70/15/15 para aproveitar mais dados no treinamento. O ponto essencial é que o teste final permaneça isolado.

Nenhuma estatística de normalização, calibração ou seleção de limiar poderá ser calculada usando o conjunto de teste.

## 6.2 Fase A — treinamento da LSTM preditora

A LSTM será treinada para aprender o comportamento esperado de `Hsig`.

A perda recomendada é Huber:

\[
\mathcal L_{\mathrm{pred}}
=
\frac{1}{N}
\sum_t
\operatorname{Huber}(Hsig_t-\widehat{Hsig}_t).
\]

A perda Huber é preferível ao MSE puro porque reduz a influência de perturbações reais desconhecidas sem eliminar automaticamente eventos extremos plausíveis.

Nesta fase:

- remover apenas falhas inequívocas;
- manter extremos fisicamente plausíveis;
- mascarar ausências e valores sentinela;
- não tratar toda sinalização estatística como erro confirmado.

## 6.3 Fase B — geração de rótulos de treinamento

Como os dados reais não possuem ground truth completo, serão utilizados três tipos de informação:

1. **rótulos de alta confiança obtidos dos dados reais**;
2. **rótulos fracos para casos ambíguos**;
3. **anomalias sintéticas controladas**.

### GOOD de alta confiança

Selecionar observações que satisfaçam simultaneamente critérios conservadores:

- dentro do range;
- sem lacuna próxima;
- sem interpolação relevante;
- sem sinalização por múltiplos detectores;
- residual central;
- coerência multivariada.

### SUSPECT

Selecionar observações com:

- discordância entre detectores;
- residual moderado;
- evento plausível, mas inesperado;
- proximidade de lacuna;
- baixa confiança entre `GOOD` e `BAD`.

### BAD de alta confiança

Incluir:

- falhas instrumentais inequívocas;
- valores impossíveis;
- códigos sentinela;
- anomalias sintéticas injetadas somente em cópias dos dados de treinamento.

## 6.4 Fase C — treinamento da cabeça QC

Inicialmente, os pesos da LSTM serão congelados. A cabeça de classificação será treinada com entropia cruzada ponderada:

\[
\mathcal L_{\mathrm{QC}}
=-\sum_t c_t
\sum_{k\in\{G,S,B\}}
w_k y_{t,k}\log p_{t,k}.
\]

onde:

- \(w_k\): peso para corrigir desbalanceamento entre classes;
- \(c_t\): confiança do rótulo.

Exemplo de pesos de confiança:

\[
c_t=
\begin{cases}
1{,}0,& \text{GOOD de consenso forte},\\
1{,}0,& \text{BAD sintético ou inequívoco},\\
0{,}5,& \text{SUSPECT ou rótulo fraco}.
\end{cases}
\]

## 6.5 Fase D — ajuste conjunto opcional

Após verificar que a classificação funciona, poderá ser realizado fine-tuning multitarefa:

\[
\mathcal L_{\mathrm{total}}
=
\lambda_{\mathrm{pred}}\mathcal L_{\mathrm{pred}}
+
\lambda_{\mathrm{QC}}\mathcal L_{\mathrm{QC}}.
\]

O primeiro experimento deve usar a LSTM congelada. Isso permite verificar se o ganho decorre da nova cabeça QC e evita que a classificação prejudique a capacidade preditiva antes da validação inicial.

---

## 7. Injeção de perturbações

A injeção de anomalias não será usada apenas para avaliação. No novo método, cópias perturbadas do conjunto de treinamento também fornecerão exemplos rotulados para a classe `BAD` e, em alguns casos, para `SUSPECT`.

As posições, magnitudes e sementes devem ser independentes entre treinamento, validação e teste.

## 7.1 Famílias de perturbação

### A. Spike aditivo

\[
Hsig_t^{*}=Hsig_t+s\,k\sigma,
\]

com:

- \(s\in\{-1,+1\}\);
- \(k\in\{0{,}5,1,2,3,4,6\}\).

Uso esperado:

- \(k\leq1\): geralmente `GOOD` ou `SUSPECT`;
- \(1<k<3\): predominantemente `SUSPECT`;
- \(k\geq3\): predominantemente `BAD`.

Esses limites são hipóteses de simulação e não devem ser impostos diretamente ao classificador.

### B. Perturbação multiplicativa

Reproduzir o artigo de referência:

\[
Hsig_t^{*}\in
\left\{
5Hsig_t,
10Hsig_t,
\frac{Hsig_t}{5},
\frac{Hsig_t}{10}
\right\}.
\]

### C. Mudança de nível

\[
Hsig_{t:t+d}^{*}=Hsig_{t:t+d}+\delta,
\]

com duração:

\[
d\in\{3,6,12,24\}\text{ horas}.
\]

### D. Drift

\[
Hsig_{t+j}^{*}=Hsig_{t+j}+j\delta,
\qquad j=0,\ldots,d.
\]

### E. Sensor travado

\[
Hsig_{t:t+d}^{*}=Hsig_t.
\]

### F. Ruído em rajada

\[
Hsig_{t:t+d}^{*}=Hsig_{t:t+d}+\epsilon_j,
\qquad
\epsilon_j\sim\mathcal N(0,\sigma_a^2).
\]

### G. Falha de ausência ou código sentinela

Simular:

- `NaN`;
- valor sentinela;
- ausência contínua;
- valor negativo impossível.

Esses casos devem ser classificados por regra determinística como `BAD`, podendo ser usados também como teste da camada de pré-QC.

---

## 8. Simulação computacional de comportamento

A simulação deve demonstrar não apenas que o modelo alcança bom F1, mas que reage conforme o comportamento esperado.

## 8.1 Objetivo da simulação

Verificar se o modelo:

- aumenta a probabilidade de `BAD` quando a perturbação se torna mais severa;
- diferencia evento físico plausível de erro isolado;
- mantém baixa taxa de falsos alarmes em dados não perturbados;
- reconhece falhas persistentes, não apenas spikes;
- utiliza as variáveis auxiliares e não apenas a magnitude de `Hsig`;
- não utiliza informação futura de forma indevida no modo causal.

## 8.2 Simulação baseada em janelas reais

A primeira simulação deve usar janelas reais do conjunto de teste.

Para cada janela limpa selecionada:

1. preservar a janela histórica;
2. calcular a previsão da LSTM;
3. criar cópias com diferentes perturbações no ponto avaliado;
4. obter \(p^G\), \(p^S\) e \(p^B\);
5. comparar a resposta entre magnitudes e tipos de perturbação.

## 8.3 Teste de monotonicidade

Para spikes de magnitude crescente, espera-se:

\[
\mathbb E[p^B(k_1)]
\leq
\mathbb E[p^B(k_2)]
\quad\text{quando}\quad k_1<k_2.
\]

Métrica:

```text
Monotonicity rate = proporção de pares consecutivos em que p(BAD) não diminui.
```

Meta inicial:

\[
\text{monotonicity rate}\geq0{,}90.
\]

## 8.4 Teste de coerência física contextual

Construir dois cenários com o mesmo aumento de `Hsig`:

### Cenário A — sem suporte físico

- `Hsig` aumenta abruptamente;
- vento, período e direção permanecem incompatíveis com o aumento.

### Cenário B — com suporte físico

- `Hsig` aumenta;
- vento e parâmetros de onda também mudam de maneira coerente.

Comportamento esperado:

\[
p_B^{A}>p_B^{B},
\]

ou, alternativamente:

\[
p_S^{B}>p_B^{B}.
\]

Ou seja, um valor extremo fisicamente apoiado deve tender a `SUSPECT`, e não automaticamente a `BAD`.

## 8.5 Teste de causalidade

No modelo causal, alterar \(Hsig_{t+1}\) não poderá modificar o rótulo de \(Hsig_t\):

\[
Q_t(Hsig_{t+1})=Q_t(Hsig_{t+1}^{*}).
\]

Se o resultado mudar, existe vazamento de informação futura.

## 8.6 Teste de sensibilidade a entrada auxiliar

Perturbar apenas uma variável auxiliar e manter `Hsig` inalterado.

O objetivo é verificar se:

- a previsão pode mudar moderadamente;
- a classificação não muda de `GOOD` para `BAD` por uma pequena perturbação irrelevante;
- a máscara de confiabilidade reduz o efeito de entrada auxiliar suspeita.

## 8.7 Teste de recuperação

Após uma anomalia isolada, verificar quantos passos são necessários para a saída retornar ao padrão normal.

Métrica:

\[
T_{\mathrm{rec}}=
\min\{j>0:p_{t+j}^{G}\geq\tau_G\}.
\]

Isso é especialmente importante se a observação corrompida for realimentada na janela seguinte.

Devem ser comparados dois modos:

1. realimentação do valor observado;
2. substituição temporária por previsão ou imputação robusta quando `BAD`.

## 8.8 Teste de falha persistente

Para drift, travamento ou mudança de nível, medir:

- tempo até a primeira sinalização `SUSPECT`;
- tempo até a primeira sinalização `BAD`;
- proporção de pontos detectados durante o episódio.

---

## 9. Experimentos comparativos

## 9.1 Baselines

O novo método deverá ser comparado com:

1. LSTM-Peak de Xie et al.;
2. Student-t robusta sobre o residual;
3. GPD-POT;
4. spike tradicional robusto;
5. Isolation Forest;
6. VAE-LSTM.

## 9.2 Ablation study

### Modelo A — residual sem IA classificadora

Regra estatística aplicada ao residual.

### Modelo B — classificador com residual apenas

Entrada:

\[
[Hsig_t,\widehat{Hsig}_t,e_t,|e_t|].
\]

### Modelo C — residual + indicadores estatísticos

Entrada:

\[
[Hsig_t,\widehat{Hsig}_t,e_t,|e_t|,\mathbf s_t].
\]

### Modelo D — estado oculto + residual

Entrada:

\[
[\mathbf h_t,Hsig_t,\widehat{Hsig}_t,e_t,|e_t|].
\]

### Modelo E — modelo completo

Entrada:

\[
[\mathbf h_t,Hsig_t,\widehat{Hsig}_t,e_t,|e_t|,
\mathbf s_t,\mathbf z_t,\mathbf m_t].
\]

### Modelo F — sem LSTM

Usar apenas indicadores estatísticos e variáveis observadas. Esse teste demonstra se a representação temporal aprendida pela LSTM acrescenta informação.

---

## 10. Métricas

## 10.1 Métricas da previsão

- MAE;
- RMSE;
- MAPE, com cuidado para valores próximos de zero;
- erro médio assinado;
- erro por estação do ano;
- erro por regime de energia de onda.

## 10.2 Métricas da classificação de três classes

Para cada classe:

- precision;
- recall;
- F1.

Métricas globais:

- macro-F1;
- weighted-F1;
- balanced accuracy;
- Matthews correlation coefficient;
- matriz de confusão;
- AUROC one-vs-rest;
- AUPRC one-vs-rest.

A macro-F1 deve ser a métrica principal, pois trata as três classes com igual importância.

## 10.3 Métricas operacionais

- taxa de `BAD` por 1.000 observações;
- taxa de `SUSPECT` por 1.000 observações;
- retenção de dados `GOOD`;
- taxa de falsos alarmes;
- carga de revisão manual;
- detecção por tipo de falha;
- atraso de detecção em falhas persistentes.

## 10.4 Métricas de calibração

Como a rede produz probabilidades, devem ser avaliados:

- Brier score;
- expected calibration error;
- reliability diagram;
- entropia média da saída;
- frequência de baixa confiança.

## 10.5 Estabilidade

Executar pelo menos:

\[
3\text{ sementes da LSTM}
\times
5\text{ sementes de perturbação}
=15\text{ execuções}.
\]

Reportar:

- média;
- desvio-padrão;
- intervalo de confiança por bootstrap;
- ranking dos modelos por semente.

---

## 11. Testes estatísticos dos resultados

### Comparação de classificadores

Utilizar:

- McNemar para comparação pareada de erros em uma mesma base;
- bootstrap pareado para diferença de macro-F1;
- teste de Wilcoxon sobre métricas por execução quando a normalidade não for plausível.

### Comparação de calibração

Comparar Brier score e ECE por bootstrap.

### Comparação da capacidade preditiva

A capacidade de previsão da LSTM deve ser comparada antes e depois do fine-tuning conjunto, verificando se a inclusão da cabeça QC degrada o RMSE.

---

## 12. Saídas esperadas

## 12.1 Saídas por observação

Cada linha do banco final deve conter, no mínimo:

```text
timestamp
Hsig_observed
Hsig_predicted
prediction_residual
absolute_residual
p_good
p_suspect
p_bad
qc_label
qc_confidence
stat_spike
stat_mad
stat_gpd
missing_mask
interpolation_flag
model_version
```

## 12.2 Saídas agregadas

- distribuição mensal dos rótulos;
- distribuição anual dos rótulos;
- frequência de `GOOD`, `SUSPECT` e `BAD`;
- rótulos próximos a lacunas;
- rótulos por regime de onda;
- rótulos por tipo de perturbação simulada;
- estabilidade entre sementes.

## 12.3 Tabelas esperadas

### Tabela 1 — desempenho preditivo

| Modelo | MAE | RMSE | MAPE | Bias |
|---|---:|---:|---:|---:|

### Tabela 2 — desempenho QC por classe

| Modelo | F1 GOOD | F1 SUSPECT | F1 BAD | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|

### Tabela 3 — desempenho por tipo de anomalia

| Perturbação | Precision BAD | Recall BAD | F1 BAD | Atraso de detecção |
|---|---:|---:|---:|---:|

### Tabela 4 — ablação

| Configuração | Macro-F1 | AUPRC BAD | ECE | Falsos alarmes/1000 |
|---|---:|---:|---:|---:|

### Tabela 5 — teste de comportamento

| Teste | Métrica | Resultado | Critério esperado |
|---|---|---:|---:|

## 12.4 Figuras esperadas

1. arquitetura do modelo;
2. observado versus previsto;
3. probabilidades `GOOD`, `SUSPECT` e `BAD` ao longo do tempo;
4. matriz de confusão;
5. curvas precision-recall por classe;
6. reliability diagram;
7. probabilidade de `BAD` versus magnitude da perturbação;
8. comparação do mesmo pico com e sem suporte físico;
9. detecção de drift e sensor travado;
10. distribuição temporal dos rótulos reais.

---

## 13. Critérios de aceitação do primeiro protótipo

Os valores abaixo são metas de engenharia para decidir se o método merece desenvolvimento adicional. Não constituem resultados esperados garantidos.

### Desempenho mínimo

- macro-F1 superior ao LSTM-Peak;
- F1 da classe `BAD` não inferior ao melhor detector estatístico por mais de 0,02;
- recall `BAD` maior ou igual a 0,90 para perturbações severas;
- taxa de falsos alarmes inferior à do spike tradicional;
- estabilidade do macro-F1 entre sementes, com desvio-padrão inferior a 0,05.

### Comportamento

- monotonicity rate maior ou igual a 0,90;
- redução de \(p^{BAD}\) quando o aumento de `Hsig` possui suporte físico coerente;
- ausência de vazamento futuro no modo causal;
- recuperação após spike isolado em até duas janelas, quando usada imputação robusta;
- detecção de drift antes da metade do episódio simulado.

### Calibração

- ECE inferior a 0,10 no primeiro protótipo;
- melhoria de calibração após temperature scaling, se necessário.

---

## 14. Pseudocódigo do experimento

```python
# 1. Divisão temporal
train, validation, test = chronological_split(data)

# 2. Ajuste do pré-processamento apenas no treino
scaler.fit(train)
train_x = scaler.transform(train)
validation_x = scaler.transform(validation)
test_x = scaler.transform(test)

# 3. Treinamento da LSTM preditora
lstm.fit(train_windows, train_hsig, loss="huber")

# 4. Extração das representações e previsões
h_train, yhat_train = lstm.encode_and_predict(train_windows)
h_val, yhat_val = lstm.encode_and_predict(validation_windows)
h_test, yhat_test = lstm.encode_and_predict(test_windows)

# 5. Geração de cópias perturbadas independentes
train_aug = inject_anomalies(train, seeds=TRAIN_SEEDS)
val_aug = inject_anomalies(validation, seeds=VALIDATION_SEEDS)
test_aug = inject_anomalies(test, seeds=TEST_SEEDS)

# 6. Rótulos de três classes
train_labels = build_weak_and_synthetic_labels(train_aug)
val_labels = build_validation_labels(val_aug)
test_labels = build_test_ground_truth(test_aug)

# 7. Construção das features QC
qc_train = build_qc_features(
    hidden_state=h_train,
    observed=train_aug.hsig,
    predicted=yhat_train,
    statistical_scores=compute_scores(train_aug),
    masks=train_aug.masks,
)

# 8. Treinamento da cabeça QC
qc_head.fit(
    qc_train,
    train_labels,
    class_weights=class_weights,
    confidence_weights=label_confidence,
)

# 9. Seleção no conjunto de validação
best_model = select_by_macro_f1_and_calibration(qc_head, val_aug)

# 10. Avaliação única no teste
results = evaluate(best_model, test_aug)

# 11. Testes comportamentais
run_monotonicity_test(best_model, test)
run_context_consistency_test(best_model, test)
run_causality_test(best_model, test)
run_drift_and_stuck_sensor_tests(best_model, test)
```

---

## 15. Riscos metodológicos e controles

### Dados reais já perturbados

Não assumir que todos os dados originais são `GOOD`. Usar somente candidatos de alta confiança como exemplos positivos.

### Aprendizado das anomalias sintéticas

O classificador pode aprender apenas a regra de injeção. Para reduzir esse risco:

- usar várias famílias de falha;
- variar magnitudes e durações;
- usar sementes independentes;
- manter teste com perturbações não vistas no treinamento;
- incluir avaliação sobre eventos reais revisados por especialista.

### Desbalanceamento

A classe `BAD` será rara. Usar:

- pesos de classe;
- focal loss como teste adicional;
- amostragem balanceada apenas no treinamento;
- prevalência operacional real no teste.

### Contaminação temporal

Garantir que janelas sobrepostas não atravessem os limites entre treino, validação e teste.

### Variáveis auxiliares incorretas

Incluir máscaras e flags básicos para evitar que uma entrada auxiliar suspeita seja tratada como completamente confiável.

### Classe SUSPECT artificial

A classe intermediária não deve ser apenas uma faixa arbitrária de magnitude. Deve representar ambiguidade real, discordância ou baixa confiança contextual.

---

## 16. Sequência prática de implementação

### Etapa 1

Reproduzir a LSTM atual com previsão direta de `Hsig_t`.

### Etapa 2

Construir o classificador com apenas:

\[
[Hsig_t,\widehat{Hsig}_t,e_t,|e_t|].
\]

### Etapa 3

Adicionar os indicadores estatísticos.

### Etapa 4

Adicionar o estado oculto da LSTM.

### Etapa 5

Adicionar variáveis auxiliares atuais e máscaras.

### Etapa 6

Executar a simulação comportamental completa.

### Etapa 7

Comparar com LSTM-Peak e demais baselines.

### Etapa 8

Somente depois, testar fine-tuning multitarefa.

---

## 17. Resultado científico esperado

Ao final, o trabalho deverá demonstrar se uma arquitetura LSTM com cabeça classificadora é capaz de:

1. prever o comportamento esperado de `Hsig`;
2. combinar previsão, memória temporal e estatística robusta;
3. gerar diretamente rótulos `GOOD`, `SUSPECT` e `BAD`;
4. reduzir a dependência de limiares fixos;
5. separar melhor eventos físicos plausíveis de falhas instrumentais;
6. manter estabilidade entre inicializações e diferentes perturbações;
7. apresentar comportamento verificável por simulação computacional, e não apenas bom desempenho agregado.

A contribuição central será uma mudança de paradigma em relação ao LSTM-Peak: a LSTM deixa de ser apenas um preditor auxiliar e passa a integrar diretamente o mecanismo de decisão QC.
