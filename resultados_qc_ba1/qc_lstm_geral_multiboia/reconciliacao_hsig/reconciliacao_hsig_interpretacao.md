# Reconciliacao do rotulo de Hsig entre o ramo estatistico classico e o ramo com IA

Regra, para cada hora, mapeia R (ramo classico, use/review/remove) e Q_t (ramo IA,
GOOD/SUSPECT/BAD) para o mesmo espaco ordinal de severidade (0, 1, 2) e adota o mais
severo dos dois como rotulo final, a mesma logica ja usada para consolidar os sete testes
classicos dentro do proprio R.

- Horas com os dois rotulos disponiveis, 8520
- Concordancia exata R vs Q_t, 0.727
- Horas escaladas pelo ramo classico (R mais severo), 2289 (26.9%)
- Horas escaladas pelo ramo IA (Q_t mais severo), 34 (0.4%)

## Matriz de concordancia (linhas R, colunas Q_t)

| R (classico)   |   GOOD |   SUSPECT |   BAD |
|:---------------|-------:|----------:|------:|
| use            |   6125 |        33 |     1 |
| review         |   1838 |        72 |     0 |
| remove         |    374 |        77 |     0 |

## Distribuicao final apos reconciliacao (%)

| Q_reconciliado   |   count |
|:-----------------|--------:|
| GOOD             |   71.89 |
| SUSPECT          |   22.81 |
| BAD              |    5.31 |