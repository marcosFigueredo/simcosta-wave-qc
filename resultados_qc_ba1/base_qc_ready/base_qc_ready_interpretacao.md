# Base QC-ready - SIMCOSTA_BA-1

Esta tabela consolida os testes de QC em uma base unica por `Timestamp` e `variable`.

## Regra de decisao

- `remove`: falha em range fisico, classe `Bad`, Dixon 4σ `BAD`, Q-Dixon `BAD` ou decisao `removed` da suite robusta.
- `review`: qualquer flag estatistica, Dixon/Q-Dixon `SUSPECT` ou `QI < 75`.
- `use`: observacao sem alertas relevantes.

## Resumo por variavel

| variable   |   remove |   review |    use |      N |   use_pct |   review_pct |   remove_pct |
|:-----------|---------:|---------:|-------:|-------:|----------:|-------------:|-------------:|
| Avg_CDOM   |     5933 |     4901 | 104644 | 115478 |   90.6181 |       4.2441 |       5.1378 |
| Avg_Chl    |     5473 |     3239 | 106766 | 115478 |   92.4557 |       2.8049 |       4.7394 |
| Avg_DO     |     3404 |     3558 | 108516 | 115478 |   93.9711 |       3.0811 |       2.9477 |
| Avg_Sal    |     1989 |     4804 | 108685 | 115478 |   94.1175 |       4.1601 |       1.7224 |
| Avg_Turb   |     8022 |     9115 |  98341 | 115478 |   85.1599 |       7.8933 |       6.9468 |
| Avg_W_Tmp1 |     3453 |     9095 | 102930 | 115478 |   89.1339 |       7.8760 |       2.9902 |
| Avg_W_Tmp2 |     5894 |     9411 | 100173 | 115478 |   86.7464 |       8.1496 |       5.1040 |
| HM0        |    21044 |    13858 |  80576 | 115478 |   69.7761 |      12.0006 |      18.2234 |
| Hmax       |    21042 |    12248 |  82188 | 115478 |   71.1720 |      10.6063 |      18.2217 |
| Hsig       |    21231 |    14295 |  79952 | 115478 |   69.2357 |      12.3790 |      18.3853 |
| Tp         |    22364 |     9788 |  83326 | 115478 |   72.1575 |       8.4761 |      19.3665 |

## Arquivos

- `base_qc_ready_long.csv`: base oficial longa para o Digital Twin.
- `base_qc_ready_summary_by_variable.csv`: resumo por variavel.