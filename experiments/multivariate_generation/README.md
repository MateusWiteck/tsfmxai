# Experimentos do gerador multisséries

Esta pasta reúne **ensaios sobre os dados que o gerador produz e sobre o custo
de produzi-los**. Ela não contém os treinamentos do Chronos-2 nem a comparação
final de previsão; esses ficam em
[`../generator_model_evaluation/`](../generator_model_evaluation/).

Os números indicam a ordem dos ensaios originais. Mantemos aqui apenas os
registros usados como evidência no relatório. Os ensaios exploratórios `01`,
`02`, `03` e `06` foram arquivados em `experiments_local/multivariate_generation/`,
que não é enviado ao GitHub. Os tamanhos de janela e as misturas de métodos
variaram durante o desenvolvimento; não leia os resultados antigos como
medidas da configuração atual de treino.

| Pasta | Pergunta que ela responde | Onde abrir primeiro |
| --- | --- | --- |
| [`04_terminal_sampling_correction`](04_terminal_sampling_correction/) | O que mudou ao sortear terminais durante a construção da árvore, em vez de substituir folhas depois? | `README.md`; o notebook `paired.ipynb` registra sua própria comparação |
| [`05_terminal_sampling_repeated`](05_terminal_sampling_repeated/) | O efeito da regra corrigida sobre descartes e tempo se repetia em três sementes? | `README.md` e `record.json`; `paired.ipynb` contém outra análise |
| [`07_ast_batch_96_96`](07_ast_batch_96_96/) | Quanto tempo a CPU levava para entregar um batch AST no formato 16 exemplos × 6 séries × 96 passos de contexto + 96 de futuro? | `record.json` |

**Para a comparação atual entre geradores**, comece pelo desenho em
[`../../report/english/chapters/experimental_design/foundation_prior_evaluation.tex`](../../report/english/chapters/experimental_design/foundation_prior_evaluation.tex).
O ensaio `07` mede apenas entrega de dados na CPU. Ele não mede atualização do
Chronos, espera efetiva da GPU nem a nova mistura AST/KernelSynth. Sua
configuração de árvores também antecede a configuração atual do braço de
treinamento; por isso, seus tempos não são uma previsão direta da duração das
próximas runs.
