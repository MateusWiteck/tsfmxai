# Workers do DataLoader na baseline Chronos-2 / LMC-Synth

## Registro e proveniência

Executado em 16/11/2026, das 20h21 às 20h27 (America/Sao_Paulo), na VM
`vm02`, com RTX 4070 de 12 GB, 12 CPUs lógicas, Python 3.11.18 e PyTorch
2.6.0+cu124. O processo integral foi pausado após o checkpoint da atualização
12.500 e retomado com a configuração original (12 workers, microbatch 16) após
os três braços do benchmark. O treinamento integral não participou do benchmark.

- Repositório tsfmxai: `e702117e951d511b19b57f98816005adaea90cab`.
- Script executado: `scripts/benchmark_lmc_workers.py`, SHA-256
  `309e5d882143760c9d0a0f1a5373ec8a9e04e436b59194784a24be4e07b39313`.
- TimePFN: `e11d5c2982b0e49b87873cbddcb7712641861268`.
- Chronos: `10afa9ebe016e514f9d7dc1aa873f66af57e116b`;
  configuração `amazon/chronos-2` na revisão
  `29ec3766d36d6f73f0696f85560a422f50e8498c`, pesos inicializados do zero.
- Seeds: 2021 para o modelo, dados e ordem das janelas. A identidade de cada
  trajetória/janela deriva da política implementada em `TimePFNBatches`.
- Configuração completa, hashes dos geradores, dados por braço e medições brutas:
  [results.json](results.json).

## Desenho da comparação

Braços sequenciais, nesta ordem: **10, 12, 14 workers**. Em todos: seis canais,
contexto/horizonte 96/96, batch efetivo e microbatch de 16, mesmo modelo,
inicialização, otimizador e IDs de janelas.

1. **Independente com cache quente:** 10 atualizações de aquecimento e 500 medidas,
   a partir da posição 10.000 do mesmo cache de trajetórias. Mede a taxa
   sustentável quando o gerador não bloqueia o treino.
2. **LMC com cache frio:** 20 atualizações medidas desde a posição zero. Cada
   braço recebeu seu próprio cache vazio, para gerar as mesmas trajetórias sem
   herdar o trabalho do braço anterior. Cada braço gerou 316 trajetórias LMC
   distintas. Mede o início da etapa, não a taxa da etapa inteira.

O script mede `next_batch_seconds` entre pedir e receber o batch, e
`train_seconds` com sincronização CUDA após o optimizer. `nvidia-smi` foi
amostrado a cada segundo. Espera pelo batch é um diagnóstico do consumidor,
não uma medição direta de todos os ciclos ociosos da GPU. O cronômetro das
atualizações exclui criação dos workers, validação e gravação de checkpoints.

## Resultados medidos

| Workers | Independente: 500 updates | Espera/update | Treino/update | GPU média | VRAM máx. | LMC frio: 20 updates | Espera/update | GPU média |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 49,85 s | 0,044 ms | 99,65 ms | 99,0% | 3385 MiB | 45,93 s | 2,180 s | 3,5% |
| 12 | 49,56 s | 0,058 ms | 99,06 ms | 99,0% | 3397 MiB | 46,32 s | 2,202 s | 4,1% |
| 14 | 49,56 s | 0,047 ms | 99,06 ms | 99,0% | 3345 MiB | 47,79 s | 2,276 s | 3,5% |

O PyTorch avisou que 14 workers excedem os 12 CPUs lógicos sugeridos pela VM.
Nessa configuração o acesso por SSH também ficou lento durante a geração LMC.

## Projeção para o treino integral

No checkpoint usado para pausar o treino, faltavam 334.496 atualizações
independentes e 98.437 LMC. Para a primeira etapa, aplicamos a taxa medida
com o cache independente pronto. Para o cálculo da segunda, aplicamos o tempo
de `train_seconds` medido no braço LMC. Estimamos separadamente o custo único
de gerar 15.000 trajetórias LMC a partir do tempo gasto com 316 trajetórias,
descontando o cálculo das 20 atualizações. Depois da geração, as trajetórias
são reutilizadas em várias janelas. Essa extrapolação é um cenário, não uma
previsão calibrada de desempenho de 15.000 GPs.

| Workers | Independente restante | Cálculo LMC | Geração LMC projetada | Soma antes de validação/checkpoints |
|---:|---:|---:|---:|---:|
| 10 | 9,26 h | 3,18 h | 0,57 h | **13,02 h** |
| 12 | 9,21 h | 3,10 h | 0,58 h | **12,90 h** |
| 14 | 9,21 h | 3,09 h | 0,60 h | **12,90 h** |

Esses valores foram calculados sobre a posição 12.500 e servem para comparar
configurações. A execução integral continuou após o benchmark; sua previsão
operacional deve usar a posição atual. Se todos os custos permanecessem como
na pequena amostra, o término ficaria em torno da manhã de 17/11/2026 no
horário de São Paulo. Para planejamento, reservar **13 a 16 horas desde a
retomada** acomoda validações, checkpoints e incerteza na geração LMC. A
projeção `naive_cold_lmc_scenario_hours` no JSON prolonga o custo dos primeiros
batches frios por todas as 98.437 atualizações LMC; ela é deliberadamente
superestimada e não deve ser usada como ETA, pois ignora o cache.

## Síntese científica

Com o cache quente, 10, 12 e 14 workers tiveram taxas praticamente iguais:
a maior diferença entre as três medições de 500 updates foi **0,6%**. O treino
usa a GPU a ~99% e espera menos de 0,06 ms por batch. No começo frio da etapa
LMC, 10 workers foi o mais rápido e 14 o mais lento, mas uma única amostra de
20 atualizações por braço não estabelece significância estatística. Não há
evidência de benefício prático de 14 workers. A execução integral permanece em
12 workers para preservar o checkpoint e evitar outro reinício.

## Reprodução

Com a mesma revisão dos repositórios, ambiente GPU e cache independente já
preenchido, pause o treino integral em um checkpoint e execute, na raiz:

```bash
.venv/bin/python -u scripts/benchmark_lmc_workers.py \
  --shared-cache outputs/chronos2_lmc/lmc-full-20261116-190347/workers-2/cache \
  --output outputs/chronos2_lmc/benchmark-workers-10-12-14-20261116 \
  --independent-steps 500 --lmc-steps 20 \
  --independent-start 10000 --remaining-independent 334496
```

O `results.json` exportado dessa execução está nesta pasta. Uma repetição com
cache independente frio responderia outra pergunta e exigiria caches isolados
também na etapa independente.
