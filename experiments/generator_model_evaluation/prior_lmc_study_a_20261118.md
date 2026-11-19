# Estudo A: Chronos-2 treinado com LMC sob exposição controlada

## Pergunta e hipótese

Esta execução fixa a referência LMC para comparar, futuramente, a qualidade da
previsão com a do nosso gerador após o mesmo número de exemplos e updates. A
hipótese comparativa ainda não pode ser julgada: o braço pareado não foi
executado. Este registro descreve um experimento de treino por script, não um
notebook executado.

## Configuração e proveniência

- Configuração completa: [`configs/experiments/prior_baseline_frozen.json`](../../configs/experiments/prior_baseline_frozen.json), SHA-256 `b18e5a493466307a33c516a9dd75fe00f7685b7572878b292ea9eda1912f60e4`.
- Seed global `2021`; o runner deriva seeds de dados por estágio, split, trajetória e janela. Batch de 16 sistemas multivariados, seis canais por sistema, contexto 96 e horizonte 96. Chronos-2 iniciado com pesos aleatórios, 119.477.664 parâmetros, sem equation head.
- LMC em dois estágios: KernelSynth independente, depois LMC correlacionado. Banco configurado com 20.000 trajetórias independentes e 15.000 correlacionadas; fração de updates `0.7790082908091677`. Total de 445.433 updates, dividido em 346.996 e 98.437. O corpus foi materializado antes do treino.
- `tsfmxai` commit `1add431872e9a17316fe5ba75836ae35ead2da2b`; TimePFN `e11d5c2982b0e49b87873cbddcb7712641861268`; Chronos `10afa9ebe016e514f9d7dc1aa873f66af57e116b`.
- Máquina VM02: RTX 4070 (12 GB), driver 550.144.03, i5-12400F, 32 GB RAM; Python 3.11.18 e PyTorch 2.6.0+cu124. Doze workers, microbatch 16. Os hashes dos dois arquivos do gerador TimePFN e do código usado constam em `train/result.json`.
- Artefatos locais ignorados pelo Git: `outputs/prior_baseline/studyA_lmc_seed2021_20261118_1832/`. Os 463 arquivos listados em `SHA256SUMS.remote` foram recalculados localmente: 463 correspondências, zero falhas. O corpus de aproximadamente 33 GB permanece na VM e não está incluído no pacote local.

## Resultados observados

| Medida | Valor |
|---|---:|
| Parada | `data_exhausted` |
| Updates independente / LMC | 346.996 / 98.437 |
| Janelas expostas | 7.126.928 |
| Preparação do corpus, fora do relógio de treino | 5.886,4 s (1,64 h) |
| Treino | 45.704,2 s (12,70 h) |
| Espera observada por batches | 24,5 s (0,054% do treino) |
| Trajetórias brutas em cache, independente / LMC | 19.998 / 15.000 |
| Pico de VRAM alocada | 3.006.546.432 bytes (2,80 GiB) |

A última **validação sintética** do estágio independente ocorreu no update
346.996: loss 6,9405, MSE 4,0176, MAE 0,7078, em 64 exemplos. A última
validação sintética do estágio correlacionado ocorreu no update **445.000**:
loss **7,2990**, MSE **3,5314**, MAE **0,6239**, em 64 exemplos. O checkpoint
final foi salvo no update **445.433**, portanto esses números **não são
métricas medidas no checkpoint final**. Há 179 registros de validação, em
`train/validation.jsonl`; nenhuma validação sintética de teste separada foi
executada por este runner.

## Avaliação zero-shot exploratória de 2019

O checkpoint final foi avaliado localmente com
`scripts/evaluate_lmc_checkpoints.py`, nos mesmos 255 pontos de origem diários
elegíveis do Ibirapuera e 357 da Bahia, com 96 horas de contexto e 96 de
previsão. Não houve ajuste nos dados reais. Os anos de 2019 já haviam sido
inspecionados durante o desenvolvimento; estes números **não são teste
confirmatório**. A média geométrica hierárquica do MAE relativo ao seasonal
naive de 24 h foi **1,1228** (menor é melhor); o modelo venceu esse controle
em **4 dos 9 canais**. MAE por canal: Ibirapuera O3 13,1562 ppbv, NO
10,7423 ppbv, NO2 9,0093 ppbv, CO 0,2279 ppmv, PM2.5 8,3646 µg/m³;
Bahia demanda 255,6400 MW, vento 0,0961, solar 0,0356 e afluência
135,3384 MWh. WAPE por canal está nas tabelas do relatório. MAEs de unidades
diferentes não foram somados.

Reprodução local dessa avaliação, junto ao checkpoint do Estudo B:

```bash
python scripts/evaluate_lmc_checkpoints.py \
  --air outputs/real_world_evaluation/ibirapuera.tsv \
  --energy outputs/real_world_evaluation/brazil_power.zip \
  --study-a outputs/prior_baseline/studyA_lmc_seed2021_20261118_1832/train/model_final \
  --study-b outputs/prior_baseline/studyB_lmc_seed2021_20261118/train/model_final \
  --output outputs/prior_baseline/lmc_real_2019_exploratory.json
```

## Interpretação e reprodução

O braço percorreu uma vez os batches completos do banco LMC. A melhora da
validação correlacionada desde sua primeira medição (MAE 0,6735 no update
347.500) até a última (0,6239) é apenas um diagnóstico sintético, não uma
estimativa de desempenho real nem uma comparação com nosso gerador. A
validação independente teve mínimo de loss antes do fim do estágio; o
checkpoint principal segue sendo `model_final/`, conforme a regra definida.

Na revisão e ambiente acima, a partir da raiz do repositório:

```bash
python scripts/run_prior_baseline.py prepare \
  --config configs/experiments/prior_baseline_frozen.json --arm lmc \
  --upstream experiments_local/TimePFN --corpus outputs/prior_a_lmc_corpus
python scripts/run_prior_baseline.py train \
  --config configs/experiments/prior_baseline_frozen.json \
  --study controlled --arm lmc --upstream experiments_local/TimePFN \
  --corpus outputs/prior_a_lmc_corpus --output outputs/prior_a_lmc
```

Modelo publicado: [Chronos-2 LMC, Estudo A](https://huggingface.co/mateuswiteck/tsfmxai-chronos2-lmc-study-a-seed2021), em repositório privado, commit Hugging Face `0a69a11be29e58b60960bb0873b5dcaa4ecf359f` (o upload inicial dos pesos foi `f1cd22cb58e12e8775d23ac6075176f873501483`). Na raiz ficam os pesos finais e `evaluation_2019_exploratory.json`; `independent_end/` contém o checkpoint de fim do primeiro estágio. O pacote publicado não inclui `latest.pt`, corpus ou telemetria completa; esses artefatos permanecem na pasta local ignorada.
