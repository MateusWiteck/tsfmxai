# Estudo B: Chronos-2 treinado com LMC sob 10 horas de máquina

## Pergunta e hipótese

Esta execução fixa a referência LMC para comparar, futuramente, a qualidade
da previsão entregue por cada pipeline após o mesmo tempo ativo de máquina.
O efeito relativo ao nosso gerador ainda não pode ser estimado: seu braço
pareado deverá rodar na mesma VM03, em sequência e com cache frio. Este é um
experimento de treino por script, não um notebook executado.

## Configuração e proveniência

- Configuração completa: [`configs/experiments/prior_baseline_frozen.json`](../../configs/experiments/prior_baseline_frozen.json), SHA-256 `b18e5a493466307a33c516a9dd75fe00f7685b7572878b292ea9eda1912f60e4`.
- Seed global `2021`, com seeds derivadas pelo runner; batch 16, seis canais, contexto 96, horizonte 96. Chronos-2 iniciado com pesos aleatórios, 119.477.664 parâmetros e sem equation head.
- Orçamento de **10 horas ativas**, incluindo geração/cache sob demanda, espera pelos dados, treino, validação e checkpoints. Estágios independente e correlacionado recebem, respectivamente, 77,9008% e 22,0992% do tempo. O banco LMC pode ser reutilizado, mas esta execução não repetiu janelas.
- `tsfmxai` commit `1add431872e9a17316fe5ba75836ae35ead2da2b`; TimePFN `e11d5c2982b0e49b87873cbddcb7712641861268`; Chronos `10afa9ebe016e514f9d7dc1aa873f66af57e116b`.
- VM03: RTX 4070 (12 GB), driver 560.35.03, i5-13400F, 64 GB RAM; Python 3.11.18, PyTorch 2.6.0+cu124, 12 workers, microbatch 16. A GPU ficou livre por 15 minutos antes do início; o registro `contention.log` foi coletado durante o treino. Os hashes dos geradores e do código constam em `train/result.json`.
- Artefatos locais ignorados pelo Git: `outputs/prior_baseline/studyB_lmc_seed2021_20261118/`. Os 463 arquivos listados em `SHA256SUMS.remote` foram recalculados localmente: 463 correspondências, zero falhas. O cache bruto de aproximadamente 568 MB permanece na VM e não foi copiado.

## Resultados observados

| Medida | Valor |
|---|---:|
| Parada | `budget_elapsed` |
| Updates independente / LMC | 241.898 / 58.093 |
| Updates totais | 299.991 |
| Janelas expostas | 4.799.856 |
| Tempo ativo | 36.012,3 s (10,003 h) |
| Espera observada por batches | 4.264,2 s (11,84% do tempo ativo) |
| Janelas LMC repetidas | 0 |
| Trajetórias brutas em cache, independente / LMC | 19.998 / 15.000 |
| Pico de VRAM alocada | 3.006.546.432 bytes (2,80 GiB) |

A última **validação sintética** do estágio independente ocorreu no update
241.898: loss 6,3194, MSE 4,2290, MAE 0,6876, em 64 exemplos. A última
validação sintética correlacionada ocorreu no update **297.500**: loss
**7,2337**, MSE **3,4763**, MAE **0,6215**, em 64 exemplos. O checkpoint
principal foi salvo no update **299.991**: as métricas acima **não foram
medidas nele**. Há 120 registros de validação em `train/validation.jsonl`.
Este runner não fez teste sintético separado.

## Avaliação zero-shot exploratória de 2019

O checkpoint final foi avaliado localmente com
`scripts/evaluate_lmc_checkpoints.py`, nas mesmas 255 origens diárias
elegíveis do Ibirapuera e 357 da Bahia, com 96 horas de contexto e 96 de
previsão. Sem ajuste em dados reais. O ano 2019 já foi inspecionado durante
o desenvolvimento; esta análise **não é teste confirmatório**. A média
geométrica do MAE relativo ao seasonal naive de 24 h foi **0,9584 no
Ibirapuera** (melhor em 4/5 canais) e **1,1800 na Bahia** (melhor em 2/4
canais). Um valor abaixo de 1 favorece o modelo; os conjuntos não são
agrupados em um único escore.
MAE por canal: Ibirapuera O3 12,7529 ppbv, NO 9,7218 ppbv, NO2 8,3857
ppbv, CO 0,2190 ppmv, PM2.5 7,9072 µg/m³; Bahia demanda 238,6866 MW,
vento 0,0976, solar 0,0361 e afluência 115,6647 MWh. WAPE por canal está
nas tabelas do relatório. MAEs de unidades diferentes não foram somados.

Reprodução local dessa avaliação, junto ao checkpoint do Estudo A:

```bash
python scripts/evaluate_lmc_checkpoints.py \
  --air outputs/real_world_evaluation/ibirapuera.tsv \
  --energy outputs/real_world_evaluation/brazil_power.zip \
  --study-a outputs/prior_baseline/studyA_lmc_seed2021_20261118_1832/train/model_final \
  --study-b outputs/prior_baseline/studyB_lmc_seed2021_20261118/train/model_final \
  --output outputs/prior_baseline/lmc_real_2019_exploratory.json
```

## Interpretação e reprodução

O tempo foi consumido dentro da política declarada; o treino não percorreu
todo o banco de janelas. Espera observada por batch é uma medida do consumidor
de dados, não uma medida direta de stall da GPU. As curvas de validação
diagnosticam o treino sintético; não estabelecem generalização nos datasets
reais. A comparação de eficiência do Estudo B só estará completa após o
treino do braço do nosso gerador na **mesma VM03**.

Na revisão e ambiente acima, com cache novo e GPU livre, a partir da raiz:

```bash
python scripts/run_prior_baseline.py train \
  --config configs/experiments/prior_baseline_frozen.json \
  --study time --arm lmc --upstream experiments_local/TimePFN \
  --output outputs/prior_b_lmc
```

Modelo publicado: [Chronos-2 LMC, Estudo B](https://huggingface.co/mateuswiteck/tsfmxai-chronos2-lmc-study-b-seed2021), em repositório privado, commit Hugging Face `106312370c233571b4ea50de66ae3a2705dc8948` (o upload inicial dos pesos foi `e3c70c73ab032ab03017ace843ee5a16d291cf7f`). Na raiz ficam os pesos finais e `evaluation_2019_exploratory.json`, com os dois datasets separados; `independent_end/` contém o checkpoint de fim do primeiro estágio. `latest.pt`, cache e telemetria completa permanecem apenas na pasta local ignorada.
