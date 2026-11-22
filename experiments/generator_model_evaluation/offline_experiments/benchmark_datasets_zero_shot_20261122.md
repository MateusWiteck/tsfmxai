# Zero-shot em ETTh1, ETTh2 e Weather

## Pergunta e hipótese

Os dois sistemas de 2019 (Ibirapuera e Bahia) têm pouca dependência dinâmica
entre canais, e foram usados para escolher a mistura do gerador. Esta avaliação
pergunta se a vantagem do nosso gerador sobre o LMC-Synth se mantém em três
datasets multivariados de uso comum, **nunca inspecionados antes** neste
projeto: ETTh1 e ETTh2, com 7 canais cada (cargas de um transformador e
temperatura do óleo), e Weather, com 21 variáveis meteorológicas, um número de
canais acima dos 6 vistos no treino. Hipótese: os checkpoints preliminares (P25
e P50) têm MAE menor que o LMC Estudo A, que recebeu a mesma exposição. É um
experimento por script, não um notebook executado.

## Dados e proveniência

- **ETTh1 e ETTh2** ([`zhou2021informer`](../../../references/library.bib)): `ETT-small/ETTh1.csv` e `ETTh2.csv` do repositório `zhouhaoyi/ETDataset`, ramo `main`, baixados em 2026-11-22. SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` e `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. Dados horários de 2016-07-01 a 2018-06-26, sem faltantes. A fonte não informa fuso; os horários são rotulados como UTC sem conversão.
- **Weather** ([`wu2021autoformer`](../../../references/library.bib), [`bgcjena2020weather`](../../../references/library.bib)): os arquivos originais da estação do telhado do Max Planck de Jena, `mpi_roof_2020a.zip` e `mpi_roof_2020b.zip`, de `https://www.bgc-jena.mpg.de/wetter/data/`. SHA-256 `6a0be0552a2a26a04c3b41a5281d4a55ae251d4c65ce630971a8662f43b31f9f` e `c37eca910ee2803b17b8c587f3194bc7625abf994576c50f32538dd7cb65ac77`. São as 21 variáveis do benchmark Weather, com CO2 no lugar de `OT`.
  - A cada 10 minutos, o sentinela −9999 vira faltante.
  - A série vira **média horária** das seis leituras que terminam em cada hora. Isso mantém o mesmo protocolo dos outros datasets: 96 h de contexto, 96 h de horizonte e seasonal naive de 24 h.
  - Por isso, os números não são comparáveis aos do benchmark a cada 10 minutos.
- Arquivos locais ignorados pelo Git, em `outputs/real_world_evaluation/`.

## Protocolo

- **Período de teste:** origens apenas dentro do teste cronológico usual de cada dataset.
  - ETTh: horas 11.520 a 14.399, ou seja, 12 meses de treino, 4 de validação e 4 de teste, com meses de 30 dias.
  - Weather: os últimos 20% da grade horária.
- **Origens:** uma por dia, com pelo menos 80% de valores observados no contexto e no alvo. O contexto pode vir de antes do período de teste. Foram usadas todas as origens elegíveis: 117 no ETTh1, 117 no ETTh2 e 70 no Weather.
- **Previsão:** zero-shot, sem ajuste aos dados, com todos os canais de cada janela no mesmo grupo.
- **Métrica:** em cada dataset, a média geométrica, entre os canais, do MAE dividido pelo MAE do seasonal naive de 24 h. Os datasets não são agregados entre si.
- **Checkpoints:** `model_final` do LMC Estudo A e do LMC Estudo B (ver os registros `prior_lmc_study_a_20261118.md` e `../online_techniques/prior_lmc_study_b_20261118.md`), e de P25 e P50 (`preliminary_hybrid_20261121.md`).
- **Execução:** CPU local, revisão `6928815` mais o código desta avaliação, commitado junto com este registro. A saída fica em `outputs/prior_baseline/benchmark_datasets_zero_shot.json`, com SHA-256 `d1165237083ec4bc59876706347ef6091fd0d1cf61eb1904c2b5a088748e6900`. Ela traz MAE, RMSE e WAPE por canal, além do seasonal naive e do persistence.

## Resultados

MAE relativo ao seasonal naive (abaixo de 1 é melhor) e, entre parênteses, os
canais em que o modelo bate o seasonal naive:

| Modelo | ETTh1 (7 canais) | ETTh2 (7) | Weather (21) |
|---|---:|---:|---:|
| Persistence | 1,3210 (1/7) | 1,0659 (0/7) | 1,0312 (13/21) |
| LMC Estudo A | 1,0801 (2/7) | 1,0398 (1/7) | 1,0408 (15/21) |
| LMC Estudo B | 1,0246 (4/7) | 0,9852 (4/7) | 1,0222 (14/21) |
| P25 (nosso) | 0,9839 (5/7) | **0,9495** (6/7) | 0,9668 (16/21) |
| P50 (nosso, selecionado) | **0,9752** (5/7) | 0,9522 (6/7) | **0,9645** (18/21) |

Variação do MAE contra o LMC Estudo A, que recebeu a mesma exposição (média
geométrica das razões por canal):

| | ETTh1 | ETTh2 | Weather |
|---|---:|---:|---:|
| P50 | −9,7% (melhor em 7/7 canais) | −8,4% (6/7) | −7,3% (21/21) |
| P25 | −8,9% (7/7) | −8,7% (6/7) | −7,1% (21/21) |

## Interpretação

O padrão do conjunto de 2019 se repete em três datasets que não participaram da
escolha. Nos três, os dois braços do nosso gerador têm MAE de 7% a 10% menor que
o LMC com a mesma exposição e ficam abaixo do seasonal naive. O LMC fica acima
dele, exceto no Estudo B no ETTh2. No Weather, com 21 canais, mais de três vezes
os 6 do treino, o modelo funciona e a vantagem aparece em todos os canais. P25 e
P50 ficam praticamente empatados; a escolha de P50 não depende destes dados.

**Limites**
- Uma única seed e nenhum intervalo de confiança.
- O ETTh tem dois transformadores com o mesmo período de teste, logo seus resultados não são independentes.
- No Weather, a direção do vento é um ângulo e o MAE a trata como número linear.
- A radiação (SWDR e PAR) fica pior que o seasonal naive em todos os modelos.
- Isto ainda não substitui os contrastes pré-especificados dos Estudos A e B.

## Reprodução

Baixe os quatro arquivos acima para `outputs/real_world_evaluation/` e rode, a
partir da raiz:

```bash
python scripts/evaluate_benchmark_datasets.py --data-dir outputs/real_world_evaluation \
  --model lmc_a=outputs/prior_baseline/studyA_lmc_seed2021_20261118_1832/train/model_final \
  --model lmc_b=outputs/prior_baseline/studyB_lmc_seed2021_20261118/train/model_final \
  --model p25=outputs/prior_baseline/preliminary_p25_seed2021_20261121/train/model_final \
  --model p50=outputs/prior_baseline/preliminary_p50_seed2021_20261121/train/model_final \
  --output outputs/prior_baseline/benchmark_datasets_zero_shot.json --device cpu
```
