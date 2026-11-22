# Preliminar do gerador híbrido: KernelSynth 25% e 50%

## Pergunta e hipótese

Qual fração de canais KernelSynth usar no nosso gerador nos Estudos A e B?
Treinam-se dois Chronos-2 do zero com exposição idêntica à do LMC Estudo A:
um com 25% de canais KernelSynth (P25) e outro com 50% (P50); os demais canais
recebem equações autorregressivas amostradas. A escolha será feita por
`scripts/select_preliminary_hybrid.py` nas origens de 2019. **Essa seleção
ainda não foi executada**: este registro guarda apenas o treino e o benchmark de
vazão. Trata-se de um experimento por script, não de um notebook executado.

## Configuração e proveniência

- Configurações: [`configs/experiments/preliminary_hybrid_p25.json`](../../../configs/experiments/preliminary_hybrid_p25.json), SHA-256 `2c7e53fea6d6838d472f0bab6df3892ca776cdb7ada9568635f269f3dff62b8c`, e [`configs/experiments/preliminary_hybrid_p50.json`](../../../configs/experiments/preliminary_hybrid_p50.json), SHA-256 `57b080f31979dae2ab5feb832b54b2062a0ac86f19e843498119a66f253e1a8d`. Os hashes coincidem com os gravados em `launch_info.txt` nas duas VMs.
- Seed global `2021`; batch de 16 sistemas, seis canais, contexto 96, horizonte 96. Chronos-2 sem modificações (patch 16, sem índice de canal e sem cabeça de equação), pesos aleatórios, 119.477.664 parâmetros. LR 5e-4 no estágio independente e 1e-4 no correlacionado.
- Exposição controlada: 445.433 updates, sendo 346.996 independentes e 98.437 correlacionados, igual ao LMC Estudo A. Corpus materializado antes do treino.
- Gerador: de 1 a 6 operadores, lag máximo 16, grau até 4, 192 passos gerados, ruído entre 0,01 e 0,1 e magnitude dos parâmetros em [0,05; 1]. Dependência aleatória; canal sem fonte externa recebe Jacobian ou Heaps (50/50), e canal com fonte externa recebe Shao–Wu.
- Código: a preparação começou no commit `ed9f1d28790f4bb4cb1c8a6fa65846891dc18c29`. Ela foi interrompida e retomada, com o mesmo protocolo, no commit `86b68448deaf71110d64d9367df3d8e21134cd1f`, que gera um único thread por worker e compila as equações (com saída bit a bit idêntica). O treino rodou inteiro em `86b6844`. Chronos commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`. Python 3.11.18, PyTorch 2.6.0+cu124.
- P25 na VM02: RTX 4070, driver 550.144.03, i5-12400F com 12 threads, 12 workers. P50 na VM03: RTX 4070, driver 560.35.03, i5-13400F com 16 threads.
- Ressalva da retomada: os lotes salvos antes da interrupção (206.817 no P25, 161.402 no P50, todos do estágio independente) foram gerados com threads BLAS no padrão da biblioteca. Por isso seus canais KernelSynth diferem nos últimos bits de uma geração com um thread só. O protocolo e as seeds são os mesmos; `corpus/manifest.json` registra a retomada.
- Artefatos locais ignorados pelo Git: `outputs/prior_baseline/preliminary_p25_seed2021_20261121/`, `outputs/prior_baseline/preliminary_p50_seed2021_20261121/`, `outputs/prior_baseline/throughput_bench_VM02/` e `outputs/prior_baseline/throughput_bench_vm03/`. Os arquivos de cada pasta listados em `*.SHA256SUMS.remote` foram conferidos localmente (24, 24, 23 e 21 arquivos), sem nenhuma falha. Os corpora de cerca de 33 GB e os `latest.pt` ficaram nas VMs.

## Resultados observados

| Medida | P25 | P50 |
|---|---:|---:|
| Parada | `data_exhausted` | `data_exhausted` |
| Fração realizada de canais KernelSynth | 0,25002 | 0,49996 |
| Preparação do corpus, total | 28.186 s (7,83 h) | 28.695 s (7,97 h) |
| antes da retomada / depois | 20.761 s / 7.425 s | 20.796 s / 7.899 s |
| Treino | 45.688,7 s (12,69 h) | 46.675,9 s (12,97 h) |
| Espera da GPU por batches | 26,3 s (0,058%) | 23,6 s (0,051%) |
| Janelas expostas | 7.126.928 | 7.126.928 |
| Pico de VRAM alocada | 2,80 GiB | 2,80 GiB |
| Última validação independente (update 346.996): loss / MSE / MAE | 7,8912 / 0,1743 / 0,1786 | 7,1244 / 0,2932 / 0,2674 |
| Última validação correlacionada (update 445.000): loss / MSE / MAE | 7,9506 / 0,1330 / 0,1544 | 7,0039 / 0,2455 / 0,2427 |

As validações sintéticas usam 64 exemplos do próprio gerador de cada braço.
Elas **não são comparáveis entre P25 e P50**, nem com o LMC, porque as
distribuições diferem. Também não foram medidas no checkpoint final
(update 445.433). Cada braço tem 179 registros em `train/validation.jsonl`.

SHA-256 do `model_final/model.safetensors`:
- P25: `deb08b0d45239d46ac3eadac4668f11781aa9efe6efed30af56172a26898baaa`
- P50: `3404e825113f7656237a188b9c0e2e67e4432ed08f87ff0788c85ce40c3c70e8`

## Benchmark de vazão para as próximas runs

O benchmark rodou automaticamente depois dos treinos, com
`experiments_local/throughput_bench/bench_next_runs.sh`, em cada VM. Ele mede
a geração do corpus sem gravar em disco (2.400 batches), o custo de GPU por
update com batches aleatórios e um estudo de tempo de 15 minutos com geração
online.

| Medida | VM02 | VM03 |
|---|---:|---:|
| Geração P25 / P50, melhor número de workers (batches/s) | 33,6 / 31,6 (12 workers) | 34,7 / 31,0 (16 workers) |
| Corpus completo projetado, P25 / P50 | 3,7 h / 3,9 h | 3,6 h / 4,0 h |
| s/update, Chronos-2 sem modificação / com índice de canal | 0,0987 / 0,0989 | 0,0985 / 0,0988 |
| Online em 15 min, P25 / P50 (updates/h) | 32.526 / 32.864 | 31.779 / 32.032 |
| Espera da GPU online, P25 / P50 | 1,5% / 1,4% | 1,5% / 1,4% |

O tempo de sistema da CPU na geração caiu para 1%; antes da correção de threads,
ficava entre 57% e 60%. Com o orçamento de 10 h, a geração online projeta entre
318 mil e 329 mil updates, de 106% a 110% dos 299.991 do LMC Estudo B. As
projeções completas estão em `SUMMARY.md` e `summary.json` de cada pasta.

## Interpretação

Os dois braços terminaram o orçamento controlado sem gargalo de dados: a espera
pela GPU ficou abaixo de 0,06%. Com o código novo, um corpus completo sai em
cerca de 4 h, contra cerca de 1,6 h no LMC. Nenhuma conclusão sobre a mistura
é tirada aqui; ela depende da avaliação de 2019.

## Reprodução

Na revisão `86b6844`, a partir da raiz do repositório (troque `p25` por `p50`
para o outro braço):

```bash
python scripts/run_prior_baseline.py prepare \
  --config configs/experiments/preliminary_hybrid_p25.json --arm own \
  --corpus outputs/preliminary_p25/corpus
python scripts/run_prior_baseline.py train \
  --config configs/experiments/preliminary_hybrid_p25.json \
  --study controlled --arm own \
  --corpus outputs/preliminary_p25/corpus --output outputs/preliminary_p25/train
```

Uma preparação interrompida continua com `prepare ... --resume`.

## Modelos publicados

Os modelos estão em repositórios privados do Hugging Face. Os pesos finais
ficam na raiz e `independent_end/` traz o checkpoint do fim do primeiro
estágio; acompanham `result.json`, `validation.jsonl`, `corpus_manifest.json` e
a configuração.

- [Chronos-2, preliminar P25](https://huggingface.co/mateuswiteck/tsfmxai-chronos2-preliminary-p25-seed2021), commit Hugging Face `95d65f35ede3eb1e597f3929b5f5028b61e623d2`.
- [Chronos-2, preliminar P50](https://huggingface.co/mateuswiteck/tsfmxai-chronos2-preliminary-p50-seed2021), commit Hugging Face `63278d3cf025e81c055649a4cf729b2054365300`.
