# Canonicalização: organização dos experimentos

Os notebooks seguem a ordem do raciocínio: primeiro valida-se o canonizador e o
codec, depois escolhe-se o sorteador de ASTs, e só então treina-se o
encoder–decoder que aproxima a canonização.

## Canonizador e codec

1. `01_reference.ipynb`: validação do canonizador e do codec (roundtrip, erro
   numérico, rejeições). É a referência de correção dos alvos.

## Análise do sorteador

Estudos de distribuição das ASTs e cobertura dos skeletons RAW/TANH:

2. `02_distribution.ipynb`: comparação das distribuições.
3. `03_exponential.ipynb`: prior exponencial do tamanho.
4. `04_random_probabilities.ipynb`: probabilidades de terminais.
5. `05_combinatorial.ipynb`: sorteio combinatório uniforme.

## Modelos de encoder–decoder

Modelo aprendido que transforma uma AST em sua representação canônica:

6. `06_train.ipynb`: treinamento no vocabulário RAW/TANH (corpus de 10 milhões de ASTs).
7. `07_refine.ipynb`: refinamento do treinamento.
8. `08_errors.ipynb`: avaliação dos erros do modelo.

Os modelos buscam aproximar a transformação determinística para uso em
inferência. O canonicalizador permanece como gerador de alvos e referência de
correção.

## Artefatos locais

`outputs/` é local e ignorado pelo Git. Contém a execução de treino
(`raw_tanh/`: `status.json`, `manifest.json`, `run.log` e checkpoints)
e o pacote para publicação (`release/canonical_ast/`, com o corpus em Parquet e os modelos em safetensors). Configurações, métricas e
reprodução de cada estudo permanecem nos notebooks.

## Treino em máquina externa e publicação

Na máquina de treino (precisa de GPU para ser viável):

```
git clone <URL-do-repositório> tsfmxai && cd tsfmxai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[learning,release,dev]" -c constraints.txt
python scripts/run_canonical_5m.py --device cuda --release-dir experiments/canonicalization/outputs/release/canonical_ast
```

O script gera o corpus (`parquet/`), remove skeletons duplicados de validação e teste,
treina as duas variantes e, ao final, grava `experiments/canonicalization/outputs/release/canonical_ast/` já no formato do
Hugging Face (`models/<variante>/` com `model.safetensors`, `config.json` com os
parâmetros do gerador, e `dataset/` com Parquet, `metadata.json` e cartões). Rodar de
novo o mesmo comando retoma shards e variantes já concluídos.

Na máquina com o token do Hugging Face:

```
python scripts/upload_to_huggingface.py --release experiments/canonicalization/outputs/release/canonical_ast \
    --models-repo USUARIO/NOME-DO-MODELO --dataset-repo USUARIO/NOME-DO-DATASET
```
