# Canonicalização: organização dos experimentos

## Análise do sorteador

`sampler/` reúne os estudos atuais de distribuição das ASTs e cobertura
dos skeletons RAW/TANH, nesta ordem:

1. `01_distribution.ipynb`: comparação das distribuições.
2. `02_exponential.ipynb`: prior exponencial do tamanho.
3. `03_random_probabilities.ipynb`: probabilidades de terminais.
4. `04_combinatorial.ipynb`: sorteio combinatório uniforme.

Artefatos futuros dessa categoria pertencem a `sampler/outputs/`.

## Modelos de encoder–decoder

`models/` reúne a referência determinística e os estudos do modelo
aprendido que transforma uma AST em sua representação canônica:

1. `01_reference.ipynb`: validação do canonicalizador e codec.
2. `02_train.ipynb`: treinamento no vocabulário RAW/TANH.
3. `03_refine.ipynb`: refinamento do treinamento.
4. `04_errors.ipynb`: avaliação dos erros do modelo.

Os modelos buscam aproximar a transformação determinística para uso em
inferência. O canonicalizador permanece como gerador de alvos e referência de
correção. Artefatos dessas execuções pertencem a `models/outputs/`.

Todas as pastas `outputs/` desta estrutura são locais e ignoradas pelo Git.
Configurações, métricas e reprodução de cada estudo permanecem nos notebooks.

## Treino em máquina externa e publicação

Na máquina de treino (precisa de GPU para ser viável):

```
git clone <URL-do-repositório> tsfmxai && cd tsfmxai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[learning,release]"
python scripts/run_canonical_5m.py --device cuda --release-dir release/canonical_ast
```

O script gera o corpus (`parquet/`), remove skeletons duplicados de validação e teste,
treina as duas variantes e, ao final, grava `release/canonical_ast/` já no formato do
Hugging Face (`models/<variante>/` com `model.safetensors`, `config.json` com os
parâmetros do gerador, e `dataset/` com Parquet, `metadata.json` e cartões). Rodar de
novo o mesmo comando retoma shards e variantes já concluídos.

Na máquina com o token do Hugging Face:

```
python scripts/upload_to_huggingface.py --release release/canonical_ast \
    --models-repo USUARIO/NOME-DO-MODELO --dataset-repo USUARIO/NOME-DO-DATASET
```
