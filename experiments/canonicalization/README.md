# Canonicalizador de seis séries

Protocolo: `configs/canonicalization_10m_six_series.json`. São **10.000.000 de
árvores amostradas**, uma equação por exemplo. Cada equação pode referenciar
qualquer uma das seis séries, com lags de 1 a 16. O codec preserva série e lag;
seu vocabulário contém 197 tokens.

Mantemos 1–6 operadores, grau máximo 4, proteção RAW/TANH com probabilidade 0,5,
seeds explícitas, batch 256, até 10 épocas com patience 3, dimensão 96 e duas
camadas Transformer. Treinamos `transformer_skeleton` e `transformer_joint`
do zero. O split por skeleton é 70/10/20; a deduplicação de validação/teste
pode reduzir a quantidade final de linhas armazenadas. Resultados pendentes.

Os notebooks 01–05 documentam o canonicalizador determinístico e os estudos
do sorteador. O treino atual usa o script abaixo e guarda configuração,
proveniência, métricas, histórico e checkpoints numa pasta ignorada pelo Git.

## Treinar na máquina remota (Linux, Python 3.11)

Na raiz do repositório, na branch `canonicalization-10m`:

```bash
git pull --ff-only
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning,release,dev]'
python -c "import torch; assert torch.cuda.is_available(), 'Instale PyTorch com CUDA compatível com o driver'; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
git rev-parse HEAD
mkdir -p experiments/canonicalization/outputs/six_series
nohup python -u scripts/run_canonical.py --device cuda > experiments/canonicalization/outputs/six_series/run.log 2>&1 < /dev/null &
tail -f experiments/canonicalization/outputs/six_series/run.log
```

A configuração padrão contém 10 milhões de árvores. A geração usa quatro
processos CPU; o treino usa uma GPU. Os logs mostram taxa e ETA. `status.json`
registra o estado; `manifest.json` é escrito ao concluir. Reexecutar o mesmo
comando retoma shards, variantes concluídas e a última época salva. Uma época
interrompida é refeita. Não execute duas instâncias sobre a mesma pasta.

Se já existe um ambiente funcional com PyTorch CUDA, use-o. A checagem CUDA
deve passar antes de iniciar. As versões efetivamente usadas são registradas
no manifest; os constraints do Windows não são impostos ao ambiente Linux.

## Conferir e transferir

Após terminar, na máquina remota:

```bash
python -c "import json; s=json.load(open('experiments/canonicalization/outputs/six_series/status.json')); assert s['state']=='completed', s['state']; print('Treino concluído')"
tar -czf canonical_six_series.tar.gz -C experiments/canonicalization/outputs six_series
sha256sum canonical_six_series.tar.gz
```

Na raiz do repositório local (PowerShell), substitua usuário, host e caminho:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
scp usuario@host:/caminho/tsfmxai/canonical_six_series.tar.gz outputs/
Get-FileHash outputs/canonical_six_series.tar.gz -Algorithm SHA256
New-Item -ItemType Directory -Force experiments/canonicalization/outputs | Out-Null
tar -xzf outputs/canonical_six_series.tar.gz -C experiments/canonicalization/outputs
```

Compare os hashes antes de extrair. O arquivo inclui corpus, checkpoints,
estado do otimizador, métricas e logs. Parquet já é comprimido; gzip pode
reduzir pouco seu tamanho. O tamanho final será medido, não presumido.
Esses artefatos e o arquivo compactado continuam ignorados pelo Git.

## Registrar resultados e publicar depois

Após revisar os resultados, registrar a execução em um notebook executado
com uma célula de registro completo e fazer o commit desse registro.
Corpus e pesos são publicados no Hugging Face, não no Git. Para preparar
o pacote, use o SHA do treino registrado no manifest:

```bash
python scripts/package_canonical_release.py --training-commit SHA_DO_TREINO
```

O pacote fica em `experiments/canonicalization/outputs/release/six_series`.
O script verifica a proveniência e inclui codec, código, pesos e dados.
Publicar depois de revisar o pacote e escolher os IDs dos repositórios de
modelos e dataset, usando `scripts/upload_to_huggingface.py`.
Nenhum upload é feito durante o treino.
