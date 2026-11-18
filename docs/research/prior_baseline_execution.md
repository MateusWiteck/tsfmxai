# Execução prospectiva das baselines de forecasting

Este runner treina o mesmo Chronos-2 com pesos aleatórios em dois braços: `lmc`
(TimePFN KernelSynth independente, depois LMC) e `own` (nosso gerador sem
dependências entre canais, depois com dependências). Nenhum braço usa a
*equation head* ou o encoder de canonicalization. Ambos usam 16 exemplos, seis
canais relacionados por exemplo, contexto 96 e horizonte 96. A configuração
`configs/experiments/prior_baseline_example.json` é um **exemplo**, não o
orçamento congelado do estudo. Copie-a, defina `time_hours` e
`controlled_updates` antes das execuções e distribua o mesmo arquivo às duas
máquinas. Guarde o hash SHA-256 desse arquivo.

O LMC requer o checkout TimePFN no commit
`e11d5c2982b0e49b87873cbddcb7712641861268`. O runner verifica o commit
antes de gerar qualquer batch. O Chronos usa a configuração de arquitetura
fixada em `chronos2_lmc.py`, sem carregar pesos. As dependências Python de
ambos os pipelines precisam estar instaladas no mesmo ambiente da GPU.

## Estudo A: exposição controlada

Faça estes dois comandos em cada máquina, mudando `ARM` para `lmc` ou `own`.
O corpus é materializado antes de ligar o relógio de treino. Os dois braços
usam o mesmo número de batches e updates, dividido entre os dois estágios pela
mesma fração. A divisão padrão do exemplo reproduz a proporção de updates
346996:98437 do loader LMC, mas o número total é decisão local. No nosso braço,
cada batch usa trajetórias novas; no LMC, as janelas são retiradas do banco
finito. Assim, o estudo iguala **exemplos expostos e updates**, não o número
de trajetórias únicas nem a regra de escolha das janelas. O corpus ocupa
aproximadamente 73,7 kB por update por braço, mais o banco bruto LMC; o runner
recusa a preparação se não houver espaço estimado com margem de 20%. O scheduler
OneCycle por estágio usa a fração de updates completada.

```bash
ARM=lmc   # na outra máquina: ARM=own
CONFIG=configs/experiments/prior_baseline_frozen.json
UPSTREAM=experiments_local/TimePFN
python scripts/run_prior_baseline.py prepare --config "$CONFIG" --arm "$ARM" \
  --upstream "$UPSTREAM" --corpus "outputs/prior_a_${ARM}_corpus"
python scripts/run_prior_baseline.py train --config "$CONFIG" --study controlled \
  --arm "$ARM" --upstream "$UPSTREAM" \
  --corpus "outputs/prior_a_${ARM}_corpus" --output "outputs/prior_a_${ARM}"
```

Para `own`, o parâmetro `--upstream` é ignorado. Não reutilize o mesmo diretório
de saída. O manifest do corpus é verificado contra a configuração do treino.

## Estudo B: orçamento de horas, cache frio

Execute uma vez em cada máquina, com diretórios vazios e exclusivos. O relógio
começa após carregar o modelo, antes da primeira criação de loader, e inclui
geração/cache sob demanda, espera por batch, updates, validação e checkpoint.
Pausas entre processos não consomem o orçamento; o tempo ativo é restaurado
de `latest.pt` com `--resume`. Atualizações registradas depois do último
checkpoint são refeitas; o tempo ativo registrado nelas continua cobrado,
mas trabalho entre a última linha de telemetria e uma queda abrupta não pode
ser recuperado. O scheduler de cada estágio usa a fração do
tempo prevista para o estágio. Ambas as máquinas precisam ter GPU, CPU, RAM,
versões de software e política de concorrência equivalentes. `time_hours` é o
mesmo nos dois braços; uma execução antiga com cache aquecido não estabelece
esse número. A fração provisória 0,779 vem da proporção histórica de
**updates** LMC; aplicá-la ao tempo é uma escolha local, igual nos dois braços.
O LMC volta ao início da sequência de janelas quando esgota uma passagem pelo
banco. Se um braço alcançar `max_updates_per_stage` antes do limite de tempo,
o runner gera erro: o protocolo precisa ser revisto antes de repetir ambos.

```bash
ARM=lmc   # na outra máquina: ARM=own
CONFIG=configs/experiments/prior_baseline_frozen.json
UPSTREAM=experiments_local/TimePFN
python scripts/run_prior_baseline.py train --config "$CONFIG" --study time \
  --arm "$ARM" --upstream "$UPSTREAM" --output "outputs/prior_b_${ARM}"
```

Se interromper, repita o último comando acrescentando `--resume`. Use um
diretório novo para cada tentativa completa; o runner recusa saída preexistente
sem `--resume`. Se o limite de segurança de updates for atingido antes do
relógio, o runner gera erro e a execução não constitui a comparação por horas.
Aumente o limite em um *novo protocolo* antes de reiniciar ambos os braços;
não altere a configuração no meio de uma run.

Em cada saída, `telemetry.jsonl` contém loss, LR, duração e espera por batch a
cada update; `validation.jsonl` traz a validação sintética disjunta;
`independent_end/` e `model_final/` guardam os pesos nos dois pontos; `latest.pt`
guarda o estado necessário para retomada; `latest_clock.json` guarda o tempo
medido após a escrita do checkpoint; e `result.json` registra o motivo de
parada, tempo ativo, updates, versões, hashes do código e informação da GPU.
A telemetria registra amostras de utilização da GPU a cada 100 updates e o
pico de memória alocada no resultado. A espera por batch resumida refere-se
aos updates preservados no checkpoint; `resume_events.jsonl` registra a espera
dos updates refeitos depois de uma queda. O checkpoint final é o ponto primário
para a avaliação real zero-shot. A validação sintética é diagnóstico e não
seleciona o checkpoint.

O estudo A responde o efeito de trocar a distribuição sob exposição igual. O
estudo B responde o desempenho do pipeline completo sob o mesmo tempo de
máquina. Não misture os resultados das duas perguntas em uma única conclusão.

## Avaliação exploratória nos dados reais

Depois de copiar os dois diretórios `model_final` para uma máquina de avaliação,
execute ambos sobre as mesmas origens de 2019. Informe o TSV da PANGAEA
940276 e o ZIP do EnDAT Zenodo 7478165:

```bash
python scripts/evaluate_prior_baselines.py \
  --air outputs/real_world_evaluation/ibirapuera.tsv \
  --energy outputs/real_world_evaluation/brazil_power.zip \
  --lmc outputs/prior_b_lmc/model_final \
  --own outputs/prior_b_own/model_final \
  --output outputs/prior_b_evaluation.json
```

O resultado inclui MAE e WAPE por canal, MAE relativo ao seasonal naive de
24 horas, média geométrica com peso igual para os dois domínios e intervalo
pareado por blocos de sete origens diárias. O script não agrega MAE bruto de
unidades diferentes. O período 2019 continua **exploratório**; ele não vira
confirmação só porque o cálculo foi automatizado. Os caminhos de dados acima
são exemplos: use os arquivos reais baixados pelo notebook 09 ou informe seus
caminhos locais.
