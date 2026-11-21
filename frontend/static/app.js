const $ = (id) => document.getElementById(id);

const PALETTE = [
  "#1f6b55", "#4488a3", "#d9692f", "#8b75a5", "#b8901a", "#c4526b",
  "#5b8c2a", "#2f5d9e", "#a0522d", "#2a9d8f", "#737b76", "#d1495b",
];
const SUBSCRIPT = "₀₁₂₃₄₅₆₇₈₉";
const channelName = (index) => `x${[...String(index)].map((digit) => SUBSCRIPT[digit]).join("")}`;
const channelColor = (index) => PALETTE[index % PALETTE.length];

const state = {
  defaults: null,
  system: null,
  data: null,
  activeChannel: 0,
  activeTrajectory: 0,
  playback: null,
  hoverTime: null,
  layout: null,
};

/* ------------------------------------------------------------------ *
 * Parameter schema. Every control, its description and its tooltip    *
 * live here; default values come from the server (/api/defaults).    *
 * ------------------------------------------------------------------ */

const DEPENDENCIES = [
  { value: "independent", label: "Independentes", tip: "Nenhum canal pode ler outro: cada equação usa apenas lags de si mesma." },
  { value: "linked_chain", label: "Cadeia ligada", tip: "Cada canal pode ler apenas o canal imediatamente anterior em uma ordem sorteada, formando uma cadeia." },
  { value: "previous_chain", label: "Cadeia acumulada (DAG)", tip: "Cada canal pode ler qualquer canal que veio antes na ordem sorteada. Não há ciclos." },
  { value: "random", label: "Aleatória", tip: "Qualquer outro canal pode ser fonte, inclusive formando ciclos com atraso." },
];

const INITIALIZATIONS = [
  { value: "random", label: "Famílias sorteadas (treino)", tip: "Para cada canal, sorteia quais famílias aparecem (constante, ruído branco, cosseno), sorteia os parâmetros de cada uma e soma. Sem nenhuma família, o histórico é zero. Cada nova tentativa sorteia tudo de novo." },
  { value: "uniform", label: "Uniforme", tip: "offset + amplitude × U(−1, 1), sorteado de forma independente por canal e por lag." },
  { value: "zero", label: "Zero", tip: "Histórico inicial igual a zero (mais o offset)." },
  { value: "constant", label: "Constante", tip: "Todos os valores iniciais iguais ao offset." },
  { value: "cosine", label: "Cosseno", tip: "Um ciclo de cosseno com o período indicado, igual em todos os canais. O prefixo precisa cobrir o maior lag." },
];

const GROUPS = [
  {
    scope: "s", step: "01", title: "Sistema multivariado", open: true,
    fields: [
      { key: "seed", label: "Seed do sistema", type: "seed", wide: true,
        tip: "Fixa a topologia, as árvores, os coeficientes e o desvio de ruído de cada canal. Mesma seed e mesma configuração reproduzem exatamente o mesmo sistema." },
      { key: "channels", label: "Canais", type: "number", min: 1, max: 12, step: 1,
        tip: "Número de séries do sistema. No treino, a contagem vem de channel_count_probs; aqui fixamos um valor para inspecionar." },
      { key: "max_external_sources", label: "Fontes externas máx.", type: "number", min: 0, max: 11, step: 1,
        tip: "Quantos outros canais, no máximo, uma equação tem permissão de ler. É permissão, não obrigação: a árvore sorteada pode não usar todas." },
      { key: "dependency", label: "Topologia de dependência", type: "select", wide: true, options: DEPENDENCIES,
        tip: "Regra que decide quais canais podem servir de fonte para cada equação. Uma ordem aleatória dos canais é sorteada para cadeias." },
      { key: "self_source_weight", label: "Peso do próprio canal", type: "number", min: 0.1, max: 20, step: 0.5,
        tip: "Peso relativo de sortear o próprio canal como fonte de cada lag, contra peso 1 para cada fonte externa. Valores altos deixam os canais mais autorregressivos." },
    ],
  },
  {
    scope: "s", step: "02", title: "Estrutura das equações", open: false,
    fields: [
      { key: "minimum_operators", label: "Operadores mín.", type: "number", min: 0, max: 12, step: 1,
        tip: "Menor número de operadores binários (+ e ×) da árvore de cada equação. Zero permite folhas isoladas." },
      { key: "maximum_operators", label: "Operadores máx.", type: "number", min: 0, max: 12, step: 1,
        tip: "Maior número de operadores binários da árvore. A estrutura é sorteada uniformemente entre todas as árvores dentro do intervalo." },
      { key: "minimum_lag", label: "Lag mínimo", type: "number", min: 1, max: 64, step: 1,
        tip: "Menor atraso que um termo pode consultar (1 = o passo imediatamente anterior)." },
      { key: "maximum_lag", label: "Lag máximo", type: "number", min: 1, max: 64, step: 1,
        tip: "Maior atraso que um termo pode consultar. Define o histórico inicial necessário. O treino atual usa 16." },
      { key: "max_degree", label: "Grau polinomial máx.", type: "number", min: 1, max: 8, step: 1, wide: true,
        tip: "Grau máximo do polinômio depois de expandir produtos, ou seja, quantos lags podem ser multiplicados entre si. Limita a não linearidade." },
      { key: "minimum_parameter_magnitude", label: "|Coeficiente| mín.", type: "number", min: 0.001, max: 5, step: 0.05,
        tip: "Menor valor absoluto de um coeficiente. O sinal é sorteado com igual probabilidade." },
      { key: "maximum_parameter_magnitude", label: "|Coeficiente| máx.", type: "number", min: 0.001, max: 5, step: 0.05,
        tip: "Maior valor absoluto de um coeficiente. Valores maiores aumentam a chance de explosão numérica." },
    ],
  },
  {
    scope: "s", step: "03", title: "Tratamento e ruído", open: false,
    fields: [
      { key: "kernel_synth_probability", label: "Probabilidade de canal KernelSynth", type: "range", min: 0, max: 0.95, step: 0.05, wide: true, percent: true,
        tip: "Cada canal é sorteado, independentemente, como série KernelSynth (processo gaussiano, sem equação) com esta probabilidade; os demais vêm de equações. Canais de equação podem ler canais KernelSynth como fonte externa." },
      { key: "jacobian_share", label: "Fração Jacobiano nos canais sem dependência", type: "range", min: 0, max: 1, step: 0.05, wide: true, percent: true,
        tip: "O tratamento segue a topologia: canal que pode ler outra série usa Shao–Wu (tanh nos produtos de lags e escala que limita o ganho) e lê ao menos uma série externa. Canal sem fonte externa usa Jacobiano com esta probabilidade e Heaps com o restante." },
      { key: "target_gain", label: "Ganho-alvo", type: "number", min: 0.05, max: 0.99, step: 0.05,
        tip: "Ganho desejado da equação tratada (entre 0 e 1). A escala aplicada é fator de segurança × min(1, ganho-alvo / g), em que g soma as cotas de derivada da equação." },
      { key: "safety_factor", label: "Fator de segurança", type: "number", min: 0.05, max: 0.99, step: 0.05,
        tip: "Fração extra de folga multiplicada à escala Shao–Wu. Mais perto de 1 significa menos redução." },
      { key: "noise_minimum", label: "Desvio do ruído mín.", type: "number", min: 0, max: 2, step: 0.01,
        tip: "Cada canal recebe um desvio padrão sorteado uniformemente entre o mínimo e o máximo. A inovação gaussiana é somada a cada passo e o desvio fica fixo para todas as realizações do canal." },
      { key: "noise_maximum", label: "Desvio do ruído máx.", type: "number", min: 0, max: 2, step: 0.01,
        tip: "Limite superior do desvio padrão sorteado para cada canal." },
    ],
  },
  {
    scope: "t", step: "04", title: "Trajetórias", open: true,
    fields: [
      { key: "seed", label: "Seed das trajetórias", type: "seed", wide: true,
        tip: "Fixa históricos iniciais e ruídos. Trocar a seed produz outras realizações do mesmo sistema." },
      { key: "trajectory_count", label: "Realizações", type: "number", min: 1, max: 64, step: 1,
        tip: "Quantas trajetórias independentes tentar gerar a partir do mesmo sistema. Realizações que explodem são descartadas, como no treino." },
      { key: "context_length", label: "Contexto", type: "number", min: 1, max: 1000, step: 1,
        tip: "Passos de entrada que o modelo vê em cada janela de treino." },
      { key: "prediction_length", label: "Horizonte", type: "number", min: 1, max: 1000, step: 1, wide: true,
        tip: "Passos futuros que o modelo deve prever. A trajetória gerada tem contexto + horizonte passos, sem burn-in." },
      { key: "initialization", label: "Inicialização do histórico", type: "select", wide: true, options: INITIALIZATIONS,
        tip: "Como é preenchido o histórico necessário para disparar a recorrência. Esses valores iniciais não entram na janela de treino." },
      { key: "constant_probability", label: "Chance: constante", type: "range", min: 0, max: 1, step: 0.05, wide: true, percent: true, family: true,
        tip: "Probabilidade de cada canal receber um nível constante somado ao histórico. O sinal é sorteado (+ ou −)." },
      { key: "constant_minimum", label: "|Constante| mín.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Menor magnitude do nível constante. O estudo de inicialização mostrou que até ±1 é seguro para todos os métodos." },
      { key: "constant_maximum", label: "|Constante| máx.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Maior magnitude do nível constante. Constantes maiores aumentaram um pouco a explosão da AST original (50% → 62% em ±16)." },
      { key: "white_noise_probability", label: "Chance: ruído branco", type: "range", min: 0, max: 1, step: 0.05, wide: true, percent: true, family: true,
        tip: "Probabilidade de cada canal receber ruído branco uniforme i.i.d., U(−r, r) em cada lag do histórico." },
      { key: "white_noise_minimum", label: "Raio do ruído mín.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Menor raio r do ruído branco do histórico. O raio de cada canal é sorteado uniformemente entre mín. e máx." },
      { key: "white_noise_maximum", label: "Raio do ruído máx.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Maior raio r do ruído branco do histórico. Não confundir com o ruído de inovação do sistema." },
      { key: "cosine_probability", label: "Chance: cosseno", type: "range", min: 0, max: 1, step: 0.05, wide: true, percent: true, family: true,
        tip: "Probabilidade de cada canal receber um cosseno A·cos(2πk/T + φ), com fase φ sorteada em [0, 2π)." },
      { key: "cosine_amplitude_minimum", label: "Amplitude cos. mín.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Menor amplitude A do cosseno." },
      { key: "cosine_amplitude_maximum", label: "Amplitude cos. máx.", type: "number", min: 0, max: 100, step: 0.1, family: true,
        tip: "Maior amplitude A do cosseno." },
      { key: "cosine_period_minimum", label: "Período cos. mín.", type: "number", min: 2, max: 512, step: 1, family: true,
        tip: "Menor período T (inteiro, em passos). O histórico tem só o maior lag em pontos (16 no treino): períodos bem maiores que isso parecem uma curva suave ou rampa." },
      { key: "cosine_period_maximum", label: "Período cos. máx.", type: "number", min: 2, max: 512, step: 1, family: true,
        tip: "Maior período T. O cosseno é calculado direto no comprimento do histórico, então não precisa cobrir o lag." },
      { key: "amplitude", label: "Amplitude", type: "number", min: 0, max: 100, step: 0.1,
        tip: "Escala do histórico inicial. Só vale para inicialização uniforme ou cosseno." },
      { key: "offset", label: "Offset", type: "number", min: -100, max: 100, step: 0.1,
        tip: "Valor somado a todo o histórico inicial." },
      { key: "period", label: "Período do cosseno", type: "number", min: 4, max: 512, step: 1,
        tip: "Número de passos de um ciclo completo do cosseno." },
      { key: "fraction", label: "Fração do ciclo", type: "select", options: [
          { value: "1", label: "1 (ciclo todo)", tip: "O histórico usa o ciclo inteiro." },
          { value: "0.5", label: "1/2", tip: "O histórico usa a primeira metade do ciclo." },
          { value: "0.25", label: "1/4", tip: "O histórico usa o primeiro quarto do ciclo." }],
        tip: "Parte inicial do ciclo usada como histórico. O prefixo resultante precisa cobrir o maior lag do sistema." },
      { key: "resample_innovations", label: "Sortear ruído a cada tentativa", type: "checkbox", wide: true,
        tip: "Por padrão, só o histórico inicial muda entre tentativas e o ruído se repete. Ligue para mudar os dois." },
    ],
  },
];

const ALL_FIELDS = GROUPS.flatMap((group) => group.fields.map((field) => ({ ...field, scope: group.scope })));
const fieldId = (field) => `${field.scope}-${field.key}`;
const findField = (scope, key) => ALL_FIELDS.find((field) => field.scope === scope && field.key === key);

const PRESETS = [
  { label: "Config. de treino (6 canais)", values: null,
    tip: "Os valores padrão usados pelo treinamento canônico de 6 séries." },
  { label: "Didático (3 canais)", tip: "Poucos canais, lags curtos e árvores pequenas: fácil de ler e conferir.",
    values: { s: { channels: 3, dependency: "previous_chain", minimum_operators: 1, maximum_operators: 3, maximum_lag: 4,
                   max_degree: 2 }, t: { context_length: 64, prediction_length: 16 } } },
  { label: "Cadeia ligada", tip: "Cada canal lê o anterior na ordem sorteada: todos, menos o primeiro, usam Shao–Wu.",
    values: { s: { dependency: "linked_chain" }, t: {} } },
  { label: "Univariado", tip: "Um único canal, sem dependências externas.",
    values: { s: { channels: 1, dependency: "independent" }, t: {} } },
];

/* ------------------------------------------------------------------ *
 * Small DOM helpers and tooltips                                      *
 * ------------------------------------------------------------------ */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function infoIcon(tip) {
  const icon = el("button", "info", "i");
  icon.type = "button";
  icon.dataset.tip = tip;
  icon.setAttribute("aria-label", tip);
  return icon;
}

const tooltip = $("tooltip");
let tooltipOwner = null;

function showTooltip(owner) {
  tooltipOwner = owner;
  tooltip.textContent = owner.dataset.tip;
  tooltip.hidden = false;
  const box = owner.getBoundingClientRect();
  const width = tooltip.offsetWidth;
  const height = tooltip.offsetHeight;
  const left = Math.max(8, Math.min(window.innerWidth - width - 8, box.left + box.width / 2 - width / 2));
  const above = box.top - height - 8 >= 8;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${above ? box.top - height - 8 : box.bottom + 8}px`;
}

function hideTooltip() {
  tooltipOwner = null;
  tooltip.hidden = true;
}

function setupTooltips() {
  document.querySelectorAll(".info[data-tip]:not(button)").forEach((icon) => {
    icon.tabIndex = 0;
    icon.setAttribute("role", "note");
    icon.setAttribute("aria-label", icon.dataset.tip);
  });
  document.addEventListener("mouseover", (event) => {
    const owner = event.target.closest?.("[data-tip]");
    if (owner) showTooltip(owner);
  });
  document.addEventListener("mouseout", (event) => {
    if (event.target.closest?.("[data-tip]")) hideTooltip();
  });
  document.addEventListener("focusin", (event) => {
    const owner = event.target.closest?.("[data-tip]");
    if (owner) showTooltip(owner);
  });
  document.addEventListener("focusout", hideTooltip);
  // Icons live inside labels and summaries; a click must neither toggle nor focus those.
  document.addEventListener("click", (event) => {
    const icon = event.target.closest?.(".info");
    if (!icon) return;
    event.preventDefault();
    tooltipOwner === icon ? hideTooltip() : showTooltip(icon);
  }, true);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTooltip(); });
  window.addEventListener("scroll", hideTooltip, true);
}

/* ------------------------------------------------------------------ *
 * Controls                                                            *
 * ------------------------------------------------------------------ */

function buildControl(field) {
  const id = fieldId(field);
  const wrapper = el("div", `field${field.wide ? " wide" : ""}${field.type === "checkbox" ? " check-field" : ""}`);
  wrapper.dataset.field = id;
  let input;
  if (field.type === "checkbox") {
    const row = el("div", "field-label");
    const label = el("label", "check");
    input = el("input");
    input.type = "checkbox";
    input.id = id;
    label.append(input, el("span", "", field.label));
    row.append(label, infoIcon(field.tip));
    wrapper.append(row);
    input.addEventListener("change", updateAvailability);
    return wrapper;
  }
  const head = el("div", "field-label");
  const label = el("label", "", field.label);
  label.htmlFor = id;
  head.append(label, infoIcon(field.tip));
  wrapper.append(head);
  if (field.type === "select") {
    input = el("select");
    field.options.forEach((option) => {
      const node = el("option", "", option.label);
      node.value = option.value;
      input.append(node);
    });
    input.id = id;
    wrapper.append(input);
    if (field.options.some((option) => option.tip)) {
      const hint = el("p", "hint");
      hint.id = `${id}-hint`;
      wrapper.append(hint);
    }
  } else if (field.type === "range") {
    const row = el("div", "range-row");
    input = el("input");
    Object.assign(input, { type: "range", min: field.min, max: field.max, step: field.step, id });
    const output = el("output");
    output.htmlFor = id;
    output.id = `${id}-output`;
    row.append(input, output);
    wrapper.append(row);
  } else {
    input = el("input");
    Object.assign(input, { type: "number", min: field.min, max: field.max, step: field.step, id });
    if (field.type === "seed") {
      const row = el("div", "seed-row");
      const dice = el("button", "dice", "↻");
      dice.type = "button";
      dice.title = "Sortear outra seed";
      dice.setAttribute("aria-label", `Sortear outra seed (${field.label})`);
      dice.addEventListener("click", () => { input.value = randomSeed(); });
      row.append(input, dice);
      wrapper.append(row);
    } else {
      wrapper.append(input);
    }
  }
  input.addEventListener("input", updateAvailability);
  input.addEventListener("change", updateAvailability);
  return wrapper;
}

function buildControls() {
  const form = $("controls");
  form.replaceChildren();
  GROUPS.forEach((group) => {
    const details = el("details", "group");
    details.open = group.open;
    const summary = el("summary");
    summary.append(el("span", "step", group.step), el("span", "group-title", group.title));
    const body = el("div", "group-body");
    group.fields.forEach((field) => body.append(buildControl({ ...field, scope: group.scope })));
    details.append(summary, body);
    form.append(details);
  });
  $("presets").replaceChildren(...PRESETS.map((preset) => {
    const button = el("button", "preset", preset.label);
    button.type = "button";
    button.dataset.tip = preset.tip;
    button.addEventListener("click", () => applyPreset(preset));
    return button;
  }));
}

function setField(scope, key, value) {
  const field = findField(scope, key);
  const input = $(fieldId(field));
  if (field.type === "checkbox") input.checked = Boolean(value);
  else input.value = String(value);
}

function applyDefaults() {
  const groups = { s: state.defaults.system, t: state.defaults.trajectory };
  ALL_FIELDS.forEach((field) => setField(field.scope, field.key, groups[field.scope][field.key]));
  updateAvailability();
}

function applyPreset(preset) {
  applyDefaults();
  if (preset.values) {
    Object.entries(preset.values).forEach(([scope, values]) =>
      Object.entries(values).forEach(([key, value]) => setField(scope, key, value)));
  }
  updateAvailability();
  setStatus(`Ponto de partida “${preset.label}” aplicado. Clique em gerar para ver o resultado.`);
  clearStatusSoon();
}

function readField(field) {
  const input = $(fieldId(field));
  if (field.type === "checkbox") return input.checked;
  if (field.type === "select" && field.key !== "fraction") return input.value;
  return Number(input.value);
}

function readScope(scope) {
  const values = {};
  ALL_FIELDS.filter((field) => field.scope === scope).forEach((field) => { values[field.key] = readField(field); });
  return values;
}

function updateAvailability() {
  const system = readScope("s");
  const trajectory = readScope("t");
  const disabled = new Set();
  if (system.channels <= 1 || system.dependency === "independent") {
    disabled.add("s-max_external_sources");
    disabled.add("s-self_source_weight");
  }
  if (system.channels <= 1) disabled.add("s-dependency");
  if (system.channels <= 1 || system.dependency === "independent") {
    disabled.add("s-target_gain");
    disabled.add("s-safety_factor");
  }
  const random = trajectory.initialization === "random";
  ALL_FIELDS.filter((field) => field.family).forEach((field) => { if (!random) disabled.add(fieldId(field)); });
  if (random) ["t-amplitude", "t-offset"].forEach((id) => disabled.add(id));
  if (["zero", "constant"].includes(trajectory.initialization)) disabled.add("t-amplitude");
  if (trajectory.initialization !== "cosine") {
    disabled.add("t-period");
    disabled.add("t-fraction");
  }
  document.querySelectorAll("#controls [data-field]").forEach((wrapper) => {
    const off = disabled.has(wrapper.dataset.field);
    wrapper.classList.toggle("disabled", off);
    wrapper.querySelectorAll("input, select").forEach((input) => { input.disabled = off; });
  });
  ALL_FIELDS.forEach((field) => {
    const id = fieldId(field);
    if (field.type === "range") {
      $(`${id}-output`).textContent = field.percent ? `${Math.round(readField(field) * 100)}%` : readField(field);
    }
    const hint = $(`${id}-hint`);
    if (hint) hint.textContent = field.options.find((option) => option.value === $(id).value)?.tip || "";
  });
}

/* ------------------------------------------------------------------ *
 * Server access                                                       *
 * ------------------------------------------------------------------ */

function randomSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.className = `status visible${isError ? " error" : ""}`;
}

let statusTimer = null;
function clearStatusSoon() {
  window.clearTimeout(statusTimer);
  statusTimer = window.setTimeout(() => $("status").classList.remove("visible"), 3200);
}

async function requestJson(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "A operação falhou.");
  return body;
}

function setBusy(busy) {
  $("generate-system").disabled = busy;
  $("generate-series").disabled = busy || !state.system;
}

async function generateSystem() {
  pausePlayback();
  setBusy(true);
  setStatus("Sorteando o sistema multivariado…");
  try {
    state.system = await requestJson("/api/systems", readScope("s"));
    state.data = null;
    state.activeChannel = 0;
    renderSystem();
    clearCanvases();
    setBusy(false);
    await generateTrajectories(true);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function generateTrajectories(afterSystem = false) {
  if (!state.system) return;
  pausePlayback();
  setBusy(true);
  setStatus("Executando as recorrências de todos os canais de forma síncrona…");
  try {
    state.data = await requestJson("/api/trajectories", { ...readScope("t"), system_id: state.system.id });
    state.activeTrajectory = 0;
    const replaced = state.data.stats.replaced_systems;
    if (state.data.replacement_system) {
      state.system = state.data.replacement_system;
      state.activeChannel = 0;
      renderSystem();
    }
    renderTrajectories();
    const { accepted, requested } = state.data.stats;
    const swap = replaced ? ` · sistema trocado ${replaced}× (seed original ${readScope("s").seed} descartada)` : "";
    setStatus(`${accepted} de ${requested} realizações aceitas${swap}${!replaced && afterSystem ? ` · sistema com seed ${state.system.seed}` : ""}.`);
    clearStatusSoon();
  } catch (error) {
    state.data = null;
    clearCanvases();
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ *
 * Formulas                                                            *
 * ------------------------------------------------------------------ */

function prettyFormula(text) {
  return text
    .replace(/[xs]_(\d+)[(\[]t-(\d+)[)\]]/g, (_, series, lag) => `${channelName(Number(series))}(t−${lag})`)
    .replace(/\*/g, "×");
}

const TREATMENT_LABEL = { jacobian: "Jacobiano", heaps: "Heaps", original: "Original", shao_wu: "Shao–Wu", kernel_synth: "KernelSynth" };
const DEPENDENCY_LABEL = Object.fromEntries(DEPENDENCIES.map((item) => [item.value, item.label]));

/* ------------------------------------------------------------------ *
 * System rendering                                                    *
 * ------------------------------------------------------------------ */

function renderSystem() {
  const system = state.system;
  const n = system.num_channels;
  $("scope-text").textContent = `${n} ${n === 1 ? "canal" : "canais"} · ${DEPENDENCY_LABEL[system.dependency].toLowerCase()}`;
  renderKpis(system);
  renderGraph(system);
  renderDependencyMap(system);
  renderEquationCards(system);
  renderActiveEquation();
  renderPhaseSelects(system);
}

function renderKpis(system) {
  const s = system.summary;
  const items = [
    [s.channels, "canais", "Quantidade de séries do sistema."],
    [s.operators, "operadores", "Soma dos operadores (+, ×, tanh) de todas as equações já tratadas."],
    [s.nodes, "nós", "Total de nós (operadores, lags e coeficientes) nas árvores executadas."],
    [`${s.effective_edges} / ${s.allowed_edges}`, "arestas usadas", "Dependências entre canais que de fato aparecem nas equações, sobre as permitidas pela topologia."],
    [`${s.shao_wu_channels} / ${s.channels}`, "canais Shao–Wu", "Quantos canais receberam saturação com tanh e escala de ganho."],
    [`${s.kernel_synth_channels ?? 0} / ${s.channels}`, "canais KernelSynth", "Quantos canais são séries KernelSynth (processo gaussiano), sem equação."],
    [s.history_length, "histórico exigido", "Maior lag consultado por qualquer equação: quantos valores iniciais são necessários para começar."],
  ];
  $("kpis").replaceChildren(...items.map(([value, label, tip]) => {
    const card = el("div", "kpi");
    card.append(el("strong", "", String(value)));
    const caption = el("span", "", label);
    caption.append(infoIcon(tip));
    card.append(caption);
    return card;
  }));
}

function svgNode(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function renderGraph(system) {
  const host = $("graph");
  host.replaceChildren();
  const n = system.num_channels;
  const size = 460;
  const center = size / 2;
  const radius = n === 1 ? 0 : 150;
  const nodeRadius = 23;
  const position = (index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / n;
    return { x: center + radius * Math.cos(angle), y: center + radius * Math.sin(angle) };
  };
  const svg = svgNode("svg", { viewBox: `0 0 ${size} ${size}`, role: "img",
    "aria-label": "Grafo de dependências entre os canais" });
  const defs = svgNode("defs");
  [["used", "#5c6a63"], ["unused", "#b5bcb6"]].forEach(([name, color]) => {
    const marker = svgNode("marker", { id: `arrow-${name}`, viewBox: "0 0 10 10", refX: 8, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    marker.append(svgNode("path", { d: "M0 0 L10 5 L0 10 z", fill: color }));
    defs.append(marker);
  });
  svg.append(defs);

  system.equations.forEach((equation) => {
    equation.allowed_sources.filter((source) => source !== equation.target).forEach((source) => {
      const used = equation.effective_sources.includes(source);
      const lags = equation.lags_by_source[String(source)] || [];
      const a = position(source);
      const b = position(equation.target);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy) || 1;
      const bend = 0.16 * length;
      const control = { x: (a.x + b.x) / 2 - (dy / length) * bend, y: (a.y + b.y) / 2 + (dx / length) * bend };
      const toward = (from, to, distance) => {
        const d = Math.hypot(to.x - from.x, to.y - from.y) || 1;
        return { x: from.x + ((to.x - from.x) / d) * distance, y: from.y + ((to.y - from.y) / d) * distance };
      };
      const start = toward(a, control, nodeRadius + 2);
      const end = toward(b, control, nodeRadius + 6);
      const path = svgNode("path", {
        d: `M${start.x} ${start.y} Q${control.x} ${control.y} ${end.x} ${end.y}`,
        class: `edge ${used ? "used" : "unused"}`,
        "stroke-width": used ? 1.4 + 0.45 * Math.min(lags.length, 4) : 1.3,
        "marker-end": `url(#arrow-${used ? "used" : "unused"})`,
      });
      const title = svgNode("title");
      title.textContent = used
        ? `${channelName(source)} → ${channelName(equation.target)} · lags ${lags.join(", ")}`
        : `${channelName(source)} → ${channelName(equation.target)} · permitida, não usada`;
      path.append(title);
      svg.append(path);
    });
  });

  system.equations.forEach((equation, index) => {
    const { x, y } = position(index);
    const group = svgNode("g", { class: "node", transform: `translate(${x} ${y})`, tabindex: 0, style: "cursor:pointer" });
    group.append(svgNode("circle", { r: nodeRadius, fill: channelColor(index) }));
    if (equation.treatment.method === "shao_wu") {
      group.append(svgNode("circle", { r: nodeRadius + 5, class: "ring", stroke: channelColor(index) }));
    }
    const label = svgNode("text", { y: 5 });
    label.textContent = channelName(index);
    const title = svgNode("title");
    title.textContent = `${channelName(index)} · ${TREATMENT_LABEL[equation.treatment.method]} · σ = ${equation.innovation.std.toPrecision(3)}`;
    group.append(label, title);
    group.addEventListener("click", () => selectChannel(index));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") selectChannel(index); });
    svg.append(group);
  });
  host.append(svg);
}

function renderDependencyMap(system) {
  const map = $("dependency-map");
  const n = system.num_channels;
  map.replaceChildren();
  map.style.gridTemplateColumns = `62px repeat(${n}, minmax(46px, 1fr))`;
  map.append(el("div", "dependency-header corner", "alvo ↓ fonte →"));
  for (let source = 0; source < n; source += 1) {
    const header = el("div", "dependency-header", channelName(source));
    header.style.color = channelColor(source);
    map.append(header);
  }
  system.dependency_lags.forEach((row, target) => {
    const label = el("div", "dependency-target", `${channelName(target)}(t)`);
    label.style.color = channelColor(target);
    map.append(label);
    row.forEach((lags, source) => {
      const equation = system.equations[target];
      const allowed = equation.allowed_sources.includes(source);
      const kind = source === target ? "self" : "cross";
      const state_ = !allowed ? "forbidden" : lags.length ? "used" : "allowed";
      const cell = el("div", `dependency-cell ${kind} ${state_}`);
      const pills = el("div", "lag-pills");
      if (lags.length) {
        const shownLags = lags.slice(0, 3).map((lag) => `−${lag}`);
        if (lags.length > 3) shownLags.push(`+${lags.length - 3}`);
        pills.append(el("span", "lag-text", shownLags.join(" ")));
      } else if (allowed) {
        pills.append(el("span", "dependency-empty", "—"));
      }
      cell.title = !allowed
        ? `${channelName(source)} não é fonte permitida de ${channelName(target)}`
        : `${channelName(source)} → ${channelName(target)}: ${lags.length ? `lags ${lags.join(", ")}` : "permitida, não usada"}`;
      cell.append(pills);
      map.append(cell);
    });
  });
}

function renderEquationCards(system) {
  const grid = $("equation-grid");
  grid.replaceChildren();
  system.equations.forEach((equation, index) => {
    const card = el("button", `equation-card${index === state.activeChannel ? " active" : ""}`);
    card.type = "button";
    card.style.setProperty("--channel", channelColor(index));
    card.setAttribute("aria-pressed", String(index === state.activeChannel));
    card.addEventListener("click", () => selectChannel(index));

    const heading = el("div", "equation-card-heading");
    heading.append(el("span", "series-badge", `${channelName(index)}(t)`));
    const shao = equation.treatment.method === "shao_wu";
    const treatment = el("span", `pill${shao ? " shao" : ""}`,
      `${TREATMENT_LABEL[equation.treatment.method]}${["shao_wu", "jacobian"].includes(equation.treatment.method) ? ` - escala ${equation.treatment.scale.toPrecision(3)}` : ""}`);
    treatment.dataset.tip = equation.treatment.method === "jacobian"
      ? "Contracao Jacobiana no dominio certificado, com inovacao uniforme limitada."
      : equation.treatment.method === "kernel_synth"
      ? `Série KernelSynth: amostra de processo gaussiano com kernel ${equation.kernel_formula}. Não tem equação nem alvo da cabeça.`
      : equation.treatment.method === "heaps"
      ? "AR est?vel constru?do pela parametriza??o Heaps, com inova??o gaussiana."
      : shao
      ? "Fatores dos produtos de lags saturados com tanh e equação multiplicada pela escala indicada para limitar o ganho."
      : "Equação executada exatamente como sorteada, sem saturação nem escala.";
    const sigma = el("span", "pill sigma", `σ ${equation.innovation.std.toPrecision(2)}`);
    sigma.dataset.tip = `Inovacao ${equation.innovation.family}: desvio padrao por passo.`;
    const sources = el("span", "pill sigma", equation.family === "kernel_synth" ? "série gerada, sem lags"
      : `lê ${equation.effective_sources.length ? equation.effective_sources.map(channelName).join(", ") : "nenhum lag"}`);
    sources.dataset.tip = "Canais cujos lags realmente aparecem nesta equação.";
    heading.append(treatment, sigma, sources);
    card.append(heading);

    card.append(formulaRow("Equação executada", `${channelName(index)}(t) = ${prettyFormula(equation.executed_formula)}`));
    if (shao) card.append(formulaRow("Antes do tratamento", prettyFormula(equation.original_formula)));
    card.append(formulaRow(`Forma canônica · alvo da cabeça (${equation.canonical_terms} termos)`, prettyFormula(equation.canonical_formula)));

    const metrics = el("div", "equation-mini-metrics");
    [[equation.summary.operators, "operadores"], [equation.summary.depth, "níveis"], [equation.summary.degree, "grau"]]
      .forEach(([value, label]) => {
        const item = el("span");
        item.append(el("strong", "", String(value)), ` ${label}`);
        metrics.append(item);
      });
    card.append(metrics);
    grid.append(card);
  });
}

function formulaRow(label, formula) {
  const row = el("div", "formula-row");
  row.append(el("p", "", label), el("code", "", formula));
  return row;
}

function selectChannel(index) {
  state.activeChannel = index;
  document.querySelectorAll(".equation-card").forEach((card, position) => {
    card.classList.toggle("active", position === index);
    card.setAttribute("aria-pressed", String(position === index));
  });
  renderActiveEquation();
}

function renderActiveEquation() {
  const equation = state.system.equations[state.activeChannel];
  const target = `${channelName(state.activeChannel)}(t)`;
  $("active-tree-target").textContent = target;
  $("active-binding-target").textContent = target;
  renderTree(equation.tree);
  renderBindings(equation.binding);
}

function nodeLabel(node) {
  if (node.type === "TANH") return ["tanh"];
  if (node.type === "ADD") return ["+"];
  if (node.type === "MULTIPLY") return ["×"];
  if (node.type === "PARAMETER") return [node.slot.replace("MULTISERIES_SCALE", "escala"), Number(node.value).toPrecision(3)];
  return [`${channelName(node.series)}(t−${node.lag})`, node.slot];
}

function renderTree(root) {
  const stage = $("tree-visualization");
  stage.replaceChildren();
  const levels = [];
  let leafCursor = 0;

  function layout(node, depth = 0) {
    const laidChildren = node.children.map((child) => layout(child, depth + 1));
    const x = laidChildren.length
      ? laidChildren.reduce((sum, child) => sum + child.x, 0) / laidChildren.length
      : leafCursor++;
    levels[depth] = (levels[depth] || 0) + 1;
    return { ...node, depth, x, children: laidChildren };
  }

  const tree = layout(root);
  const leafCount = Math.max(leafCursor, 1);
  const width = Math.max(520, stage.clientWidth || 520, leafCount * 66);
  const height = Math.max(240, levels.length * 88 + 20);
  const marginX = 48;
  const scaleX = leafCount === 1 ? 0 : (width - marginX * 2) / (leafCount - 1);
  const px = (item) => (leafCount === 1 ? width / 2 : marginX + item.x * scaleX);
  const py = (item) => 42 + item.depth * 84;

  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, height });
  svg.style.minWidth = `${Math.min(width, leafCount * 66)}px`;
  (function drawEdges(item) {
    item.children.forEach((child) => {
      svg.append(svgNode("line", { x1: px(item), y1: py(item), x2: px(child), y2: py(child), class: "tree-edge" }));
      drawEdges(child);
    });
  })(tree);
  (function drawNodes(item) {
    const group = svgNode("g", { class: `tree-node node-${item.type}`, transform: `translate(${px(item)} ${py(item)})` });
    group.append(svgNode("circle", { r: item.type === "LAG" || item.type === "PARAMETER" ? 29 : 24 }));
    const lines = nodeLabel(item);
    lines.forEach((text, index) => {
      const label = svgNode("text", { y: lines.length === 1 ? 4 : index * 12 - 2 });
      label.textContent = text;
      group.append(label);
    });
    svg.append(group);
    item.children.forEach(drawNodes);
  })(tree);
  stage.append(svg);
}

function renderBindings(binding) {
  const rows = [
    ...Object.entries(binding.parameters).map(([slot, value]) => ["coeficiente", slot, Number(value).toPrecision(5)]),
    ...Object.entries(binding.lags).map(([slot, value]) => ["lag", slot, String(value)]),
  ];
  const table = $("binding-table");
  table.replaceChildren();
  if (!rows.length) {
    table.append(el("p", "empty", "A expressão não possui slots numéricos."));
    return;
  }
  rows.forEach(([kind, slot, value]) => {
    const row = el("div", "binding-row");
    row.append(el("span", "binding-kind", kind), el("span", "binding-slot", slot.replace("MULTISERIES_SCALE", "escala")),
      el("span", "binding-value", value));
    table.append(row);
  });
}

/* ------------------------------------------------------------------ *
 * Canvas helpers                                                      *
 * ------------------------------------------------------------------ */

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function paddedExtent(values) {
  let minimum = Infinity;
  let maximum = -Infinity;
  values.forEach((value) => {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  });
  if (minimum === maximum) {
    minimum -= 1;
    maximum += 1;
  }
  const padding = (maximum - minimum) * 0.08;
  return [minimum - padding, maximum + padding];
}

function chartAxis(values) {
  const logarithmic = $("trajectory-scale").value === "symlog";
  const transform = logarithmic
    ? (value) => Math.sign(value) * Math.log1p(Math.abs(value)) / Math.LN10
    : (value) => value;
  // Normalize before subtraction so opposite, very large finite values remain drawable.
  const transformed = values.map(transform);
  const magnitude = transformed.reduce((best, value) => Math.max(best, Math.abs(value)), 0) || 1;
  const [minimum, maximum] = paddedExtent(transformed.map((value) => value / magnitude));
  return {
    position: (value) => (transform(value) / magnitude - minimum) / (maximum - minimum),
    tick: (fraction) => {
      const value = (minimum + fraction * (maximum - minimum)) * magnitude;
      const original = logarithmic ? Math.sign(value) * Math.expm1(Math.abs(value) * Math.LN10) : value;
      return Number.isFinite(original) ? original.toPrecision(3) : (value < 0 ? "−" : "") + ">1.80e308";
    },
  };
}

const INK = "#18221e";
const MUTED = "#6a766f";
const GRID = "#e6e3d8";
const FONT = "system-ui, sans-serif";

/* ------------------------------------------------------------------ *
 * Trajectory player                                                   *
 * ------------------------------------------------------------------ */

function pausePlayback() {
  window.clearInterval(state.playback);
  state.playback = null;
  $("trajectory-play").textContent = "▶ Reproduzir";
  $("trajectory-play").setAttribute("aria-pressed", "false");
}

function updatePlaybackControls() {
  const count = state.data?.trajectories.length || 0;
  const index = state.data?.trajectory_indices?.[state.activeTrajectory];
  $("trajectory-position").textContent = count
    ? `Realização ${state.activeTrajectory + 1} de ${count} · índice sorteado #${index}`
    : "Nenhuma trajetória";
  $("trajectory-select").max = String(Math.max(1, count));
  $("trajectory-select").value = String(state.activeTrajectory + 1);
  $("trajectory-select").disabled = !count;
  $("trajectory-play").disabled = count < 2;
  $("trajectory-previous").disabled = !count || state.activeTrajectory === 0;
  $("trajectory-next").disabled = !count || state.activeTrajectory === count - 1;
}

function selectTrajectory(index) {
  if (!state.data) return;
  const count = state.data.trajectories.length;
  state.activeTrajectory = Math.max(0, Math.min(count - 1, Math.round(index)));
  renderTrajectories();
}

function playTrajectories() {
  pausePlayback();
  if (!state.data || state.data.trajectories.length < 2) return;
  if (state.activeTrajectory === state.data.trajectories.length - 1) selectTrajectory(0);
  $("trajectory-play").textContent = "⏸ Pausar";
  $("trajectory-play").setAttribute("aria-pressed", "true");
  state.playback = window.setInterval(() => {
    selectTrajectory(state.activeTrajectory + 1);
    if (state.activeTrajectory === state.data.trajectories.length - 1) pausePlayback();
  }, Number($("trajectory-speed").value));
}

const FAILURE_LABEL = {
  nonfinite_simulation: "explosão numérica (valor não finito)",
  arithmetic_overflow: "estouro aritmético",
  nonfinite_output_dtype: "valor fora do alcance de float32",
  nonfinite_inputs: "entradas não finitas",
};

function renderAcceptance() {
  const box = $("acceptance");
  box.replaceChildren();
  const stats = state.data?.stats;
  if (!stats) return;
  const rate = el("span", `pill${stats.skipped ? " shao" : ""}`, `${stats.accepted}/${stats.requested} aceitas`);
  rate.dataset.tip = "Realizações que ficaram finitas até o fim. As demais são descartadas, exatamente como o gerador de treino faz.";
  box.append(rate);
  if (stats.replaced_systems) {
    const swapped = el("span", "pill", `${stats.replaced_systems} troca${stats.replaced_systems > 1 ? "s" : ""} de sistema`);
    swapped.dataset.tip = `Todas as tentativas de uma realização explodiram, então o sistema foi descartado e outro sorteado ${stats.replaced_systems}× (${stats.discarded_trajectories} realizações já aceitas foram descartadas junto). O sistema exibido é o aceito; sua seed é ${state.system.seed}.`;
    swapped.tabIndex = 0;
    box.append(swapped);
  }
  if (stats.skipped) {
    const reasons = Object.entries(stats.failure_reasons)
      .map(([reason, count]) => `${count}× ${FAILURE_LABEL[reason] || reason}`).join("; ");
    const skipped = el("span", "pill sigma", `${stats.skipped} descartadas`);
    skipped.dataset.tip = `Tentativas que falharam: ${reasons}. Reduza os coeficientes, o grau ou o horizonte, ou ative Shao–Wu, para explodir menos.`;
    skipped.tabIndex = 0;
    box.append(skipped);
  }
  rate.tabIndex = 0;
}

function renderTrajectories() {
  const data = state.data;
  renderAcceptance();
  drawLineChart();
  if ($("ensemble-overview").open) drawHeatmap();
  drawPhaseChart();
  updatePlaybackControls();
  const history = data.initial_history[state.activeTrajectory][0]?.length || 0;
  const label = data.initialization === "random" ? "famílias sorteadas por canal"
    : `inicialização ${(INITIALIZATIONS.find((option) => option.value === data.initialization)?.label || data.initialization).toLowerCase()}`;
  $("trajectory-caption").textContent =
    `Contexto de ${data.context_length} passos e horizonte de ${data.prediction_length}; cada painel tem escala própria. ` +
    `O histórico inicial (${history} valores, ${label}) só dispara a recorrência.`;
  renderInitializationComponents();
}

function describeComponent(part) {
  if (part.family === "constant") return `constante ${part.level >= 0 ? "+" : "−"}${Math.abs(part.level).toFixed(2)}`;
  if (part.family === "noise") return `ruído U(±${part.radius.toFixed(2)})`;
  return `cosseno A=${part.amplitude.toFixed(2)}, T=${part.period}, φ=${part.phase.toFixed(2)}`;
}

function renderInitializationComponents() {
  const box = $("initialization-components");
  box.replaceChildren();
  const components = state.data.initialization_components?.[state.activeTrajectory];
  if (!components) { box.hidden = true; return; }
  box.hidden = false;
  box.append(el("span", "components-title", "Histórico sorteado nesta realização:"));
  components.forEach((parts, channel) => {
    const item = el("span", "component");
    const name = el("strong", null, channelName(channel));
    name.style.color = channelColor(channel);
    item.append(name, document.createTextNode(" " + (parts.length ? parts.map(describeComponent).join(" + ") : "zero (nenhuma família)")));
    box.append(item);
  });
}

function historyShown() {
  if (!state.data || !$("show-history").checked) return 0;
  return state.data.initial_history[state.activeTrajectory][0]?.length || 0;
}

function drawLineChart() {
  const data = state.data;
  const channels = data.trajectories[0].length;
  const panelHeight = channels > 8 ? 78 : channels > 4 ? 92 : 128;
  const gap = 24;
  const margin = { left: 62, right: 16, top: 26, bottom: 34 };
  const canvas = $("trajectory-chart");
  canvas.style.height = `${margin.top + channels * (panelHeight + gap) - gap + margin.bottom}px`;
  const { context: ctx, width, height } = setupCanvas(canvas);
  const plotWidth = width - margin.left - margin.right;
  const steps = data.trajectories[state.activeTrajectory][0].length;
  const shown = historyShown();
  const span = Math.max(shown + steps - 1, 1);
  const x = (time) => margin.left + ((time + shown) / span) * plotWidth;
  state.layout = { left: margin.left, plotWidth, shown, steps };

  ctx.clearRect(0, 0, width, height);
  const trajectory = data.trajectories[state.activeTrajectory];
  const histories = data.initial_history[state.activeTrajectory];
  const contextEnd = data.context_length;

  for (let channel = 0; channel < channels; channel += 1) {
    const top = margin.top + channel * (panelHeight + gap);
    const color = channelColor(channel);
    const past = shown ? histories[channel].slice(-shown) : [];
    const axis = chartAxis([...past, ...trajectory[channel]]);
    const y = (value) => top + (1 - axis.position(value)) * panelHeight;

    ctx.fillStyle = "rgba(68,136,163,0.10)";
    ctx.fillRect(x(0), top, x(contextEnd) - x(0), panelHeight);
    ctx.fillStyle = "rgba(236,122,69,0.13)";
    ctx.fillRect(x(contextEnd), top, x(steps - 1) - x(contextEnd), panelHeight);

    ctx.font = `10px ${FONT}`;
    ctx.lineWidth = 1;
    for (let tick = 0; tick <= 2; tick += 1) {
      const yy = top + (1 - tick / 2) * panelHeight;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(margin.left, yy);
      ctx.lineTo(width - margin.right, yy);
      ctx.stroke();
      ctx.fillStyle = MUTED;
      ctx.textAlign = "right";
      ctx.fillText(axis.tick(tick / 2), margin.left - 6, yy + 3);
    }
    ctx.textAlign = "left";

    if (shown) {
      ctx.beginPath();
      past.forEach((value, index) => {
        const time = index - past.length;
        index ? ctx.lineTo(x(time), y(value)) : ctx.moveTo(x(time), y(value));
      });
      ctx.lineTo(x(0), y(trajectory[channel][0]));
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    trajectory[channel].forEach((value, index) => (index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value))));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    ctx.strokeStyle = "rgba(24,34,30,0.35)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(contextEnd), top);
    ctx.lineTo(x(contextEnd), top + panelHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    const equation = state.system?.equations[channel];
    ctx.fillStyle = color;
    ctx.font = `700 12px ${FONT}`;
    ctx.fillText(`${channelName(channel)}(t)`, margin.left, top - 8);
    if (equation) {
      ctx.fillStyle = MUTED;
      ctx.font = `10px ${FONT}`;
      ctx.fillText(`${TREATMENT_LABEL[equation.treatment.method]} · σ ${equation.innovation.std.toPrecision(2)}`,
        margin.left + 46, top - 8);
    }
    if (channel === 0) {
      ctx.fillStyle = "#2f6b86";
      ctx.fillText("contexto", x(0) + 6, top + 13);
      ctx.fillStyle = "#b5551f";
      ctx.fillText("horizonte", x(contextEnd) + 6, top + 13);
    }
  }

  const baseline = height - margin.bottom + 16;
  ctx.fillStyle = MUTED;
  ctx.font = `10px ${FONT}`;
  [[0, "t=0"], [contextEnd, `t=${contextEnd}`], [steps - 1, `t=${steps - 1}`]].forEach(([time, label], index) => {
    ctx.textAlign = index === 2 ? "right" : index === 0 ? "left" : "center";
    ctx.fillText(label, x(time), baseline);
  });
  if (shown) {
    ctx.textAlign = "left";
    ctx.fillText(`t=−${shown}`, x(-shown), baseline);
  }
  ctx.textAlign = "left";

  if (state.hoverTime !== null) {
    const time = Math.max(-shown, Math.min(steps - 1, state.hoverTime));
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(time), margin.top - 4);
    ctx.lineTo(x(time), height - margin.bottom + 2);
    ctx.stroke();
    updateReadout(time);
  } else {
    $("readout").textContent = "";
  }
}

function updateReadout(time) {
  const trajectory = state.data.trajectories[state.activeTrajectory];
  const histories = state.data.initial_history[state.activeTrajectory];
  const values = trajectory.map((series, channel) => {
    const value = time >= 0 ? series[time] : histories[channel][histories[channel].length + time];
    return `${channelName(channel)} ${Number(value).toPrecision(3)}`;
  });
  $("readout").textContent = `t=${time} · ${values.join(" · ")}`;
}

function heatColor(value, maximumMagnitude) {
  const normalized = Math.min(Math.abs(value) / maximumMagnitude, 1);
  const base = value >= 0 ? [236, 122, 69] : [68, 136, 163];
  const light = [247, 245, 237];
  const mix = 0.12 + normalized * 0.88;
  return `rgb(${base.map((channel, index) => Math.round(light[index] * (1 - mix) + channel * mix)).join(",")})`;
}

function drawHeatmap() {
  const data = state.data;
  const channels = data.trajectories[0].length;
  const { context: ctx, width, height } = setupCanvas($("heatmap"));
  const margin = { left: 46, right: 8, top: 4, bottom: 18 };
  const gap = 8;
  const rows = data.trajectories.length;
  const columns = data.trajectories[0][0].length;
  const blockHeight = (height - margin.top - margin.bottom - gap * (channels - 1)) / channels;
  const cellWidth = (width - margin.left - margin.right) / columns;
  const cellHeight = blockHeight / rows;
  ctx.clearRect(0, 0, width, height);
  for (let channel = 0; channel < channels; channel += 1) {
    const blockTop = margin.top + channel * (blockHeight + gap);
    const maximumMagnitude = Math.max(Math.abs(data.summary.minimum[channel]), Math.abs(data.summary.maximum[channel]), 1e-12);
    data.trajectories.forEach((trajectory, row) =>
      trajectory[channel].forEach((value, column) => {
        ctx.fillStyle = heatColor(value, maximumMagnitude);
        ctx.fillRect(margin.left + column * cellWidth, blockTop + row * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
      }));
    const split = margin.left + data.context_length * cellWidth;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(split, blockTop);
    ctx.lineTo(split, blockTop + blockHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = channelColor(channel);
    ctx.font = `700 11px ${FONT}`;
    ctx.fillText(channelName(channel), 6, blockTop + 13);
    ctx.fillStyle = MUTED;
    ctx.font = `9px ${FONT}`;
    ctx.fillText(`±${maximumMagnitude.toPrecision(2)}`, 6, blockTop + 25);
  }
  ctx.fillStyle = MUTED;
  ctx.font = `9px ${FONT}`;
  ctx.fillText("tempo → (linha tracejada: início do horizonte)", margin.left, height - 5);
}

function renderPhaseSelects(system) {
  const n = system.num_channels;
  ["phase-x", "phase-y"].forEach((id, position) => {
    const select = $(id);
    select.replaceChildren();
    for (let channel = 0; channel < n; channel += 1) {
      const option = el("option", "", n === 1 && position === 0 ? `${channelName(0)}(t−1)` : channelName(channel));
      option.value = String(channel);
      select.append(option);
    }
    select.value = String(n === 1 ? 0 : Math.min(position, n - 1));
    select.disabled = n === 1;
  });
}

function drawPhaseChart() {
  const data = state.data;
  const { context: ctx, width, height } = setupCanvas($("phase-chart"));
  const margin = { left: 58, right: 20, top: 16, bottom: 40 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const trajectory = data.trajectories[state.activeTrajectory];
  const channels = trajectory.length;
  const xChannel = Number($("phase-x").value) || 0;
  const yChannel = Number($("phase-y").value) || 0;
  const series = channels === 1
    ? [trajectory[0].slice(0, -1), trajectory[0].slice(1)]
    : [trajectory[Math.min(xChannel, channels - 1)], trajectory[Math.min(yChannel, channels - 1)]];
  const xAxis = chartAxis(series[0]);
  const yAxis = chartAxis(series[1]);
  const x = (value) => margin.left + xAxis.position(value) * plotWidth;
  const y = (value) => margin.top + (1 - yAxis.position(value)) * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.font = `10px ${FONT}`;
  for (let tick = 0; tick <= 4; tick += 1) {
    const xx = margin.left + (tick / 4) * plotWidth;
    const yy = margin.top + (1 - tick / 4) * plotHeight;
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xx, margin.top);
    ctx.lineTo(xx, height - margin.bottom);
    ctx.moveTo(margin.left, yy);
    ctx.lineTo(width - margin.right, yy);
    ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.textAlign = "center";
    ctx.fillText(xAxis.tick(tick / 4), xx, height - margin.bottom + 15);
    ctx.textAlign = "right";
    ctx.fillText(yAxis.tick(tick / 4), margin.left - 6, yy + 3);
  }
  ctx.textAlign = "left";

  const split = Math.min(data.context_length, series[0].length - 1);
  const segment = (from, to, color) => {
    ctx.beginPath();
    for (let index = from; index <= to; index += 1) {
      index === from ? ctx.moveTo(x(series[0][index]), y(series[1][index])) : ctx.lineTo(x(series[0][index]), y(series[1][index]));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.stroke();
  };
  segment(0, split, "#4488a3");
  segment(split, series[0].length - 1, "#d9692f");
  ctx.beginPath();
  ctx.arc(x(series[0][0]), y(series[1][0]), 3.2, 0, Math.PI * 2);
  ctx.fillStyle = INK;
  ctx.fill();

  const xName = channels === 1 ? `${channelName(0)}(t−1)` : channelName(xChannel);
  const yName = channels === 1 ? `${channelName(0)}(t)` : channelName(yChannel);
  ctx.fillStyle = channelColor(channels === 1 ? 0 : yChannel);
  ctx.font = `700 11px ${FONT}`;
  ctx.fillText(`${yName} ↑`, 6, margin.top + 2);
  ctx.fillStyle = channelColor(channels === 1 ? 0 : xChannel);
  ctx.textAlign = "right";
  ctx.fillText(`${xName} →`, width - margin.right, height - 6);
  ctx.textAlign = "left";
  ctx.fillStyle = "#2f6b86";
  ctx.fillText("● contexto", margin.left, height - 6);
  ctx.fillStyle = "#b5551f";
  ctx.fillText("● horizonte", margin.left + 72, height - 6);
}

function clearCanvases() {
  pausePlayback();
  state.activeTrajectory = 0;
  state.hoverTime = null;
  updatePlaybackControls();
  ["trajectory-chart", "heatmap", "phase-chart"].forEach((id) => {
    const canvas = $(id);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  });
  $("acceptance").replaceChildren();
  $("readout").textContent = "";
  $("trajectory-caption").textContent = "Cada painel usa sua própria escala; o tempo permanece alinhado entre os canais.";
}

/* ------------------------------------------------------------------ *
 * Wiring                                                              *
 * ------------------------------------------------------------------ */

function redraw() {
  if (state.data) renderTrajectories();
}

function wire() {
  $("generate-system").addEventListener("click", generateSystem);
  $("generate-series").addEventListener("click", () => generateTrajectories());
  $("trajectory-previous").addEventListener("click", () => { pausePlayback(); selectTrajectory(state.activeTrajectory - 1); });
  $("trajectory-next").addEventListener("click", () => { pausePlayback(); selectTrajectory(state.activeTrajectory + 1); });
  $("trajectory-select").addEventListener("input", () => { pausePlayback(); selectTrajectory(Number($("trajectory-select").value) - 1); });
  $("trajectory-play").addEventListener("click", () => { state.playback !== null ? pausePlayback() : playTrajectories(); });
  $("trajectory-speed").addEventListener("change", () => { if (state.playback !== null) playTrajectories(); });
  $("trajectory-scale").addEventListener("change", redraw);
  $("show-history").addEventListener("change", redraw);
  $("phase-x").addEventListener("change", () => { if (state.data) drawPhaseChart(); });
  $("phase-y").addEventListener("change", () => { if (state.data) drawPhaseChart(); });
  $("ensemble-overview").addEventListener("toggle", () => { if ($("ensemble-overview").open && state.data) drawHeatmap(); });
  $("trajectory-chart").addEventListener("mousemove", (event) => {
    if (!state.data || !state.layout) return;
    const { left, plotWidth, shown, steps } = state.layout;
    const rect = $("trajectory-chart").getBoundingClientRect();
    const fraction = (event.clientX - rect.left - left) / plotWidth;
    state.hoverTime = Math.round(fraction * Math.max(shown + steps - 1, 1) - shown);
    if (state.hoverTime < -shown || state.hoverTime > steps - 1) state.hoverTime = null;
    drawLineChart();
  });
  $("trajectory-chart").addEventListener("mouseleave", () => {
    state.hoverTime = null;
    if (state.data) drawLineChart();
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pausePlayback(); });
  window.addEventListener("resize", () => {
    if (state.system) renderActiveEquation();
    redraw();
  });
}

async function start() {
  setupTooltips();
  buildControls();
  wire();
  try {
    state.defaults = await requestJson("/api/defaults");
  } catch (error) {
    setStatus("Não foi possível carregar os valores padrão do servidor.", true);
    return;
  }
  applyDefaults();
  await generateSystem();
}

start();
