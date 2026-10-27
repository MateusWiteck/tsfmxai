const $ = (id) => document.getElementById(id);

const state = { mechanism: null, trajectories: null, activeEquation: 0, activeTrajectory: 0, playback: null };
const seriesNames = ["x₀", "x₁"];
const seriesColors = ["#1f6b55", "#4488a3"];

function numberValue(id) {
  return Number($(id).value);
}

function randomSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.className = `status visible${isError ? " error" : ""}`;
}

function clearStatusSoon() {
  window.setTimeout(() => $("status").classList.remove("visible"), 2600);
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "A operação falhou.");
  return body;
}

function mechanismRequest() {
  return {
    stability_method: $("stability-method").value,
    shao_wu_mode: $("shao-wu-mode").value,
    innovation_std: numberValue("innovation"),
    seed: numberValue("tree-seed"),
    minimum_operators: numberValue("min-operators"),
    maximum_operators: numberValue("max-operators"),
    minimum_lag: numberValue("min-lag"),
    maximum_lag: numberValue("max-lag"),
    parameter_probability: numberValue("p-parameter"),
    time_probability: numberValue("p-time"),
    lag_probability: numberValue("p-lag"),
    add_probability: numberValue("p-add"),
    require_lag: $("require-lag").checked,
    require_time: $("require-time").checked,
  };
}

function trajectoryRequest() {
  return {
    mechanism_id: state.mechanism.id,
    seed: numberValue("series-seed"),
    trajectory_count: numberValue("trajectory-count"),
    generated_steps: numberValue("generated-steps"),
    initial_value_minimum: numberValue("initial-min"),
    initial_value_maximum: numberValue("initial-max"),
    innovation_std: numberValue("innovation"),
  };
}

async function generateMechanism() {
  pausePlayback();
  const button = $("generate-tree");
  button.disabled = true;
  $("generate-series").disabled = true;
  setStatus("Sorteando as duas equações…");
  try {
    state.mechanism = await postJson("/api/mechanisms", mechanismRequest());
    state.trajectories = null;
    state.activeEquation = 0;
    renderMechanism(state.mechanism);
    clearCanvases();
    $("generate-series").disabled = false;
    setStatus(`Sistema reproduzível criado com seed ${state.mechanism.seed}.`);
    clearStatusSoon();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function generateTrajectories() {
  if (!state.mechanism) return;
  pausePlayback();
  const button = $("generate-series");
  button.disabled = true;
  setStatus("Executando as duas recorrências de forma síncrona…");
  try {
    state.trajectories = await postJson("/api/trajectories", trajectoryRequest());
    state.activeTrajectory = 0;
    renderTrajectories(state.trajectories);
    setStatus(
      `${state.trajectories.trajectories.length} trajetórias multivariadas geradas com o sistema fixo.`,
    );
    clearStatusSoon();
  } catch (error) {
    setStatus(
      `${error.message} Tente outro sistema, menos passos ou condições iniciais menores.`,
      true,
    );
  } finally {
    button.disabled = false;
  }
}

function renderMechanism(mechanism) {
  const certified = mechanism.certificates?.length > 0 && mechanism.certificates.every((c) => c.holds);
  const amplitudeBound = mechanism.certificates?.some((c) => c.guarantee === "uniform_second_moment_bound");
  const guarantee = amplitudeBound ? "Limite uniforme do segundo momento verificado."
    : "Condições da garantia matemática verificadas.";
  $("stability-summary").textContent = `${$("stability-method").selectedOptions[0].textContent} · ${certified ? guarantee : "Sem verificação matemática de estabilidade."}`;
  renderEquationCards(mechanism.equations);
  const labels = {
    equations: "equações",
    operators: "operadores",
    nodes: "nós totais",
    maximum_depth: "profundidade máx.",
    required_history: "histórico comum",
  };
  $("tree-metrics").innerHTML = Object.entries(mechanism.summary)
    .map(
      ([key, value]) =>
        `<div class="metric"><strong>${value}</strong><span>${labels[key]}</span></div>`,
    )
    .join("");
  renderEquationTabs(mechanism.equations);
  renderActiveEquation();
  renderDependencyMap(mechanism);
}

function renderEquationCards(equations) {
  const grid = $("equation-grid");
  grid.replaceChildren();
  equations.forEach((equation, index) => {
    const card = document.createElement("article");
    card.className = `equation-card series-${index}`;

    const heading = document.createElement("div");
    heading.className = "equation-card-heading";
    const badge = document.createElement("span");
    badge.className = "series-badge";
    badge.textContent = `${seriesNames[index]}(t)`;
    const seed = document.createElement("span");
    seed.className = "equation-seed";
    seed.textContent = `seed derivada ${equation.seed}`;
    heading.append(badge, seed);

    const skeleton = formulaRow("Estrutura sem valores", equation.skeleton_formula);
    const bound = formulaRow(
      "Recorrência com valores",
      `${seriesNames[index]}(t) = ${equation.formula}`,
    );
    const miniMetrics = document.createElement("div");
    miniMetrics.className = "equation-mini-metrics";
    miniMetrics.innerHTML = [
      [equation.summary.operators, "operadores"],
      [equation.summary.depth, "níveis"],
      [equation.summary.required_history, "lag máximo"],
    ]
      .map(([value, label]) => `<span><strong>${value}</strong> ${label}</span>`)
      .join("");

    card.append(heading, skeleton, bound, miniMetrics);
    grid.appendChild(card);
  });
}

function formulaRow(label, formula) {
  const row = document.createElement("div");
  row.className = "formula-row";
  const title = document.createElement("p");
  title.textContent = label;
  const code = document.createElement("code");
  code.textContent = formula;
  row.append(title, code);
  return row;
}

function renderEquationTabs(equations) {
  const tabs = $("equation-tabs");
  tabs.replaceChildren();
  equations.forEach((equation, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `equation-tab${index === state.activeEquation ? " active" : ""}`;
    button.dataset.series = String(index);
    button.textContent = `${seriesNames[index]}(t)`;
    button.setAttribute("aria-pressed", String(index === state.activeEquation));
    button.addEventListener("click", () => {
      state.activeEquation = equation.target_series;
      renderEquationTabs(state.mechanism.equations);
      renderActiveEquation();
    });
    tabs.appendChild(button);
  });
}

function renderActiveEquation() {
  const equation = state.mechanism.equations[state.activeEquation];
  const target = `${seriesNames[state.activeEquation]}(t)`;
  $("active-tree-target").textContent = target;
  $("active-binding-target").textContent = target;
  renderTree(equation.tree);
  renderBindings(equation.binding);
}

function renderDependencyMap(mechanism) {
  const map = $("dependency-map");
  map.replaceChildren();
  const corner = document.createElement("div");
  corner.className = "dependency-header";
  corner.textContent = "alvo ↓ fonte →";
  map.appendChild(corner);
  seriesNames.forEach((name) => {
    const header = document.createElement("div");
    header.className = "dependency-header";
    header.textContent = name;
    map.appendChild(header);
  });

  mechanism.dependency_lags.forEach((sourceLags, target) => {
    const targetLabel = document.createElement("div");
    targetLabel.className = "dependency-target";
    targetLabel.textContent = `${seriesNames[target]}(t)`;
    map.appendChild(targetLabel);
    sourceLags.forEach((lags, source) => {
      const cell = document.createElement("div");
      cell.className = `dependency-cell${source === target ? "" : " cross"}`;
      const relation = document.createElement("strong");
      relation.textContent = `${seriesNames[source]} → ${seriesNames[target]}`;
      const pills = document.createElement("div");
      pills.className = "lag-pills";
      if (lags.length) {
        lags.forEach((lag) => {
          const pill = document.createElement("span");
          pill.className = "lag-pill";
          pill.textContent = `t−${lag}`;
          pills.appendChild(pill);
        });
      } else {
        const empty = document.createElement("span");
        empty.className = "dependency-empty";
        empty.textContent = "—";
        pills.appendChild(empty);
      }
      cell.append(relation, pills);
      map.appendChild(cell);
    });
  });
}

function nodeLabel(node) {
  if (node.type === "TANH") return ["tanh"];
  if (node.type === "ADD") return ["+"];
  if (node.type === "MULTIPLY") return ["×"];
  if (node.type === "TIME") return ["t"];
  if (node.type === "PARAMETER") return [node.slot, Number(node.value).toPrecision(3)];
  return [`x${node.series}(t−${node.lag})`, node.slot];
}

function renderTree(root) {
  const stage = $("tree-visualization");
  stage.replaceChildren();
  const namespace = "http://www.w3.org/2000/svg";
  const width = Math.max(660, stage.clientWidth || 660);
  const levels = [];
  let leafCursor = 0;

  function layout(node, depth = 0) {
    const laidChildren = node.children.map((child) => layout(child, depth + 1));
    const x = laidChildren.length
      ? laidChildren.reduce((sum, child) => sum + child.x, 0) / laidChildren.length
      : leafCursor++;
    const item = { ...node, depth, x, children: laidChildren };
    if (!levels[depth]) levels[depth] = [];
    levels[depth].push(item);
    return item;
  }

  const tree = layout(root);
  const leafCount = Math.max(leafCursor, 1);
  const height = Math.max(260, levels.length * 94);
  const marginX = 60;
  const scaleX = leafCount === 1 ? 0 : (width - marginX * 2) / (leafCount - 1);
  const px = (item) => (leafCount === 1 ? width / 2 : marginX + item.x * scaleX);
  const py = (item) => 44 + item.depth * 88;

  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("height", height);

  function drawEdges(item) {
    item.children.forEach((child) => {
      const line = document.createElementNS(namespace, "line");
      line.setAttribute("x1", px(item));
      line.setAttribute("y1", py(item));
      line.setAttribute("x2", px(child));
      line.setAttribute("y2", py(child));
      line.setAttribute("class", "tree-edge");
      svg.appendChild(line);
      drawEdges(child);
    });
  }

  function drawNodes(item) {
    const group = document.createElementNS(namespace, "g");
    group.setAttribute("class", `tree-node node-${item.type}`);
    group.setAttribute("transform", `translate(${px(item)} ${py(item)})`);
    const circle = document.createElementNS(namespace, "circle");
    circle.setAttribute("r", item.type === "LAG" || item.type === "PARAMETER" ? 29 : 25);
    group.appendChild(circle);
    nodeLabel(item).forEach((label, index, all) => {
      const text = document.createElementNS(namespace, "text");
      text.setAttribute("y", all.length === 1 ? "4" : String(index * 12 - 2));
      text.textContent = label;
      group.appendChild(text);
    });
    svg.appendChild(group);
    item.children.forEach(drawNodes);
  }

  drawEdges(tree);
  drawNodes(tree);
  stage.appendChild(svg);
}

function renderBindings(binding) {
  const rows = [
    ...Object.entries(binding.parameters).map(([slot, value]) => [
      "parâmetro",
      slot,
      Number(value).toPrecision(5),
    ]),
    ...Object.entries(binding.lags).map(([slot, value]) => ["lag", slot, String(value)]),
  ];
  $("binding-table").innerHTML = rows.length
    ? rows
        .map(
          ([kind, slot, value]) =>
            `<div class="binding-row"><span class="binding-kind">${kind}</span><span class="binding-slot">${slot}</span><span class="binding-value">${value}</span></div>`,
        )
        .join("")
    : '<p class="empty">A expressão não possui slots numéricos.</p>';
}

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
  const magnitude = Math.max(...transformed.map(Math.abs)) || 1;
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

function pausePlayback() {
  window.clearInterval(state.playback);
  state.playback = null;
  $("trajectory-play").textContent = "Reproduzir";
  $("trajectory-play").setAttribute("aria-pressed", "false");
}

function updatePlaybackControls() {
  const count = state.trajectories?.trajectories.length || 0;
  $("trajectory-position").textContent = count ? `Trajetória ${state.activeTrajectory + 1} de ${count}` : "Nenhuma trajetória";
  $("trajectory-select").max = String(Math.max(1, count));
  $("trajectory-select").value = String(state.activeTrajectory + 1);
  $("trajectory-select").disabled = !count;
  $("trajectory-play").disabled = count < 2;
  $("trajectory-previous").disabled = !count || state.activeTrajectory === 0;
  $("trajectory-next").disabled = !count || state.activeTrajectory === count - 1;
}

function selectTrajectory(index) {
  if (!state.trajectories) return;
  const count = state.trajectories.trajectories.length;
  state.activeTrajectory = Math.max(0, Math.min(count - 1, Math.round(index)));
  renderTrajectories(state.trajectories);
}

function playTrajectories() {
  pausePlayback();
  if (!state.trajectories || state.trajectories.trajectories.length < 2) return;
  if (state.activeTrajectory === state.trajectories.trajectories.length - 1) selectTrajectory(0);
  $("trajectory-play").textContent = "Pausar";
  $("trajectory-play").setAttribute("aria-pressed", "true");
  state.playback = window.setInterval(() => {
    selectTrajectory(state.activeTrajectory + 1);
    if (state.activeTrajectory === state.trajectories.trajectories.length - 1) pausePlayback();
  }, Number($("trajectory-speed").value));
}

function renderTrajectories(data) {
  drawLineChart(data);
  if ($("ensemble-overview").open) drawHeatmap(data);
  drawPhaseChart(data);
  updatePlaybackControls();
  const trajectory = data.trajectories[state.activeTrajectory];
  const ranges = seriesNames.map(
    (name, index) =>
      `${name} [${Math.min(...trajectory[index]).toPrecision(3)}, ${Math.max(...trajectory[index]).toPrecision(3)}]`,
  );
  $("trajectory-caption").textContent = ranges.join(" · ");
}

function drawLineChart(data) {
  const { context: ctx, width, height } = setupCanvas($("trajectory-chart"));
  const margin = { left: 58, right: 18, top: 20, bottom: 34 };
  const gap = 48;
  const plotWidth = width - margin.left - margin.right;
  const panelHeight = (height - margin.top - margin.bottom - gap) / 2;
  const x = (index) =>
    margin.left + (index / Math.max(data.times.length - 1, 1)) * plotWidth;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px system-ui";

  for (let series = 0; series < 2; series += 1) {
    const panelTop = margin.top + series * (panelHeight + gap);
    const trajectory = data.trajectories[state.activeTrajectory];
    const axis = chartAxis(trajectory[series]);
    const y = (value) => panelTop + (1 - axis.position(value)) * panelHeight;

    ctx.strokeStyle = "#e3e0d6";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#768079";
    for (let tick = 0; tick <= 4; tick += 1) {
      const yy = panelTop + (1 - tick / 4) * panelHeight;
      ctx.beginPath();
      ctx.moveTo(margin.left, yy);
      ctx.lineTo(width - margin.right, yy);
      ctx.stroke();
      ctx.fillText(axis.tick(tick / 4), 3, yy + 3);
    }

    ctx.beginPath();
    trajectory[series].forEach((value, index) =>
      index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)),
    );
    ctx.strokeStyle = seriesColors[series];
    ctx.lineWidth = 3;
    ctx.stroke();

    const startX = x(data.generated_start);
    ctx.beginPath();
    ctx.setLineDash([5, 5]);
    ctx.moveTo(startX, panelTop);
    ctx.lineTo(startX, panelTop + panelHeight);
    ctx.strokeStyle = "#6f7973";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = seriesColors[series];
    ctx.font = "700 12px system-ui";
    ctx.fillText(`${seriesNames[series]}(t)`, margin.left, panelTop - 7);
    ctx.font = "10px system-ui";
    ctx.fillStyle = "#59645e";
    ctx.fillText(
      `t=${data.generated_start}`,
      Math.min(startX + 5, width - 45),
      panelTop + panelHeight + 16,
    );
  }
  ctx.fillStyle = "#59645e";
  ctx.fillText("tempo →", width - 58, height - 8);
}

function heatColor(value, maximumMagnitude) {
  const normalized = Math.min(Math.abs(value) / maximumMagnitude, 1);
  const base = value >= 0 ? [236, 122, 69] : [68, 136, 163];
  const light = [247, 245, 237];
  const mix = 0.12 + normalized * 0.88;
  return `rgb(${base
    .map((channel, index) => Math.round(light[index] * (1 - mix) + channel * mix))
    .join(",")})`;
}

function drawHeatmap(data) {
  const { context: ctx, width, height } = setupCanvas($("heatmap"));
  const margin = { left: 46, right: 8, top: 22, bottom: 24 };
  const gap = 34;
  const rows = data.trajectories.length;
  const columns = data.times.length;
  const blockHeight = (height - margin.top - margin.bottom - gap) / 2;
  const cellWidth = (width - margin.left - margin.right) / columns;
  const cellHeight = blockHeight / rows;
  ctx.clearRect(0, 0, width, height);

  for (let series = 0; series < 2; series += 1) {
    const blockTop = margin.top + series * (blockHeight + gap);
    const maximumMagnitude = Math.max(
      Math.abs(data.summary.minimum[series]),
      Math.abs(data.summary.maximum[series]),
      1e-12,
    );
    data.trajectories.forEach((trajectory, row) =>
      trajectory[series].forEach((value, column) => {
        ctx.fillStyle = heatColor(value, maximumMagnitude);
        ctx.fillRect(
          margin.left + column * cellWidth,
          blockTop + row * cellHeight,
          Math.ceil(cellWidth),
          Math.ceil(cellHeight),
        );
      }),
    );
    const startX = margin.left + data.generated_start * cellWidth;
    ctx.strokeStyle = "#17221d";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(startX, blockTop);
    ctx.lineTo(startX, blockTop + blockHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = seriesColors[series];
    ctx.font = "700 11px system-ui";
    ctx.fillText(seriesNames[series], 7, blockTop + 12);
    ctx.fillStyle = "#66716a";
    ctx.font = "9px system-ui";
    ctx.fillText(`${maximumMagnitude.toPrecision(2)} max`, 7, blockTop + 25);
  }
  ctx.fillStyle = "#66716a";
  ctx.font = "9px system-ui";
  ctx.fillText("tempo →", width - 48, height - 7);
}

function drawPhaseChart(data) {
  const { context: ctx, width, height } = setupCanvas($("phase-chart"));
  const margin = { left: 62, right: 24, top: 18, bottom: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const trajectory = data.trajectories[state.activeTrajectory];
  const xAxis = chartAxis(trajectory[0]);
  const yAxis = chartAxis(trajectory[1]);
  const x = (value) => margin.left + xAxis.position(value) * plotWidth;
  const y = (value) => margin.top + (1 - yAxis.position(value)) * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px system-ui";
  for (let tick = 0; tick <= 5; tick += 1) {
    const xx = margin.left + (tick / 5) * plotWidth;
    const yy = margin.top + (1 - tick / 5) * plotHeight;
    ctx.strokeStyle = "#e3e0d6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xx, margin.top);
    ctx.lineTo(xx, height - margin.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin.left, yy);
    ctx.lineTo(width - margin.right, yy);
    ctx.stroke();
    ctx.fillStyle = "#768079";
    ctx.fillText(xAxis.tick(tick / 5), xx - 12, height - margin.bottom + 18);
    ctx.fillText(yAxis.tick(tick / 5), 4, yy + 3);
  }

  const start = Math.min(data.generated_start, data.times.length - 1);
  ctx.beginPath();
  for (let index = 0; index < data.times.length; index += 1) {
    const xx = x(trajectory[0][index]);
    const yy = y(trajectory[1][index]);
    if (index === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  }
  ctx.strokeStyle = seriesColors[0];
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(x(trajectory[0][start]), y(trajectory[1][start]), 2.4, 0, Math.PI * 2);
  ctx.fillStyle = "#18221e";
  ctx.fill();

  ctx.fillStyle = "#315f73";
  ctx.font = "700 11px system-ui";
  ctx.fillText("x₁(t)", 7, margin.top + 2);
  ctx.fillStyle = "#1f6b55";
  ctx.fillText("x₀(t) →", width - 66, height - 10);
}

function clearCanvases() {
  pausePlayback();
  state.activeTrajectory = 0;
  updatePlaybackControls();
  ["trajectory-chart", "heatmap", "phase-chart"].forEach((id) => {
    const canvas = $(id);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  });
  $("trajectory-caption").textContent =
    "Cada painel usa sua própria escala; o tempo permanece alinhado.";
}

$("innovation").addEventListener("input", () => {
  $("innovation-value").textContent = numberValue("innovation").toFixed(2);
  if (!["none", "coefficient_gate"].includes($("stability-method").value)) {
    $("generate-series").disabled = true;
    setStatus("A inovação mudou. Gere duas equações para atualizar o certificado.");
  }
});
function updateStabilityControls() {
  const method = $("stability-method").value;
  const free = ["none", "coefficient_gate"].includes(method);
  const linear = ["heaps", "stationary_var1"].includes(method);
  const descriptions = {
    none: "Sem restrição de estabilidade ou corte por magnitude. Valores não finitos interrompem a geração.",
    coefficient_gate: "Adiciona coeficientes aos lags; pode reduzir explosões, mas não certifica estabilidade.",
    contraction: "Duas recorrências independentes, certificadas em [−1, 1], com inovação uniforme limitada. Não usa o terminal tempo.",
    shao_wu: $("shao-wu-mode").value === "product_lags"
      ? "Tanh nos fatores dos produtos de lags. Verifica contração global e esquecimento do histórico sob o mesmo ruído. Termos lineares ficam livres."
      : "Tanh após o produto completo. Verifica um limite uniforme do segundo momento da amplitude; a contração global requer análise adicional. Termos lineares ficam livres.",
    shao_wu_all_tanh: "Cada lag passa por tanh antes da equação. Permite produtos de lags com ganho global certificado, mas também satura dependências lineares.",
    shao_wu_product_lags: "Tanh apenas nos lags de produtos não lineares; termos com um único lag continuam lineares. Produtos de até quatro lags podem ser amostrados, conforme a configuração de operadores.",
    heaps: "VAR(p) linear com histórico gaussiano estacionário. Lag mínimo/máximo escolhem a ordem p; os operadores da AST livre não se aplicam.",
    stationary_var1: "VAR(1) linear com histórico gaussiano estacionário. Usa somente lag 1; os operadores da AST livre não se aplicam.",
  };
  $("stability-description").textContent = descriptions[method];
  $("shao-wu-mode-field").hidden = method !== "shao_wu";
  ["initial-min", "initial-max"].forEach((id) => { $(id).disabled = !free; });
  ["min-operators", "max-operators", "p-parameter", "p-lag", "p-add"].forEach((id) => { $(id).disabled = linear; });
  $("require-lag").disabled = !free;
  ["p-time", "require-time"].forEach((id) => { $(id).disabled = !free; });
  ["min-lag", "max-lag"].forEach((id) => { $(id).disabled = method === "stationary_var1"; });
  if (!free) { $("p-time").value = "0"; $("require-time").checked = false; $("require-lag").checked = true; }
  const positiveNoise = ["heaps", "stationary_var1", "shao_wu", "shao_wu_all_tanh", "shao_wu_product_lags"].includes(method);
  $("innovation").min = positiveNoise ? "0.01" : "0";
  $("innovation").max = method === "contraction" ? "0.99" : "1";
  if (positiveNoise && numberValue("innovation") === 0) $("innovation").value = "0.05";
  if (method === "contraction" && numberValue("innovation") >= 1) $("innovation").value = "0.05";
  $("innovation-label").textContent = method === "contraction" ? "Limite da inovação uniforme" : "Desvio da inovação";
  $("innovation-value").textContent = numberValue("innovation").toFixed(2);
}
$("stability-method").addEventListener("change", () => {
  pausePlayback();
  state.mechanism = null;
  state.trajectories = null;
  clearCanvases();
  ["equation-grid", "equation-tabs", "tree-visualization", "binding-table", "dependency-map", "tree-metrics"].forEach((id) => $(id).replaceChildren());
  $("stability-summary").textContent = "";
  $("generate-series").disabled = true;
  updateStabilityControls();
  setStatus("Método alterado. Clique em Gerar duas equações.");
});
$("shao-wu-mode").addEventListener("change", () => {
  $("stability-method").dispatchEvent(new Event("change"));
});
$("randomize-tree-seed").addEventListener("click", () => {
  $("tree-seed").value = randomSeed();
});
$("randomize-series-seed").addEventListener("click", () => {
  $("series-seed").value = randomSeed();
});
$("generate-tree").addEventListener("click", generateMechanism);
$("generate-series").addEventListener("click", generateTrajectories);
$("trajectory-previous").addEventListener("click", () => { pausePlayback(); selectTrajectory(state.activeTrajectory - 1); });
$("trajectory-next").addEventListener("click", () => { pausePlayback(); selectTrajectory(state.activeTrajectory + 1); });
$("trajectory-select").addEventListener("input", () => { pausePlayback(); selectTrajectory(numberValue("trajectory-select") - 1); });
$("trajectory-play").addEventListener("click", () => { if (state.playback !== null) pausePlayback(); else playTrajectories(); });
$("trajectory-speed").addEventListener("change", () => { if (state.playback !== null) playTrajectories(); });
$("trajectory-scale").addEventListener("change", () => { if (state.trajectories) renderTrajectories(state.trajectories); });
$("ensemble-overview").addEventListener("toggle", () => { if ($("ensemble-overview").open && state.trajectories) drawHeatmap(state.trajectories); });
document.addEventListener("visibilitychange", () => { if (document.hidden) pausePlayback(); });
window.addEventListener("resize", () => {
  if (state.mechanism) renderActiveEquation();
  if (state.trajectories) renderTrajectories(state.trajectories);
});

updateStabilityControls();
generateMechanism().then(generateTrajectories);
