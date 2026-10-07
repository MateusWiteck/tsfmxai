const $ = (id) => document.getElementById(id);

const state = { mechanism: null, trajectories: null, activeEquation: 0 };
const seriesNames = ["x₀", "x₁"];
const seriesColors = ["#1f6b55", "#4488a3"];
const seriesFaintColors = ["rgba(31, 107, 85, 0.20)", "rgba(68, 136, 163, 0.20)"];

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
  const button = $("generate-tree");
  button.disabled = true;
  $("generate-series").disabled = true;
  setStatus("Sorteando e canonizando as duas equações…");
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
  const button = $("generate-series");
  button.disabled = true;
  setStatus("Executando as duas recorrências de forma síncrona…");
  try {
    state.trajectories = await postJson("/api/trajectories", trajectoryRequest());
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

function seriesValues(data, series) {
  return data.trajectories.flatMap((trajectory) => trajectory[series]);
}

function renderTrajectories(data) {
  drawLineChart(data);
  drawHeatmap(data);
  drawPhaseChart(data);
  const count = data.trajectories.length;
  const ranges = seriesNames.map(
    (name, index) =>
      `${name} [${Number(data.summary.minimum[index]).toPrecision(3)}, ${Number(data.summary.maximum[index]).toPrecision(3)}]`,
  );
  $("trajectory-caption").textContent = `${count} trajetórias · ${ranges.join(" · ")}`;
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
    const [minimum, maximum] = paddedExtent(seriesValues(data, series));
    const y = (value) => panelTop + ((maximum - value) / (maximum - minimum)) * panelHeight;

    ctx.strokeStyle = "#e3e0d6";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#768079";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = minimum + ((maximum - minimum) * tick) / 4;
      const yy = y(value);
      ctx.beginPath();
      ctx.moveTo(margin.left, yy);
      ctx.lineTo(width - margin.right, yy);
      ctx.stroke();
      ctx.fillText(value.toPrecision(3), 3, yy + 3);
    }

    data.trajectories.forEach((trajectory) => {
      ctx.beginPath();
      trajectory[series].forEach((value, index) =>
        index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)),
      );
      ctx.strokeStyle = seriesFaintColors[series];
      ctx.lineWidth = 1.15;
      ctx.stroke();
    });

    ctx.beginPath();
    data.summary.mean[series].forEach((value, index) =>
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
  const [xMinimum, xMaximum] = paddedExtent(seriesValues(data, 0));
  const [yMinimum, yMaximum] = paddedExtent(seriesValues(data, 1));
  const x = (value) => margin.left + ((value - xMinimum) / (xMaximum - xMinimum)) * plotWidth;
  const y = (value) => margin.top + ((yMaximum - value) / (yMaximum - yMinimum)) * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px system-ui";
  for (let tick = 0; tick <= 5; tick += 1) {
    const xValue = xMinimum + ((xMaximum - xMinimum) * tick) / 5;
    const yValue = yMinimum + ((yMaximum - yMinimum) * tick) / 5;
    const xx = x(xValue);
    const yy = y(yValue);
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
    ctx.fillText(xValue.toPrecision(3), xx - 12, height - margin.bottom + 18);
    ctx.fillText(yValue.toPrecision(3), 4, yy + 3);
  }

  const start = Math.min(data.generated_start, data.times.length - 1);
  data.trajectories.forEach((trajectory) => {
    ctx.beginPath();
    for (let index = Math.max(0, start - 1); index < data.times.length; index += 1) {
      const xx = x(trajectory[0][index]);
      const yy = y(trajectory[1][index]);
      if (index === Math.max(0, start - 1)) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    }
    ctx.strokeStyle = "rgba(31, 107, 85, 0.22)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x(trajectory[0][start]), y(trajectory[1][start]), 2.4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(24, 34, 30, 0.45)";
    ctx.fill();
  });

  ctx.beginPath();
  for (let index = Math.max(0, start - 1); index < data.times.length; index += 1) {
    const xx = x(data.summary.mean[0][index]);
    const yy = y(data.summary.mean[1][index]);
    if (index === Math.max(0, start - 1)) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  }
  ctx.strokeStyle = "#ec7a45";
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = "#315f73";
  ctx.font = "700 11px system-ui";
  ctx.fillText("x₁(t)", 7, margin.top + 2);
  ctx.fillStyle = "#1f6b55";
  ctx.fillText("x₀(t) →", width - 66, height - 10);
}

function clearCanvases() {
  ["trajectory-chart", "heatmap", "phase-chart"].forEach((id) => {
    const canvas = $(id);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  });
  $("trajectory-caption").textContent =
    "Cada painel usa sua própria escala; o tempo permanece alinhado.";
}

$("innovation").addEventListener("input", () => {
  $("innovation-value").textContent = numberValue("innovation").toFixed(2);
});
$("randomize-tree-seed").addEventListener("click", () => {
  $("tree-seed").value = randomSeed();
});
$("randomize-series-seed").addEventListener("click", () => {
  $("series-seed").value = randomSeed();
});
$("generate-tree").addEventListener("click", generateMechanism);
$("generate-series").addEventListener("click", generateTrajectories);
window.addEventListener("resize", () => {
  if (state.mechanism) renderActiveEquation();
  if (state.trajectories) renderTrajectories(state.trajectories);
});

generateMechanism().then(generateTrajectories);
