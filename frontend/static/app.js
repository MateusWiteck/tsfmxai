const $ = (id) => document.getElementById(id);

const state = { mechanism: null, trajectories: null };

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
  setStatus("Sorteando e canonizando a árvore…");
  try {
    state.mechanism = await postJson("/api/mechanisms", mechanismRequest());
    state.trajectories = null;
    renderMechanism(state.mechanism);
    clearCanvases();
    $("generate-series").disabled = false;
    setStatus(`Árvore reproduzível criada com seed ${state.mechanism.seed}.`);
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
  setStatus("Executando a recorrência para cada trajetória…");
  try {
    state.trajectories = await postJson("/api/trajectories", trajectoryRequest());
    renderTrajectories(state.trajectories);
    setStatus(`${state.trajectories.trajectories.length} trajetórias geradas com a árvore fixa.`);
    clearStatusSoon();
  } catch (error) {
    setStatus(`${error.message} Tente outra árvore, menos passos ou condições iniciais menores.`, true);
  } finally {
    button.disabled = false;
  }
}

function renderMechanism(mechanism) {
  $("skeleton-formula").textContent = mechanism.skeleton_formula;
  $("bound-formula").textContent = `x₀(t) = ${mechanism.formula}`;
  const labels = {
    operators: "operadores",
    leaves: "folhas",
    nodes: "nós",
    depth: "profundidade",
    required_history: "histórico inicial",
  };
  $("tree-metrics").innerHTML = Object.entries(mechanism.summary)
    .map(([key, value]) => `<div class="metric"><strong>${value}</strong><span>${labels[key]}</span></div>`)
    .join("");
  renderTree(mechanism.tree);
  renderBindings(mechanism.binding);
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
  const px = (item) => leafCount === 1 ? width / 2 : marginX + item.x * scaleX;
  const py = (item) => 44 + item.depth * 88;

  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("height", height);

  function drawEdges(item) {
    item.children.forEach((child) => {
      const line = document.createElementNS(namespace, "line");
      line.setAttribute("x1", px(item)); line.setAttribute("y1", py(item));
      line.setAttribute("x2", px(child)); line.setAttribute("y2", py(child));
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
    ...Object.entries(binding.parameters).map(([slot, value]) => ["parâmetro", slot, Number(value).toPrecision(5)]),
    ...Object.entries(binding.lags).map(([slot, value]) => ["lag", slot, String(value)]),
  ];
  $("binding-table").innerHTML = rows.length
    ? rows.map(([kind, slot, value]) => `<div class="binding-row"><span class="binding-kind">${kind}</span><span class="binding-slot">${slot}</span><span class="binding-value">${value}</span></div>`).join("")
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

function renderTrajectories(data) {
  drawLineChart(data);
  drawHeatmap(data);
  const count = data.trajectories.length;
  $("trajectory-caption").textContent = `${count} trajetórias · intervalo total [${data.summary.minimum.toPrecision(3)}, ${data.summary.maximum.toPrecision(3)}]`;
}

function drawLineChart(data) {
  const { context: ctx, width, height } = setupCanvas($("trajectory-chart"));
  const margin = { left: 54, right: 18, top: 16, bottom: 34 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const values = data.trajectories.flat();
  let minimum = Math.min(...values), maximum = Math.max(...values);
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const padding = (maximum - minimum) * 0.08;
  minimum -= padding; maximum += padding;
  const x = (index) => margin.left + index / Math.max(data.times.length - 1, 1) * plotW;
  const y = (value) => margin.top + (maximum - value) / (maximum - minimum) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#e3e0d6"; ctx.lineWidth = 1;
  ctx.fillStyle = "#768079"; ctx.font = "10px system-ui";
  for (let i = 0; i <= 4; i++) {
    const value = minimum + (maximum - minimum) * i / 4;
    const yy = y(value);
    ctx.beginPath(); ctx.moveTo(margin.left, yy); ctx.lineTo(width - margin.right, yy); ctx.stroke();
    ctx.fillText(value.toPrecision(3), 3, yy + 3);
  }

  data.trajectories.forEach((trajectory) => {
    ctx.beginPath();
    trajectory.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)));
    ctx.strokeStyle = "rgba(31, 107, 85, 0.22)"; ctx.lineWidth = 1.2; ctx.stroke();
  });

  ctx.beginPath();
  data.summary.mean.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)));
  ctx.strokeStyle = "#ec7a45"; ctx.lineWidth = 2.8; ctx.stroke();

  const startX = x(data.generated_start);
  ctx.beginPath(); ctx.setLineDash([5, 5]); ctx.moveTo(startX, margin.top); ctx.lineTo(startX, height - margin.bottom);
  ctx.strokeStyle = "#6f7973"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#59645e"; ctx.fillText(`t=${data.generated_start}`, Math.min(startX + 5, width - 45), height - 13);
  ctx.fillText("tempo", width - 48, height - 13);
}

function heatColor(value, maximumMagnitude) {
  const normalized = Math.min(Math.abs(value) / maximumMagnitude, 1);
  const base = value >= 0 ? [236, 122, 69] : [68, 136, 163];
  const light = [247, 245, 237];
  const mix = 0.12 + normalized * 0.88;
  return `rgb(${base.map((channel, index) => Math.round(light[index] * (1 - mix) + channel * mix)).join(",")})`;
}

function drawHeatmap(data) {
  const { context: ctx, width, height } = setupCanvas($("heatmap"));
  const margin = { left: 34, right: 8, top: 8, bottom: 24 };
  const rows = data.trajectories.length, columns = data.times.length;
  const cellW = (width - margin.left - margin.right) / columns;
  const cellH = (height - margin.top - margin.bottom) / rows;
  const maximumMagnitude = Math.max(Math.abs(data.summary.minimum), Math.abs(data.summary.maximum), 1e-12);
  ctx.clearRect(0, 0, width, height);
  data.trajectories.forEach((trajectory, row) => trajectory.forEach((value, column) => {
    ctx.fillStyle = heatColor(value, maximumMagnitude);
    ctx.fillRect(margin.left + column * cellW, margin.top + row * cellH, Math.ceil(cellW), Math.ceil(cellH));
  }));
  ctx.fillStyle = "#66716a"; ctx.font = "9px system-ui";
  ctx.fillText("traj.", 2, 12); ctx.fillText("tempo →", width - 48, height - 7);
  const startX = margin.left + data.generated_start * cellW;
  ctx.strokeStyle = "#17221d"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(startX, margin.top); ctx.lineTo(startX, height - margin.bottom); ctx.stroke(); ctx.setLineDash([]);
}

function clearCanvases() {
  ["trajectory-chart", "heatmap"].forEach((id) => {
    const canvas = $(id); canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  });
  $("trajectory-caption").textContent = "Condições iniciais aparecem antes da linha tracejada.";
}

$("innovation").addEventListener("input", () => {
  $("innovation-value").textContent = numberValue("innovation").toFixed(2);
});
$("randomize-tree-seed").addEventListener("click", () => { $("tree-seed").value = randomSeed(); });
$("randomize-series-seed").addEventListener("click", () => { $("series-seed").value = randomSeed(); });
$("generate-tree").addEventListener("click", generateMechanism);
$("generate-series").addEventListener("click", generateTrajectories);
window.addEventListener("resize", () => {
  if (state.mechanism) renderTree(state.mechanism.tree);
  if (state.trajectories) renderTrajectories(state.trajectories);
});

generateMechanism().then(generateTrajectories);
