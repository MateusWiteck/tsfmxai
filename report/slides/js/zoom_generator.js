/* Zoom 1: the generator.
 *
 * One real mechanism, exported from the project library by
 * scripts/export_generator_demo.py (CombinatorialTreeSampler, selective tanh
 * protection, certified Shao-Wu scale, simulated trajectory). Nothing on this
 * slide is illustrative except the choice of an example with five operators.
 */
(function () {
  const D = window.GEN_DATA, Z = window.ZC;
  if (!D || !Z) return;
  const { col, seg, ease, text, wrap, box, arrow, fmt, lagName } = Z;
  const MINUS = "−";
  const W = 1200, H = 500;
  const num = (s) => Number(s).toLocaleString("en-US");

  const tree = Z.collapse(D.protected_tree);          // same shape as the raw tree; wrap = tanh on a lag
  const rawPretty = Z.pretty(D.raw_tree);
  const layCache = {};
  const lay = (b) => {
    const k = [b.x, b.y, b.w, b.h].join();
    return layCache[k] || (layCache[k] = Z.layout(tree, b));
  };
  const STAGES = ["① Draw", "② Bind", "③ Stabilize", "④ Run", "⑤ Keep"];

  function leafLabel(nd, mode) {
    const n = nd.n;
    if (n.t === "add") return "+";
    if (n.t === "mul") return "×";
    if (mode === "blank") return "?";
    if (mode === "slots") return n.t === "par" ? n.slot : `x(t${MINUS}${n.slot})`;
    const base = n.t === "par" ? fmt(n.v, 3) : lagName(n.lag);
    return n.wrap && mode === "wrap" ? `tanh(${base})` : base;
  }
  function leafStyle(nd, bound, wrapped) {
    const n = nd.n;
    if (n.t === "add" || n.t === "mul") return { stroke: col.ink };
    if (n.wrap && wrapped) return { fill: col.osoft, stroke: col.ours, color: col.ours, lw: 2.4 };
    if (n.t === "par") return bound ? { fill: col.soft, stroke: col.blue, color: col.blue } : { fill: col.soft, stroke: col.line };
    return bound ? { fill: col.tsoft, stroke: col.teal, color: col.teal } : {};
  }

  function pill(ctx, x, y, w, label, c) {
    Z.rrect(ctx, x, y, w, 34, 17);
    ctx.fillStyle = c;
    ctx.fill();
    text(ctx, label, x + w / 2, y + 23, 17, "#fff", { align: "center", weight: 700 });
  }

  function cards(ctx, p) {
    const info = [
      ["Draw", "a random equation: a tree of + and ×"],
      ["Bind", "give every coefficient and lag a value"],
      ["Stabilize", "make sure the series cannot blow up"],
      ["Run", "start from a history, add noise, iterate"],
      ["Keep", "store the series and its equation"],
    ];
    info.forEach((c, i) => {
      const a = ease(seg(p, i * 0.12, i * 0.12 + 0.25)), sw = i === 2;
      const x = 20 + i * 236, y = 62;
      ctx.globalAlpha = a;
      box(ctx, x, y, 208, 170, { fill: sw ? col.osoft : col.panel, stroke: sw ? col.ours : col.blue, lw: sw ? 3 : 2.4 });
      ctx.beginPath();
      ctx.arc(x + 34, y + 36, 19, 0, Math.PI * 2);
      ctx.fillStyle = sw ? col.ours : col.blue;
      ctx.fill();
      text(ctx, String(i + 1), x + 34, y + 44, 21, "#fff", { align: "center", weight: 700 });
      text(ctx, c[0], x + 64, y + 44, 23, col.ink, { weight: 700 });
      wrap(ctx, c[1], x + 18, y + 90, 176, 25, 19, col.muted);
      if (i < 4) arrow(ctx, x + 212, y + 85, x + 232, y + 85, col.muted, 2);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 0.85));
    text(ctx, "③ is a switch: for each series, one of four constructions is drawn at random, so the data mixes different kinds of stability.", 20, 272, 19, col.ink, { weight: 700 });
    [["Plain tree", 20], ["Jacobian", 190], ["Shao–Wu", 360], ["Heaps VAR", 530]].forEach(([n, x]) => pill(ctx, x, 292, 150, n, n === "Plain tree" ? col.muted : col.ours));
    wrap(ctx, "Each is a black box with the same contract: random numbers in, a recurrence and its series out. Series that diverge are discarded.", 700, 308, 480, 22, 16, col.muted);
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    box(ctx, 20, 372, 1160, 84, { fill: col.osoft, stroke: col.ours });
    text(ctx, "Output: a series and the exact equation that made it.", 44, 408, 24, col.ink, { weight: 600 });
    text(ctx, "Every stage is stored with the mechanism, so the ground truth is known without estimating anything.", 44, 438, 18, col.muted);
    ctx.globalAlpha = 1;
  }

  function step1(ctx, p) {
    // left: how many trees are there?
    box(ctx, 20, 60, 480, 380);
    text(ctx, "How many valid trees?", 40, 96, 22, col.ink, { weight: 700 });
    text(ctx, num(D.structures_total), 40, 150, 40, col.blue, { weight: 700 });
    wrap(ctx, "1–6 operators, lags 1–12, one series, every product of at most 4 lag factors.", 40, 184, 440, 24, 18, col.muted);
    const total = Number(D.structures_total);
    const rows = [
      ["6 operators", Number(D.size_counts["6"]) / total],
      ["5 operators", Number(D.size_counts["5"]) / total],
      ["1–4 operators", (Number(D.size_counts["4"]) + Number(D.size_counts["3"]) + Number(D.size_counts["2"]) + Number(D.size_counts["1"])) / total],
    ];
    rows.forEach((r, i) => {
      const y = 262 + i * 44, a = seg(p, 0.1 + i * 0.12, 0.35 + i * 0.12);
      text(ctx, r[0], 40, y + 18, 18, col.ink);
      ctx.fillStyle = i === 1 ? col.ours : col.blue;
      ctx.fillRect(170, y, Math.max(2, 230 * r[1] * a), 24);
      const pct = r[1] >= 0.1 ? (r[1] * 100).toFixed(1) : r[1] >= 0.001 ? (r[1] * 100).toFixed(2) : (r[1] * 100).toFixed(3);
      text(ctx, `${pct}%`, 174 + 230 * r[1] * a, y + 19, 17, col.muted);
    });
    wrap(ctx, "Draw one uniformly: almost every tree has 6 operators. This example has 5, so it fits on a slide.", 40, 410, 440, 22, 16, col.muted);

    // right: the tree grows from the root
    const L = lay({ x: 560, y: 110, w: 600, h: 290 });
    const reveal = Math.min(1, p / 0.7);
    const blank = p < 0.75;
    text(ctx, `this draw: ${D.operators} operators, ${D.operators + 1} leaves`, 560, 76, 20, col.ink, { weight: 600 });
    Z.drawTree(ctx, L, {
      alpha: (nd) => seg(reveal, nd.depth * 0.18, nd.depth * 0.18 + 0.2),
      label: (nd) => leafLabel(nd, blank ? "blank" : "slots"),
      style: (nd) => leafStyle(nd, false, false),
    });
    const a = seg(p, 0.8, 1);
    ctx.globalAlpha = a;
    text(ctx, "P = a coefficient slot,  x(t−L) = a lag slot. The numbers come next.", 560, 436, 17, col.muted);
    text(ctx, "Jacobian and Shao–Wu draw trees like this one; Heaps VAR draws a linear model instead.", 560, 462, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  function step2(ctx, p) {
    const L = lay({ x: 40, y: 90, w: 620, h: 300 });
    Z.drawTree(ctx, L, {
      alpha: () => 1,
      label: (nd) => {
        const bound = nd.n.t === "par" || nd.n.t === "lag" ? p > 0.1 + (nd.leaf / L.leaves) * 0.7 : true;
        return leafLabel(nd, bound ? "values" : "slots");
      },
      style: (nd) => leafStyle(nd, p > 0.1 + ((nd.leaf || 0) / L.leaves) * 0.7, false),
    });
    box(ctx, 700, 60, 480, 380);
    text(ctx, "Where the numbers come from", 722, 98, 22, col.ink, { weight: 700 });
    text(ctx, "P  coefficient", 722, 142, 20, col.blue, { weight: 700 });
    wrap(ctx, "magnitude uniform in [0.05, 0.90], random sign", 722, 168, 430, 24, 18, col.ink);
    text(ctx, "L  lag", 722, 224, 20, col.teal, { weight: 700 });
    wrap(ctx, "integer uniform in {1, …, 12}", 722, 250, 430, 24, 18, col.ink);
    const a = ease(seg(p, 0.7, 1));
    ctx.globalAlpha = a;
    text(ctx, "The equation the tree computes (not yet stabilized)", 722, 312, 18, col.muted, { weight: 600 });
    wrap(ctx, `x(t) = ${rawPretty} + ε`, 722, 342, 440, 28, 21, col.ink, { family: Z.MATH });
    ctx.globalAlpha = 1;
  }

  // ---- why stabilise: a feedback loop fades or explodes depending on its gain ----
  function lcg(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const loopRun = (g) => {
    const r = lcg(3), ys = [1];
    for (let t = 1; t < 31; t++) ys.push(g * ys[t - 1] + 0.05 * (2 * r() - 1));
    return ys;
  };
  function problem(ctx, p) {
    box(ctx, 20, 60, 700, 388);
    text(ctx, "A recurrence feeds its past back into itself", 40, 96, 22, col.ink, { weight: 700 });
    text(ctx, "x(t) = g · x(t−1) + ε(t)", 40, 142, 30, col.ink, { family: Z.MATH });
    text(ctx, "g = loop gain, as in an amplifier or a control loop", 40, 170, 17, col.muted);
    [[0.7, col.teal, "g = 0.7: the past fades"], [1.15, col.ours, "g = 1.15: the past is amplified"]].forEach(([g, c, title], k) => {
      const ys = loopRun(g), x0 = 40 + k * 338, w = 318, top = 214, h = 170, hi = Math.max(...ys.map(Math.abs));
      const n = Math.max(2, Math.floor(ys.length * seg(p, k * 0.1, 0.8 + k * 0.1)));
      ctx.strokeStyle = col.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, top + h); ctx.lineTo(x0 + w, top + h); ctx.stroke();
      ctx.strokeStyle = c; ctx.lineWidth = 2.6; ctx.beginPath();
      ys.slice(0, n).forEach((v, i) => { const X = x0 + (i / 30) * w, Y = top + h - (Math.abs(v) / hi) * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
      text(ctx, title, x0, top - 12, 17, c, { weight: 700 });
      text(ctx, `peak ${hi.toFixed(hi < 10 ? 1 : 0)} within 30 steps`, x0, top + h + 24, 15, col.muted);
    });
    text(ctx, "illustrative: |x| shown, same small noise in both", 40, 438, 13, col.muted);
    box(ctx, 740, 60, 440, 388);
    text(ctx, "Random equations hide their gain", 760, 96, 22, col.ink, { weight: 700 });
    wrap(ctx, "Ours have up to 12 lags and products of lags, so there is no single g to read off.", 760, 128, 400, 25, 18, col.ink);
    ctx.globalAlpha = ease(seg(p, 0.5, 0.85));
    text(ctx, "A plain random tree, nothing added:", 760, 222, 17, col.muted);
    text(ctx, "47.7 %", 760, 280, 54, col.ours, { weight: 700 });
    wrap(ctx, "of 128 mechanisms cross |x| > 100 within 384 steps (final-method experiment).", 760, 314, 400, 24, 17, col.ink);
    text(ctx, "Next: four constructions.", 760, 420, 18, col.blue, { weight: 700 });
    ctx.globalAlpha = 1;
  }

  // ---- hub: the three constructions as black boxes (click one for its own detail view) ----
  const METHODS = [
    { key: null, name: "Plain tree", fam: "Nonlinear tree, nothing added. Gaussian noise; uniform start.",
      think: "No safeguard. It can produce any dynamics, including explosive ones.",
      guar: "None. Paths that turn non-finite or cross the diagnostic threshold are counted as failures.", expl: "47.7 %", near: "14.8 %", rate: "3 066", ref: true },
    { key: "jac", name: "Jacobian", fam: "Nonlinear tree, products allowed. Bounded uniform noise; start inside the box.",
      think: "A funnel with guard rails: states stay in a box and get closer each step.",
      guar: "Lipschitz bound below 1 on the box (target 0.9).", expl: "0 %", near: "37.5 %", rate: "1 558" },
    { key: "sw", name: "Shao–Wu", fam: "Nonlinear tree, tanh on lag products. Gaussian noise; uniform start.",
      think: "An echo that dies out: past influence adds up to less than 1, so it fades.",
      guar: "Lag sensitivities add up below 1 (target 0.76).", expl: "0 %", near: "25.0 %", rate: "1 427" },
    { key: "hvar", name: "Heaps VAR", fam: "Linear model only. Gaussian noise; stationary start.",
      think: "A stable filter built to order: random numbers map to coefficients that cannot be unstable.",
      guar: "All poles inside the unit circle; no rejection needed.", expl: "0 %", near: "44.5 %", rate: "349" },
  ];
  const HUB = { x0: 20, w: 272, gap: 24, y: 62, h: 392 };
  const hubRect = (i) => ({ x: HUB.x0 + i * (HUB.w + HUB.gap), y: HUB.y, w: HUB.w, h: HUB.h });
  function hub(ctx, p) {
    METHODS.forEach((m, i) => {
      const r = hubRect(i), ref = m.ref;
      ctx.globalAlpha = ease(seg(p, i * 0.1, i * 0.1 + 0.3));
      box(ctx, r.x, r.y, r.w, r.h, { fill: ref ? col.grey : col.panel, stroke: ref ? col.muted : col.blue, lw: 2.4, dash: ref ? [6, 5] : [] });
      text(ctx, m.name, r.x + 16, r.y + 36, 23, ref ? col.muted : col.ink, { weight: 700 });
      if (!ref) {
        box(ctx, r.x + r.w - 124, r.y + 14, 108, 28, { r: 14, fill: col.blue, stroke: col.blue });
        text(ctx, "how it works ▸", r.x + r.w - 70, r.y + 33, 13, "#fff", { align: "center", weight: 700 });
      }
      wrap(ctx, m.fam, r.x + 16, r.y + 64, 240, 18, 14, col.muted);
      text(ctx, "THINK OF IT AS", r.x + 16, r.y + 136, 12.5, ref ? col.muted : col.blue, { weight: 700 });
      wrap(ctx, m.think, r.x + 16, r.y + 158, 240, 21, 15.5, col.ink);
      text(ctx, "GUARANTEE", r.x + 16, r.y + 262, 12.5, ref ? col.muted : col.blue, { weight: 700 });
      wrap(ctx, m.guar, r.x + 16, r.y + 284, 240, 21, 15.5, col.ink);
      text(ctx, `${m.expl} explode`, r.x + 16, r.y + 354, 17, ref ? col.ours : col.teal, { weight: 700 });
      text(ctx, `${m.near} near zero · ${m.rate} /s`, r.x + 16, r.y + 376, 14, col.muted);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "The plain tree is kept as one option: 47.7 % of its sampled paths crossed the threshold. Click a card for details.", 20, 482, 17, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  const SHOWN = 48;
  function step5(ctx, p) {
    const ys = D.trajectory.slice(0, SHOWN);
    const hist = D.history_len;
    // the first `order` values are the starting history; the rest are generated
    const lo = Math.min(...ys) - 0.1, hi = Math.max(...ys) + 0.1;
    const bx = 40, by = 90, bw = 1120, bh = 280;
    box(ctx, bx - 14, by - 30, bw + 28, bh + 140);
    const X = (i) => bx + (i + 0.5) * (bw / SHOWN);
    const Y = (v) => by + bh - ((v - lo) / (hi - lo)) * bh;
    const cur = Math.min(SHOWN - 1, Math.floor(ease(p) * (SHOWN - 1)));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx, Y(0));
    ctx.lineTo(bx + bw, Y(0));
    ctx.stroke();
    // lag arrows into the current point
    const lags = [...new Set(collectLags(tree))].sort((a, b) => a - b);
    if (cur >= hist) {
      lags.forEach((lg, j) => {
        const src = cur - lg;
        if (src < 0) return;
        ctx.strokeStyle = ["#1d4ed8", "#0f766e", "#6d5bd0", "#be185d"][j % 4];
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(X(src), Y(ys[src]));
        ctx.quadraticCurveTo((X(src) + X(cur)) / 2, Math.min(Y(ys[src]), Y(ys[cur])) - 30 - j * 8, X(cur), Y(ys[cur]));
        ctx.stroke();
        ctx.setLineDash([]);
        text(ctx, lagName(lg), X(src), Y(ys[src]) - 16 - (j % 2) * 16, 14, ctx.strokeStyle, { align: "center", weight: 700 });
      });
    }
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= cur; i++) (i ? ctx.lineTo(X(i), Y(ys[i])) : ctx.moveTo(X(i), Y(ys[i])));
    ctx.stroke();
    for (let i = 0; i <= cur; i++) {
      ctx.beginPath();
      ctx.arc(X(i), Y(ys[i]), i === cur ? 6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = i === cur ? col.ours : i < hist ? "#fff" : col.ink;
      ctx.strokeStyle = col.ink;
      ctx.lineWidth = 1.6;
      ctx.fill();
      if (i < hist) ctx.stroke();
    }
    text(ctx, cur < hist ? "history" : `step ${cur - hist + 1}`, X(cur), by + bh + 24, 16, col.ours, { align: "center", weight: 700 });
    text(ctx, "x(t) = f(x(t−1), …, x(t−" + D.order + ")) + ε(t),    ε ∼ 𝒩(0, " + D.innovation_std + "²)", bx, by + bh + 58, 21, col.ink, { family: Z.MATH });
    text(ctx, `Example drawn with Shao–Wu (f includes its scale). The first ${hist} values are the starting history, uniform in [−${D.initial_radius}, ${D.initial_radius}]; arrows show the past values the next one reads.`, bx, by + bh + 88, 16, col.muted);
  }
  function collectPars(n) {
    if (n.t === "par") return [n.v];
    return (n.k || []).flatMap(collectPars);
  }
  function collectLags(n) {
    if (n.t === "lag") return [n.lag];
    return (n.k || []).flatMap(collectLags);
  }

  function step6(ctx, p) {
    // left: whole series
    box(ctx, 20, 60, 440, 380);
    text(ctx, "series", 40, 96, 20, col.ink, { weight: 700 });
    const ys = D.trajectory, lo = Math.min(...ys), hi = Math.max(...ys);
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const n = Math.floor(ys.length * seg(p, 0, 0.6));
    for (let i = 0; i < n; i++) {
      const x = 40 + (i / (ys.length - 1)) * 400, y = 130 + (1 - (ys[i] - lo) / (hi - lo)) * 270;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    // right: the stored record
    box(ctx, 490, 60, 690, 380);
    text(ctx, "stored with it: the mechanism", 512, 96, 20, col.ink, { weight: 700 });
    const poly = Z.polyText(D.final_poly, 3);
    const rows = [
      ["tree", "ADD / MUL / tanh over lag and coefficient slots"],
      ["binding", `${collectPars(tree).map((v, i) => `P${i} = ${fmt(v, 3)}`).join(", ")};  lags ${[...new Set(collectLags(tree))].sort((a, b) => a - b).join(", ")}`],
      ["stabilizer", `which of the four was drawn, with its certificate if any; here Shao–Wu, s = ${D.scale.toFixed(4)}, sensitivity ${D.certificate.total_gain.toFixed(2)} < 1`],
      ["equation", poly.join("  +  ")],
    ];
    let y = 128;
    rows.forEach((r, i) => {
      ctx.globalAlpha = ease(seg(p, 0.2 + i * 0.14, 0.4 + i * 0.14));
      text(ctx, r[0], 512, y, 18, col.blue, { weight: 700 });
      y = wrap(ctx, r[1], 612, y, 548, 24, 17, col.ink, { family: i === 3 ? Z.MATH : Z.SANS }) + 12;
      ctx.globalAlpha = 1;
    });
    const a = ease(seg(p, 0.8, 1));
    ctx.globalAlpha = a;
    box(ctx, 512, 360, 310, 64, { fill: col.osoft, stroke: col.ours });
    text(ctx, "target of the equation head", 667, 389, 18, col.ours, { align: "center", weight: 700 });
    text(ctx, "next slide", 667, 411, 15, col.muted, { align: "center" });
    box(ctx, 846, 360, 316, 64, { fill: col.grey, stroke: col.muted, dash: [6, 5] });
    text(ctx, "ground truth for explanations", 1004, 389, 18, col.muted, { align: "center", weight: 700 });
    text(ctx, "future work", 1004, 411, 15, col.muted, { align: "center" });
    ctx.globalAlpha = 1;
  }


  // ---- step 0: Chronos-2 autoregressive data vs this project, as two pipelines ----
  const LANE_C2 = [
    ["① Structure", "AR(p): a weighted sum of the last p values", "textbook form; order p not published"],
    ["② Parameters", "Draw the weights φ at random", "priors not published"],
    ["③ Stability", "Not described in the paper", "no stationarity test or resampling rule published"],
    ["④ Run", "Start values, then add noise and iterate", "start values, σ not published"],
    ["⑤ Output", "A series, mixed with other generators", "the forecasting loss needs only the series"],
  ];
  const LANE_OURS = [
    ["① Structure", "Random tree: sums of products of lags, degree ≤ 4", `${(Number(D.structures_total) / 1e9).toFixed(0)} billion trees`],
    ["② Parameters", "Coefficients in [0.05, 0.90], lags 1…12", "every prior is in the config"],
    ["③ Stability", "One of 4 constructions per series; diverging draws dropped", "3 of the 4 are certified"],
    ["④ Run", `History in [−${D.initial_radius}, ${D.initial_radius}], noise σ = ${D.innovation_std}, iterate`, "same recurrence idea"],
    ["⑤ Output", "A series and the exact equation that made it", "the equation is the label"],
  ];
  const DIFFS = [["linear → nonlinear", true], ["unknown → documented", true], ["unspecified → stated", true], ["same recurrence", false], ["no label → equation label", true]];

  function lane(ctx, rows, y, p, a0, a1, o) {
    rows.forEach((r, i) => {
      const k = (a1 - a0) / rows.length, a = ease(seg(p, a0 + i * k * 0.8, a0 + i * k * 0.8 + k * 0.9));
      const x = 20 + i * 236;
      ctx.globalAlpha = a;
      box(ctx, x, y, 208, 150, { fill: o.fill, stroke: o.stroke, lw: 2.4 });
      text(ctx, r[0], x + 16, y + 30, 19, o.stroke, { weight: 700 });
      wrap(ctx, r[1], x + 16, y + 58, 178, 21, 16, col.ink);
      wrap(ctx, r[2], x + 16, y + 124, 178, 17, 14, col.muted);
      if (i < rows.length - 1) arrow(ctx, x + 211, y + 75, x + 233, y + 75, col.muted, 2);
      ctx.globalAlpha = 1;
    });
  }

  function compare(ctx, p) {
    text(ctx, "Chronos-2 training data · autoregressive generator", 20, 26, 20, col.blue, { weight: 700 });
    text(ctx, "textbook AR(p); the paper lists AR as a base generator but publishes no parameters", 540, 26, 15, col.muted);
    lane(ctx, LANE_C2, 36, p, 0, 0.38, { fill: col.soft, stroke: col.blue });
    const d = ease(seg(p, 0.74, 0.92));
    ctx.globalAlpha = d;
    DIFFS.forEach((df, i) => {
      const x = 20 + i * 236, w = 208;
      Z.rrect(ctx, x, 192, w, 28, 14);
      ctx.fillStyle = df[1] ? col.ours : col.grey;
      ctx.fill();
      text(ctx, "↕ " + df[0], x + w / 2, 212, 15, df[1] ? "#fff" : col.muted, { align: "center", weight: 700 });
    });
    ctx.globalAlpha = 1;
    text(ctx, "This project · random nonlinear recurrence", 20, 256, 20, col.ours, { weight: 700 });
    lane(ctx, LANE_OURS, 266, p, 0.38, 0.74, { fill: col.osoft, stroke: col.ours });
    ctx.globalAlpha = ease(seg(p, 0.9, 1));
    wrap(ctx, "Same idea, each value computed from its own past plus noise. What changes: the equation is nonlinear, stable by construction or filtered, and kept, so it can serve as a label.", 20, 446, 1160, 27, 21, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  const STEPS = [
    { narr: "Zoom 1 · Generator", sub: "autoregressive data in Chronos-2 versus ours", dur: 9500,
      foot: "Top lane: the Chronos-2 paper (Ansari et al., 2025) lists AR models as a base generator but publishes no orders, coefficient priors or noise scales, so the textbook AR(p) is shown; nothing is said about a stationarity test or about resampling unstable draws (Section 4.2 only lists AR as a base generator). The official repository (amazon-science/chronos-forecasting) publishes only kernel-synth.py for data generation, no AR code. Bottom lane: this project's samplers (src/tsfmxai/generators)." },
    { narr: "Our generator in five stages", sub: "a random equation, then its series", dur: 5200 },
    { narr: "① Draw a tree shape", sub: "uniformly among all valid trees", dur: 4200 },
    { narr: "② Bind the numbers", sub: "each slot gets a value", dur: 4200 },
    { narr: "③ Why stabilize?", sub: "a feedback loop fades or explodes depending on its gain", dur: 5200,
      foot: "Left: illustrative linear loop. Right: final_methods/00_final_methods_tradeoff.ipynb, 128 random mechanisms without protection, 384 steps; explosion = non-finite or |x| > 100." },
    { narr: "③ Four constructions, one contract", sub: "drawn at random per series; click one to see how it works", dur: 5600,
      foot: "Comparison numbers: final_methods/00_final_methods_tradeoff.ipynb (128 mechanisms per method, 384 steps; Shao-Wu product_lags). Near zero: every value within ?0.15 over the last 24 steps. Speed measured on one machine." },
    { narr: "④ Run the recurrence", sub: "history, then one step at a time", dur: 9000 },
    { narr: "⑤ Keep the record", sub: "the equation is the label", dur: 3800 },
  ];

  Z.mount({
    id: "zoom-gen", W, H, steps: STEPS,
    foot: "Real example from the project's sampler (seed " + D.seed + ", sample " + D.index + "); formulas and numbers exported by scripts/export_generator_demo.py. The run and record steps use a draw from the Shao–Wu construction (Shao and Wu, 2007, Thm 5.1, as adapted in this project).",
    hits: (step) => (step === 5 ? METHODS.map((m, i) => Object.assign({ key: m.key }, hubRect(i))).filter((h) => h.key) : []),
    onHit: (key) => window.GenDetail && window.GenDetail.open(key, 0),
    paint(ctx, step, p) {
      if (step === 0) return compare(ctx, p);
      // chips: 0 Draw, 1 Bind, 2 Stabilize (two steps), 3 Run, 4 Keep
      Z.chips(ctx, STAGES, [-1, -1, 0, 1, 2, 2, 3, 5][step], 8);
      if (step === 1) cards(ctx, p);
      else if (step === 2) step1(ctx, p);
      else if (step === 3) step2(ctx, p);
      else if (step === 4) problem(ctx, p);
      else if (step === 5) hub(ctx, p);
      else if (step === 6) step5(ctx, p);
      else step6(ctx, p);
    },
  });
})();
