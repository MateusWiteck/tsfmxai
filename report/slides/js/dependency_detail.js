/* Click-through detail view for the multivariatizer step (slide "From one series to many",
 * Options A and B): given three sampled source series x1, x2, x3, how the variates y1, y2, y3
 * are built, which operations the Chronos-2 paper states and which it does not, and how the
 * equation is represented there versus in this project's AST.
 *
 * Provenance is part of the content:
 *  - Chronos-2 paper, Section 4.2 (pp. 7-8): two classes of multivariatizers, in two sentences.
 *    Nothing else is published: no formulas, no list of functions, no sampler. Checked on 5 Nov 2026: the main
 *    branch of amazon-science/chronos-forecasting (full tree) has no multivariatizer code, only scripts/kernel-synth.py
 *    and chronos2/dataset.py (windowing of given series); arXiv 2510.15821 has v1 only.
 *    Rows marked "stated" quote those sentences; rows marked "not stated" are questions the paper
 *    leaves open. The formulas of the two examples are the slide's own illustration (data/multivariate_demo.js).
 *  - The AST side is read from src/tsfmxai/generators/trajectory.py and representations/expression.py.
 * Opened with ?gen=dep&gstep=N for screenshots.
 */
(function () {
  const G = window.GenDetail, Z = window.ZC, MV = window.MV_DATA;
  if (!G || !Z || !MV) return;
  const { col, seg, ease, text, wrap, box, arrow } = Z;
  const MINUS = "−";
  const W = 1200, H = 400;
  const COLORS = [col.blue, col.teal, "#6d5bd0"];
  const STATED = col.teal, OPEN = col.muted, ILLU = col.ours;
  const L = MV.lag;

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const line = (ctx, pts, color, lw) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw || 2;
    ctx.beginPath();
    pts.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
    ctx.stroke();
  };
  const axisLine = (ctx, x1, y1, x2, y2, color, dash) => {
    ctx.strokeStyle = color || col.line;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  const std = (a) => {
    const m = a.reduce((s, v) => s + v, 0) / a.length;
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length) || 1;
  };
  const unit = (a) => {
    const m = a.reduce((s, v) => s + v, 0) / a.length, s = std(a);
    return a.map((v) => (v - m) / s);
  };
  const smooth = (seed, n, a) => {
    const r = rng(seed), y = [0];
    for (let t = 1; t < n; t++) y.push(a * y[t - 1] + gauss(r));
    return unit(y);
  };

  // ---- small shared pieces ----
  function lane(ctx, x, y, w, h, arr, color, f, name) {
    const lo = Math.min(...arr), hi = Math.max(...arr), n = Math.max(2, Math.floor(arr.length * f));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const X = x + (i / (arr.length - 1)) * w, Y = y + h - 4 - ((arr[i] - lo) / (hi - lo || 1)) * (h - 8);
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.stroke();
    text(ctx, name, x + 6, y + 15, 13.5, color, { weight: 700 });
  }
  function chip(ctx, x, y, kind) {
    const stated = kind === "stated", label = stated ? "stated" : "not stated";
    Z.rrect(ctx, x, y, 96, 26, 13);
    ctx.fillStyle = stated ? col.tsoft : col.grey;
    ctx.fill();
    ctx.setLineDash(stated ? [] : [4, 3]);
    ctx.strokeStyle = stated ? STATED : OPEN;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, label, x + 48, y + 18, 13.5, stated ? STATED : OPEN, { align: "center", weight: 700 });
  }
  /** rows: [kind, what, quote]; stated rows quote the paper, open rows are questions it leaves unanswered. */
  function opsTable(ctx, p, x, y, w, rows) {
    const rh = Math.min(58, (H - 24 - y) / rows.length);
    rows.forEach((r, i) => {
      const a = ease(seg(p, 0.15 + i * 0.1, 0.4 + i * 0.1)), top = y + i * rh;
      ctx.globalAlpha = a;
      chip(ctx, x, top + 4, r[0]);
      text(ctx, r[1], x + 112, top + 20, 16, r[0] === "stated" ? col.ink : col.muted, { weight: r[0] === "stated" ? 700 : 500 });
      if (r[2]) text(ctx, r[2], x + 112, top + 40, 13, col.muted);
      axisLine(ctx, x, top + rh - 2, x + w, top + rh - 2);
      ctx.globalAlpha = 1;
    });
  }
  /** x1..x3 on the left, y1..y3 on the right, with the slide's own series. */
  function io(ctx, p, ys, label) {
    const f = ease(seg(p, 0, 0.6));
    text(ctx, "sampled sources", 34, 36, 14, col.muted, { weight: 600 });
    text(ctx, label, 250, 36, 14, col.muted, { weight: 600 });
    [0, 1, 2].forEach((i) => {
      const y = 46 + i * 78;
      lane(ctx, 34, y, 160, 62, MV.base[i], COLORS[i], f, `x${"₁₂₃"[i]}`);
      arrow(ctx, 200, y + 31, 242, y + 31, col.muted, 2, 1);
      lane(ctx, 250, y, 160, 62, ys[i], COLORS[i], seg(p, 0.35, 0.95), `y${"₁₂₃"[i]}`);
    });
  }

  // ---- step 0: what goes in, what comes out, what the paper says ----
  function stepInside(ctx, p) {
    box(ctx, 14, 8, 1172, 384);
    text(ctx, "Three independent sources go in, three coupled variates come out", 34, 40, 20, col.ink, { weight: 700 });
    [0, 1, 2].forEach((i) => {
      const a = ease(seg(p, i * 0.08, i * 0.08 + 0.25)), y = 62 + i * 52;
      ctx.globalAlpha = a;
      box(ctx, 34, y, 130, 40, { r: 10, fill: col.soft, stroke: COLORS[i] });
      text(ctx, `x${"₁₂₃"[i]}`, 99, y + 27, 20, COLORS[i], { align: "center", weight: 700 });
      box(ctx, 1022, y, 130, 40, { r: 10, fill: col.soft, stroke: COLORS[i] });
      text(ctx, `y${"₁₂₃"[i]}`, 1087, y + 27, 20, COLORS[i], { align: "center", weight: 700 });
      arrow(ctx, 168, y + 20, 262, 138, col.muted, 1.8, 1);
      arrow(ctx, 940, 138, 1018, y + 20, col.muted, 1.8, 1);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.3, 0.7));
    box(ctx, 266, 62, 672, 150, { fill: col.osoft, stroke: ILLU, lw: 2.4 });
    text(ctx, "multivariatizer", 602, 90, 19, ILLU, { align: "center", weight: 700 });
    [["Option A: cotemporaneous", "same time step"], ["Option B: sequential", "across time"]].forEach((o, k) => {
      const x = 286 + k * 326;
      box(ctx, x, 104, 306, 90, { r: 10, fill: col.panel, stroke: col.line });
      text(ctx, o[0], x + 153, 138, 17, col.ink, { align: "center", weight: 700 });
      text(ctx, o[1], x + 153, 166, 15, col.muted, { align: "center" });
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 1));
    text(ctx, "Everything the paper says about the operations, in its own words (Section 4.2):", 34, 246, 16, col.muted, { weight: 600 });
    [["A", "“apply linear or nonlinear transformations at the same time step across time series”"],
      ["B", "“induce dependencies across time, generating richer multivariate properties such as lead–lag effects and cointegration”"]].forEach((q, i) => {
      text(ctx, q[0], 34, 280 + i * 34, 17, ILLU, { weight: 700 });
      text(ctx, q[1], 62, 280 + i * 34, 16.5, col.ink);
    });
    text(ctx, "No formulas, no list of functions, no sampler are published. The next steps separate what is stated from what is left open.", 34, 362, 16, col.ink, { weight: 700 });
    ctx.globalAlpha = 1;
  }

  // ---- steps 1 and 2: the operations of Option A and Option B ----
  function stepA(ctx, p) {
    box(ctx, 14, 8, 560, 384);
    io(ctx, p, MV.cot, "variates y (the slide's example)");
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "the slide's illustration, not the paper's:", 34, 300, 13, ILLU, { weight: 700 });
    ["y₁ = x₁", "y₂ = 0.8·x₁ + 0.6·x₂", "y₃ = tanh(x₃ + x₁)"].forEach((s, i) => text(ctx, s, 34, 322 + i * 22, 16, col.ink, { family: Z.MATH }));
    text(ctx, "no product of two series in it", 300, 366, 14, col.muted);
    ctx.globalAlpha = 1;
    box(ctx, 590, 8, 596, 384);
    text(ctx, "Option A: which operations?", 610, 40, 20, col.ink, { weight: 700 });
    opsTable(ctx, p, 610, 56, 560, [
      ["stated", "A weighted sum of the series at one time step", "“linear … transformations”"],
      ["stated", "A nonlinear function of the series at one time step", "“nonlinear transformations”"],
      ["stated", "Only the same time step: no earlier values", "“at the same time step”"],
      ["open", "Which nonlinear functions: tanh? a product x₁·x₂?", ""],
      ["open", "Weights, how many series enter each variate, noise", ""],
    ]);
  }
  function stepB(ctx, p) {
    box(ctx, 14, 8, 560, 384);
    io(ctx, p, MV.seq, "variates y (the slide's example)");
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "the slide's illustration, not the paper's:", 34, 300, 13, ILLU, { weight: 700 });
    ["y₁(t) = x₁(t)", `y₂(t) = 0.4·x₂(t) + 0.9·y₁(t − ${L})`, `y₃(t) = 0.4·x₃(t) + 0.9·y₂(t − ${L})`].forEach((s, i) => text(ctx, s, 34, 322 + i * 22, 16, col.ink, { family: Z.MATH }));
    ctx.globalAlpha = 1;
    box(ctx, 590, 8, 596, 384);
    text(ctx, "Option B: which operations?", 610, 40, 20, col.ink, { weight: 700 });
    opsTable(ctx, p, 610, 56, 560, [
      ["stated", "Dependencies across time", "“induce dependencies across time”"],
      ["stated", "Lead–lag effects", "“such as lead–lag effects”"],
      ["stated", "Cointegration", "“and cointegration”"],
      ["open", "Which series read which, and with what delays", ""],
      ["open", "Products or tanh inside the dependencies", ""],
      ["open", "Whether A and B are combined in one variate", ""],
    ]);
  }

  // ---- steps 3 and 4: the two effects the paper names for Option B (computed live) ----
  function xc(a, b, k) {
    let s = 0, c = 0;
    for (let t = 0; t + k < a.length; t++) { s += a[t] * b[t + k]; c++; }
    return s / c;
  }
  function stepLag(ctx, p) {
    const x = smooth(5, 340, 0.95);
    [[0, 14, "ℓ = 0: no delay"], [12, 606, "ℓ = 12: y follows x 12 steps later"]].forEach(([Lg, x0, title], k) => {
      const r = rng(30 + k), y = unit(x.map((_, t) => 0.9 * (x[t - Lg] === undefined ? 0 : x[t - Lg]) + 0.35 * gauss(r)));
      const xs = x.slice(40, 170), ys = y.slice(40, 170);
      box(ctx, x0, 8, 580, 366);
      text(ctx, title, x0 + 20, 40, 20, col.ink, { weight: 700 });
      const px = x0 + 20, pw = 540, X = (t) => px + (t / 129) * pw, n = Math.max(2, Math.floor(130 * seg(p, 0, 0.6)));
      const Y = (v, top) => top + 36 - (v / 3) * 36;
      line(ctx, xs.slice(0, n).map((v, t) => [X(t), Y(v, 78)]), col.blue, 2);
      line(ctx, ys.slice(0, n).map((v, t) => [X(t), Y(v, 140)]), col.teal, 2);
      text(ctx, "x", px - 2, 70, 14, col.blue, { weight: 700 });
      text(ctx, "y", px - 2, 134, 14, col.teal, { weight: 700 });
      const base = 350, hmax = 110, ks = 21, slot = pw / ks;
      axisLine(ctx, px, base, px + pw, base);
      text(ctx, "correlation of x(t) with y(t + k), by k", px, 212, 14, col.muted, { weight: 600 });
      const vals = Array.from({ length: ks }, (_, kk) => xc(unit(x), y, kk));
      const peak = vals.indexOf(Math.max(...vals)), grow = ease(seg(p, 0.5, 1));
      vals.forEach((v, kk) => {
        const h = Math.max(0, v) * hmax * grow;
        ctx.fillStyle = kk === peak ? ILLU : "#a9b2c6";
        ctx.fillRect(px + kk * slot + 3, base - h, slot - 6, h);
        if (kk % 4 === 0 || kk === peak) text(ctx, String(kk), px + kk * slot + slot / 2, base + 18, 12.5, kk === peak ? ILLU : col.muted, { align: "center", weight: kk === peak ? 700 : 500 });
      });
      ctx.globalAlpha = seg(p, 0.8, 1);
      text(ctx, `peak at k = ${peak}`, px + peak * slot + slot / 2 + (peak > 14 ? -10 : 14), base - hmax - 8, 15, ILLU, { align: peak > 14 ? "right" : "left", weight: 700 });
      ctx.globalAlpha = 1;
    });
    text(ctx, "illustration: y = 0.9·x(t − ℓ) + noise. A delayed dependency is what produces a lead–lag effect.", 20, 392, 13, col.muted);
  }

  let coint = null;
  function cointData() {
    if (coint) return coint;
    const r = rng(17), N = 320, y1 = [0], y2 = [0.4];
    for (let t = 1; t < N; t++) {
      y1.push(y1[t - 1] + 0.32 * gauss(r));
      y2.push(y2[t - 1] + 0.3 * (y1[t - 1] - y2[t - 1]) + 0.1 * gauss(r));
    }
    const gap = y1.map((v, t) => v - y2[t]);
    const rg = (a) => Math.max(...a) - Math.min(...a);
    coint = { y1, y2, gap, N, r1: rg(y1), rgap: rg(gap) };
    return coint;
  }
  function stepCoint(ctx, p) {
    const C = cointData(), n = Math.max(2, Math.floor(C.N * seg(p, 0, 0.8)));
    box(ctx, 14, 8, 760, 384);
    const bx = 40, bw = 710, X = (t) => bx + (t / (C.N - 1)) * bw;
    const lo = Math.min(...C.y1, ...C.y2) - 0.3, hi = Math.max(...C.y1, ...C.y2) + 0.3, top = 40, h = 190;
    const Y = (v) => top + h - ((v - lo) / (hi - lo)) * h;
    text(ctx, "two series that wander", 40, 30, 16, col.muted, { weight: 600 });
    line(ctx, C.y1.slice(0, n).map((v, t) => [X(t), Y(v)]), col.blue, 2.4);
    line(ctx, C.y2.slice(0, n).map((v, t) => [X(t), Y(v)]), col.teal, 2.4);
    text(ctx, "y₁", X(C.N - 1) - 4, Y(C.y1[C.N - 1]) - 8, 15, col.blue, { weight: 700 });
    text(ctx, "y₂", X(C.N - 1) - 4, Y(C.y2[C.N - 1]) + 20, 15, col.teal, { weight: 700 });
    const gl = -1.7, gh = 1.7, gt = 268, gH = 100, GY = (v) => gt + gH - ((v - gl) / (gh - gl)) * gH;
    text(ctx, "their gap  y₁ − y₂", 40, 256, 16, col.muted, { weight: 600 });
    axisLine(ctx, bx, GY(0), bx + bw, GY(0));
    line(ctx, C.gap.slice(0, n).map((v, t) => [X(t), GY(Math.max(gl, Math.min(gh, v)))]), ILLU, 2.4);
    box(ctx, 796, 8, 390, 384);
    text(ctx, "Cointegration", 816, 44, 21, col.ink, { weight: 700 });
    wrap(ctx, "Each series drifts without a fixed level, yet one keeps pulling back toward the other. Their difference stays bounded.", 816, 74, 350, 22, 16, col.ink);
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, `range of y₁:  ${C.r1.toFixed(1)}`, 816, 196, 18, col.blue, { weight: 700 });
    text(ctx, `range of the gap:  ${C.rgap.toFixed(1)}`, 816, 226, 18, ILLU, { weight: 700 });
    wrap(ctx, "It needs a delayed dependency: y₂ reads y₁ one step back. The paper names the effect and gives no recipe; this construction is ours.", 816, 262, 350, 22, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- steps 5 and 6: how the equation is represented ----
  const leaf = (name) => ({ t: "leaf", name });
  const par = (v) => ({ t: "par", v });
  const add = (a, b) => ({ t: "add", k: [a, b] });
  const mul = (a, b) => ({ t: "mul", k: [a, b] });
  function tree(ctx, t, bx, by, bw, bh, size, alpha) {
    const Lo = Z.layout(t, { x: bx, y: by, w: bw, h: bh });
    Z.drawTree(ctx, Lo, {
      size,
      alpha: () => alpha,
      label: (nd) => (nd.n.t === "add" ? "+" : nd.n.t === "mul" ? "×" : nd.n.t === "par" ? Z.fmt(nd.n.v, 2) : nd.n.name),
      style: (nd) => {
        if (nd.kids.length) return { stroke: col.ink };
        if (nd.n.t === "par") return { fill: col.soft, stroke: col.blue, color: col.blue };
        return { fill: col.tsoft, stroke: col.teal, color: col.teal };
      },
    });
  }
  function stepRepr(ctx, p) {
    box(ctx, 14, 8, 560, 384);
    text(ctx, "Chronos-2: a recipe applied to the sampled series", 34, 40, 20, col.ink, { weight: 700 });
    ctx.globalAlpha = ease(seg(p, 0, 0.4));
    [0, 1, 2].forEach((i) => {
      const y = 64 + i * 40;
      box(ctx, 34, y, 70, 32, { r: 8, fill: col.soft, stroke: COLORS[i] });
      text(ctx, `x${"₁₂₃"[i]}`, 69, y + 22, 17, COLORS[i], { align: "center", weight: 700 });
      box(ctx, 470, y, 70, 32, { r: 8, fill: col.soft, stroke: COLORS[i] });
      text(ctx, `y${"₁₂₃"[i]}`, 505, y + 22, 17, COLORS[i], { align: "center", weight: 700 });
    });
    box(ctx, 190, 64, 190, 112, { r: 12, fill: col.grey, stroke: col.muted, dash: [6, 5] });
    text(ctx, "?", 285, 134, 44, col.muted, { align: "center", weight: 700 });
    arrow(ctx, 108, 120, 186, 120, col.muted, 2, 1);
    arrow(ctx, 384, 120, 466, 120, col.muted, 2, 1);
    ctx.globalAlpha = ease(seg(p, 0.35, 1));
    [["described in words", "a transformation across series, or a dependence across time"],
      ["no grammar published", "no node types, no operator list: nothing to compare term by term"],
      ["not part of the record", "only the resulting values are stored; the recipe is not (slide 9)"]].forEach((r, i) => {
      text(ctx, r[0], 34, 220 + i * 56, 16, col.ink, { weight: 700 });
      wrap(ctx, r[1], 34, 242 + i * 56, 510, 19, 14.5, col.muted);
    });
    ctx.globalAlpha = 1;
    box(ctx, 590, 8, 596, 384);
    text(ctx, "This project: one AST per variate, the equation is stored", 610, 40, 20, col.ink, { weight: 700 });
    const a = ease(seg(p, 0.3, 0.7));
    text(ctx, "Option A example: y₂ = 0.8·x₁ + 0.6·x₂", 610, 74, 15, ILLU, { weight: 700 });
    tree(ctx, add(mul(par(0.8), leaf("x₁(t)")), mul(par(0.6), leaf("x₂(t)"))), 640, 96, 520, 78, 14, a);
    text(ctx, `Option B example: y₂ = 0.4·x₂ + 0.9·y₁(t − ${L})`, 610, 222, 15, ILLU, { weight: 700 });
    tree(ctx, add(mul(par(0.4), leaf("x₂(t)")), mul(par(0.9), leaf(`y₁(t−${L})`))), 640, 252, 520, 70, 14, a);
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    wrap(ctx, "Same-time terms (lag 0) are outside the lags 1 to 16 that the generator draws: this project writes next values from past values.", 610, 366, 560, 16, 13, col.muted);
    ctx.globalAlpha = 1;
  }

  function stepProduct(ctx, p) {
    box(ctx, 14, 8, 560, 384);
    text(ctx, "A product between two series, as an AST", 34, 40, 20, col.ink, { weight: 700 });
    text(ctx, "x₂(t) = −0.55 · x₁(t−3) · x₂(t−1)", 34, 76, 20, col.ink, { family: Z.MATH });
    tree(ctx, mul(par(-0.55), mul(leaf("x₁(t−3)"), leaf("x₂(t−1)"))), 60, 120, 480, 150, 15, ease(seg(p, 0, 0.6)));
    ctx.globalAlpha = ease(seg(p, 0.55, 1));
    wrap(ctx, "The × joins a lag of series 1 and a lag of series 2: an interaction between two series, with the delays and the coefficient stored in the tree.", 34, 320, 520, 21, 15.5, col.ink);
    ctx.globalAlpha = 1;
    box(ctx, 590, 8, 596, 384);
    text(ctx, "How the tree is read", 610, 40, 20, col.ink, { weight: 700 });
    [["who reads whom", "the variable index of a Lag leaf"], ["when", "the lag value of that leaf"], ["interaction", "a Multiply of two Lag leaves (two series here)"], ["squashing", "a Tanh node around a lag"], ["check", "an index beyond the number of series is rejected"]].forEach((r, i) => {
      ctx.globalAlpha = ease(seg(p, 0.15 + i * 0.1, 0.4 + i * 0.1));
      const y = 78 + i * 46;
      text(ctx, r[0], 610, y, 15.5, STATED, { weight: 700 });
      text(ctx, r[1], 780, y, 15.5, col.ink);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    box(ctx, 610, 318, 560, 60, { fill: col.osoft, stroke: ILLU });
    text(ctx, "Product between two series", 626, 342, 16, col.ink, { weight: 700 });
    text(ctx, "Chronos-2 paper: not stated.   This project: a first-class term.", 626, 364, 15, ILLU, { weight: 700 });
    ctx.globalAlpha = 1;
  }

  G.add("dep", "Multivariatizer", {
    title: "Inside the multivariatizer",
    foot: "Chronos-2 (Ansari et al., 2025), Section 4.2, pp. 7–8: the only description of the two classes; no formulas, list of functions or sampler. Checked on 5 Nov 2026: the main branch of amazon-science/chronos-forecasting (full file tree) has only scripts/kernel-synth.py for data generation, and chronos2/dataset.py only cuts windows from given series; autogluon/chronos_datasets has the kernel_synth and tsmixup corpora, no multivariate synthetic one; arXiv has v1 only. Series and formulas of the examples: the slide's own illustration (data/multivariate_demo.js). AST: src/tsfmxai/generators/trajectory.py.",
    steps: [
      { narr: "What goes in, what comes out", sub: "sources x₁, x₂, x₃ become variates y₁, y₂, y₃", formula: "", dur: 5200 },
      { narr: "Option A, cotemporaneous: which operations?", sub: "what the paper states and what it leaves open", formula: "yᵢ(t) = gᵢ( x₁(t), x₂(t), x₃(t) )", dur: 5600 },
      { narr: "Option B, sequential: which operations?", sub: "what the paper states and what it leaves open", formula: "yᵢ(t) depends on values at earlier times", dur: 5600 },
      { narr: "Option B effect 1: lead–lag", sub: "a delayed dependency shifts one series against another", formula: "y(t) = 0.9 · x(t − ℓ) + noise", dur: 6000 },
      { narr: "Option B effect 2: cointegration", sub: "two wandering series held together", formula: "y₂(t) = y₂(t−1) + 0.3 · ( y₁(t−1) − y₂(t−1) ) + ε(t)", dur: 7000 },
      { narr: "How the equation is represented", sub: "a recipe in words versus an AST per variate", formula: "", dur: 5600 },
      { narr: "What the AST adds: products between series", sub: "who reads whom, when, and how they interact", formula: "", dur: 5600 },
    ],
    paint(ctx, step, p) {
      [stepInside, stepA, stepB, stepLag, stepCoint, stepRepr, stepProduct][step](ctx, p);
    },
  }, "dep");

  document.querySelectorAll("#mv .dep-open").forEach((b) => b.addEventListener("click", () => G.open("dep", 0)));
})();
