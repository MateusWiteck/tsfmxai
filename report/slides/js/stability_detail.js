/* Click-through detail views for the three explosion-avoidance methods compared in the
 * project (opened from the "Four constructions" step of the generator zoom):
 *   sw   Shao-Wu global moment contraction
 *   jac  Jacobian contraction by construction
 *   hvar Heaps symmetric stationary VAR
 *
 * Everything numeric comes from the executed notebooks in experiments/abstract_syntax_tree
 * (the final-method notebook for the comparison, the method notebooks for the details) or from the exported
 * demo mechanism (data/generator_demo.js). Pictures marked "schematic" or "illustrative"
 * use made-up numbers on purpose and say so. Opened with ?gen=sw&gstep=2 for screenshots.
 */
(function () {
  const G = window.GenDetail, Z = window.ZC, D = window.GEN_DATA;
  if (!G || !Z || !D) return;
  const { col, seg, ease, text, wrap, box, arrow } = Z;
  const MINUS = "−";
  const W = 1200, H = 400;
  const GREY_BAR = "#a9b2c6";

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

  /** Vertical bars; p scales every bar. */
  function vbars(ctx, b, vals, maxV, labels, colorFn, p) {
    const n = vals.length, slot = b.w / n, bw = slot * 0.56;
    axisLine(ctx, b.x, b.y + b.h, b.x + b.w, b.y + b.h);
    vals.forEach((v, i) => {
      const h = Math.max(0, (b.h * v) / maxV) * (p === undefined ? 1 : p);
      const x = b.x + i * slot + (slot - bw) / 2;
      ctx.fillStyle = colorFn(i);
      ctx.fillRect(x, b.y + b.h - h, bw, h);
      text(ctx, labels[i], x + bw / 2, b.y + b.h + 22, 14, col.muted, { align: "center" });
      if (v > 0.001) text(ctx, v.toFixed(2), x + bw / 2, b.y + b.h - h - 8, 14, col.ink, { align: "center", weight: 700 });
    });
  }
  /** Horizontal meter with a tick at 1. */
  function meter(ctx, x, y, w, val, maxV, color, tickLabel) {
    ctx.fillStyle = col.grey;
    ctx.fillRect(x, y, w, 26);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, Math.max(2, (w * val) / maxV), 26);
    const tx = x + (w * 1) / maxV;
    axisLine(ctx, tx, y - 10, tx, y + 36, col.teal);
    ctx.lineWidth = 2.4;
    text(ctx, tickLabel || "limit 1", tx + 6, y - 14, 15, col.teal, { weight: 700 });
  }

  // ---- shared: the four-method comparison (final_methods/00_final_methods_tradeoff.ipynb, 128 mechanisms each, 384 steps) ----
  const TRADE = [
    { k: "orig", name: "no protection", expl: 47.66, near: 14.84, rate: 2691 },
    { k: "jac", name: "Jacobian", expl: 0, near: 37.5, rate: 1622 },
    { k: "sw", name: "Shao–Wu", expl: 0, near: 25, rate: 1395 },
    { k: "hvar", name: "Heaps VAR", expl: 0, near: 44.53, rate: 327 },
  ];
  function tradeoff(ctx, p, me) {
    const metrics = [
      ["Crossed |x| > 100", "within 384 steps", "expl", 100, (v) => v.toFixed(1) + " %"],
      ["Low-amplitude tail", "every |x| ≤ 0.15 in the last 24 steps", "near", 100, (v) => v.toFixed(1) + " %"],
      ["Mechanisms per second", "sampling and certificate", "rate", 3500, (v) => String(Math.round(v))],
    ];
    metrics.forEach((m, i) => {
      const x = 14 + i * 396, w = 376;
      box(ctx, x, 8, w, 384);
      text(ctx, m[0], x + 18, 42, 20, col.ink, { weight: 700 });
      text(ctx, m[1], x + 18, 64, 14, col.muted);
      TRADE.forEach((t, j) => {
        const y = 92 + j * 72, a = seg(p, j * 0.1, j * 0.1 + 0.5);
        const mine = t.k === me;
        text(ctx, t.name, x + 18, y + 16, 16, mine ? col.ours : col.ink, { weight: mine ? 700 : 500 });
        const len = Math.max(2, 230 * (t[m[2]] / m[3]) * a);
        ctx.fillStyle = mine ? col.ours : t.k === "orig" ? "#d9d6cc" : GREY_BAR;
        ctx.fillRect(x + 18, y + 26, len, 20);
        text(ctx, m[4](t[m[2]]), x + 18 + len + 8, y + 42, 15, col.ink, { weight: 700 });
      });
    });
  }

  // ---- shared: "fine print" table. Row: [specification, used here, source, status] ----
  // status: true = from the paper, false = project choice, null = neutral note.
  function table(ctx, p, rows) {
    const cx = [14, 236, 792], cw = [208, 540, 394];
    ["Specification", "Used in this project", "Source or status"].forEach((h, i) => text(ctx, h, cx[i], 18, 14, col.muted, { weight: 700 }));
    axisLine(ctx, 10, 28, 1190, 28);
    const rh = (H - 36) / rows.length;
    rows.forEach((r, i) => {
      const y = 36 + i * rh, a = seg(p, i * 0.1, i * 0.1 + 0.3);
      ctx.globalAlpha = a;
      text(ctx, r[0], cx[0], y + 22, 15, col.ink, { weight: 700 });
      wrap(ctx, r[1], cx[1], y + 22, cw[1], 19, 15, col.ink);
      const c = r[3] === true ? col.teal : r[3] === false ? col.ours : col.muted;
      wrap(ctx, r[2], cx[2], y + 22, cw[2], 19, 15, c);
      axisLine(ctx, 10, y + rh - 4, 1190, y + rh - 4);
      ctx.globalAlpha = 1;
    });
  }

  // ---- shared: step 0 of every method, "in plain words": a picture on the left, three short
  // blocks on the right (an everyday picture, the rule, the engineering counterpart) ----
  function plainRight(ctx, p, cfg) {
    box(ctx, 600, 8, 586, 384);
    text(ctx, cfg.title, 622, 40, 21, col.ink, { weight: 700 });
    const blocks = [["THINK OF IT AS", cfg.think], ["THE RULE", cfg.rule], ["IN ENGINEERING TERMS", cfg.eng]];
    let y = 68;
    blocks.forEach((b, i) => {
      ctx.globalAlpha = ease(seg(p, 0.15 + i * 0.2, 0.4 + i * 0.2));
      text(ctx, b[0], 622, y, 12.5, col.blue, { weight: 700 });
      y = wrap(ctx, b[1], 622, y + 22, 546, 21, 15.5, col.ink) + 14;
      ctx.globalAlpha = 1;
    });
  }
  function plainEcho(ctx, p) {
    const rows = [[0.76, col.teal, "total pull 0.76 < 1: the echoes die out"], [1.2, col.ours, "total pull 1.2 > 1: the echoes build up"]];
    box(ctx, 14, 8, 572, 384);
    rows.forEach(([c, color, title], r) => {
      const y0 = 36 + r * 180, base = y0 + 118;
      text(ctx, title, 34, y0 + 4, 17, color, { weight: 700 });
      axisLine(ctx, 34, base, 566, base);
      for (let k = 0; k < 8; k++) {
        const v = Math.pow(c, k), h = Math.min(1, v / 2.2) * 78 * ease(seg(p, k * 0.09, k * 0.09 + 0.25));
        ctx.fillStyle = k === 0 ? col.ink : color;
        ctx.fillRect(40 + k * 66, base - h, 36, h);
        text(ctx, v.toFixed(2), 58 + k * 66, base - h - 6, 13, col.ink, { align: "center", weight: 700 });
      }
    });
    text(ctx, "a single pulse of size 1 at time 0, then its echoes; illustrative", 34, 382, 12.5, col.muted);
  }
  function plainFunnel(ctx, p) {
    box(ctx, 14, 8, 572, 384);
    const bx = 150, by = 50, bs = 280;
    ctx.fillStyle = col.grey;
    ctx.fillRect(bx, by, bs, bs);
    ctx.strokeStyle = col.teal;
    ctx.lineWidth = 3;
    ctx.strokeRect(bx, by, bs, bs);
    text(ctx, "box: the state never leaves it", bx, by - 10, 15, col.teal, { weight: 700 });
    const A = [-0.8, 0.7], B = [0.7, -0.6], k = 0.82, cx = bx + bs / 2, cy = by + bs / 2;
    const steps = Math.floor(seg(p, 0, 0.85) * 12);
    const pos = (q, t) => [cx + q[0] * Math.pow(k, t) * (bs / 2), cy - q[1] * Math.pow(k, t) * (bs / 2)];
    [[A, col.ink, "A"], [B, col.ours, "B"]].forEach(([q, c, n]) => {
      ctx.strokeStyle = c;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let t = 0; t <= steps; t++) { const z = pos(q, t); t ? ctx.lineTo(z[0], z[1]) : ctx.moveTo(z[0], z[1]); }
      ctx.stroke();
      ctx.setLineDash([]);
      const z = pos(q, steps);
      ctx.beginPath();
      ctx.arc(z[0], z[1], 8, 0, 7);
      ctx.fillStyle = c;
      ctx.fill();
      text(ctx, n, z[0] + 12, z[1] - 10, 16, c, { weight: 700 });
    });
    const a = pos(A, steps), b = pos(B, steps);
    axisLine(ctx, a[0], a[1], b[0], b[1], col.blue, [2, 3]);
    text(ctx, `the gap shrinks by 0.82 each step: now ${(Math.hypot(a[0] - b[0], a[1] - b[1]) / (bs / 2)).toFixed(2)}`, 34, 360, 15, col.blue, { weight: 700 });
    text(ctx, "illustrative: two states in 2-D", 34, 382, 12.5, col.muted);
  }
  function plainCircle(ctx, p) {
    box(ctx, 14, 8, 572, 384);
    const cx = 300, cy = 200, R = 130;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(15,118,110,0.08)";
    ctx.fill();
    ctx.strokeStyle = col.teal;
    ctx.lineWidth = 3;
    ctx.stroke();
    axisLine(ctx, cx - R - 30, cy, cx + R + 30, cy);
    axisLine(ctx, cx, cy - R - 24, cx, cy + R + 24);
    text(ctx, "unit circle", cx + R * 0.72, cy - R * 0.72 - 10, 15, col.teal, { weight: 700 });
    text(ctx, "real", cx + R + 6, cy + 20, 13, col.muted);
    text(ctx, "imaginary", cx + 8, cy - R - 28, 13, col.muted);
    const r = rng(99), pts = [];
    for (let i = 0; i < 14; i++) { const rad = 0.9 * Math.sqrt(r()), ang = 2 * Math.PI * r(); pts.push([rad * Math.cos(ang), rad * Math.sin(ang)]); }
    const n = Math.floor(pts.length * ease(seg(p, 0, 0.8)));
    pts.slice(0, n).forEach((q) => {
      ctx.beginPath();
      ctx.arc(cx + q[0] * R, cy - q[1] * R, 6, 0, 7);
      ctx.fillStyle = col.teal;
      ctx.fill();
    });
    ctx.globalAlpha = seg(p, 0.7, 1);
    ctx.beginPath();
    ctx.arc(cx + 1.25 * R, cy - 0.3 * R, 6, 0, 7);
    ctx.strokeStyle = col.ours;
    ctx.lineWidth = 2.4;
    ctx.stroke();
    text(ctx, "never produced", cx + 1.25 * R + 12, cy - 0.3 * R + 5, 13, col.ours, { weight: 700 });
    ctx.globalAlpha = 1;
    text(ctx, "teal dots: the roots (eigenvalues) of random models; illustrative", 34, 382, 12.5, col.muted);
  }
  const PLAIN = {
    sw: { title: "In plain words: the past must fade", think: "An echo in a room. Every past value comes back with some strength. If the echoes together are weaker than the original sound, it dies out instead of building up.",
      rule: "Take the largest effect each past value can have on the next one. Add these up. Keep the sum below 1.",
      eng: "A small-gain condition: total gain around a feedback loop below 1, here worked out for a nonlinear equation with many lags." },
    jac: { title: "In plain words: states get closer and stay inside", think: "Start two copies of the system in different states. Every step pulls them closer, and guard rails keep both inside a box.",
      rule: "Bound how much one step can stretch distances (a Lipschitz constant below 1) on the box, and bound the noise so the state never leaves it.",
      eng: "A contraction mapping, as in the Banach fixed-point theorem: iterate a shrinking map and the starting point is forgotten." },
    hvar: { title: "In plain words: build it stable from the start", think: "Do not draw a random filter and test it. Pass random numbers through a mapping whose output is always a stable set of coefficients.",
      rule: "Every root of the model (eigenvalue of its companion matrix) sits inside the unit circle.",
      eng: "The usual stability condition of a digital filter: poles inside the unit circle. It covers linear models only." },
  };

  // =============================== Shao–Wu ===================================
  const GAINS = D.certificate.lag_gains;
  const RAW = GAINS.map((g) => g / D.scale);
  const LAGS = GAINS.map((_, i) => `x(t${MINUS}${i + 1})`);
  const barColor = (vals) => (i) => (vals[i] > 0 ? col.teal : col.line);

  // Two runs of the demo's own final equation: same noise, different starting histories.
  let coupling = null;
  function runCoupling() {
    if (coupling) return coupling;
    const N = 240, L = D.order, R = D.initial_radius;
    const evalAt = (h, t) => D.final_poly.reduce((acc, term) => {
      let v = term.coef;
      term.atoms.forEach((a) => { const x = h[t - a[1]]; v *= a[2] === "tanh" ? Math.tanh(x) : x; });
      return acc + v;
    }, 0);
    const noise = rng(5), eps = Array.from({ length: N }, () => D.innovation_std * gauss(noise));
    const mk = (seed) => {
      const r = rng(seed), h = Array.from({ length: L }, () => (2 * r() - 1) * R);
      for (let t = L; t < L + N; t++) h.push(evalAt(h, t) + eps[t - L]);
      return h;
    };
    const a = mk(11), b = mk(12);
    coupling = { a, b, gap: a.map((v, i) => Math.abs(v - b[i])), N, L };
    return coupling;
  }

  function swIdea(ctx, p) {
    const L = D.order;
    text(ctx, "the next value reads the past …", 40, 40, 18, col.muted, { weight: 600 });
    for (let i = 0; i < L; i++) {
      const lag = L - i, x = 30 + i * 86, y = 70;
      const used = GAINS[lag - 1] > 0;
      ctx.globalAlpha = seg(p, i * 0.05, i * 0.05 + 0.2);
      box(ctx, x, y, 72, 38, { r: 8, fill: used ? col.tsoft : col.panel, stroke: used ? col.teal : col.line });
      text(ctx, `x(t${MINUS}${lag})`, x + 36, y + 25, 15, used ? col.teal : col.muted, { align: "center", weight: 600 });
      const a = seg(p, 0.35 + i * 0.04, 0.55 + i * 0.04);
      ctx.globalAlpha = a;
      arrow(ctx, x + 36, y + 40, 360, 262, used ? col.teal : col.line, 2, 1);
      text(ctx, `a${"₀₁₂₃₄₅₆₇₈"[lag]}`, x + 36, y + 66 + (i % 2) * 14, 15, used ? col.teal : col.muted, { align: "center", weight: 700 });
      ctx.globalAlpha = 1;
    }
    box(ctx, 300, 262, 120, 44, { fill: col.osoft, stroke: col.ours });
    text(ctx, "x(t)", 360, 292, 22, col.ours, { align: "center", weight: 700 });
    const a = seg(p, 0.6, 0.95);
    ctx.globalAlpha = a;
    box(ctx, 740, 50, 450, 330);
    text(ctx, "aℓ  =  largest effect of x(t−ℓ)", 764, 92, 19, col.teal, { weight: 700 });
    wrap(ctx, "on x(t): the biggest possible change of the next value when x(t−ℓ) moves by one unit, over every possible input.", 764, 120, 400, 24, 17, col.ink);
    text(ctx, "c  =  a₁ + a₂ + … + a" + "₀₁₂₃₄₅₆₇₈"[L], 764, 232, 20, col.ink, { weight: 700, family: Z.MATH });
    wrap(ctx, "the total pull of the past. If c < 1, a difference in the past is multiplied by less than 1 at every step, so it fades.", 764, 262, 400, 24, 17, col.ink);
    ctx.globalAlpha = 1;
  }

  function swBounds(ctx, p, scaled) {
    const f = scaled ? 1 + (D.scale - 1) * ease(seg(p, 0.15, 0.7)) : 1;
    const vals = RAW.map((v) => v * (scaled ? f : 1));
    vbars(ctx, { x: 40, y: 50, w: 560, h: 270 }, vals, 1.8, LAGS, barColor(vals), scaled ? 1 : ease(seg(p, 0, 0.6)));
    text(ctx, "ℓ: how far back  ·  bar = aℓ", 40, 28, 16, col.muted, { weight: 600 });
    const total = vals.reduce((s, v) => s + v, 0);
    box(ctx, 650, 20, 540, 360);
    text(ctx, scaled ? "After scaling" : "Total pull of the past", 672, 56, 22, col.ink, { weight: 700 });
    const ok = total < 1;
    text(ctx, `c = ${total.toFixed(2)}`, 672, 118, 40, ok ? col.teal : col.ours, { weight: 700, family: Z.MATH });
    meter(ctx, 672, 170, 480, total, 5, ok ? col.teal : col.ours, "must stay below 1");
    if (!scaled) {
      wrap(ctx, `The bounds come from the tree itself. Lag 2 multiplies the factor tanh x(t${MINUS}8) + ${(RAW[1] - 1).toFixed(2)}, so its bound reaches ${RAW[1].toFixed(2)}. The sum is far above 1: left alone, the past could be amplified.`, 672, 254, 490, 24, 17, col.ink);
    } else {
      const a = seg(p, 0.55, 0.9);
      ctx.globalAlpha = a;
      text(ctx, `s = ${D.target_gain} × ${D.safety} / ${(D.target_gain * D.safety / D.scale).toFixed(2)} = ${D.scale.toFixed(4)}`, 672, 250, 22, col.blue, { weight: 700, family: Z.MATH });
      wrap(ctx, `Every coefficient is multiplied by s, so every bar shrinks by the same factor. Target 0.8, times a 0.95 safety margin, gives ${D.certificate.total_gain.toFixed(2)} < 1. The shape of the equation does not change.`, 672, 282, 490, 24, 17, col.ink);
      ctx.globalAlpha = 1;
    }
  }

  function swFade(ctx, p) {
    const C = runCoupling(), n = Math.max(2, Math.floor((C.L + C.N) * seg(p, 0, 0.85)));
    const all = C.a.concat(C.b), lo = Math.min(...all) - 0.1, hi = Math.max(...all) + 0.1;
    const bx = 30, bw = 740, X = (i) => bx + (i / (C.L + C.N - 1)) * bw;
    const top = { y: 14, h: 190 }, Y = (v) => top.y + top.h - ((v - lo) / (hi - lo)) * top.h;
    box(ctx, 14, 4, 770, 212);
    axisLine(ctx, bx, Y(0), bx + bw, Y(0));
    axisLine(ctx, X(C.L - 0.5), top.y, X(C.L - 0.5), top.y + top.h, col.muted, [4, 4]);
    text(ctx, "dashed line: end of the starting history", X(C.L) + 10, top.y + 18, 13, col.muted);
    line(ctx, C.a.slice(0, n).map((v, i) => [X(i), Y(v)]), col.ink, 2.2);
    line(ctx, C.b.slice(0, n).map((v, i) => [X(i), Y(v)]), col.ours, 2.2);
    const bot = { y: 246, h: 130 }, floor = -14, ceil = 1;
    const LY = (g) => bot.y + bot.h - ((Math.max(floor, Math.log10(Math.max(g, 1e-300))) - floor) / (ceil - floor)) * bot.h;
    box(ctx, 14, 224, 770, 168);
    text(ctx, "log₁₀ of the gap between the two runs", 30, 242, 14, col.muted, { weight: 600 });
    axisLine(ctx, X(C.L - 0.5), bot.y, X(C.L - 0.5), bot.y + bot.h, col.muted, [4, 4]);
    line(ctx, C.gap.slice(0, n).map((g, i) => [X(i), LY(g)]), col.blue, 2.4);
    box(ctx, 806, 4, 384, 388);
    text(ctx, "Same equation, same noise", 826, 40, 20, col.ink, { weight: 700 });
    wrap(ctx, "Two runs of this example's final equation start from different random histories (black and orange) but receive exactly the same noise.", 826, 70, 346, 24, 16, col.ink);
    wrap(ctx, "The gap between them keeps shrinking, at least geometrically: the starting history is forgotten. This is what c < 1 means.", 826, 178, 346, 24, 16, col.ink);
    text(ctx, "Computed live from this example's equation.", 826, 350, 13, col.muted);
  }

  function swTanh(ctx, p) {
    box(ctx, 14, 8, 590, 384);
    text(ctx, "Why tanh, and only inside products", 34, 42, 20, col.ink, { weight: 700 });
    const a = seg(p, 0, 0.35);
    ctx.globalAlpha = a;
    box(ctx, 34, 66, 550, 120, { fill: "#fdeaea", stroke: col.ours });
    text(ctx, "x(t−1) · x(t−2)", 54, 102, 24, col.ours, { weight: 700, family: Z.MATH });
    text(ctx, "∂/∂x(t−1) = x(t−2): no finite bound", 54, 136, 17, col.ink);
    text(ctx, "no global certificate", 54, 164, 16, col.ours, { weight: 700 });
    ctx.globalAlpha = seg(p, 0.3, 0.65);
    box(ctx, 34, 204, 550, 120, { fill: col.tsoft, stroke: col.teal });
    text(ctx, "tanh x(t−1) · tanh x(t−2)", 54, 240, 24, col.teal, { weight: 700, family: Z.MATH });
    text(ctx, "each derivative is at most 1 × 1", 54, 274, 17, col.ink);
    text(ctx, "certificate holds", 54, 302, 16, col.teal, { weight: 700 });
    ctx.globalAlpha = seg(p, 0.6, 0.9);
    text(ctx, "Terms that are linear in the past keep their raw lags.", 34, 356, 16, col.muted);
    ctx.globalAlpha = 1;
    // right: measured explosions
    box(ctx, 626, 8, 564, 384);
    text(ctx, "Explosions in 64 trajectories per case", 646, 42, 18, col.ink, { weight: 700 });
    text(ctx, "384 steps, |x| > 100, stress start in [2, 3]; k = lags in the product", 646, 64, 13, col.muted);
    const groups = [["k = 2", 17.19], ["k = 3", 100], ["k = 4", 100]];
    groups.forEach((g, i) => {
      const x = 660 + i * 170, base = 316, hmax = 190, k = ease(seg(p, 0.4 + i * 0.08, 0.8 + i * 0.08));
      const h1 = (g[1] / 100) * hmax * k;
      ctx.fillStyle = col.ours;
      ctx.fillRect(x, base - h1, 56, h1);
      ctx.fillStyle = col.teal;
      ctx.fillRect(x + 64, base - 3, 56, 3);
      text(ctx, g[1].toFixed(g[1] < 100 ? 1 : 0) + " %", x + 28, base - h1 - 8, 14, col.ink, { align: "center", weight: 700 });
      text(ctx, "0 %", x + 92, base - 12, 14, col.teal, { align: "center", weight: 700 });
      text(ctx, g[0], x + 60, base + 22, 15, col.ink, { align: "center", weight: 600 });
    });
    axisLine(ctx, 650, 316, 1170, 316);
    ctx.fillStyle = col.ours; ctx.fillRect(660, 350, 14, 14);
    text(ctx, "no tanh", 680, 363, 14, col.ink);
    ctx.fillStyle = col.teal; ctx.fillRect(770, 350, 14, 14);
    text(ctx, "tanh on each factor", 790, 363, 14, col.ink);
    text(ctx, "default start in [−1, 1]: none exploded", 960, 363, 13, col.muted);
  }

  const SW_ROWS = [
    ["Criterion", "Σℓ ‖Dℓ‖₂ < 1, where Dℓ bounds the derivatives of the next value with respect to lag ℓ", "Shao and Wu (2007), Theorem 5.1, pp. 11–12", true],
    ["Noise", "Gaussian, σ = 0.05", "theorem condition: finite second moment", true],
    ["Target gain, safety", "0.8 × 0.95 = 0.76 (maximum certified gain in the 128-mechanism run)", "project choice", false],
    ["Trees and tanh", "random ADD / MULTIPLY trees; tanh only on lag factors inside nonlinear products", "project adaptation; the paper does not use trees", false],
    ["Starting history", "uniform in [−1, 1], drawn directly", "project choice: the theorem gives a stationary solution, this start is not drawn from it, so an initial transient is possible", false],
    ["What is shown", "0 of 128 mechanisms crossed |x| > 100 within 384 steps", "an observation, not a proof for all times", null],
  ];

  // =============================== Jacobian ==================================
  function jacIdea(ctx, p) {
    text(ctx, "state s(t): the last L values", 40, 40, 18, col.muted, { weight: 600 });
    const names = ["x(t)", "x(t−1)", "x(t−2)", "x(t−3)", "x(t−4)"];
    const cw = 92, gap = 16;
    names.forEach((n, i) => {
      const x = 30 + i * (cw + gap), a = seg(p, i * 0.05, i * 0.05 + 0.2);
      ctx.globalAlpha = a;
      box(ctx, x, 60, cw, 44, { r: 8, fill: col.tsoft, stroke: col.teal });
      text(ctx, n, x + cw / 2, 88, 16, col.teal, { align: "center", weight: 600 });
      ctx.globalAlpha = 1;
    });
    text(ctx, "…", 30 + 5 * (cw + gap) + 4, 90, 24, col.muted, { weight: 700 });
    const b = seg(p, 0.35, 0.8);
    ctx.globalAlpha = b;
    text(ctx, "next state s(t+1)", 40, 218, 18, col.muted, { weight: 600 });
    const names2 = ["x(t+1)", "x(t)", "x(t−1)", "x(t−2)", "x(t−3)"];
    names2.forEach((n, i) => {
      const x = 30 + i * (cw + gap), y = 236;
      box(ctx, x, y, cw, 44, { r: 8, fill: i === 0 ? col.osoft : col.tsoft, stroke: i === 0 ? col.ours : col.teal });
      text(ctx, n, x + cw / 2, y + 28, 16, i === 0 ? col.ours : col.teal, { align: "center", weight: 600 });
      if (i === 0) {
        names.forEach((_, j) => arrow(ctx, 30 + j * (cw + gap) + cw / 2, 106, x + cw / 2, y - 2, "#d8a07a", 1.4, 1));
      } else {
        arrow(ctx, 30 + (i - 1) * (cw + gap) + cw / 2, 106, x + cw / 2, y - 2, col.line, 1.6, 1);
      }
    });
    text(ctx, "tree(all cells) + noise", 30 + cw / 2, 308, 14, col.ours, { align: "center", weight: 700 });
    text(ctx, "shift: copies", 30 + 2.5 * (cw + gap), 308, 14, col.muted, { align: "center" });
    ctx.globalAlpha = seg(p, 0.65, 1);
    box(ctx, 650, 28, 540, 350);
    text(ctx, "A map that shrinks distances", 672, 66, 21, col.ink, { weight: 700 });
    wrap(ctx, "If the map from s(t) to s(t+1) has Lipschitz bound k < 1 on a box, two different states get closer at every step, whatever they were.", 672, 96, 496, 24, 17, col.ink);
    wrap(ctx, "The first row of the Jacobian holds the tree's derivatives; the other rows only shift. Weights q^j with q = 1.25 make the shift rows cost 1/q = 0.8. The certificate is the larger of the two numbers.", 672, 200, 496, 24, 17, col.ink);
    ctx.globalAlpha = 1;
  }

  function jacLip(ctx, p) {
    const panels = [[14, 0.7, col.teal, "k = 0.7: the gap shrinks"], [316, 1.3, col.ours, "k = 1.3: the gap grows"]];
    panels.forEach(([x0, k, c, title], j) => {
      box(ctx, x0, 8, 292, 384);
      text(ctx, title, x0 + 16, 40, 17, c, { weight: 700 });
      text(ctx, "gap between two states, step by step (starts at 1)", x0 + 16, 62, 12, col.muted);
      const base = 340, hmax = 240, n = 6, slot = 252 / n;
      axisLine(ctx, x0 + 16, base, x0 + 276, base);
      for (let t = 0; t < n; t++) {
        const g = Math.pow(k, t), a = seg(p, t * 0.12, t * 0.12 + 0.25), h = Math.min(1, g / 4) * hmax * a;
        ctx.fillStyle = c;
        ctx.fillRect(x0 + 20 + t * slot, base - h, slot * 0.6, h);
        ctx.globalAlpha = a;
        text(ctx, g.toFixed(2), x0 + 20 + t * slot + slot * 0.3, base - h - 8, 13, col.ink, { align: "center", weight: 700 });
        text(ctx, "t=" + t, x0 + 20 + t * slot + slot * 0.3, base + 20, 12, col.muted, { align: "center" });
        ctx.globalAlpha = 1;
      }
    });
    ctx.globalAlpha = seg(p, 0.55, 1);
    box(ctx, 630, 8, 560, 384);
    text(ctx, "Lipschitz bound k", 652, 44, 21, col.ink, { weight: 700 });
    wrap(ctx, "A map F has Lipschitz bound k if two inputs never end up farther apart than k times their original distance:", 652, 74, 516, 23, 16, col.ink);
    text(ctx, "|F(s) − F(s′)| ≤ k · |s − s′|", 652, 150, 22, col.blue, { weight: 700, family: Z.MATH });
    wrap(ctx, "Here F is one step of the recurrence and s, s′ are two states. With k < 1 every step brings them closer, so the starting state is forgotten. With k > 1 gaps can grow, which is how series explode.", 652, 180, 516, 23, 16, col.ink);
    text(ctx, "Where the Jacobian comes in", 652, 292, 18, col.ink, { weight: 700 });
    wrap(ctx, "The Jacobian J(s) is the matrix of all partial derivatives of F at s. k is at most the largest norm of J over the whole box, so bounding the derivatives bounds k.", 652, 318, 516, 23, 16, col.ink);
    ctx.globalAlpha = 1;
  }

  function jacBox(ctx, p) {
    const a = seg(p, 0, 0.3);
    ctx.globalAlpha = a;
    text(ctx, "1  Gate every lag", 40, 44, 20, col.ink, { weight: 700 });
    box(ctx, 40, 66, 180, 54, { fill: col.tsoft, stroke: col.teal });
    text(ctx, "x(t−3)", 130, 100, 20, col.teal, { align: "center", weight: 700 });
    arrow(ctx, 224, 93, 286, 93, col.muted, 2);
    box(ctx, 290, 66, 280, 54, { fill: col.soft, stroke: col.blue });
    text(ctx, "P_g · x(t−3)", 430, 100, 20, col.blue, { align: "center", weight: 700 });
    text(ctx, "P_g is a signed coefficient drawn per lag (coefficient gating).", 40, 150, 15, col.muted);
    ctx.globalAlpha = seg(p, 0.3, 0.65);
    text(ctx, "2  Work on a box", 40, 214, 20, col.ink, { weight: 700 });
    box(ctx, 40, 236, 530, 110, { fill: col.grey });
    text(ctx, "every lag stays in [−1, 1]  (radius 1)", 62, 276, 19, col.ink, { weight: 600 });
    text(ctx, "interval arithmetic bounds the value and every derivative", 62, 308, 16, col.muted);
    text(ctx, "of every node of the tree on the whole box.", 62, 330, 16, col.muted);
    ctx.globalAlpha = seg(p, 0.65, 1);
    box(ctx, 640, 28, 550, 350);
    text(ctx, "Why products are allowed here", 662, 66, 21, col.ink, { weight: 700 });
    wrap(ctx, "On a bounded box, x(t−1)·x(t−2) has a finite derivative: |x(t−2)| ≤ 1. So the tree needs no tanh and can keep unsaturated products.", 662, 96, 506, 24, 17, col.ink);
    wrap(ctx, "The price is that the guarantee is only valid inside the box. The next two steps keep the state inside it.", 662, 220, 506, 24, 17, col.ink);
    ctx.globalAlpha = 1;
  }

  function jacScale(ctx, p) {
    const X = (v) => 600 + v * 400, R = 1, nb = Math.sqrt(3) * D.innovation_std;
    ctx.fillStyle = col.grey;
    ctx.fillRect(X(-R), 60, X(R) - X(-R), 300);
    text(ctx, "box: radius 1", X(-R) + 10, 80, 15, col.muted, { weight: 700 });
    axisLine(ctx, X(0), 60, X(0), 276, col.muted, [4, 4]);
    const k = ease(seg(p, 0.2, 0.7));
    const half = 1.3 + (R - nb - 1.3) * k;
    ctx.fillStyle = k < 0.99 ? col.ours : col.blue;
    ctx.fillRect(X(-half), 120, X(half) - X(-half), 44);
    text(ctx, "range of the tree output", 600, 112, 16, col.ink, { align: "center", weight: 600 });
    text(ctx, k < 0.5 ? "too wide: leaves the box" : "scaled to fit", 600, 150, 16, "#fff", { align: "center", weight: 700 });
    const n = seg(p, 0.7, 1);
    ctx.globalAlpha = n;
    ctx.fillStyle = col.ours;
    ctx.fillRect(X(-half) - (X(nb) - X(0)), 220, X(nb) - X(0), 44);
    ctx.fillRect(X(half), 220, X(nb) - X(0), 44);
    ctx.fillStyle = col.blue;
    ctx.fillRect(X(-half), 220, X(half) - X(-half), 44);
    text(ctx, `plus the noise bound ${nb.toFixed(3)}: still inside the box`, 600, 300, 17, col.ink, { align: "center", weight: 600 });
    ctx.globalAlpha = 1;
    text(ctx, "schematic: the widths are not from a particular mechanism", 600, 384, 13, col.muted, { align: "center" });
  }

  function jacNoise(ctx, p) {
    const x0 = 60, x1 = 660, sx = (v) => x0 + ((v + 0.3) / 0.6) * (x1 - x0), sd = D.innovation_std, ub = Math.sqrt(3) * sd;
    const base = 330, hs = 36, pdfG = (v) => Math.exp(-(v * v) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
    axisLine(ctx, x0, base, x1, base);
    const k = seg(p, 0, 0.7);
    const pts = [];
    for (let v = -0.3; v <= 0.3001; v += 0.004) pts.push([sx(v), base - pdfG(v) * hs * k]);
    line(ctx, pts, col.ours, 3);
    ctx.fillStyle = col.ours; ctx.fillRect(60, 50, 26, 5);
    text(ctx, "Gaussian, σ = 0.05: unbounded", 96, 58, 17, col.ours, { weight: 700 });
    ctx.globalAlpha = seg(p, 0.35, 0.9);
    const hu = (1 / (2 * ub)) * hs;
    line(ctx, [[sx(-ub), base], [sx(-ub), base - hu], [sx(ub), base - hu], [sx(ub), base]], col.blue, 3);
    ctx.fillStyle = col.blue; ctx.fillRect(60, 80, 26, 5);
    text(ctx, "uniform, same σ: bounded", 96, 88, 17, col.blue, { weight: 700 });
    text(ctx, `${MINUS}${ub.toFixed(3)}`, sx(-ub), base + 22, 15, col.blue, { align: "center", weight: 700 });
    text(ctx, `+${ub.toFixed(3)}`, sx(ub), base + 22, 15, col.blue, { align: "center", weight: 700 });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = seg(p, 0.6, 1);
    box(ctx, 700, 28, 490, 350);
    text(ctx, "Noise must be bounded too", 722, 66, 21, col.ink, { weight: 700 });
    wrap(ctx, "A box is invariant only if no shock can push the state out of it. A Gaussian shock can be arbitrarily large, so no finite box works.", 722, 96, 446, 24, 17, col.ink);
    wrap(ctx, `Uniform noise on ±√3·σ = ±${ub.toFixed(3)} has the same standard deviation 0.05 and never leaves the allowance left over by the scale.`, 722, 214, 446, 24, 17, col.ink);
    ctx.globalAlpha = 1;
  }

  const JAC_ROWS = [
    ["Certificate", "weighted Lipschitz bound < 1 on the box [−1, 1]^L: derivative sum with weights q^j (q = 1.25), shift rows 1/q; target 0.9, safety 0.95", "project proposal; no paper theorem is claimed", false],
    ["Scale", "s = 0.95 · min(1, 0.9 / derivative gain, (radius − noise bound) / largest tree value)", "project construction", false],
    ["Trees", "random ADD / MULTIPLY trees, every lag multiplied by a signed coefficient, no tanh", "project; products allowed because the box is bounded", false],
    ["Noise", "uniform in ±0.0866 (σ = 0.05)", "needed for a deterministic box; Gaussian would not be invariant", false],
    ["Starting history", "uniform in the box [−1, 1]", "project choice", false],
    ["Checks", "analytic certificate, plus Monte Carlo evaluation of local Jacobian norms", "Monte Carlo checks the code; it does not create the guarantee", null],
  ];

  // ================================ Heaps ====================================
  function hvFlow(ctx, p) {
    const items = [
      ["Free numbers", "u ∼ 𝒩(0, 0.45²): any matrix U per lag"],
      ["Partial autocorrelations", "P = (I + UUᵀ)^(−½) U: every singular value below 1"],
      ["Coefficients φ", "symmetric recursions also give the autocovariances"],
      ["Stationary VAR(p)", "linear equations, stored as ASTs"],
    ];
    items.forEach((it, i) => {
      const x = 20 + i * 296, a = ease(seg(p, i * 0.17, i * 0.17 + 0.3));
      ctx.globalAlpha = a;
      box(ctx, x, 70, 266, 170, { fill: i === 3 ? col.osoft : col.panel, stroke: i === 3 ? col.ours : col.blue, lw: 2.4 });
      text(ctx, it[0], x + 16, 106, 20, i === 3 ? col.ours : col.blue, { weight: 700 });
      wrap(ctx, it[1], x + 16, 138, 236, 23, 16, col.ink);
      if (i < 3) arrow(ctx, x + 268, 155, x + 294, 155, col.muted, 2);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    box(ctx, 20, 272, 1160, 100, { fill: col.tsoft, stroke: col.teal });
    text(ctx, "No rejection step: every draw gives a stable model.", 44, 312, 23, col.ink, { weight: 700 });
    text(ctx, "The map from free numbers to coefficients can only produce stationary VARs (Heaps, 2023).", 44, 344, 17, col.muted);
    ctx.globalAlpha = 1;
  }

  let heapsPts = null;
  function hvPoints() {
    if (heapsPts) return heapsPts;
    const r = rng(2024), N = 400, naive = [], heaps = [];
    for (let i = 0; i < N; i++) {
      const a = (4 * r() - 2), b = (2 * r() - 1);
      naive.push([a, b, a + b < 1 && b - a < 1 && Math.abs(b) < 1]);
      const sq = (u) => u / Math.sqrt(1 + u * u), p1 = sq(0.45 * gauss(r)), p2 = sq(0.45 * gauss(r));
      heaps.push([p1 * (1 - p2), p2, true]);
    }
    heapsPts = { naive, heaps, N };
    return heapsPts;
  }
  function hvPlane(ctx, p) {
    const P = hvPoints();
    const panel = (x0, title, pts, shown) => {
      const w = 560, h = 300, X = (v) => x0 + ((v + 2.2) / 4.4) * w, Y = (v) => 60 + h - ((v + 1.2) / 2.4) * h;
      box(ctx, x0 - 12, 6, w + 24, 388);
      text(ctx, title, x0, 36, 18, col.ink, { weight: 700 });
      ctx.beginPath();
      ctx.moveTo(X(-2), Y(-1)); ctx.lineTo(X(2), Y(-1)); ctx.lineTo(X(0), Y(1)); ctx.closePath();
      ctx.fillStyle = "rgba(15,118,110,0.10)";
      ctx.fill();
      ctx.strokeStyle = col.teal;
      ctx.lineWidth = 2;
      ctx.stroke();
      axisLine(ctx, X(-2.2), Y(0), X(2.2), Y(0));
      axisLine(ctx, X(0), Y(1.2), X(0), Y(-1.2));
      text(ctx, "φ₁", X(2.2) - 22, Y(0) - 8, 15, col.muted, { weight: 700 });
      text(ctx, "φ₂", X(0) + 8, Y(1.2) + 14, 15, col.muted, { weight: 700 });
      const n = Math.floor(pts.length * shown);
      let ins = 0;
      for (let i = 0; i < n; i++) {
        const q = pts[i];
        if (q[2]) ins++;
        ctx.beginPath();
        ctx.arc(X(q[0]), Y(q[1]), 3.2, 0, Math.PI * 2);
        ctx.fillStyle = q[2] ? col.teal : "#c9b8b0";
        ctx.fill();
      }
      return [ins, n];
    };
    const sh = seg(p, 0, 0.85);
    const a = panel(30, "Uniform draws, then keep the stable ones", P.naive, sh);
    const b = panel(610, "Heaps: free draws mapped to coefficients", P.heaps, sh);
    text(ctx, `${a[0]} of ${a[1]} stable`, 30, 384, 17, col.ours, { weight: 700 });
    text(ctx, `${b[0]} of ${b[1]} stable`, 610, 384, 17, col.teal, { weight: 700 });
  }

  function hvChecks(ctx, p) {
    const items = [
      ["spectral radius", "ρ(B) < 1", "B is the companion matrix: the lags stacked into one big matrix"],
      ["covariance", "V positive definite", "the stationary covariance of the stacked state"],
      ["Lyapunov equation", "V = B V B? + Q", "residual checked for 128 mechanisms"],
    ];
    items.forEach((it, i) => {
      const x = 20 + i * 396, a = ease(seg(p, i * 0.2, i * 0.2 + 0.3));
      ctx.globalAlpha = a;
      box(ctx, x, 40, 368, 200, { stroke: col.teal, lw: 2.4 });
      text(ctx, "✓  " + it[0], x + 18, 80, 20, col.teal, { weight: 700 });
      text(ctx, it[1], x + 18, 124, 26, col.ink, { weight: 700, family: Z.MATH });
      wrap(ctx, it[2], x + 18, 160, 330, 23, 16, col.muted);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    box(ctx, 20, 268, 1160, 104, { fill: col.tsoft, stroke: col.teal });
    text(ctx, "128 of 128 mechanisms certified.", 44, 308, 23, col.ink, { weight: 700 });
    text(ctx, "Each mechanism is checked after construction, so a bug in the mapping would show up here.", 44, 340, 17, col.muted);
    ctx.globalAlpha = 1;
  }

  let startRuns = null;
  function hvRuns() {
    if (startRuns) return startRuns;
    const f1 = 0.5, f2 = -0.3, sd = D.innovation_std, N = 60;
    const g0 = (sd * sd * (1 - f2)) / ((1 + f2) * ((1 - f2) * (1 - f2) - f1 * f1)), rho = f1 / (1 - f2);
    const mk = (kind, seed) => {
      const r = rng(seed), h = [];
      if (kind === "uniform") h.push(2 * r() - 1, 2 * r() - 1);
      else { const z1 = gauss(r), z2 = gauss(r), s = Math.sqrt(g0); h.push(s * z1, s * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)); }
      for (let t = 2; t < N; t++) h.push(f1 * h[t - 1] + f2 * h[t - 2] + sd * gauss(r));
      return h;
    };
    startRuns = { u: [1, 2, 3].map((s) => mk("uniform", 40 + s)), s: [1, 2, 3].map((s) => mk("stat", 70 + s)), N };
    return startRuns;
  }
  function hvStart(ctx, p) {
    const R = hvRuns(), n = Math.max(2, Math.floor(R.N * seg(p, 0, 0.85)));
    const cols = [col.ink, col.ours, col.blue];
    [[R.u, 14, "Start uniform in [−1, 1]", "an initial transient, then the stationary level"], [R.s, 610, "Start from the stationary distribution", "stationary from the first value"]].forEach(([runs, x0, title, sub]) => {
      const w = 570, top = 70, h = 250, X = (i) => x0 + 14 + (i / (R.N - 1)) * (w - 28), Y = (v) => top + h / 2 - (v / 1.1) * (h / 2);
      box(ctx, x0, 6, w, 388);
      text(ctx, title, x0 + 16, 36, 18, col.ink, { weight: 700 });
      text(ctx, sub, x0 + 16, 58, 14, col.muted);
      axisLine(ctx, X(0), Y(0), X(R.N - 1), Y(0));
      runs.forEach((ys, k) => line(ctx, ys.slice(0, n).map((v, i) => [X(i), Y(v)]), cols[k], 2));
      text(ctx, "t = 0", X(0), top + h + 24, 13, col.muted);
    });
    text(ctx, "illustrative AR(2): φ = (0.5, −0.3), σ = 0.05; three runs each. Heaps draws the start from the stationary normal of each mechanism.", 20, 392, 13, col.muted);
  }

  const HV_ROWS = [
    ["Mapping", "P = (I + UUᵀ)^(−½) U through the SVD identity u → u / √(1 + u²); symmetric recursions give φ and the autocovariances", "Heaps (2023), appendix pp. 11–12, supplement S1.3; written from the equations", true],
    ["Free numbers", "entries ∼ 𝒩(0, 0.45²), order 1 to 4 (stationary-VAR notebook)", "project choice of the sampling law", false],
    ["Family", "linear Gaussian VAR(p), stored as linear ASTs. No products, no tanh", "a different family from the trees: it cannot express nonlinear mechanisms", null],
    ["Starting history", "drawn from the joint stationary normal", "project choice; no initial transient", false],
    ["Certificate", "ρ(B) < 1, V positive definite, Lyapunov residual 1.6 × 10⁻¹⁵", "checked for each mechanism: 128 of 128", null],
    ["Cost", "slowest sampler: 327 mechanisms per second, against 2 691 without protection", "final-method experiment, one machine", null],
  ];

  // ============================== registrations ==============================
  const FOOT6 = "Comparison numbers: final_methods/00_final_methods_tradeoff.ipynb (128 mechanisms per method, one series, 384 steps; Shao-Wu product_lags). Explosion = non-finite or |x| > 100.";
  G.add("sw", "Shao–Wu", {
    title: "Shao–Wu contraction",
    foot: "Shao and Wu (2007), Theorem 5.1, adapted in src/tsfmxai/generators/shao_wu.py; experiments/abstract_syntax_tree/final_methods/00_final_methods_tradeoff.ipynb. Bars and gap plot use this example's equation (seed " + D.seed + "). " + FOOT6,
    steps: [
      { narr: "In plain words: the past must fade", sub: "one of the four constructions", formula: "", dur: 5200 },
      { narr: "The criterion in symbols: the total pull of the past is below 1", sub: "global moment contraction", formula: "x(t) = s · f(x(t−1), …, x(t−L)) + ε(t),&nbsp;&nbsp; Σ<sub>ℓ</sub> a<sub>ℓ</sub> &lt; 1", dur: 4800 },
      { narr: "① Bound the pull of every lag", sub: "from the tree, over all inputs", formula: "a<sub>ℓ</sub> ≥ |∂x(t) / ∂x(t−ℓ)|&nbsp; for every possible past", dur: 3600 },
      { narr: "② Scale until the sum is 0.76", sub: "one factor for the whole equation", formula: "s = 0.95 · min( 1 , 0.8 / Σ a<sub>ℓ</sub> )&nbsp;&nbsp;<span class='mut'>never above 0.95: an already stable tree is not enlarged</span>", dur: 4200 },
      { narr: "③ What c < 1 buys: the past is forgotten", sub: "two histories, same noise", formula: "gap(t) shrinks geometrically", dur: 8000 },
      { narr: "④ Why tanh, and only inside products", sub: "products of lags have no derivative bound", formula: "tanh x(t−1) · tanh x(t−2),&nbsp; with the linear terms left raw", dur: 4800 },
      { narr: "What it costs", sub: "compared with the other methods", formula: "", dur: 3200 },
      { narr: "⑤ Fine print: paper and project", sub: "what the theorem gives and what was adapted", formula: "", dur: 3600 },
    ],
    paint(ctx, step, p) {
      if (step === 0) { plainEcho(ctx, p); plainRight(ctx, p, PLAIN.sw); }
      else if (step === 1) swIdea(ctx, p);
      else if (step === 2) swBounds(ctx, p, false);
      else if (step === 3) swBounds(ctx, p, true);
      else if (step === 4) swFade(ctx, p);
      else if (step === 5) swTanh(ctx, p);
      else if (step === 6) tradeoff(ctx, p, "sw");
      else table(ctx, p, SW_ROWS);
    },
  }, "stab");
  G.add("jac", "Jacobian", {
    title: "Jacobian contraction",
    foot: "Project proposal: src/tsfmxai/generators/contraction.py; experiments/abstract_syntax_tree/contraction_by_construction. " + FOOT6,
    steps: [
      { narr: "In plain words: states get closer and stay inside", sub: "one of the four constructions", formula: "", dur: 5200 },
      { narr: "The idea: a map that shrinks distances inside a box", sub: "the state is the last L values", formula: "s(t+1) = F(s(t)) + ε(t),&nbsp;&nbsp; Lipschitz bound of F &lt; 1 on the box", dur: 4800 },
      { narr: "What is a Lipschitz bound?", sub: "the number k that must stay below 1", formula: "|F(s) − F(s′)| ≤ k · |s − s′|,&nbsp;&nbsp; k ≤ largest ‖J(s)‖ on the box", dur: 4800 },
      { narr: "① Gate the lags and bound derivatives on the box", sub: "interval propagation", formula: "inputs in [−1, 1]<sup>L</sup>", dur: 4200 },
      { narr: "② Scale so the output fits the box", sub: "with room left for the noise", formula: "s = 0.95 · min(1, 0.9 / g, (1 − 0.0866) / B)&nbsp;&nbsp;<span class='mut'>g: derivative gain, B: largest tree value</span>", dur: 4800 },
      { narr: "③ Bounded noise", sub: "a Gaussian could throw the state out of any box", formula: "ε ∼ uniform(−√3·σ, +√3·σ),&nbsp; σ = 0.05", dur: 4200 },
      { narr: "What it costs", sub: "compared with the other methods", formula: "", dur: 3200 },
      { narr: "④ Fine print: what is proven and what is chosen", sub: "", formula: "", dur: 3600 },
    ],
    paint(ctx, step, p) {
      if (step === 0) { plainFunnel(ctx, p); plainRight(ctx, p, PLAIN.jac); }
      else if (step === 1) jacIdea(ctx, p);
      else if (step === 2) jacLip(ctx, p);
      else if (step === 3) jacBox(ctx, p);
      else if (step === 4) jacScale(ctx, p);
      else if (step === 5) jacNoise(ctx, p);
      else if (step === 6) tradeoff(ctx, p, "jac");
      else table(ctx, p, JAC_ROWS);
    },
  }, "stab");
  G.add("hvar", "Heaps VAR", {
    title: "Heaps stationary VAR",
    foot: "Heaps (2023), appendix; implementation src/tsfmxai/generators/heaps.py; experiments/abstract_syntax_tree/heaps_stationary_var. " + FOOT6,
    steps: [
      { narr: "In plain words: build it stable from the start", sub: "one of the four constructions", formula: "", dur: 5200 },
      { narr: "The idea: draw free numbers, map them to a stable model", sub: "no rejection needed", formula: "u → p = u / √(1 + u²) ∈ (−1, 1) → coefficients φ", dur: 4800 },
      { narr: "① One series: every mapped draw is stationary", sub: "AR(2) as the smallest picture", formula: "φ₂ = p₂,&nbsp; φ₁ = p₁ (1 − p₂)&nbsp;&nbsp;<span class='mut'>one series: the classic partial-autocorrelation recursion</span>", dur: 5200 },
      { narr: "② Each mechanism is checked anyway", sub: "three tests", formula: "", dur: 4000 },
      { narr: "③ The start is already stationary", sub: "no initial transient", formula: "history ∼ joint stationary normal", dur: 7000 },
      { narr: "What it costs", sub: "compared with the other methods", formula: "", dur: 3200 },
      { narr: "④ Fine print: paper and project", sub: "", formula: "", dur: 3600 },
    ],
    paint(ctx, step, p) {
      if (step === 0) { plainCircle(ctx, p); plainRight(ctx, p, PLAIN.hvar); }
      else if (step === 1) hvFlow(ctx, p);
      else if (step === 2) hvPlane(ctx, p);
      else if (step === 3) hvChecks(ctx, p);
      else if (step === 4) hvStart(ctx, p);
      else if (step === 5) tradeoff(ctx, p, "hvar");
      else table(ctx, p, HV_ROWS);
    },
  }, "stab");
})();
