/* Zoom 2: the equation head (symbolic regression).
 *
 * The target and the example come from the project library through
 * scripts/export_generator_demo.py (canonicalize_polynomial, CanonicalCodec).
 * The last step is a plan: no model has been trained on the current RAW/TANH corpus yet,
 * so it shows what will be measured and carries no numbers.
 */
(function () {
  const D = window.GEN_DATA, Z = window.ZC;
  if (!D || !Z) return;
  const { col, seg, ease, text, wrap, box, arrow, fmt, lagName } = Z;
  const W = 1200, H = 500;
  const STAGES = ["① Many trees", "② Canonical form", "③ RAW / TANH", "④ Symbols", "⑤ Model", "⑥ Evidence"];

  const termText = (t) => {
    const atoms = t.atoms.map((a) => (a[2] === "tanh" ? "tanh " : "") + lagName(a[1]));
    return atoms.length ? atoms.join("·") : "(constant)";
  };
  const sameTerm = (a, b) => !!b && termText(a) === termText(b);

  function treeStyle(nd) {
    const n = nd.n;
    if (n.t === "add" || n.t === "mul") return { stroke: col.ink };
    return n.t === "par" ? { fill: col.soft, stroke: col.blue, color: col.blue } : { fill: col.tsoft, stroke: col.teal, color: col.teal };
  }
  const treeLabel = (nd) => {
    const n = nd.n;
    if (n.t === "add") return "+";
    if (n.t === "mul") return "×";
    return n.t === "par" ? fmt(n.v, 2) : lagName(n.lag);
  };

  function scene0(ctx, p) {
    const a1 = ease(seg(p, 0, 0.3)), a2 = ease(seg(p, 0.25, 0.55)), a3 = ease(seg(p, 0.5, 0.85));
    ctx.globalAlpha = a1;
    box(ctx, 20, 90, 300, 130, { fill: col.soft, stroke: col.blue, lw: 2.4 });
    text(ctx, "Shared encoder", 170, 145, 26, col.blue, { align: "center", weight: 700 });
    text(ctx, "reads the input", 170, 178, 19, col.muted, { align: "center" });
    ctx.globalAlpha = a2;
    arrow(ctx, 326, 155, 396, 155, col.muted, 2.4);
    box(ctx, 400, 90, 300, 130, { fill: col.osoft, stroke: col.ours, lw: 2.4 });
    text(ctx, "Equation head", 550, 145, 26, col.ours, { align: "center", weight: 700 });
    text(ctx, "writes the mechanism", 550, 178, 19, col.muted, { align: "center" });
    ctx.globalAlpha = a3;
    arrow(ctx, 706, 155, 756, 155, col.muted, 2.4);
    box(ctx, 760, 70, 420, 170, { fill: col.panel, stroke: col.ours });
    text(ctx, "output: the equation as text", 780, 102, 18, col.muted, { weight: 600 });
    const terms = Z.polyText(D.raw_poly, 3);
    let y = 134;
    terms.forEach((s, i) => {
      text(ctx, (i ? "+ " : "  ") + s, 780, y, 17, col.ink, { family: Z.MATH });
      y += 21;
    });
    ctx.globalAlpha = a3;
    text(ctx, "coefficients of the sampled tree, before the global scale s", 780, 228, 14, col.muted);
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    box(ctx, 20, 290, 1160, 120, { fill: col.grey, stroke: col.line });
    text(ctx, "Why symbolic regression?", 44, 328, 22, col.ink, { weight: 700 });
    wrap(ctx, "A forecast only says what happens. The equation says how the next step is computed, and which past values enter it directly. Because the generator knows its equation, the head can be trained and graded exactly.", 44, 360, 1110, 26, 19, col.ink);
    ctx.globalAlpha = 1;
  }

  function scene1(ctx, p) {
    const names = ["factored", "expanded", "factors swapped"];
    D.same_trees.forEach((t, i) => {
      const a = ease(seg(p, i * 0.16, i * 0.16 + 0.24));
      const x = 20 + i * 396;
      ctx.globalAlpha = a;
      box(ctx, x, 60, 368, 250, { fill: col.panel });
      text(ctx, names[i], x + 16, 88, 18, col.muted, { weight: 600 });
      const L = Z.layout(t, { x: x + 20, y: 130, w: 328, h: 100 });
      Z.drawTree(ctx, L, { label: treeLabel, style: treeStyle, size: 15 });
      wrap(ctx, Z.pretty(t, 2), x + 16, 280, 340, 22, 17, col.ink, { family: Z.MATH });
      ctx.globalAlpha = 1;
    });
    const b = ease(seg(p, 0.55, 0.8));
    [0, 1, 2].forEach((i) => arrow(ctx, 204 + i * 396, 316, 600, 352, col.muted, 2, b));
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    box(ctx, 220, 356, 760, 76, { fill: col.osoft, stroke: col.ours, lw: 2.4 });
    text(ctx, Z.polyText(D.same_poly, 3).join("  +  "), 600, 392, 24, col.ink, { align: "center", family: Z.MATH });
    text(ctx, "Same function, one canonical answer. Otherwise the head would be punished for a valid rewrite.", 600, 420, 15, col.muted, { align: "center" });
    ctx.globalAlpha = 1;
  }

  function scene2(ctx, p) {
    box(ctx, 20, 60, 1160, 84);
    text(ctx, "sampled tree", 40, 88, 16, col.muted, { weight: 600 });
    wrap(ctx, Z.pretty(D.raw_tree, 3), 40, 118, 1120, 24, 20, col.ink, { family: Z.MATH });
    const rules = ["1  distribute products over sums", "2  merge equal terms", "3  sort: degree, then series and lag"];
    rules.forEach((s, i) => {
      const a = ease(seg(p, i * 0.12, i * 0.12 + 0.2));
      ctx.globalAlpha = a;
      box(ctx, 20 + i * 392, 160, 376, 46, { fill: col.soft, stroke: col.blue });
      text(ctx, s, 38 + i * 392, 190, 18, col.blue, { weight: 700 });
      ctx.globalAlpha = 1;
    });
    box(ctx, 20, 222, 1160, 224);
    ["degree", "term", "coefficient"].forEach((h, i) => text(ctx, h, [44, 150, 800][i], 252, 16, col.muted, { weight: 700 }));
    D.raw_poly.forEach((t, i) => {
      const a = ease(seg(p, 0.35 + i * 0.1, 0.5 + i * 0.1)), y = 284 + i * 32;
      ctx.globalAlpha = a;
      text(ctx, String(t.atoms.length), 56, y, 20, col.ink, { weight: 700 });
      text(ctx, termText(t), 150, y, 20, col.ink, { family: Z.MATH });
      text(ctx, fmt(t.coef, 4), 800, y, 20, col.ours, { weight: 700, family: Z.MATH });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.9, 1));
    text(ctx, "at most 4 factors per term", 960, 252, 16, col.muted);
    ctx.globalAlpha = 1;
  }

  function scene3(ctx, p) {
    const cols = [["Without protection (RAW)", D.raw_poly, 20], ["With protection (TANH on some lags)", D.protected_poly, 620]];
    cols.forEach(([title, poly, x], c) => {
      box(ctx, x, 60, 560, 250);
      text(ctx, title, x + 20, 92, 20, c ? col.ours : col.ink, { weight: 700 });
      poly.forEach((t, i) => {
        const a = ease(seg(p, 0.1 + i * 0.07, 0.25 + i * 0.07)), y = 130 + i * 34;
        const diff = c && !sameTerm(t, D.raw_poly[i]);
        ctx.globalAlpha = a;
        if (diff) {
          ctx.fillStyle = col.osoft;
          ctx.fillRect(x + 12, y - 22, 536, 30);
        }
        text(ctx, termText(t), x + 24, y, 19, diff ? col.ours : col.ink, { family: Z.MATH, weight: diff ? 700 : 400 });
        text(ctx, fmt(t.coef, 3), x + 470, y, 19, col.muted, { align: "right", family: Z.MATH });
        ctx.globalAlpha = 1;
      });
    });
    arrow(ctx, 586, 190, 614, 190, col.muted, 2.2, ease(seg(p, 0.4, 0.6)));
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    box(ctx, 20, 330, 1160, 100, { fill: col.grey });
    text(ctx, "x(t−2) and tanh x(t−2) are different symbols.", 44, 366, 21, col.ink, { weight: 700 });
    text(ctx, `Vocabulary: ${(D.vocab_size - 5) / 2} lags × {RAW, TANH} = ${D.vocab_size - 5} factor symbols, plus 5 special symbols = ${D.vocab_size}.`, 44, 396, 19, col.muted);
    text(ctx, "Orange rows: terms rewritten by the protection (x(t−3) and x(t−6) are linear, so they stay RAW).", 44, 420, 16, col.muted);
    ctx.globalAlpha = 1;
  }

  function chip(ctx, s, x, y, kind, a) {
    const w = Z.measure(ctx, s, 16, 600) + 22;
    ctx.globalAlpha = a;
    const st = kind === "end" ? { fill: col.osoft, stroke: col.ours, color: col.ours }
      : kind === "sp" ? { fill: col.grey, stroke: col.muted, color: col.muted }
      : s.startsWith("tanh") ? { fill: col.osoft, stroke: col.ours, color: col.ours }
      : { fill: col.soft, stroke: col.blue, color: col.blue };
    box(ctx, x, y, w, 32, { r: 8, fill: st.fill, stroke: st.stroke, lw: 2 });
    text(ctx, s, x + w / 2, y + 22, 16, st.color, { align: "center", weight: 600 });
    ctx.globalAlpha = 1;
    return w;
  }
  function tokenStrip(ctx, x0, y0, maxW, upto, a0) {
    let x = x0, y = y0;
    D.target.forEach((t, i) => {
      const kind = t.tok === "END_TERM" ? "end" : t.tok === "BOS" || t.tok === "EOS" ? "sp" : "atom";
      const label = t.tok === "END_TERM" ? (i > 0 && D.target[i - 1].tok === "BOS" ? "const ;" : ";") : t.atom ? (t.atom[2] === "tanh" ? "tanh " : "") + lagName(t.atom[1]) : t.tok;
      const w = Z.measure(ctx, label, 16, 600) + 22;
      if (x + w > x0 + maxW) { x = x0; y += 74; }
      const a = a0 === undefined ? 1 : ease(seg(upto, i / D.target.length, i / D.target.length + 0.08));
      if (a > 0) {
        chip(ctx, label, x, y, kind, a);
        if (t.coef !== undefined) {
          ctx.globalAlpha = a;
          text(ctx, fmt(t.coef, 3), x + w / 2, y + 54, 16, col.ours, { align: "center", weight: 700, family: Z.MATH });
          ctx.globalAlpha = 1;
        }
      }
      x += w + 8;
    });
    return y;
  }

  function scene4(ctx, p) {
    box(ctx, 20, 60, 1160, 170);
    text(ctx, "The equation as a sequence of symbols", 40, 92, 20, col.ink, { weight: 700 });
    tokenStrip(ctx, 40, 116, 1110, p * 0.9, 1);
    ctx.globalAlpha = ease(seg(p, 0.85, 1));
    box(ctx, 20, 250, 1160, 126, { fill: col.grey });
    text(ctx, "One term = its factors, then the end-of-term symbol ;", 44, 284, 20, col.ink, { weight: 700 });
    text(ctx, "The coefficient (orange) belongs to that end-of-term symbol. An empty term is the constant, and it comes first (degree 0).", 44, 314, 18, col.muted);
    text(ctx, "Terms come in canonical order, so the model has exactly one correct sequence to learn.", 44, 342, 18, col.muted);
    ctx.globalAlpha = 1;
  }

  function scene5(ctx, p) {
    text(ctx, "input: the tree, read in prefix order", 20, 80, 17, col.muted, { weight: 600 });
    let x = 20;
    D.prefix.forEach((t, i) => {
      const a = ease(seg(p, i * 0.025, i * 0.025 + 0.12));
      const label = t.v !== null && t.v !== undefined ? `PAR ${fmt(t.v, 2)}` : t.tok.startsWith("LAG") ? lagName(t.tok.slice(3)) : t.tok;
      const kind = t.tok === "ADD" || t.tok === "MUL" || t.tok === "TANH" ? "sp" : "atom";
      const w = chip(ctx, label, x, 92, kind, a);
      x += w + 8;
    });
    const b = ease(seg(p, 0.3, 0.5));
    ctx.globalAlpha = b;
    arrow(ctx, 160, 134, 160, 186, col.muted, 2.4);
    box(ctx, 20, 190, 330, 130, { fill: col.soft, stroke: col.blue, lw: 2.4 });
    text(ctx, "Encoder", 185, 232, 26, col.blue, { align: "center", weight: 700 });
    text(ctx, "Transformer, 2 layers, d = 96", 185, 262, 17, col.muted, { align: "center" });
    text(ctx, "(a tree-LSTM is also compared)", 185, 286, 16, col.muted, { align: "center" });
    arrow(ctx, 356, 255, 416, 255, col.muted, 2.4);
    box(ctx, 420, 190, 330, 130, { fill: col.osoft, stroke: col.ours, lw: 2.4 });
    text(ctx, "Decoder", 585, 232, 26, col.ours, { align: "center", weight: 700 });
    text(ctx, "GRU + attention over the tree", 585, 262, 17, col.muted, { align: "center" });
    text(ctx, "one symbol per step", 585, 286, 16, col.muted, { align: "center" });
    ctx.globalAlpha = 1;
    const c = ease(seg(p, 0.55, 0.7));
    ctx.globalAlpha = c;
    arrow(ctx, 756, 255, 806, 255, col.muted, 2.4);
    box(ctx, 810, 150, 370, 290, { fill: col.panel });
    ctx.globalAlpha = 1;
    text(ctx, "output, symbol by symbol", 826, 188, 16, col.muted, { weight: 600 });
    tokenStrip(ctx, 826, 200, 340, seg(p, 0.6, 1), 1);
    ctx.globalAlpha = ease(seg(p, 0.9, 1));
    text(ctx, "Two models will be trained: skeleton (symbols only)", 20, 360, 18, col.ink);
    text(ctx, "and joint (symbols plus a number at every ; symbol).", 20, 384, 18, col.ink);
    text(ctx, "Planned: 10 million random trees; validation and test skeletons never appear in training.", 20, 416, 17, col.muted);
    ctx.globalAlpha = 1;
  }

  function stat(ctx, x, y, w, big, label, color, a, muted) {
    ctx.globalAlpha = a;
    box(ctx, x, y, w, 150, { stroke: color, lw: 2.2, fill: muted ? col.grey : col.panel, dash: muted ? [6, 5] : [] });
    text(ctx, big, x + 16, y + 62, muted ? 40 : 46, color, { weight: 700 });
    wrap(ctx, label, x + 16, y + 94, w - 28, 21, 16, muted ? col.muted : col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }
  function model(ctx, x, w, title, sub, tone, a) {
    ctx.globalAlpha = a;
    box(ctx, x, 60, w, 262, { fill: tone === "joint" ? col.osoft : col.soft, stroke: tone === "joint" ? col.ours : col.blue, lw: 2.6 });
    text(ctx, title, x + 18, 92, 22, tone === "joint" ? col.ours : col.blue, { weight: 700 });
    text(ctx, sub, x + 18, 116, 15, col.muted);
    ctx.globalAlpha = 1;
  }
  function scene6(ctx, p) {
    const plan = [
      ["Equations with every term right", "the whole equation must match", col.teal],
      ["Terms right, counted one by one (F1)", "each term counts alone", col.teal],
      ["Mean absolute error of the coefficients", "on the terms the model found", col.ours],
    ];
    ctx.globalAlpha = ease(seg(p, 0, 0.3));
    box(ctx, 20, 60, 1160, 70, { fill: col.osoft, stroke: col.ours, dash: [6, 5], lw: 2.2 });
    text(ctx, "PLANNED: no model has been trained on the RAW/TANH corpus yet", 44, 104, 22, col.ours, { weight: 700 });
    ctx.globalAlpha = 1;
    plan.forEach((m, i) => {
      const a = ease(seg(p, 0.2 + i * 0.15, 0.5 + i * 0.15)), x = 20 + i * 392;
      ctx.globalAlpha = a;
      box(ctx, x, 156, 376, 150, { stroke: m[2], lw: 2.2, dash: [6, 5] });
      text(ctx, "–", x + 18, 222, 46, col.muted, { weight: 700 });
      wrap(ctx, m[0], x + 18, 252, 340, 22, 17, col.ink, { weight: 700 });
      text(ctx, m[1], x + 18, 292, 14.5, col.muted);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 1));
    box(ctx, 20, 330, 1160, 154, { fill: col.grey });
    text(ctx, "What will be trained and measured", 44, 362, 21, col.ink, { weight: 700 });
    const lines = [
      "Two models, as in the diagram: skeleton (symbols only) and joint (symbols plus a number at every ; symbol).",
      "Corpus: random trees with lags 1 to 16; half of them keep raw products, half use tanh(u)·tanh(v).",
      "Validation and test skeletons never appear in training. A coefficient counts if within 1% or 0.01.",
      "The input is the tree itself, not the series: reading the equation from a series is a later step.",
    ];
    lines.forEach((s, i) => text(ctx, "• " + s, 44, 392 + i * 25, 16.5, col.muted));
    ctx.globalAlpha = 1;
  }

  const STEPS = [
    { narr: "Zoom 2 · Equation head", sub: "write the mechanism as an equation", dur: 3800 },
    { narr: "① Many trees compute one function", sub: "so the answer must be canonical", dur: 4200 },
    { narr: "② The canonical form", sub: "distribute, merge, sort", dur: 4600 },
    { narr: "③ RAW and TANH factors", sub: "protection changes the symbols, not the idea", dur: 4200 },
    { narr: "④ A sequence the model can write", sub: "factors, end of term, coefficient", dur: 5200 },
    { narr: "⑤ The model", sub: "encoder reads, decoder writes", dur: 6500 },
    { narr: "⑥ What will be measured", sub: "a plan: no results yet", dur: 3800,
      foot: "Design only. Corpus and training configuration: configs/canonicalization_5m_raw_tanh.json (10,000,000 sampled ASTs, 1–6 operators, lags 1–16, degree ≤ 4, protection probability 0.5, split by skeleton)." },
  ];

  Z.mount({
    id: "zoom-head", W, H, steps: STEPS,
    foot: "Example from the project's sampler (seed " + D.seed + "); target and symbols from canonicalize_polynomial and CanonicalCodec.",
    paint(ctx, step, p) {
      Z.chips(ctx, STAGES, step === 0 ? -1 : step - 1, 8);
      [scene0, scene1, scene2, scene3, scene4, scene5, scene6][step](ctx, p);
    },
  });
})();
