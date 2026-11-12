/* Pipeline slide: the whole method in one story, one zoom per step.
 *
 * The pipeline starts without the fixing (tree sampler, series generator, encoder with two
 * heads, and the canonicalizer that writes the label). Then, in order: the generator (zoom
 * into the tree and its series, initial values discarded, raw series that are not stable);
 * the fixing enters the diagram, with its four methods and the protection; the canonicalizer
 * (how it iterates, what it costs); the swap for a canonicalizer encoder and the model; and,
 * last, the same steps repeated for N series.
 *
 * The scenes of the former generator zoom (ZGEN, zoom_generator.js), of the former
 * equation-head zoom (ZHEAD, zoom_head.js) and the click-through method views
 * (GenDetail, stability_detail.js) are shown here, each at its own place.
 * Data: GEN_DATA, PIPE_DATA, CANON_TIMING. Stability shares: experiments/abstract_syntax_tree/
 * 6_stability_methods_tradeoff.ipynb; model numbers: experiments/canonicalization/06_train.ipynb.
 */
(function () {
  const D = window.GEN_DATA, T = window.CANON_TIMING, PIPE = window.PIPE_DATA, Z = window.ZC;
  const ZG = window.ZGEN, ZH = window.ZHEAD, GD = window.GenDetail;
  if (!D || !T || !PIPE || !Z || !ZG || !ZH || !GD) return;
  const { col, seg, ease, text, wrap, box, arrow, lagName } = Z;
  const W = 1200, H = 500;

  // ---------------------------------------------------------------- canonicalization by node
  const keyOf = (atoms) => atoms.map((a) => a.join(",")).join("|");
  const sortTerms = (terms) => terms.sort((a, b) => a.atoms.length - b.atoms.length || (keyOf(a.atoms) < keyOf(b.atoms) ? -1 : 1));
  const merge = (terms) => {
    const m = new Map();
    terms.forEach((t) => {
      const k = keyOf(t.atoms);
      if (m.has(k)) m.get(k).coef += t.coef;
      else m.set(k, { atoms: t.atoms, coef: t.coef });
    });
    return sortTerms([...m.values()].filter((t) => Math.abs(t.coef) > 0));
  };
  const orderAtoms = (atoms) => atoms.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1] || (a[2] ? 1 : 0) - (b[2] ? 1 : 0));

  /** Post-order visit: one record per node with the polynomial after that node. */
  function trace(tree) {
    const steps = [];
    const visit = (n) => {
      let poly, info;
      if (n.t === "par") { poly = [{ atoms: [], coef: n.v }]; info = "read a parameter"; }
      else if (n.t === "lag") { poly = [{ atoms: [[n.series || 0, n.lag]], coef: 1 }]; info = "read a lag"; }
      else if (n.t === "tanh") { const l = n.k[0]; poly = [{ atoms: [[l.series || 0, l.lag, "tanh"]], coef: 1 }]; info = "read tanh of a lag"; }
      else {
        const a = visit(n.k[0]), b = visit(n.k[1]);
        if (n.t === "add") {
          poly = merge(a.concat(b));
          info = `add: ${a.length} + ${b.length} terms, merge equal ones`;
        } else {
          const products = [];
          a.forEach((x) => b.forEach((y) => products.push({ atoms: orderAtoms(x.atoms.concat(y.atoms)), coef: x.coef * y.coef })));
          poly = merge(products);
          info = `multiply: ${a.length} × ${b.length} = ${products.length} products, merge, sort`;
        }
      }
      steps.push({ node: n, poly, info });
      return poly;
    };
    visit(tree);
    return steps;
  }
  const STEPS_TRACE = trace(D.raw_tree);

  const treeLabel = (nd) => {
    const n = nd.n;
    if (n.t === "add") return "+";
    if (n.t === "mul") return "×";
    return n.t === "par" ? Z.fmt(n.v, 2) : lagName(n.lag);
  };

  // ---------------------------------------------------------------------------------- pieces
  function card(ctx, x, y, w, h, title, sub, tone, a) {
    const c = { blue: [col.soft, col.blue], ours: [col.osoft, col.ours], teal: [col.tsoft, col.teal], plain: [col.panel, col.line], grey: [col.grey, col.line] }[tone];
    ctx.globalAlpha = a;
    box(ctx, x, y, w, h, { fill: c[0], stroke: c[1], lw: 2.4 });
    const accent = tone === "plain" || tone === "grey" ? col.ink : c[1];
    text(ctx, title, x + w / 2, y + h / 2 - (sub ? 4 : -8), 21, accent, { align: "center", weight: 700 });
    if (sub) text(ctx, sub, x + w / 2, y + h / 2 + 22, 15, col.muted, { align: "center" });
    ctx.globalAlpha = 1;
  }
  function loopGlyph(ctx, cx, cy, r, color, a) {
    ctx.globalAlpha = a;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -0.4, Math.PI * 1.45);
    ctx.stroke();
    const ang = Math.PI * 1.45, tx = cx + r * Math.cos(ang), ty = cy + r * Math.sin(ang);
    ctx.beginPath();
    ctx.moveTo(tx + 7, ty - 6);
    ctx.lineTo(tx - 8, ty - 4);
    ctx.lineTo(tx + 2, ty + 9);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  function tag(ctx, label, x, y, w, color) {
    box(ctx, x, y, w, 26, { fill: color, stroke: color, lw: 1, r: 13 });
    text(ctx, label, x + w / 2, y + 18, 13, col.panel, { align: "center", weight: 700 });
  }
  function stat(ctx, x, y, w, big, label, color, a, tagText) {
    ctx.globalAlpha = a;
    box(ctx, x, y, w, 150, { stroke: color, lw: 2.2, fill: col.panel });
    text(ctx, big, x + 16, y + 66, 46, color, { weight: 700 });
    wrap(ctx, label, x + 16, y + 98, w - 28, 21, 16, col.ink, { weight: 600 });
    if (tagText) tag(ctx, tagText, x + w - 16 - (tagText.length * 8 + 28), y + 12, tagText.length * 8 + 28, color);
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------- visual building blocks
  const LANE_COLORS = [col.blue, col.teal, col.ours];

  /** Small tree icon: dots and lines, no labels. tanh nodes get an orange ring. */
  function miniTree(ctx, tree, x, y, w, h, a, scale) {
    const lay = Z.layout(tree, { x, y, w, h });
    const r = 6.5 * (scale || 1);
    ctx.globalAlpha = a;
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = 2.2;
    lay.nodes.forEach((nd) => {
      if (!nd.parent) return;
      ctx.beginPath();
      ctx.moveTo(nd.parent.x, nd.parent.y);
      ctx.lineTo(nd.x, nd.y);
      ctx.stroke();
    });
    lay.nodes.forEach((nd) => {
      const t = nd.n.t;
      ctx.beginPath();
      if (t === "lag" || t === "par") {
        const s = r * 1.5;
        Z.rrect(ctx, nd.x - s / 2, nd.y - s / 2, s, s, 3);
        ctx.fillStyle = t === "lag" ? col.tsoft : col.soft;
        ctx.strokeStyle = t === "lag" ? col.teal : col.blue;
      } else {
        ctx.arc(nd.x, nd.y, t === "tanh" ? r * 0.85 : r, 0, Math.PI * 2);
        ctx.fillStyle = t === "tanh" ? col.osoft : col.panel;
        ctx.strokeStyle = t === "tanh" ? col.ours : col.ink;
      }
      ctx.lineWidth = 2.2;
      ctx.fill();
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  /** The "fixed" mark: a mechanic's wrench on a tinted rounded badge, like the other icons. */
  function spanner(ctx, cx, cy, s, color, a) {
    const side = s * 1.3, k = s * 0.5;
    ctx.save();
    ctx.globalAlpha = a;
    Z.rrect(ctx, cx - side / 2, cy - side / 2, side, side, side * 0.28);
    ctx.fillStyle = col.tsoft;
    ctx.fill();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.translate(cx, cy);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = color;
    // open-end jaw: a disc with a slot cut into its top
    const hy = -0.62 * k, R = 0.34 * k, slot = 0.14 * k, d = Math.asin(slot / R);
    ctx.beginPath();
    ctx.arc(0, hy, R, -Math.PI / 2 + d, 1.5 * Math.PI - d, false);
    ctx.lineTo(-slot, hy + 0.12 * k);
    ctx.lineTo(slot, hy + 0.12 * k);
    ctx.closePath();
    ctx.fill();
    // shaft, tapering into a ring end with a hole
    Z.rrect(ctx, -0.1 * k, hy + 0.2 * k, 0.2 * k, 1.1 * k, 0.1 * k);
    ctx.fill();
    const ry = 0.8 * k;
    ctx.beginPath();
    ctx.arc(0, ry, 0.3 * k, 0, Math.PI * 2);
    ctx.arc(0, ry, 0.12 * k, 0, Math.PI * 2, true);
    ctx.fill("evenodd");
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  /**
   * A series line, optionally split into observed context and horizon. o.discard = number of
   * initial values that are thrown away: they are drawn grey inside a hatched band, and the
   * vertical scale is taken from the kept values only.
   */
  function series(ctx, x, y, w, h, ys, color, a, o) {
    o = o || {};
    ctx.globalAlpha = a;
    const keep = o.discard || 0, kept = ys.slice(keep);
    const lo = Math.min.apply(null, kept), hi = Math.max.apply(null, kept), span = hi - lo || 1;
    const cut = o.horizon ? x + w * 0.8 : x + w;
    const step = w / (ys.length - 1), bandW = Math.max(0, (keep - 0.5) * step);
    if (o.horizon) {
      Z.rrect(ctx, cut, y, x + w - cut, h, 6);
      ctx.fillStyle = col.soft;
      ctx.fill();
    }
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.2;
    [0.25, 0.5, 0.75].forEach((f) => { ctx.beginPath(); ctx.moveTo(x, y + h * f); ctx.lineTo(x + w, y + h * f); ctx.stroke(); });
    if (o.horizon) {
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = col.blue;
      ctx.beginPath(); ctx.moveTo(cut, y); ctx.lineTo(cut, y + h); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(x - 2, y, w + 4, h);
    ctx.clip();
    if (bandW > 0) {  // hatched band over the discarded values
      ctx.fillStyle = col.grey;
      ctx.fillRect(x, y, bandW, h);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, bandW, h);
      ctx.clip();
      ctx.strokeStyle = col.muted;
      ctx.lineWidth = 1;
      for (let k = -h; k < bandW; k += 7) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
      ctx.restore();
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x, y, bandW, h);
      ctx.setLineDash([]);
    }
    const n = Math.max(2, Math.floor(ys.length * (o.upto === undefined ? 1 : o.upto)));
    ctx.lineWidth = o.lw || 2.6;
    ctx.lineJoin = "round";
    [[0, Math.min(n, keep + 1), col.muted, [4, 4]], [keep, n, color, []]].forEach((s) => {
      if (s[1] - s[0] < 2) return;
      ctx.strokeStyle = s[2];
      ctx.setLineDash(s[3]);
      ctx.beginPath();
      for (let i = s[0]; i < s[1]; i++) {
        const px = x + step * i, py = y + h - 6 - ((ys[i] - lo) / span) * (h - 12);
        i === s[0] ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function cardFrame(ctx, x, y, w, h, tone, a, title) {
    const c = { blue: [col.soft, col.blue], ours: [col.osoft, col.ours], teal: [col.tsoft, col.teal], plain: [col.panel, col.line], grey: [col.grey, col.line] }[tone];
    ctx.globalAlpha = a;
    box(ctx, x, y, w, h, { fill: c[0], stroke: c[1], lw: 2.4 });
    if (title) text(ctx, title, x + 16, y + 26, 13.5, tone === "plain" || tone === "grey" ? col.muted : c[1], { weight: 700 });
    ctx.globalAlpha = 1;
  }

  /** The label box of the pipeline: the iterative canonicalizer, or the learned one that replaces it. */
  function canonCard(ctx, k, learned) {
    if (k <= 0) return;
    cardFrame(ctx, 270, 62, 230, 170, learned ? "blue" : "ours", k, learned ? "CANONICALIZER ENCODER" : "CANONICALIZER");
    if (learned) {
      ctx.globalAlpha = k;
      box(ctx, 408, 211, 80, 22, { fill: col.blue, stroke: col.blue, lw: 1, r: 11 });
      text(ctx, "LEARNED", 448, 227, 12.5, col.panel, { align: "center", weight: 700 });
    } else {
      loopGlyph(ctx, 470, 90, 11, col.ours, k);
    }
    ctx.globalAlpha = k;
    Z.polyText(D.final_poly, 2).slice(0, 5).forEach((s, i) => text(ctx, (i ? "+ " : "  ") + s, 286, 112 + i * 21, 15, col.ink, { family: Z.MATH }));
    text(ctx, learned ? "same equation" : "one canonical equation", 286, 226, 14, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------------------------- scenes
  /** Empty chart frame: the series is not there yet. */
  function emptyChart(ctx, x, y, w, h, a) {
    ctx.globalAlpha = a;
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.2;
    [0.25, 0.5, 0.75].forEach((f) => { ctx.beginPath(); ctx.moveTo(x, y + h * f); ctx.lineTo(x + w, y + h * f); ctx.stroke(); });
    text(ctx, "x(t)", x + w / 2, y + h / 2 + 8, 24, col.muted, { align: "center", weight: 600, family: Z.MATH });
    ctx.globalAlpha = 1;
  }
  const RAW = PIPE.raw;  // [exploding (the slide's tree), fading, stable]; values are asinh(x)

  /**
   * The whole pipeline. o.fix in [0, 1]: the fixing stage is in the diagram; o.show in [0, 1]:
   * the series card shows a series (the raw one without the fixing, the fixed one with it);
   * o.morph: the canonicalizer is swapped for the canonicalizer encoder.
   */
  function scene0(ctx, p, o) {
    o = o || {};
    const morph = o.morph || 0, fix = o.fix === undefined ? 1 : o.fix, show = o.show === undefined ? 1 : o.show;
    const A = (a, b) => ease(seg(p, a, b));
    const ys = D.trajectory;
    // top lane: the label
    const built = A(0.3, 0.45);
    canonCard(ctx, built * (1 - morph), false);
    canonCard(ctx, built * morph, true);
    const lc = morph > 0.5 ? col.blue : col.ours;
    // the label comes from the sampled tree: straight from the sampler, then from the fixed tree
    ctx.globalAlpha = A(0.25, 0.4) * (1 - fix);
    ctx.strokeStyle = lc;
    ctx.lineWidth = 2.6;
    ctx.beginPath(); ctx.moveTo(130, 250); ctx.lineTo(130, 147); ctx.stroke();
    ctx.globalAlpha = 1;
    arrow(ctx, 130, 147, 264, 147, lc, 2.6, A(0.25, 0.4) * (1 - fix));
    arrow(ctx, 385, 252, 385, 236, lc, 2.6, A(0.25, 0.4) * fix);
    ctx.globalAlpha = A(0.7, 0.9);
    ctx.setLineDash([8, 7]);
    arrow(ctx, 504, 147, 912, 147, col.ours, 2.6);
    ctx.setLineDash([]);
    text(ctx, "the equation is the label", 708, 136, 15, col.ours, { align: "center", weight: 700 });
    ctx.globalAlpha = 1;
    cardFrame(ctx, 920, 62, 260, 170, "ours", A(0.68, 0.82), "EQUATION HEAD");
    ctx.globalAlpha = A(0.7, 0.85);
    text(ctx, "writes the equation", 1050, 138, 19, col.ours, { align: "center", weight: 700 });
    text(ctx, "from the series alone", 1050, 164, 15, col.muted, { align: "center" });
    ctx.globalAlpha = 1;

    // bottom lane: the data
    cardFrame(ctx, 20, 252, 220, 190, "plain", A(0, 0.15), "TREE SAMPLER");
    miniTree(ctx, D.raw_tree, 40, 296, 180, 100, A(0.05, 0.2), 1);
    text(ctx, "a random equation", 130, 428, 15, col.muted, { align: "center" });
    arrow(ctx, 244, 347, 536, 347, col.muted, 2.4, A(0.15, 0.3) * (1 - fix));  // no fixing: tree straight to series
    arrow(ctx, 244, 347, 266, 347, col.muted, 2.4, A(0.15, 0.25) * fix);

    if (fix > 0) {
      cardFrame(ctx, 270, 252, 230, 190, "plain", fix, "FIXED TREE");
      ctx.globalAlpha = fix;
      box(ctx, 418, 262, 72, 24, { fill: col.tsoft, stroke: col.teal, lw: 1.2, r: 12 });
      text(ctx, "FIXED", 454, 279, 13, col.teal, { align: "center", weight: 700 });
      ctx.globalAlpha = 1;
      miniTree(ctx, D.protected_tree, 288, 304, 138, 92, fix, 1);
      spanner(ctx, 468, 352, 38, col.teal, fix);
      ctx.globalAlpha = fix;
      text(ctx, "made safe to simulate", 385, 428, 15, col.muted, { align: "center" });
      ctx.globalAlpha = 1;
      arrow(ctx, 504, 347, 536, 347, col.muted, 2.4, fix);
    }

    cardFrame(ctx, 540, 252, 340, 190, "plain", A(0.35, 0.5), "SERIES GENERATOR");
    const sw = A(0.4, 0.55);
    if (show < 1) {  // before the raw series is examined: any series, here a tree that stays alive
      const any = RAW[2];
      series(ctx, 560, 290, 300, 110, any.asinh_trajectory, col.blue, sw * (1 - show), { discard: any.history_len });
      ctx.globalAlpha = sw * (1 - show);
      text(ctx, "discarded", 590, 416, 14, col.ours, { align: "center", weight: 600 });
      text(ctx, "the series", 760, 416, 14, col.muted, { align: "center" });
      ctx.globalAlpha = 1;
    }
    if (show > 0) {
      const raw = RAW[0].asinh_trajectory;
      if (fix < 1) {  // without the fixing the series of this very tree blows up
        series(ctx, 560, 290, 300, 110, raw, col.ours, sw * show * (1 - fix), { discard: RAW[0].history_len });
        ctx.globalAlpha = sw * show * (1 - fix);
        text(ctx, "discarded", 586, 416, 14, col.ours, { align: "center", weight: 600 });
        text(ctx, "diverges", 826, 416, 14, col.ours, { align: "center", weight: 600 });
        ctx.globalAlpha = 1;
      }
      if (fix > 0) {
        series(ctx, 560, 290, 300, 110, ys, col.blue, sw * show * fix, { horizon: true, discard: D.history_len });
        ctx.globalAlpha = sw * show * fix;
        text(ctx, "discarded", 586, 416, 14, col.ours, { align: "center", weight: 600 });
        text(ctx, "observed context", 706, 416, 14, col.muted, { align: "center" });
        text(ctx, "horizon", 832, 416, 14, col.blue, { align: "center", weight: 600 });
        ctx.globalAlpha = 1;
      }
    }
    arrow(ctx, 884, 347, 916, 347, col.muted, 2.4, A(0.55, 0.65));

    cardFrame(ctx, 920, 252, 260, 84, "blue", A(0.6, 0.75), "SHARED ENCODER");
    ctx.globalAlpha = A(0.62, 0.78);
    text(ctx, "reads the series", 1050, 316, 17, col.blue, { align: "center", weight: 700 });
    ctx.globalAlpha = 1;
    arrow(ctx, 1050, 252, 1050, 236, col.muted, 2.4, A(0.65, 0.8));
    arrow(ctx, 1050, 338, 1050, 358, col.muted, 2.4, A(0.65, 0.8));
    cardFrame(ctx, 920, 362, 260, 80, "grey", A(0.68, 0.82), "FORECAST HEAD");
    ctx.globalAlpha = A(0.7, 0.85);
    text(ctx, "predicts the horizon", 1050, 422, 17, col.ink, { align: "center", weight: 700 });
    ctx.globalAlpha = 1;

    ctx.globalAlpha = A(0.88, 1);
    box(ctx, 20, 450, 1160, 48, { fill: col.grey });
    text(ctx, fix > 0.5
      ? "One tree gives one series and one label: the tree is fixed, simulated for the series, and canonicalized for the equation."
      : "One tree gives one series and one label: the tree is simulated for the series and canonicalized for the equation.", 40, 469, 16, col.ink);
    text(ctx, "The first values, the initial history the recurrence starts from, are thrown away: their distribution is very different.", 40, 489, 15, col.ours, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  /** Zoom: from the tree to a series, and the initial values that are thrown away. */
  function sceneSim(ctx, p) {
    const r = RAW[0], SHOWN = 22, ys = r.asinh_trajectory.slice(0, SHOWN).map(Math.sinh);  // first steps, linear scale
    box(ctx, 20, 60, 1160, 64);
    text(ctx, "x(t) = " + Z.pretty(D.raw_tree, 2) + "  +  ε(t)", 40, 100, 19, col.ink, { family: Z.MATH });
    text(ctx, "noise ε ~ N(0, 0.05²)", 1160, 100, 17, col.muted, { align: "right" });
    box(ctx, 20, 140, 1160, 220);
    series(ctx, 44, 160, 1112, 180, ys, col.ours, 1, { discard: r.history_len, upto: ease(seg(p, 0.1, 0.85)), lw: 2.8 });
    ctx.globalAlpha = ease(seg(p, 0.55, 0.8));
    text(ctx, "the first " + r.history_len + " values: random draw in [−1, 1]", 52, 154, 15.5, col.ours, { weight: 700 });
    text(ctx, "thrown away", 52, 354, 15.5, col.ours, { weight: 700 });
    text(ctx, "it keeps growing and diverges at step " + r.diverged_at, 1150, 154, 15.5, col.ours, { align: "right", weight: 700 });
    text(ctx, "first " + SHOWN + " steps, linear scale", 1150, 354, 14, col.muted, { align: "right" });
    ctx.globalAlpha = 1;
    [["1  start from random values", "as many as the recurrence reaches back"], ["2  step the recurrence", "past values in, the next value out, plus noise"], ["3  drop the initial values", "their distribution is very different from the rest"]].forEach((s, i) => {
      const a = ease(seg(p, 0.2 + i * 0.2, 0.45 + i * 0.2)), x = 20 + i * 392;
      ctx.globalAlpha = a;
      box(ctx, x, 376, 376, 74, { fill: i === 2 ? col.osoft : col.soft, stroke: i === 2 ? col.ours : col.blue });
      text(ctx, s[0], x + 16, 406, 18, i === 2 ? col.ours : col.blue, { weight: 700 });
      text(ctx, s[1], x + 16, 432, 14.5, col.muted);
      ctx.globalAlpha = 1;
    });
  }

  /** The raw series are not stable. */
  function sceneUnstable(ctx, p) {
    const names = { exploding: ["explodes", "|x| > 100, here not even finite"], fading: ["fades", "settles inside ±0.15"], stable: ["stays alive", "neither of the two"] };
    const tone = { exploding: col.ours, fading: col.blue, stable: col.teal };
    RAW.forEach((r, i) => {
      const a = ease(seg(p, i * 0.14, i * 0.14 + 0.3)), x = 20 + i * 396;
      ctx.globalAlpha = a;
      box(ctx, x, 60, 368, 230, { stroke: tone[r.kind], lw: 2.4 });
      miniTree(ctx, r.raw_tree, x + 14, 78, 110, 60, 1, 0.8);
      text(ctx, names[r.kind][0], x + 140, 104, 24, tone[r.kind], { weight: 700 });
      text(ctx, names[r.kind][1], x + 140, 130, 14.5, col.muted);
      ctx.globalAlpha = 1;
      series(ctx, x + 14, 156, 340, 110, r.asinh_trajectory, tone[r.kind], a, { discard: r.history_len, upto: ease(seg(p, 0.15 + i * 0.12, 0.7 + i * 0.1)), lw: 2.4 });
    });
    const tiles = [["52.2%", "of raw trees explode", col.ours], ["10.4%", "fade to almost zero", col.blue], ["37.4%", "are neither: the only usable ones", col.teal]];
    tiles.forEach((s, i) => stat(ctx, 20 + i * 396, 306, 368, s[0], s[1], s[2], ease(seg(p, 0.55 + i * 0.12, 0.85 + i * 0.1)), null));
    ctx.globalAlpha = ease(seg(p, 0.9, 1));
    text(ctx, "5,000 raw trees, 384 steps, explosion = |x| > 100. Nothing makes a random recurrence stable.", 20, 478, 15.5, col.muted);
    ctx.globalAlpha = 1;
  }

  /** Zoom: the fixing. */
  function sceneFix(ctx, p) {
    box(ctx, 20, 60, 560, 390);
    text(ctx, "Fix the tree, keep its idea", 40, 96, 22, col.ink, { weight: 700 });
    const a1 = ease(seg(p, 0, 0.3)), a2 = ease(seg(p, 0.25, 0.55));
    miniTree(ctx, D.raw_tree, 50, 150, 200, 120, a1, 1.5);
    text(ctx, "the sampled tree", 150, 296, 16, col.muted, { align: "center" });
    arrow(ctx, 262, 210, 318, 210, col.teal, 2.6, a2);
    spanner(ctx, 290, 166, 40, col.teal, a2);
    miniTree(ctx, D.protected_tree, 330, 150, 220, 120, a2, 1.5);
    ctx.globalAlpha = a2;
    text(ctx, "tanh on the lag factors of products", 440, 296, 16, col.ours, { align: "center", weight: 600 });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = ease(seg(p, 0.5, 0.8));
    box(ctx, 40, 318, 520, 112, { fill: col.osoft, stroke: col.ours });
    text(ctx, `then one global scale s = ${D.scale.toFixed(3)}`, 56, 348, 19, col.ink, { weight: 700 });
    wrap(ctx, `The scale makes the certified gain ${D.certificate.total_gain.toFixed(2)}, below the target ${D.target_gain.toFixed(1)}: the recurrence cannot blow up. The equation keeps its shape; only the coefficients shrink.`, 56, 376, 490, 22, 15.5, col.muted);
    ctx.globalAlpha = 1;

    box(ctx, 600, 60, 580, 390);
    text(ctx, "One of four fixes, drawn per series", 620, 96, 22, col.ink, { weight: 700 });
    const rows = [["Plain tree", 52.24, 10.36], ["Jacobian", 0, 56.22], ["Shao–Wu", 0, 23.68], ["Heaps VAR", 0, 41.54]];
    rows.forEach((r, i) => {
      const y = 124 + i * 70, a = ease(seg(p, 0.2 + i * 0.12, 0.5 + i * 0.12));
      ctx.globalAlpha = a;
      text(ctx, r[0], 620, y + 22, 18, r[0] === "Shao–Wu" ? col.ours : col.ink, { weight: r[0] === "Shao–Wu" ? 700 : 500 });
      [[r[1], col.ours, 0], [r[2], col.blue, 26]].forEach((b) => {
        ctx.fillStyle = b[1];
        ctx.fillRect(760, y + b[2], Math.max(2, 320 * (b[0] / 100) * a), 20);
        text(ctx, b[0].toFixed(1) + "%", 766 + 320 * (b[0] / 100) * a, y + b[2] + 16, 14.5, col.muted);
      });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    [[col.ours, "explodes"], [col.blue, "fades to almost zero"]].forEach((l, i) => {
      ctx.fillStyle = l[0];
      ctx.fillRect(620 + i * 130, 410, 14, 14);
      text(ctx, l[1], 640 + i * 130, 423, 14.5, col.muted);
    });
    ctx.globalAlpha = 1;
  }

  function scene0b(ctx, p) {
    const A = (a, b) => ease(seg(p, a, b));
    const lanes = [
      { raw: D.raw_tree, fixed: D.protected_tree, ys: D.trajectory, hist: D.history_len },
      { raw: PIPE.lanes[0].raw_tree, fixed: PIPE.lanes[0].protected_tree, ys: PIPE.lanes[0].trajectory, hist: PIPE.lanes[0].history_len },
      { raw: PIPE.lanes[1].raw_tree, fixed: PIPE.lanes[1].protected_tree, ys: PIPE.lanes[1].trajectory, hist: PIPE.lanes[1].history_len },
    ];
    const y0 = 62, laneH = 96, gap = 8;
    ["TREE SAMPLER", "FIXED TREE", "SERIES GENERATOR"].forEach((s, i) => {
      ctx.globalAlpha = A(0, 0.15);
      text(ctx, s, [130, 385, 700][i], 56, 13.5, col.muted, { align: "center", weight: 700 });
      ctx.globalAlpha = 1;
    });
    lanes.forEach((ln, i) => {
      const y = y0 + 8 + i * (laneH + gap), color = LANE_COLORS[i], a = A(0.1 + i * 0.22, 0.4 + i * 0.22);
      cardFrame(ctx, 20, y, 220, laneH, "plain", a);
      miniTree(ctx, ln.raw, 40, y + 22, 180, laneH - 44, a, 0.9);
      arrow(ctx, 244, y + laneH / 2, 266, y + laneH / 2, col.muted, 2.2, a);
      cardFrame(ctx, 270, y, 230, laneH, "plain", a);
      miniTree(ctx, ln.fixed, 290, y + 22, 140, laneH - 44, a, 0.9);
      spanner(ctx, 466, y + laneH / 2 + 2, 44, col.teal, a);
      arrow(ctx, 504, y + laneH / 2, 536, y + laneH / 2, col.muted, 2.2, a);
      cardFrame(ctx, 540, y, 340, laneH, "plain", a);
      series(ctx, 556, y + 12, 308, laneH - 24, ln.ys, color, a, { discard: ln.hist, upto: A(0.2 + i * 0.22, 0.6 + i * 0.22), lw: 2.4 });
      arrow(ctx, 884, y + laneH / 2, 916, 128 + i * 50, color, 2.2, a);
      ctx.globalAlpha = a;
      text(ctx, "x" + "₁₂₃"[i], 40 + 188, y + 22, 18, color, { weight: 700, align: "right" });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = A(0.7, 0.9);
    text(ctx, "⋮", 130, y0 + 8 + 3 * (laneH + gap) + 12, 26, col.muted, { align: "center", weight: 700 });
    text(ctx, "⋮", 385, y0 + 8 + 3 * (laneH + gap) + 12, 26, col.muted, { align: "center", weight: 700 });
    text(ctx, "⋮", 700, y0 + 8 + 3 * (laneH + gap) + 12, 26, col.muted, { align: "center", weight: 700 });
    ctx.globalAlpha = 1;

    ctx.globalAlpha = A(0.55, 0.8);
    box(ctx, 920, 62, 260, 292, { fill: col.panel, stroke: col.blue, lw: 2.4 });
    text(ctx, "MULTIVARIATE SERIES", 936, 88, 13.5, col.blue, { weight: 700 });
    lanes.forEach((ln, i) => series(ctx, 936, 108 + i * 50, 228, 40, ln.ys.slice(ln.hist), LANE_COLORS[i], 1, { lw: 2.2 }));
    text(ctx, "x₁ … x_N", 1050, 290, 22, col.ink, { align: "center", weight: 700 });
    text(ctx, "N series, side by side", 1050, 318, 15, col.muted, { align: "center" });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = A(0.85, 1);
    box(ctx, 20, 424, 1160, 66, { fill: col.grey });
    text(ctx, "To get N series, repeat the same steps N times: one tree per series, each fixed and simulated on its own.", 40, 452, 17, col.ink, { weight: 600 });
    text(ctx, "Every series drops its initial values and has its own canonical equation, which is its label.", 40, 478, 15.5, col.muted);
    ctx.globalAlpha = 1;
  }

  function scene1(ctx, p) {
    const n = STEPS_TRACE.length;
    const cur = Math.min(n - 1, Math.floor(ease(seg(p, 0.05, 0.95)) * n));
    const done = new Set(STEPS_TRACE.slice(0, cur).map((s) => s.node));
    const here = STEPS_TRACE[cur];
    box(ctx, 20, 60, 560, 290);
    text(ctx, "the tree, visited leaves first", 40, 88, 16, col.muted, { weight: 600 });
    const lay = Z.layout(D.raw_tree, { x: 70, y: 130, w: 470, h: 170 });
    const byNode = new Map(lay.nodes.map((nd) => [nd.n, nd]));
    Z.drawTree(ctx, lay, {
      label: treeLabel, size: 14,
      style: (nd) => (nd.n === here.node ? { fill: col.osoft, stroke: col.ours, color: col.ours, lw: 3.4 }
        : done.has(nd.n) ? { fill: col.tsoft, stroke: col.teal, color: col.teal } : { stroke: col.line }),
    });
    const at = byNode.get(here.node);
    if (at && at.kids.length) { ctx.strokeStyle = col.ours; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(at.x, at.y, 27, 0, Math.PI * 2); ctx.stroke(); }
    text(ctx, `node ${cur + 1} of ${n}`, 40, 336, 16, col.ours, { weight: 700 });
    text(ctx, here.info, 150, 336, 16, col.ink);

    box(ctx, 600, 60, 580, 290, { fill: col.panel });
    text(ctx, "polynomial after this node", 620, 88, 16, col.muted, { weight: 600 });
    text(ctx, `${here.poly.length} term${here.poly.length === 1 ? "" : "s"}`, 1160, 88, 16, col.ours, { align: "right", weight: 700 });
    const lines = Z.polyText(here.poly, 2);
    lines.slice(0, 8).forEach((s, i) => text(ctx, (i ? "+ " : "  ") + s, 620, 120 + i * 27, 17, col.ink, { family: Z.MATH }));
    if (lines.length > 8) text(ctx, `… and ${lines.length - 8} more`, 620, 120 + 8 * 27, 16, col.muted);

    ["1  distribute products over sums", "2  merge equal terms", "3  sort by degree, series, lag"].forEach((s, i) => {
      const x = 20 + i * 392;
      box(ctx, x, 366, 376, 44, { fill: col.soft, stroke: col.blue });
      text(ctx, s, x + 18, 395, 17, col.blue, { weight: 700 });
    });
    ctx.globalAlpha = cur === n - 1 ? 1 : 0.55;
    box(ctx, 20, 426, 1160, 64, { fill: col.grey });
    wrap(ctx, "The same three rules are applied at every node, from the leaves to the root, and each node's result feeds its parent. The last node holds the canonical equation.", 40, 454, 1120, 22, 17, col.ink);
    ctx.globalAlpha = 1;
  }

  const ms = (v) => (v < 0.095 ? v.toFixed(3) : v.toFixed(2)) + " ms";
  function scene2(ctx, p) {
    const sweep = T.sweep, top = Math.max.apply(null, sweep.map((r) => r.median_ms));
    box(ctx, 20, 60, 600, 360);
    text(ctx, "median time of one canonicalization, one CPU core", 40, 88, 16, col.muted, { weight: 600 });
    const x0 = 60, baseY = 356, hMax = 215, bw = 58, gap = 28;
    sweep.forEach((r, i) => {
      const grow = ease(seg(p, i * 0.08, i * 0.08 + 0.35));
      const h = (r.median_ms / top) * hMax * grow, x = x0 + i * (bw + gap);
      const corpus = r.operators === 6;
      box(ctx, x, baseY - h, bw, Math.max(h, 1), { r: 4, lw: 1.5, fill: corpus ? col.ours : col.blue, stroke: corpus ? col.ours : col.blue });
      ctx.globalAlpha = grow;
      text(ctx, ms(r.median_ms), x + bw / 2, baseY - h - 8, 14, corpus ? col.ours : col.ink, { align: "center", weight: 700 });
      text(ctx, String(r.operators), x + bw / 2, baseY + 22, 17, col.ink, { align: "center", weight: 600 });
      ctx.globalAlpha = 1;
    });
    text(ctx, "operators in the tree (the corpus uses up to 6, in orange)", 320, 408, 15, col.muted, { align: "center" });

    const c = T.corpus, a = ease(seg(p, 0.5, 0.9));
    ctx.globalAlpha = a;
    box(ctx, 640, 60, 540, 360, { fill: col.panel });
    text(ctx, "What it costs and what it needs", 660, 92, 20, col.ink, { weight: 700 });
    const facts = [
      [col.ours, `${ms(c.median_ms)} per tree on the corpus (1 to 6 operators): about ${Math.round(c.trees_per_second / 100) * 100} trees per second per core.`],
      [col.ours, `Grows with tree size, from ${ms(sweep[0].median_ms)} at 2 operators to ${ms(sweep[sweep.length - 1].median_ms)} at 12; slowest of ${c.trees.toLocaleString("en-US")} trees: ${c.max_ms.toFixed(0)} ms.`],
      [col.blue, "One tree at a time, in Python with exact fractions: no batching, no GPU."],
      [col.blue, "Needs the explicit tree and its parameter values, and only covers tanh around a single lag."],
      [col.blue, "Refuses trees whose expansion passes the degree or term limits."],
    ];
    let y = 124;
    facts.forEach((f) => {
      ctx.fillStyle = f[0];
      ctx.beginPath();
      ctx.arc(670, y - 5, 5, 0, Math.PI * 2);
      ctx.fill();
      y = wrap(ctx, f[1], 686, y, 478, 22, 16.5, col.ink) + 12;
    });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    box(ctx, 20, 436, 1160, 54, { fill: col.grey });
    text(ctx, "Cheap for one small tree, but sequential and tied to the tree itself. Timed with Python " + T.python + ", one core, same sampler as the corpus.", 40, 470, 16.5, col.ink);
    ctx.globalAlpha = 1;
  }

  // ----------------------------------------------------------------------------- zoom machinery
  const FOCUS = {
    sampler: { x: 20, y: 252, w: 220, h: 190 },
    series: { x: 540, y: 252, w: 340, h: 190 },
    fix: { x: 270, y: 252, w: 230, h: 190 },
    canon: { x: 270, y: 62, w: 230, h: 170 },
  };
  const veil = (ctx, a) => {
    if (a <= 0) return;
    ctx.globalAlpha = Math.min(1, a);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
  };
  /** Draw fn() with the camera moved a fraction tt of the way into the card F. */
  function camera(ctx, tt, F, fn) {
    const S = Math.min(1160 / F.w, 430 / F.h), s = 1 + (S - 1) * tt;
    const cx = 600 + (F.x + F.w / 2 - 600) * tt, cy = 250 + (F.y + F.h / 2 - 250) * tt;
    ctx.save();
    ctx.translate(600, 250);
    ctx.scale(s, s);
    ctx.translate(-cx, -cy);
    fn();
    ctx.restore();
  }
  const pipe = (ctx, o) => () => scene0(ctx, 1, o);
  function zoomIn(ctx, p, F, o, detail) {
    if (p < 0.5) {
      camera(ctx, ease(seg(p, 0, 0.42)), F, pipe(ctx, o));
      veil(ctx, seg(p, 0.3, 0.5));
    } else {
      detail(ctx, seg(p, 0.5, 1));
      veil(ctx, 1 - seg(p, 0.5, 0.65));
    }
  }
  function zoomOut(ctx, p, F, o, detail) {
    if (p < 0.25) {
      detail(ctx, 1);
      veil(ctx, seg(p, 0, 0.25));
    } else {
      camera(ctx, 1 - ease(seg(p, 0.25, 0.85)), F, pipe(ctx, o));
      veil(ctx, 1 - seg(p, 0.25, 0.5));
    }
  }
  /** Zoom out of one card and into another, without stopping at the pipeline. */
  function hop(ctx, p, F1, d1, F2, d2, o) {
    if (p < 0.18) { d1(ctx, 1); veil(ctx, seg(p, 0, 0.18)); }
    else if (p < 0.4) { camera(ctx, 1 - ease(seg(p, 0.18, 0.4)), F1, pipe(ctx, o)); veil(ctx, 1 - seg(p, 0.18, 0.26)); }
    else if (p < 0.62) camera(ctx, ease(seg(p, 0.4, 0.62)), F2, pipe(ctx, o));
    else if (p < 0.76) { camera(ctx, 1, F2, pipe(ctx, o)); veil(ctx, seg(p, 0.62, 0.76)); }
    else { d2(ctx, seg(p, 0.76, 1)); veil(ctx, 1 - seg(p, 0.76, 0.86)); }
  }

  const NOFIX = { fix: 0, show: 1 }, NOFIX_BLANK = { fix: 0, show: 0 }, FIXED = { fix: 1, show: 1 };
  const GEN = "Real sampled mechanisms: report/slides/scripts/export_generator_demo.py and export_pipeline_demo.py.";
  const STAB = "Shares: 5,000 raw trees, 384 steps (experiments/abstract_syntax_tree/6_stability_methods_tradeoff.ipynb); example series: export_pipeline_demo.py.";
  const TRACE_FOOT = "The polynomial after each node is recomputed here with the rules of canonicalize_polynomial; the last node equals the canonical polynomial exported by export_generator_demo.py.";
  const COST_FOOT = "Cost figures: report/slides/scripts/export_canonicalizer_timing.py (Python " + T.python + ", one core, same sampler as the corpus).";
  const MANY_FOOT = GEN;

  /** HTML of the method views (sub, sup, span) drawn on the canvas, in the band above the scene. */
  function drawFormula(ctx, html) {
    if (!html) return;
    const dec = (s) => s.replace(/&nbsp;/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
    const re = /<(sub|sup|span)[^>]*>([\s\S]*?)<\/\1>|([^<]+)/g;
    let x = 24, m;
    while ((m = re.exec(html))) {
      let size = 24, dy = 0, color = col.ink, family = Z.MATH, s;
      if (m[3] !== undefined) s = dec(m[3]);
      else {
        s = dec(m[2].replace(/<[^>]*>/g, ""));
        if (m[1] === "sub") { size = 16; dy = 7; } else if (m[1] === "sup") { size = 16; dy = -9; } else { size = 17; color = col.muted; family = Z.SANS; }
      }
      text(ctx, s, x, 40 + dy, size, color, { family });
      x += Z.measure(ctx, s, size, 400, family);
    }
  }
  /** The click-through view of one method (Jacobian, Shao–Wu, Heaps VAR), one pipeline step per view step. */
  function methodSteps(key) {
    const def = GD.get(key);
    return def.steps.map((st, i) => ({
      narr: st.narr, sub: st.sub || "", dur: st.dur || 3000, foot: def.foot,
      run: (ctx, p) => {
        drawFormula(ctx, st.formula);
        ctx.save();
        ctx.translate(0, st.formula ? 70 : 40);
        def.paint(ctx, i, ease(p));
        ctx.restore();
      },
    }));
  }
  const gStep = (i, run, over) => Object.assign({ narr: ZG.steps[i].narr, sub: ZG.steps[i].sub, dur: ZG.steps[i].dur, foot: ZG.steps[i].foot || ZG.foot, run }, over || {});
  const hStep = (i, run, over) => Object.assign({ narr: ZH.steps[i].narr, sub: ZH.steps[i].sub, dur: ZH.steps[i].dur, foot: ZH.steps[i].foot || ZH.foot, run }, over || {});
  const GS = ZG.scenes, HS = ZH.scenes;

  const FLOW = [
    { narr: "The pipeline", sub: "a tree, its series, and its label", dur: 6500, foot: GEN, run: (c, p) => scene0(c, p, NOFIX_BLANK) },
    // ---- the generator
    gStep(0, (c, p) => zoomIn(c, p, FOCUS.sampler, NOFIX_BLANK, GS.compare), { narr: "Zoom · the generator", sub: "autoregressive data in Chronos-2 versus ours" }),
    gStep(2, GS.tree),
    gStep(3, GS.bind),
    { narr: "Zoom · from the tree to a series", sub: "random initial values, the recurrence, and what is thrown away", dur: 6500, foot: GEN, run: sceneSim },
    { narr: "Zoom · the raw series are not stable", sub: "a random recurrence explodes or fades", dur: 6500, foot: STAB,
      run: (c, p) => hop(c, p, FOCUS.sampler, sceneSim, FOCUS.series, sceneUnstable, NOFIX) },
    gStep(4, GS.problem),
    // ---- the fixing
    { narr: "Introduce the fixing", sub: "the tree is repaired before it is simulated", dur: 5200, foot: GEN,
      run: (c, p) => (p < 0.45 ? zoomOut(c, p / 0.45, FOCUS.series, NOFIX, GS.problem) : scene0(c, 1, { fix: ease(seg(p, 0.5, 0.95)), show: 1 })) },
    { narr: "Zoom · the fixing", sub: "tanh on the lag factors, a certified scale, one of four methods", dur: 7000, foot: STAB,
      run: (c, p) => zoomIn(c, p, FOCUS.fix, FIXED, sceneFix) },
    gStep(5, GS.hub),
    ...methodSteps("jac"),
    ...methodSteps("sw"),
    ...methodSteps("hvar"),
    gStep(1, GS.cards, { narr: "Recap · the generator in five stages" }),
    gStep(6, GS.run),
    gStep(7, GS.keep),
    // ---- the canonicalizer
    hStep(0, (c, p) => hop(c, p, FOCUS.fix, GS.keep, FOCUS.canon, HS[0], FIXED), { narr: "Zoom · the equation head" }),
    hStep(1, HS[1]),
    hStep(2, HS[2]),
    { narr: "Zoom · the canonicalizer", sub: "iterative: the same rules at every node, leaves to root", dur: 12500, foot: TRACE_FOOT, run: scene1 },
    { narr: "Zoom · what iterating costs", sub: "cheap per tree, but sequential and tied to the tree", dur: 4600, foot: COST_FOOT, run: scene2 },
    hStep(3, HS[3], { narr: "Protection · RAW and TANH factors" }),
    { narr: "Back to the pipeline", sub: "the label still comes from the iterative canonicalizer", dur: 3800, foot: GEN,
      run: (c, p) => zoomOut(c, p, FOCUS.canon, FIXED, (cc) => HS[3](cc, 1)) },
    { narr: "Swap in a canonicalizer encoder", sub: "a learned model writes the same equation", dur: 4800, foot: GEN,
      run: (c, p) => scene0(c, 1, { fix: 1, show: 1, morph: ease(seg(p, 0.1, 0.7)) }) },
    hStep(4, (c, p) => zoomIn(c, p, FOCUS.canon, { fix: 1, show: 1, morph: 1 }, HS[4]), { narr: "Zoom · the canonicalizer encoder", sub: "a sequence the model can write: factors, end of term, coefficient" }),
    hStep(5, HS[5]),
    hStep(6, HS[6]),
    // ---- many series
    { narr: "Many series", sub: "repeat the same steps once per series", dur: 6500, foot: MANY_FOOT,
      run: (c, p) => { scene0b(c, p); veil(c, 1 - seg(p, 0, 0.2)); } },
  ];

  // one empty fragment per step after the first: the slide advances through them
  const host = document.querySelector("#pipeline .canvas");
  if (host) {
    for (let i = host.querySelectorAll(".ks-frag").length + 1; i < FLOW.length; i++) {
      const s = document.createElement("span");
      s.className = "fragment ks-frag";
      s.setAttribute("data-fragment-index", String(i));
      host.appendChild(s);
    }
  }

  Z.mount({
    id: "pipeline", W, H, steps: FLOW, foot: "",
    paint(ctx, step, p) { FLOW[step].run(ctx, p); },
  });
})();
