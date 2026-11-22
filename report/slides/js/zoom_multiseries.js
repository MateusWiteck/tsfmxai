/* Multiseries generator: the last part of the pipeline slide (pipeline_canon.js).
 *
 * One system of coupled equations, x_i(t) = f_i(past of the series in S_i) + eps_i(t), told in
 * six stages: topology (who may read whom), one tree per channel with a source drawn per lag,
 * one fix drawn per channel, noise and startup history, joint simulation with the retry and
 * replacement policy, and the labels and batch contract.
 *
 * Data: MULTI_DATA, exported by report/slides/scripts/export_multiseries_demo.py from
 * sample_multiseries_system and generate_training_batches (src/tsfmxai/generators/multiseries.py,
 * src/tsfmxai/datasets/multiseries.py). Rules on the slides follow docs/multiseries_training.md.
 */
(function () {
  const M = window.MULTI_DATA, Z = window.ZC;
  if (!M || !Z) return;
  const { col, seg, ease, text, wrap, box, arrow } = Z;
  const D = M.demo, S = M.stats;
  const MINUS = "−";
  // channel colours stay clear of the treatment colours (Shao–Wu orange, Heaps blue, Jacobian teal)
  const CH = ["#6d5bd0", "#be185d", "#4d7c0f", "#475569"];
  const CSOFT = ["#ece8fb", "#fce7f3", "#ecf5dc", "#e9edf2"];
  const SUBS = "₀₁₂₃₄₅₆₇₈₉";
  const sub = (n) => String(n).split("").map((c) => SUBS[+c]).join("");
  const xs = (i) => "x" + sub(i + 1);
  const lagS = (s, l) => `${xs(s)}(t${MINUS}${l})`;
  const NAME = { shao_wu: "Shao–Wu", heaps: "Heaps", jacobian: "Jacobian" };
  const TONE = { shao_wu: col.ours, heaps: col.blue, jacobian: col.teal };
  const n = D.channels, fmtInt = (v) => v.toLocaleString("en-US");
  const STAGES = ["① Topology", "② Trees, sources", "③ Fix per channel", "④ Noise, start", "⑤ Simulate, retry", "⑥ Labels, batch"];
  const maxLag = (i) => Math.max.apply(null, D.canonical[i].flatMap((t) => t.atoms.map((a) => a[1])));

  /** A line chart; o.hist = number of leading startup values drawn grey and dashed. */
  function line(ctx, x, y, w, h, ys, color, o) {
    o = o || {};
    const lo = o.lo !== undefined ? o.lo : Math.min.apply(null, ys), hi = o.hi !== undefined ? o.hi : Math.max.apply(null, ys);
    const span = hi - lo || 1, step = w / (ys.length - 1), k = o.hist || 0;
    const upto = Math.max(2, Math.floor(ys.length * (o.upto === undefined ? 1 : o.upto)));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y + h); ctx.lineTo(x + w, y + h); ctx.stroke();
    if (k > 1) {
      ctx.fillStyle = col.grey;
      ctx.fillRect(x, y, (k - 1) * step, h);
    }
    [[0, Math.min(upto, k), col.muted, [4, 4]], [Math.max(0, k - 1), upto, color, []]].forEach((s) => {
      if (s[1] - s[0] < 2) return;
      ctx.strokeStyle = s[2];
      ctx.setLineDash(s[3]);
      ctx.lineWidth = o.lw || 2.2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (let i = s[0]; i < s[1]; i++) {
        const px = x + i * step, py = y + h - 3 - ((ys[i] - lo) / span) * (h - 6);
        i === s[0] ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }
  function pill(ctx, label, x, y, color, o) {
    o = o || {};
    const size = o.size || 14, w = Z.measure(ctx, label, size, 700) + 22;
    box(ctx, x, y, w, size + 12, { fill: o.fill || color, stroke: color, lw: 1.4, r: (size + 12) / 2 });
    text(ctx, label, x + w / 2, y + size + 3, size, o.fill ? color : col.panel, { align: "center", weight: 700 });
    return w;
  }
  function node(ctx, cx, cy, r, i, a) {
    ctx.globalAlpha = a;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = CSOFT[i];
    ctx.fill();
    ctx.lineWidth = 2.6;
    ctx.strokeStyle = CH[i];
    ctx.stroke();
    text(ctx, xs(i), cx, cy + r * 0.32, r * 0.9, CH[i], { align: "center", weight: 700 });
    ctx.globalAlpha = 1;
  }
  /** Arrow from (x1, y1) to (x2, y2) bent through a control point lifted by `bend`, cut at both radii. */
  function curved(ctx, x1, y1, x2, y2, bend, color, o) {
    o = o || {};
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    const cx = mx - (dy / L) * bend, cy = my + (dx / L) * bend;
    const cut = (px, py, qx, qy, r) => { const d = Math.hypot(qx - px, qy - py) || 1; return [px + ((qx - px) / d) * r, py + ((qy - py) / d) * r]; };
    const [sx, sy] = cut(x1, y1, cx, cy, o.r || 0), [ex, ey] = cut(x2, y2, cx, cy, o.r || 0);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = o.lw || 2;
    ctx.setLineDash(o.dash || []);
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(cx, cy, ex, ey); ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(ey - cy, ex - cx), s = o.head || 9;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - s * Math.cos(ang - 0.42), ey - s * Math.sin(ang - 0.42));
    ctx.lineTo(ex - s * Math.cos(ang + 0.42), ey - s * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
  }
  const chips = (ctx, k) => Z.chips(ctx, STAGES, k, 10);
  const atomText = (a) => (a[2] === "tanh" ? "tanh " : "") + lagS(a[0], a[1]);
  const termBody = (t) => Math.abs(t.coef).toFixed(3) + "·" + t.atoms.map(atomText).join("·");
  /** As many canonical terms as fit in w pixels, then "+ … (k terms)". */
  function polyFit(ctx, terms, w, size) {
    let s = "";
    for (let i = 0; i < terms.length; i++) {
      const neg = terms[i].coef < 0;
      const next = (s ? s + (neg ? " " + MINUS + " " : " + ") : neg ? MINUS : "") + termBody(terms[i]);
      const rest = i < terms.length - 1 ? ` + … (${terms.length} terms)` : "";
      if (Z.measure(ctx, next + rest, size, 400, Z.MATH) > w && s) return s + ` + … (${terms.length} terms)`;
      s = next;
    }
    return s;
  }

  // ------------------------------------------------------------------------------ overview
  const GRAPH = [[150, 250], [390, 250], [150, 380], [390, 380]];
  function overview(ctx, p) {
    const A = (a, b) => ease(seg(p, a, b));
    box(ctx, 20, 60, 520, 388);
    text(ctx, "One system of coupled equations", 40, 96, 22, col.ink, { weight: 700 });
    text(ctx, "xᵢ(t) = fᵢ( past of the series in Sᵢ ) + εᵢ(t)", 40, 136, 22, col.ink, { family: Z.MATH });
    text(ctx, "Sᵢ: the series channel i may read; all read the same past", 40, 162, 15, col.muted);
    ctx.globalAlpha = A(0.1, 0.3);
    text(ctx, "this sampled system: j → i means equation i uses the past of j", 40, 190, 14, col.muted);
    ctx.globalAlpha = 1;
    const ga = A(0.1, 0.35);
    D.effective.forEach((srcs, i) => srcs.forEach((j) => {
      ctx.globalAlpha = ga;
      if (j === i) {  // reads its own past: a small loop on the outer side
        const dir = GRAPH[i][0] < 270 ? -1 : 1;
        ctx.strokeStyle = CH[i];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(GRAPH[i][0] + dir * 38, GRAPH[i][1], 14, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        curved(ctx, GRAPH[j][0], GRAPH[j][1], GRAPH[i][0], GRAPH[i][1], 18, CH[j], { r: 34, lw: 3, head: 12 });
      }
      ctx.globalAlpha = 1;
    }));
    GRAPH.forEach((g, i) => {
      node(ctx, g[0], g[1], 28, i, A(0.05, 0.25));
      ctx.globalAlpha = A(0.2, 0.4);
      const t = D.treatments[i], label = NAME[t], top = g[1] < 300;
      text(ctx, label, g[0], top ? g[1] - 40 : g[1] + 50, 15, TONE[t], { align: "center", weight: 700 });
      ctx.globalAlpha = 1;
    });

    box(ctx, 560, 60, 620, 388);
    text(ctx, "Its four series, simulated together", 580, 96, 22, col.ink, { weight: 700 });
    D.trajectory.forEach((ys, i) => {
      const y = 116 + i * 78, a = A(0.15 + i * 0.08, 0.4 + i * 0.08);
      ctx.globalAlpha = a;
      text(ctx, xs(i), 580, y + 40, 20, CH[i], { weight: 700 });
      line(ctx, 624, y, 536, 62, D.initial_history[i].concat(ys), CH[i], { hist: D.history_length, upto: A(0.2 + i * 0.06, 0.75 + i * 0.06) });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = A(0.6, 0.8);
    text(ctx, `grey: the ${D.history_length} startup values, never in a training window`, 624, 436, 14, col.muted);
    text(ctx, `${D.trajectory[0].length} generated steps`, 1160, 436, 14, col.muted, { align: "right" });
    ctx.globalAlpha = 1;

    ctx.globalAlpha = A(0.85, 1);
    box(ctx, 20, 456, 1160, 40, { fill: col.grey });
    text(ctx, "Before: N independent trees. Now: one system whose N equations may read each other's past; each channel keeps its own label.", 36, 482, 16, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------------------ ① topology
  const TOPO = [
    ["independent", "Independent", "no external source"],
    ["linked_chain", "Linked chain", "the predecessor in a random order"],
    ["previous_chain", "Previous chain", "any earlier channel in that order"],
    ["random", "Random", "any other channel, cycles allowed"],
  ];
  function topology(ctx, p) {
    chips(ctx, 0);
    const order = D.order;
    TOPO.forEach((tp, k) => {
      const x = 20 + k * 294, y = 60, w = 278, a = ease(seg(p, k * 0.1, 0.3 + k * 0.1)), chosen = tp[0] === D.dependency;
      ctx.globalAlpha = a;
      box(ctx, x, y, w, 230, chosen ? { fill: col.osoft, stroke: col.ours, lw: 2.4 } : {});
      text(ctx, tp[1], x + 16, y + 30, 19, chosen ? col.ours : col.ink, { weight: 700 });
      text(ctx, tp[2], x + 16, y + 52, 14, col.muted);
      const px = (pos) => x + 40 + pos * 66, ny = y + 150;
      const pairs = [];
      order.forEach((target, pos) => {
        const cands = tp[0] === "linked_chain" ? order.slice(Math.max(0, pos - 1), pos)
          : tp[0] === "previous_chain" ? order.slice(0, pos)
            : tp[0] === "random" ? order.filter((c) => c !== target) : [];
        cands.forEach((s) => pairs.push([order.indexOf(s), pos]));
      });
      pairs.forEach(([from, to]) => {
        const d = Math.abs(px(to) - px(from)), bend = Math.min(58, 14 + 0.24 * d);
        // the same signed bend puts forward arcs above the row and backward arcs below it
        curved(ctx, px(from), ny, px(to), ny, -bend, chosen ? col.ours : col.muted, { r: 19, lw: 1.6, dash: [5, 4], head: 8 });
      });
      order.forEach((c, pos) => node(ctx, px(pos), ny, 18, c, a));
      if (!pairs.length) text(ctx, "each channel reads only itself", x + w / 2, y + 206, 14, col.muted, { align: "center" });
      ctx.globalAlpha = 1;
    });

    ctx.globalAlpha = ease(seg(p, 0.45, 0.7));
    box(ctx, 20, 306, 560, 142);
    text(ctx, "The draw for this system (random topology)", 36, 334, 17, col.ink, { weight: 700 });
    D.allowed.forEach((src, i) => {
      const y = 344 + i * 26;
      text(ctx, xs(i) + " may read", 40, y + 17, 15, CH[i], { weight: 700 });
      let x = 150;
      src.forEach((j) => { x += pill(ctx, xs(j), x, y, CH[j], { size: 12, fill: CSOFT[j] }) + 6; });
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 0.85));
    box(ctx, 600, 306, 580, 142, { fill: col.soft, stroke: col.blue });
    [
      "its own past is always permitted",
      `number of other sources: uniform from 0 to min(${D.max_external}, candidates)`,
      "a permission is not an edge: the source counts only if the tree draws it",
      "cycles are allowed: every equation reads only the past",
    ].forEach((s, i) => {
      ctx.fillStyle = col.blue;
      ctx.beginPath(); ctx.arc(622, 334 + i * 30 - 5, 4.5, 0, Math.PI * 2); ctx.fill();
      text(ctx, s, 636, 334 + i * 30, 15, col.ink);
    });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = ease(seg(p, 0.85, 1));
    text(ctx, "Each system draws a random channel order; training uses the random topology, with six channels.", 20, 478, 15.5, col.muted);
    ctx.globalAlpha = 1;
  }

  // -------------------------------------------------------------------- ② trees and sources
  const DEMO_CH = D.treatments.indexOf("shao_wu");
  function sources(ctx, p) {
    chips(ctx, 1);
    const tree = D.original_trees[DEMO_CH], srcs = D.allowed[DEMO_CH], sw = D.source_weights[DEMO_CH];
    box(ctx, 20, 60, 660, 388);
    text(ctx, `Channel ${xs(DEMO_CH)}: grow a tree for one placeholder series`, 40, 96, 20, col.ink, { weight: 700 });
    text(ctx, "the same combinatorial sampler as before; then every lag leaf draws its source", 40, 120, 15, col.muted);
    const lay = Z.layout(tree, { x: 90, y: 160, w: 540, h: 200 });
    const leaves = lay.nodes.filter((nd) => nd.n.t === "lag");
    const assigned = (nd) => p > 0.3 + leaves.indexOf(nd) * 0.1;
    Z.drawTree(ctx, lay, {
      size: 16,
      label: (nd) => {
        const t = nd.n;
        if (t.t === "add") return "+";
        if (t.t === "mul") return "×";
        if (t.t === "par") return Z.fmt(t.v, 2);
        return assigned(nd) ? lagS(t.series, t.lag) : `x?(t${MINUS}${t.lag})`;
      },
      style: (nd) => (nd.n.t === "lag" && assigned(nd) ? { fill: CSOFT[nd.n.series], stroke: CH[nd.n.series], color: CH[nd.n.series], lw: 2.6 } : nd.n.t === "lag" ? { color: col.muted } : {}),
    });
    const counts = srcs.map((j) => leaves.filter((nd) => nd.n.series === j).length);
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    box(ctx, 40, 384, 620, 48, { fill: col.grey });
    text(ctx, `here: ${counts[srcs.indexOf(DEMO_CH)]} of ${leaves.length} lags on itself, ` +
      srcs.filter((j) => j !== DEMO_CH).map((j) => `${counts[srcs.indexOf(j)]} on ${xs(j)}`).join(", ") + " · still before its fix", 56, 414, 16, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;

    box(ctx, 700, 60, 480, 388);
    text(ctx, "Which source for a lag?", 720, 96, 20, col.ink, { weight: 700 });
    text(ctx, `own past weight ${D.self_weight}, every other permitted series 1`, 720, 120, 15, col.muted);
    srcs.forEach((j, k) => {
      const y = 140 + k * 44, a = ease(seg(p, 0.1 + k * 0.08, 0.35 + k * 0.08)), v = sw.weights[k];
      ctx.globalAlpha = a;
      text(ctx, xs(j) + (j === DEMO_CH ? " (self)" : ""), 720, y + 22, 16, CH[j], { weight: 700 });
      box(ctx, 820, y + 4, 280 * v * a, 26, { fill: CH[j], stroke: CH[j], r: 5 });
      text(ctx, Math.round(100 * v) + "%", 828 + 280 * v * a, y + 23, 15, col.ink, { weight: 700 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.5, 0.75));
    box(ctx, 716, 282, 448, 150, { fill: col.soft, stroke: col.blue });
    text(ctx, "At least one lag reads another series", 732, 308, 16, col.blue, { weight: 700 });
    wrap(ctx, "A channel with a permitted external source is a Shao–Wu channel. One of its lag leaves, chosen at random, takes an external source; the others use the weights above. So every such channel really reads another series.", 732, 332, 420, 20, 14.5, col.ink);
    text(ctx, "every tree has at least one lag", 732, 424, 14, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- ③ one fix per channel
  function treatments(ctx, p) {
    chips(ctx, 2);
    box(ctx, 20, 60, 1160, 58);
    text(ctx, "The topology sets the fix", 40, 96, 20, col.ink, { weight: 700 });
    const so = D.self_only_treatment_probs, ba = ease(seg(p, 0, 0.25));
    ctx.globalAlpha = ba;
    pill(ctx, "another series permitted → Shao–Wu", 330, 76, TONE.shao_wu, { size: 15 });
    pill(ctx, `only itself → Jacobian ${Math.round(100 * so.jacobian)}% · Heaps ${Math.round(100 * so.heaps)}%`, 690, 76, col.teal, { size: 15 });
    ctx.globalAlpha = 1;

    const desc = (i) => {
      const t = D.treatments[i];
      if (t === "shao_wu") return `reads ${D.effective[i].filter((j) => j !== i).map(xs).join(", ")}; tanh on the lag factors of products; row scale s = ${D.safety_factor} · min(1, ${D.target_gain} / g), derivative bound g = ${D.gains[i].toFixed(2)}, so s = ${D.scales[i].toFixed(3)}`;
      if (t === "heaps") {
        const h = D.heaps[i], kind = { consecutive: "consecutive lags", isolated: "one isolated lag", periodic: "periodic lags" }[h.support];
        return `no external source; stable additive AR on ${kind}, order ${h.order}${h.period > 1 ? `, period ${h.period}` : ""}, drawn directly (40% consecutive, 20% isolated, 40% periodic)`;
      }
      const s = D.scales[i];
      return `no external source; contraction certified on |x| ≤ ${D.domain_radius}, scale ${s < 0.001 ? s.toExponential(1) : s.toFixed(3)}; bounded uniform noise`;
    };
    D.treatments.forEach((t, i) => {
      const y = 132 + i * 72, a = ease(seg(p, 0.2 + i * 0.13, 0.45 + i * 0.13));
      ctx.globalAlpha = a;
      box(ctx, 20, y, 1160, 64, { stroke: TONE[t], lw: 2 });
      text(ctx, xs(i), 40, y + 42, 26, CH[i], { weight: 700 });
      pill(ctx, NAME[t], 92, y + 19, TONE[t], { size: 15 });
      let x = 216;
      D.allowed[i].forEach((j) => { x += pill(ctx, xs(j), x, y + 20, CH[j], { size: 13, fill: CSOFT[j] }) + 6; });
      const one = Z.measure(ctx, desc(i), 15) <= 760;
      wrap(ctx, desc(i), 400, y + (one ? 38 : 27), 760, 20, 15, col.ink);
      ctx.globalAlpha = 1;
    });
    const sh = S.treatment_shares;
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    box(ctx, 20, 428, 1160, 66, { fill: col.osoft, stroke: col.ours });
    text(ctx, `Six channels, at most ${D.max_external} external sources: ${Math.round(100 * sh.shao_wu)}% Shao–Wu, ${Math.round(100 * sh.jacobian)}% Jacobian, ${Math.round(100 * sh.heaps)}% Heaps (measured).`, 40, 456, 17, col.ours, { weight: 700 });
    text(ctx, "Each channel is certified on its own; the mixed system has no joint certificate, so every trajectory is still checked (⑤).", 40, 480, 15, col.ink);
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------ ④ noise and startup
  const compText = (c) => c.family === "constant" ? `constant ${Z.fmt(c.level, 2)}`
    : c.family === "noise" ? `noise ±${c.radius.toFixed(2)}` : `cosine A ${c.amplitude.toFixed(2)}, T ${c.period}`;
  const COMP_TONE = { constant: col.blue, noise: col.muted, cosine: col.teal };
  function start(ctx, p) {
    chips(ctx, 3);
    const H = D.history_spec;
    box(ctx, 20, 60, 430, 388);
    text(ctx, "Noise: one scale per channel", 40, 96, 20, col.ink, { weight: 700 });
    text(ctx, `σ ~ U(${D.noise_std_range[0]}, ${D.noise_std_range[1]}), kept for all its trajectories`, 40, 120, 15, col.muted);
    D.innovations.forEach((s, i) => {
      const y = 140 + i * 56, a = ease(seg(p, 0.05 + i * 0.08, 0.3 + i * 0.08));
      ctx.globalAlpha = a;
      text(ctx, xs(i), 40, y + 24, 20, CH[i], { weight: 700 });
      ctx.fillStyle = col.grey;
      ctx.fillRect(86, y + 8, 200, 22);
      ctx.fillStyle = CH[i];
      ctx.fillRect(86, y + 8, 200 * (s.std / D.noise_std_range[1]) * a, 22);
      text(ctx, "σ = " + s.std.toFixed(3), 296, y + 25, 15, col.ink, { weight: 700 });
      text(ctx, s.family === "uniform" ? "uniform ±√3σ" : "Gaussian", 86, y + 48, 13.5, s.family === "uniform" ? col.teal : col.muted, { weight: 600 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.4, 0.6));
    wrap(ctx, "Jacobian channels get bounded uniform noise: their certificate needs a bound. The head is told σ and the family.", 40, 386, 390, 20, 14.5, col.ink);
    ctx.globalAlpha = 1;

    box(ctx, 470, 60, 710, 388);
    text(ctx, "Startup history: a random sum per channel", 490, 96, 20, col.ink, { weight: 700 });
    text(ctx, `constant, white noise and cosine each appear with probability ${H.constant_probability}, then are summed`, 490, 120, 15, col.muted);
    const all = D.initial_history.flat(), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    D.initial_history.forEach((h, i) => {
      const y = 140 + i * 66, a = ease(seg(p, 0.25 + i * 0.1, 0.5 + i * 0.1));
      ctx.globalAlpha = a;
      text(ctx, xs(i), 490, y + 30, 20, CH[i], { weight: 700 });
      let x = 534;
      D.initialization_components[i].forEach((c) => { x += pill(ctx, compText(c), x, y + 12, COMP_TONE[c.family], { size: 13, fill: col.panel }) + 6; });
      line(ctx, 900, y + 4, 260, 52, h, CH[i], { lo, hi, lw: 2.2 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    wrap(ctx, `${D.history_length} values per channel, as deep as the deepest lag. They only start the recurrence: no window contains them and there is no extra burn-in.`, 490, 412, 670, 20, 15, col.ink);
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------- ⑤ simulate and retry
  function simulate(ctx, p) {
    chips(ctx, 4);
    const A = (a, b) => ease(seg(p, a, b));
    box(ctx, 20, 60, 620, 388);
    text(ctx, "Run, check, retry, replace", 40, 96, 20, col.ink, { weight: 700 });
    const step = (x, y, w, h, title, subt, tone, a) => {
      const c = { blue: [col.soft, col.blue], teal: [col.tsoft, col.teal], ours: [col.osoft, col.ours], plain: [col.panel, col.ink] }[tone];
      ctx.globalAlpha = a;
      box(ctx, x, y, w, h, { fill: c[0], stroke: c[1], lw: 2.2 });
      text(ctx, title, x + 14, y + (subt ? 24 : 30), 16, c[1], { weight: 700 });
      if (subt) text(ctx, subt, x + 14, y + 44, 13.5, col.muted);
      ctx.globalAlpha = 1;
    };
    step(40, 112, 270, 54, "draw a startup history", "and the channel noise", "plain", A(0, 0.12));
    arrow(ctx, 175, 168, 175, 186, col.muted, 2.2, A(0.08, 0.16));
    step(40, 188, 270, 54, "step all channels at once", "every equation reads the same past", "blue", A(0.1, 0.22));
    arrow(ctx, 175, 244, 175, 262, col.muted, 2.2, A(0.18, 0.26));
    step(40, 264, 270, 54, "all values finite?", "float64, and after the cast to float32", "plain", A(0.2, 0.32));
    arrow(ctx, 312, 291, 350, 291, col.teal, 2.4, A(0.3, 0.4));
    ctx.globalAlpha = A(0.3, 0.4);
    text(ctx, "yes", 318, 310, 13.5, col.teal, { weight: 700 });
    ctx.globalAlpha = 1;
    step(352, 264, 268, 54, "keep it", "cut into context / horizon windows", "teal", A(0.32, 0.44));
    ctx.globalAlpha = A(0.42, 0.52);
    ctx.strokeStyle = col.blue;
    ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(312, 276); ctx.lineTo(336, 276); ctx.lineTo(336, 230); ctx.lineTo(350, 230); ctx.stroke();
    text(ctx, "no", 330, 262, 13.5, col.blue, { weight: 700, align: "right" });
    ctx.globalAlpha = 1;
    arrow(ctx, 336, 230, 352, 230, col.blue, 2.2, A(0.42, 0.52));
    step(352, 188, 268, 62, "new history, same noise", "up to 3 retries", "blue", A(0.44, 0.56));
    ctx.globalAlpha = A(0.5, 0.6);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = col.blue;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(486, 186); ctx.lineTo(486, 139); ctx.stroke();
    arrow(ctx, 486, 139, 314, 139, col.blue, 2, 1);
    ctx.setLineDash([]);
    text(ctx, "try again", 494, 168, 13.5, col.blue, { weight: 600 });
    ctx.globalAlpha = 1;
    arrow(ctx, 175, 320, 175, 352, col.ours, 2.4, A(0.6, 0.68));
    ctx.globalAlpha = A(0.6, 0.68);
    text(ctx, "4 attempts failed", 186, 342, 13.5, col.ours, { weight: 700 });
    ctx.globalAlpha = 1;
    step(40, 354, 580, 78, "replace the whole system", "a new seed draws new trees for every channel; its accepted trajectories are dropped too", "ours", A(0.62, 0.74));

    const d = S;
    const rescued = (s) => (s.failures_by_attempt["0"] || 0) - (s.failures_by_attempt["3"] || 0);
    ctx.globalAlpha = A(0.7, 0.85);
    box(ctx, 660, 60, 520, 388);
    text(ctx, "Measured", 680, 96, 20, col.ink, { weight: 700 });
    text(ctx, `${S.systems} systems of ${S.channels} channels × 2 trajectories`, 680, 120, 15, col.muted);
    [
      ["trajectories failing a first try", d.failures_by_attempt["0"] || 0],
      ["rescued by a new history", rescued(d)],
      ["systems replaced", d.replaced_systems],
      ["equations redrawn", d.resampled_equations],
      ["channels reading another series", `${Math.round(100 * d.cross_series_share)}%`],
    ].forEach((r, i) => {
      const y = 166 + i * 40;
      ctx.strokeStyle = col.line;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(680, y + 12); ctx.lineTo(1164, y + 12); ctx.stroke();
      text(ctx, r[0], 680, y, 15.5, col.ink);
      text(ctx, typeof r[1] === "number" ? fmtInt(r[1]) : r[1], 1164, y, 18, col.teal, { align: "right", weight: 700 });
    });
    wrap(ctx, `With certified fixes and the forecasting arm's settings, no trajectory needed a retry and no system was replaced. The retry and replacement remain because the mixed system has no joint certificate. The run yields ${fmtInt(d.windows)} windows.`, 680, 372, 484, 20, 14.5, col.ink);
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------ ⑥ labels and batch
  function labels(ctx, p) {
    chips(ctx, 5);
    box(ctx, 20, 60, 680, 388);
    text(ctx, "One label per channel: its canonical equation", 40, 96, 20, col.ink, { weight: 700 });
    D.canonical.forEach((terms, i) => {
      const y = 116 + i * 66, a = ease(seg(p, 0.05 + i * 0.1, 0.3 + i * 0.1));
      ctx.globalAlpha = a;
      text(ctx, xs(i) + "(t) =", 40, y + 24, 17, CH[i], { weight: 700, family: Z.MATH });
      text(ctx, polyFit(ctx, terms, 572, 15), 112, y + 24, 15, col.ink, { family: Z.MATH });
      text(ctx, "reads " + D.effective[i].map(xs).join(" ") + "  ·  " + NAME[D.treatments[i]] + `  ·  ${D.target_tokens[i]} tokens`, 112, y + 48, 13.5, col.muted);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.5, 0.7));
    box(ctx, 36, 384, 648, 52, { fill: col.osoft, stroke: col.ours });
    wrap(ctx, "The equation actually executed, after its fix. The sources it reads come from the canonical terms, not from the permissions.", 52, 405, 620, 19, 14.5, col.ink);
    ctx.globalAlpha = 1;

    const sh = S.batch_shapes;
    ctx.globalAlpha = ease(seg(p, 0.45, 0.7));
    box(ctx, 720, 60, 460, 388);
    text(ctx, "One training batch", 740, 96, 20, col.ink, { weight: 700 });
    text(ctx, `${S.channels}-channel run: batch, time, channel`, 740, 120, 15, col.muted);
    [
      ["context", sh.context, "the observed window"],
      ["future", sh.future, "the horizon to forecast"],
      ["channel_mask", sh.channel_mask, "real channel or padding"],
      ["equation_targets", sh.equation_targets, "skeleton tokens per channel"],
      ["noise std", sh.noise_std, "σ of every channel"],
      ["noise family", sh.noise_std, "Gaussian 0, uniform 1"],
    ].forEach((r, i) => {
      const y = 144 + i * 40;
      text(ctx, r[0], 740, y + 14, 15.5, col.blue, { weight: 700, family: "Consolas, monospace" });
      text(ctx, "[" + r[1].join(", ") + "]", 1160, y + 14, 15.5, col.ink, { align: "right", weight: 700, family: "Consolas, monospace" });
      text(ctx, r[2], 740, y + 32, 13.5, col.muted);
    });
    wrap(ctx, "Training fixes six channels, so no channel is padding. The head learns the skeleton, not the coefficients.", 740, 414, 420, 19, 14.5, col.ink);
    ctx.globalAlpha = 1;
  }

  const DEMO_FOOT = `Real system: sample_multiseries_system, seed ${D.seed}, ${n} channels for legibility (training uses six), random topology, ≤ 4 operators, dependency-driven treatments; simulated by generate_training_batches. Exported by report/slides/scripts/export_multiseries_demo.py.`;
  const STATS_FOOT = `${S.systems} systems × ${S.channels} channels × 2 trajectories of 128 steps, random topology, 1–6 operators, lags 1–16, degree 4, magnitudes 0.05–1, seed ${S.seed} (export_multiseries_demo.py).`;
  window.ZMULTI = {
    steps: [
      { narr: "Many series", sub: "one system of coupled equations, one label per channel", dur: 6500, foot: DEMO_FOOT, run: overview },
      { narr: "Many series · ① topology", sub: "which series a channel is allowed to read", dur: 6500, foot: DEMO_FOOT + " Topology rules: src/tsfmxai/generators/multiseries.py.", run: topology },
      { narr: "Many series · ② trees and sources", sub: "a scalar tree first, then one source drawn per lag", dur: 6500, foot: DEMO_FOOT, run: sources },
      { narr: "Many series · ③ one fix per channel", sub: "Shao–Wu where another series is permitted; Jacobian or Heaps otherwise", dur: 7000, foot: DEMO_FOOT + " Shares: " + STATS_FOOT, run: treatments },
      { narr: "Many series · ④ noise and startup", sub: "per-channel noise scale, random startup history", dur: 6500, foot: DEMO_FOOT, run: start },
      { narr: "Many series · ⑤ simulate and retry", sub: "three new histories, then a new system", dur: 7500, foot: STATS_FOOT, run: simulate },
      { narr: "Many series · ⑥ labels and batch", sub: "the canonical equation of every channel goes with the windows", dur: 6000, foot: DEMO_FOOT + " Shapes: first batch of the six-channel statistics run.", run: labels },
    ],
  };
})();
