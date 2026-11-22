/* Final section of the deck: LMC-Synth (TimePFN's multivariate generator) and the planned comparison.
 *
 * Slide "lmc": how the generator works, step by step, with a real example. The construction follows
 * synthetic_data_generation/LMC_Synth.py of github.com/egetaga/TimePFN (read on 9 Nov 2026) and
 * Taga, Ildiz and Oymak (2025). Curves and matrices come from data/lmc_demo.js, written by
 * scripts/export_lmc_demo.py (illustrative sizes: 5 channels, 4 latents; official: 160 channels).
 * Corpus and training numbers, and the ablation bars, are transcribed from the paper.
 *
 * Slide "evalplan": the comparative evaluation (preliminary mixture, Studies A and B). The LMC rows
 * are measured; every cell of our generator that is still pending is a red dash.
 */
(function () {
  const D = window.LMC_DATA, Z = window.ZC;
  if (!D || !Z) return;
  const { col, seg, ease, text, wrap, box, arrow } = Z;
  const MINUS = "−";
  const W = 1200, H = 500;
  const COLORS = [col.blue, col.teal, "#6d5bd0", col.ours, "#be185d"];
  const SUBS = "₁₂₃₄₅₆₇₈₉";
  const NL = D.latents.length, NC = D.examples.mid.channels.length;
  const rows = (flat, n) => Array.from({ length: flat.length / n }, (_, i) => flat.slice(i * n, (i + 1) * n));
  const WEIGHTS = { low: rows(D.examples.low.weights, NL), mid: rows(D.examples.mid.weights, NL), high: rows(D.examples.high.weights, NL) };
  const CORR = { low: rows(D.examples.low.corr, NC), high: rows(D.examples.high.corr, NC) };

  function thumb(ctx, x, y, w, h, arr, color, f, lw) {
    const lo = Math.min(...arr), hi = Math.max(...arr), n = Math.max(2, Math.floor(arr.length * f));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw || 1.8;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const X = x + (i / (arr.length - 1)) * w, Y = y + h - 4 - ((arr[i] - lo) / (hi - lo || 1)) * (h - 8);
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.stroke();
  }
  const heat = (v) => {
    const a = Math.max(0, Math.min(1, v));
    return `rgb(${Math.round(255 - a * 226)}, ${Math.round(255 - a * 181)}, ${Math.round(255 - a * 16)})`;
  };
  function matrix(ctx, x, y, cell, vals, dec, label, abs) {
    vals.forEach((r, i) => r.forEach((v, j) => {
      ctx.fillStyle = heat(abs ? Math.abs(v) : v);
      ctx.fillRect(x + j * cell, y + i * cell, cell - 2, cell - 2);
      text(ctx, v.toFixed(dec), x + j * cell + (cell - 2) / 2, y + i * cell + cell / 2 + 5, 13.5, (abs ? Math.abs(v) : v) > 0.55 ? "#fff" : col.ink, { align: "center", weight: 600 });
    }));
    if (label) text(ctx, label, x, y - 10, 14, col.muted, { weight: 600 });
  }
  const mixed = (w) => D.latents[0].series.map((_, t) => w.reduce((s, wj, j) => s + wj * D.latents[j].series[t], 0));
  const sum = (a, b) => a.map((v, i) => v + b[i]);

  // ---- step 0: the whole construction, as a flow from latents to channels ----
  function overview(ctx, p) {
    const lx = 24, lw = 200, cx = 640, cw = 230, lh = 62;
    const ly = (i) => 84 + i * 84, cy = (i) => 84 + i * 68;
    text(ctx, `${NL} latent functions`, lx, 68, 17, col.ink, { weight: 700 });
    text(ctx, `${NC} channels`, cx, 68, 17, col.ink, { weight: 700 });
    text(ctx, "mixing weights α", 330, 68, 17, col.ink, { weight: 700 });
    // edges: one line per (latent, channel), width and opacity proportional to the weight
    const eP = ease(seg(p, 0.25, 0.65));
    WEIGHTS.mid.forEach((row, c) => row.forEach((a, l) => {
      if (eP <= 0) return;
      const x0 = lx + lw, y0 = ly(l) + lh / 2, x1 = cx, y1 = cy(c) + 27;
      ctx.globalAlpha = 0.15 + 0.7 * a;
      ctx.strokeStyle = COLORS[l];
      ctx.lineWidth = 0.8 + 7 * a;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(x0 + 190, y0, x1 - 190, y1, x0 + (x1 - x0) * eP, y0 + (y1 - y0) * eP);
      ctx.stroke();
    }));
    ctx.globalAlpha = 1;
    D.latents.forEach((l, i) => {
      const a = ease(seg(p, i * 0.07, i * 0.07 + 0.25));
      ctx.globalAlpha = a;
      box(ctx, lx - 6, ly(i) - 6, lw + 12, lh + 12, { r: 10, fill: col.panel, stroke: COLORS[i], lw: 2 });
      thumb(ctx, lx, ly(i), lw, lh, l.series, COLORS[i], ease(seg(p, i * 0.07, i * 0.07 + 0.4)));
      ctx.fillStyle = "#fff";
      ctx.fillRect(lx + 2, ly(i) + 2, 40, 20);
      text(ctx, `l${SUBS[i]}(t)`, lx + 6, ly(i) + 17, 13.5, COLORS[i], { weight: 700 });
      ctx.globalAlpha = 1;
    });
    D.examples.mid.channels.forEach((c, i) => {
      const a = ease(seg(p, 0.55 + i * 0.05, 0.85 + i * 0.03));
      ctx.globalAlpha = a;
      box(ctx, cx - 6, cy(i) - 4, cw + 12, 62, { r: 10, fill: col.panel, stroke: col.line, lw: 1.6 });
      thumb(ctx, cx, cy(i), cw, 54, c, COLORS[i % 5], ease(seg(p, 0.6 + i * 0.05, 1)));
      ctx.fillStyle = "#fff";
      ctx.fillRect(cx + 2, cy(i) + 2, 40, 20);
      text(ctx, `y${SUBS[i]}(t)`, cx + 6, cy(i) + 17, 13.5, COLORS[i % 5], { weight: 700 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    box(ctx, 900, 70, 280, 316, { fill: col.osoft, stroke: col.ours });
    text(ctx, "Official defaults", 918, 102, 18, col.ours, { weight: 700 });
    [["15,000", "datasets"], ["160", "channels each"], ["1,024", "steps each"], ["8 to 160", "latent functions"]].forEach((r, i) => {
      text(ctx, r[0], 918, 150 + i * 56, 26, col.ink, { weight: 700 });
      text(ctx, r[1], 918, 172 + i * 56, 15, col.muted);
    });
    text(ctx, "C_i(t) = Σ_j α_ij · l_j(t),     α_ij ≥ 0,   Σ_j α_ij = 1", 24, 430, 22, col.ink, { family: Z.MATH });
    text(ctx, "Line width = weight α_ij. Illustration: 5 channels and 4 latents; the official generator draws 160 channels.", 24, 462, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 1: the latent functions (drawn like the bank-to-pipeline of the KernelSynth slide) ----
  function latents(ctx, p) {
    // a stack of "× N" ghosts behind the first row says: the same procedure is repeated per latent
    D.latents.forEach((l, i) => {
      const a = ease(seg(p, i * 0.14, i * 0.14 + 0.3)), y = 70 + i * 92, slide = (1 - a) * 40;
      ctx.globalAlpha = a;
      box(ctx, 20 - slide, y, 700, 80, { stroke: COLORS[i], lw: 2 });
      thumb(ctx, 34 - slide, y + 8, 250, 64, l.series, COLORS[i], ease(seg(p, 0.1 + i * 0.12, 0.6 + i * 0.12)));
      text(ctx, `l${SUBS[i]}`, 300 - slide, y + 32, 22, COLORS[i], { weight: 700 });
      text(ctx, "kernel expression", 340 - slide, y + 28, 13, col.muted);
      wrap(ctx, l.label, 340 - slide, y + 52, 370, 20, 17.5, col.ink, { weight: 600 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.4, 0.8));
    box(ctx, 740, 70, 440, 360);
    text(ctx, "Exactly KernelSynth, once per latent", 760, 104, 19, col.ink, { weight: 700 });
    wrap(ctx, "Each latent function is a draw from a Gaussian process whose kernel is built from 1 to 5 bank kernels, combined left to right by + or ×. Raw draw: no normalisation.", 760, 134, 400, 24, 17, col.ink);
    text(ctx, "How many latents?", 760, 262, 17, col.blue, { weight: 700 });
    text(ctx, "L = clip( round( Weibull·scale + 1 ),  max(2, C/20),  C )", 760, 292, 17, col.ink, { family: Z.MATH });
    wrap(ctx, "For C = 160 channels: between 8 and 160 latents, never more latents than channels.", 760, 322, 400, 22, 16, col.muted);
    wrap(ctx, "The script has no default for the Weibull shape and scale: they are arguments.", 760, 380, 400, 22, 14.5, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 2: the Dirichlet weights, each channel as a bar split among the latents ----
  function weightBars(ctx, x, y, w, rowsW, grow, rowH) {
    rowsW.forEach((r, i) => {
      let acc = 0;
      text(ctx, `y${SUBS[i]}`, x - 8, y + i * rowH + rowH / 2 + 5, 14, COLORS[i % 5], { align: "right", weight: 700 });
      r.forEach((v, j) => {
        const bw = v * w * grow;
        ctx.fillStyle = COLORS[j];
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x + acc, y + i * rowH + 3, Math.max(0, bw - 1.5), rowH - 6);
        ctx.globalAlpha = 1;
        if (v > 0.12 && grow > 0.9) text(ctx, v.toFixed(2), x + acc + bw / 2, y + i * rowH + rowH / 2 + 5, 12.5, "#fff", { align: "center", weight: 700 });
        acc += bw;
      });
    });
  }
  function mixing(ctx, p) {
    [["low", 20, "small concentration"], ["high", 600, "large concentration"]].forEach(([k, x0, title], i) => {
      const a = ease(seg(p, i * 0.25, i * 0.25 + 0.35)), g = ease(seg(p, 0.15 + i * 0.25, 0.6 + i * 0.25));
      ctx.globalAlpha = a;
      box(ctx, x0, 62, 560, 290);
      text(ctx, `${title}:  d = ${D.examples[k].d}`, x0 + 20, 94, 20, k === "low" ? col.ours : col.teal, { weight: 700 });
      text(ctx, "each bar: how one channel splits its weight over the latents", x0 + 20, 118, 13.5, col.muted);
      weightBars(ctx, x0 + 52, 134, 380, WEIGHTS[k], g, 34);
      D.latents.forEach((_, j) => {
        ctx.fillStyle = COLORS[j];
        ctx.fillRect(x0 + 452, 140 + j * 24, 12, 12);
        text(ctx, `l${SUBS[j]}`, x0 + 470, 151 + j * 24, 14, col.ink, { weight: 600 });
      });
      text(ctx, k === "low" ? "→ channels nearly copy one latent" : "→ every channel mixes all latents", x0 + 20, 332, 15, col.ink, { weight: 700 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 1));
    text(ctx, "α_i ∼ Dirichlet( d · 1 ),  one draw per channel;    d ∼ Uniform( d_min , d_max ),  one draw per dataset", 20, 384, 20, col.ink, { family: Z.MATH });
    wrap(ctx, "A Dirichlet draw is a list of non-negative numbers that sum to 1: a convex combination. The concentration d decides how evenly the weight is spread. The script has no default for d_min and d_max.", 20, 414, 1150, 23, 16.5, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 3: a channel is a weighted sum, scanned at one time step ----
  function building(ctx, p) {
    const ch = 1, w = WEIGHTS.mid[ch], f = ease(seg(p, 0, 0.35));
    const scan = ease(seg(p, 0.5, 0.95));
    text(ctx, "channel y₂ as a weighted sum of the latents", 20, 70, 19, col.ink, { weight: 700 });
    const n = D.latents[0].series.length, idx = Math.min(n - 1, Math.floor(scan * (n - 1)));
    const total = mixed(w);
    const rowsXY = [];
    D.latents.forEach((l, i) => {
      const y = 88 + i * 86, a = ease(seg(p, i * 0.08, i * 0.08 + 0.25));
      ctx.globalAlpha = a;
      thumb(ctx, 20, y, 220, 60, l.series, COLORS[i], 1);
      text(ctx, `l${SUBS[i]}`, 26, y + 15, 13.5, COLORS[i], { weight: 700 });
      arrow(ctx, 246, y + 30, 330, y + 30, col.muted, 2, 1);
      box(ctx, 336, y + 8, 92, 44, { r: 10, fill: col.soft, stroke: COLORS[i] });
      text(ctx, `× ${w[i].toFixed(2)}`, 382, y + 37, 20, COLORS[i], { align: "center", weight: 700 });
      // the scaled curve shrinks from its own amplitude to the weighted one
      const scaled = l.series.map((v) => v * (1 - f + f * w[i]));
      const lo = Math.min(...l.series), hi = Math.max(...l.series), mid = (lo + hi) / 2, half = (hi - lo) / 2 || 1;
      ctx.strokeStyle = col.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(446, y, 200, 60);
      ctx.strokeStyle = COLORS[i];
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      scaled.forEach((v, t) => {
        const X = 446 + (t / (n - 1)) * 200, Y = y + 30 - ((v - mid * (1 - f + f * w[i])) / half) * 26;
        t ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      });
      ctx.stroke();
      arrow(ctx, 652, y + 30, 716, 244, col.line, 1.6, 1);
      rowsXY.push({ y, scaled, mid: mid * (1 - f + f * w[i]), half });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.35, 0.6));
    text(ctx, "Σ", 700, 252, 34, col.ink, { align: "center", weight: 700 });
    const tlo = Math.min(...total), thi = Math.max(...total);
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(740, 160, 440, 140);
    text(ctx, "y₂(t)", 750, 182, 16, COLORS[1], { weight: 700 });
    ctx.strokeStyle = COLORS[1];
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    for (let t = 0; t <= idx; t++) {
      const X = 740 + (t / (n - 1)) * 440, Y = 160 + 140 - 8 - ((total[t] - tlo) / (thi - tlo || 1)) * 124;
      t ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.stroke();
    // the scan line: the same time step in every panel
    if (scan > 0) {
      const li = scan < 1 ? idx : Math.floor(0.62 * (n - 1)), tx = (x0, wd) => x0 + (li / (n - 1)) * wd;
      ctx.strokeStyle = col.ours;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 4]);
      [[20, 220], [446, 200]].forEach(([x0, wd]) => {
        ctx.beginPath();
        ctx.moveTo(tx(x0, wd), 88);
        ctx.lineTo(tx(x0, wd), 88 + 3 * 86 + 60);
        ctx.stroke();
      });
      ctx.beginPath();
      ctx.moveTo(tx(740, 440), 160);
      ctx.lineTo(tx(740, 440), 300);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = ease(seg(p, 0.85, 1));
    text(ctx, `y₂ = ${w.map((v, i) => `${v.toFixed(2)}·l${SUBS[i]}`).join(" + ")}`, 740, 336, 17, col.ink, { family: Z.MATH });
    wrap(ctx, "The orange line is one time step t: y₂(t) uses only the latent values at that same t. No lag, no product.", 740, 372, 440, 22, 16, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 4: which channels move together? Two pairs, then the effect of d ----
  const zs = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length, sd = Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1; return a.map((v) => (v - m) / sd); };
  function pairRow(ctx, p, y, i, j, why, color, t0) {
    const a = ease(seg(p, t0, t0 + 0.25)), g = ease(seg(p, t0 + 0.1, t0 + 0.5)), w = WEIGHTS.low;
    ctx.globalAlpha = a;
    text(ctx, `channels y${SUBS[i]} and y${SUBS[j]}`, 40, y + 20, 17, col.ink, { weight: 700 });
    [i, j].forEach((c, r) => {
      text(ctx, `y${SUBS[c]}`, 40, y + 50 + r * 26, 14, COLORS[c % 5], { weight: 700 });
      let acc = 0;
      w[c].forEach((v, k) => {
        ctx.fillStyle = COLORS[k];
        ctx.fillRect(66 + acc * 130 * g, y + 38 + r * 26, Math.max(0, v * 130 * g - 1.5), 18);
        if (v > 0.25 && g > 0.9) text(ctx, `l${SUBS[k]}`, 66 + (acc + v / 2) * 130, y + 52 + r * 26, 12.5, "#fff", { align: "center", weight: 700 });
        acc += v;
      });
    });
    text(ctx, "weights used", 66, y + 100, 12.5, col.muted);
    // the two channels, overlaid (each rescaled to mean 0 and unit spread)
    const X0 = 230, W0 = 330, H0 = 96, n = D.examples.low.channels[0].length, f = ease(seg(p, t0 + 0.2, t0 + 0.7));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(X0, y + 8, W0, H0);
    [i, j].forEach((c, r) => {
      const z = zs(D.examples.low.channels[c]), cnt = Math.max(2, Math.floor(n * f));
      ctx.strokeStyle = COLORS[c % 5];
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      for (let t = 0; t < cnt; t++) {
        const X = X0 + (t / (n - 1)) * W0, Y = y + 8 + H0 / 2 - Math.max(-2.6, Math.min(2.6, z[t])) * (H0 / 6);
        t ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = a * ease(seg(p, t0 + 0.55, t0 + 0.85));
    text(ctx, `correlation ${CORR.low[i][j].toFixed(2)}`, X0, y + 128, 19, color, { weight: 700 });
    text(ctx, why, X0 + 150, y + 128, 14.5, col.ink);
    ctx.globalAlpha = 1;
  }
  function correlation(ctx, p) {
    box(ctx, 20, 62, 560, 336);
    text(ctx, "Do two channels move together?  (the d = 0.15 example)", 40, 90, 17.5, col.ink, { weight: 700 });
    pairRow(ctx, p, 98, 0, 2, "both lean on l₂: they share it", col.teal, 0);
    pairRow(ctx, p, 248, 0, 1, "no latent in common", col.ours, 0.12);
    // right: the effect of d
    const a2 = ease(seg(p, 0.5, 0.7));
    ctx.globalAlpha = a2;
    box(ctx, 600, 62, 580, 336);
    text(ctx, "More even weights: channels share more latents", 620, 90, 17.5, col.ink, { weight: 700 });
    const bx = 668, by = 118, bw = 476, bh = 190, sw = D.sweep, lx = (d) => bx + ((Math.log(d) - Math.log(0.05)) / (Math.log(10) - Math.log(0.05))) * bw, Y = (v) => by + bh - v * bh;
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(bx, by + bh);
    ctx.lineTo(bx + bw, by + bh);
    ctx.moveTo(bx, by);
    ctx.lineTo(bx, by + bh);
    ctx.stroke();
    const q = seg(p, 0.6, 1) * (sw.length - 1), n0 = Math.floor(q), fr = q - n0;
    const pts = sw.slice(0, n0 + 1).map((s) => [lx(s.d), Y(s.mean_abs_corr)]);
    if (n0 < sw.length - 1) pts.push([lx(sw[n0].d) + (lx(sw[n0 + 1].d) - lx(sw[n0].d)) * fr, Y(sw[n0].mean_abs_corr) + (Y(sw[n0 + 1].mean_abs_corr) - Y(sw[n0].mean_abs_corr)) * fr]);
    ctx.strokeStyle = col.blue;
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    pts.forEach((pt, k) => (k ? ctx.lineTo(pt[0], pt[1]) : ctx.moveTo(pt[0], pt[1])));
    ctx.stroke();
    sw.forEach((s, k) => {
      if (k % 2 === 0 || k === sw.length - 1) text(ctx, String(s.d), lx(s.d), by + bh + 20, 12.5, col.muted, { align: "center" });
      if (k > q) return;
      ctx.beginPath();
      ctx.arc(lx(s.d), Y(s.mean_abs_corr), 4, 0, 7);
      ctx.fillStyle = col.blue;
      ctx.fill();
    });
    [0, 0.5, 1].forEach((v) => text(ctx, v.toFixed(1), bx - 10, Y(v) + 5, 12.5, col.muted, { align: "right" }));
    text(ctx, "d: how evenly a channel spreads its weight (small = on one latent, large = on all)", bx + bw / 2, by + bh + 42, 13, col.muted, { align: "center" });
    text(ctx, "average correlation between channels", bx - 40, by - 12, 13, col.muted);
    ctx.globalAlpha = a2 * ease(seg(p, 0.85, 1));
    text(ctx, "channels differ", bx + 8, Y(0.23) + 30, 15, col.ours, { weight: 700 });
    text(ctx, "channels alike", bx + bw - 8, Y(0.92) - 16, 15, col.teal, { align: "right", weight: 700 });
    ctx.globalAlpha = 1;
    ctx.globalAlpha = ease(seg(p, 0.85, 1));
    text(ctx, `Whole 5-channel example: average correlation ${D.examples.low.mean_abs_corr.toFixed(2)} at d = ${D.examples.low.d}, ${D.examples.high.mean_abs_corr.toFixed(2)} at d = ${D.examples.high.d}.`, 20, 432, 16.5, col.ink, { weight: 600 });
    text(ctx, "d is drawn per dataset, so the corpus holds everything from almost independent to almost identical channels.", 20, 458, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 5: the corpus and the training, with counted-up figures ----
  function corpus(ctx, p) {
    const cards = [
      ["Corpus", col.blue, [[15000, "datasets", 0], [160, "channels each", 0], [1024, "steps each", 0], [1.5, "million windows", 1]], "windows of 192 (96 in, 96 out)"],
      ["Model: TimePFN", col.teal, [[8, "transformer layers", 0], [10, "hours on one L40S", 0]], "tokens of all channels attend to each other; MSE loss; independent channels first, then correlated"],
      ["Evaluation in the paper", "#6d5bd0", [[9, "multivariate datasets", 0], [96, "input and horizon", 0]], "ETTh1, ETTh2, ETTm1, ETTm2, Weather, Solar, ECL, Exchange, Traffic; MSE and MAE"],
    ];
    cards.forEach((c, i) => {
      const a = ease(seg(p, i * 0.15, i * 0.15 + 0.3)), x = 20 + i * 392, k = ease(seg(p, 0.2 + i * 0.15, 0.7 + i * 0.1));
      ctx.globalAlpha = a;
      box(ctx, x, 70, 372, 310, { stroke: c[1], lw: 2.4 });
      text(ctx, c[0], x + 20, 106, 21, c[1], { weight: 700 });
      c[2].forEach((s, r) => {
        const v = s[0] * k, str = s[2] ? v.toFixed(1) : Math.round(v).toLocaleString("en-US");
        text(ctx, str, x + 20, 156 + r * 46, 30, col.ink, { weight: 700 });
        text(ctx, s[1], x + 170, 156 + r * 46, 16.5, col.muted);
      });
      wrap(ctx, c[3], x + 20, 156 + c[2].length * 46 + 6, 332, 20, 15, col.ink);
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    wrap(ctx, "These are the paper's numbers, for the paper's own model. Our comparison reuses the generator, not this model.", 20, 416, 1160, 24, 18, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  // ---- step 6: the paper's own check on cross-channel dependence ----
  const ABL = [["ECL", 0.315, 0.350], ["Weather", 0.209, 0.214], ["Traffic", 1.108, 1.180], ["Solar", 0.941, 1.197], ["Exchange", 0.105, 0.113], ["ETTh1", 0.453, 0.468], ["ETTh2", 0.328, 0.326], ["ETTm1", 0.637, 0.761], ["ETTm2", 0.212, 0.215]];
  function ablation(ctx, p) {
    box(ctx, 20, 62, 1160, 330);
    text(ctx, "Zero-shot MSE with independent channels only, relative to full LMC-Synth", 40, 94, 19, col.ink, { weight: 700 });
    const bx = 160, bw = 900, zero = bx + bw * 0.12, scale = (bw * 0.88) / 30;
    ctx.strokeStyle = col.muted;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(zero, 112);
    ctx.lineTo(zero, 372);
    ctx.stroke();
    ABL.forEach((r, i) => {
      const pct = (r[2] / r[1] - 1) * 100, y = 114 + i * 28, a = ease(seg(p, i * 0.06, i * 0.06 + 0.3));
      text(ctx, r[0], 150, y + 14, 15, col.ink, { align: "right", weight: 600 });
      ctx.fillStyle = pct >= 0 ? col.ours : col.teal;
      const w = Math.abs(pct) * scale * a;
      ctx.fillRect(pct >= 0 ? zero : zero - w, y + 2, w, 18);
      ctx.globalAlpha = a;
      text(ctx, `${pct >= 0 ? "+" : MINUS}${Math.abs(pct).toFixed(1)}%`, pct >= 0 ? zero + w + 8 : zero + 8, y + 16, 14, col.ink, { weight: 700 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "Dependence between channels helps on 8 of 9 datasets (ETTh2 is the exception).", 40, 420, 18, col.ink, { weight: 700 });
    text(ctx, "Bars are worse-than-LMC-Synth: the model trained on independent channels only.", 40, 446, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  // ---- step 7: what it can and cannot express ----
  const CAN = [
    ["Dependence between channels", "shared latent functions", true, "lagged reads of up to two other series", true],
    ["Form of the dependence", "linear, a convex combination", false, "sums, products, tanh of lags", true],
    ["Lags between channels", "none: same time step", false, "x_j(t − ℓ) is a leaf", true],
    ["Products between channels", "none", false, "x_i(t − a) · x_j(t − b)", true],
    ["Law of the data", "Gaussian (the paper's noise is not in the code)", false, "not Gaussian in general", true],
    ["Stored with a sample", "start date and target values only", false, "the equation, as the label", true],
    ["Reproducible draws", "seeds never set in the script", false, "fixed seeds, recorded", true],
  ];
  function capability(ctx, p) {
    const cx = [24, 330, 740];
    ["Capability", "LMC-Synth (TimePFN)", "This project's generator"].forEach((h, i) => text(ctx, h, cx[i], 78, 15, col.muted, { weight: 700 }));
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(20, 90);
    ctx.lineTo(1180, 90);
    ctx.stroke();
    CAN.forEach((r, i) => {
      const y = 100 + i * 50, a = ease(seg(p, i * 0.09, i * 0.09 + 0.3));
      ctx.globalAlpha = a;
      text(ctx, r[0], cx[0], y + 30, 17, col.ink, { weight: 700 });
      text(ctx, (r[2] ? "✓  " : "✗  ") + r[1], cx[1], y + 30, 17, r[2] ? col.teal : col.ours);
      text(ctx, (r[4] ? "✓  " : "✗  ") + r[3], cx[2], y + 30, 17, r[4] ? col.teal : col.ours);
      ctx.strokeStyle = col.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(20, y + 46);
      ctx.lineTo(1180, y + 46);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    text(ctx, "Read from LMC_Synth.py: the output is np.dot(weights, latent_functions), returned as {start, target}.", 24, 478, 15, col.muted);
    ctx.globalAlpha = 1;
  }

  const LSTEPS = [
    { narr: "LMC-Synth in one picture", sub: "many channels from a few shared functions", dur: 6500 },
    { narr: "① Latent functions", sub: "KernelSynth draws, as before", dur: 5200 },
    { narr: "② Mixing weights", sub: "a Dirichlet draw per channel", dur: 5600 },
    { narr: "③ A channel is a weighted sum", sub: "same time step, no lags", dur: 7000 },
    { narr: "④ Which channels move together?", sub: "channels that share a latent are correlated", dur: 7000 },
    { narr: "The corpus and the model that learns from it", sub: "TimePFN, as reported in the paper", dur: 4200 },
    { narr: "Does the dependence between channels matter?", sub: "the paper's own ablation", dur: 5200 },
    { narr: "What it can and cannot express", sub: "against this project's generator", dur: 6500 },
  ];
  const LFOOT = "LMC_Synth.py (egetaga/TimePFN, read 9 Nov 2026) and Taga, Ildiz and Oymak (2025, AAAI). Curves from data/lmc_demo.js (scripts/export_lmc_demo.py): 5 channels, 4 nearly uncorrelated latents of similar amplitude, fixed seeds.";
  Z.mount({
    id: "lmc", W, H, steps: LSTEPS, foot: LFOOT,
    paint(ctx, step, p) {
      [overview, latents, mixing, building, correlation, corpus, ablation, capability][step](ctx, p);
    },
  });
  LSTEPS[4].foot = LFOOT + " Right: pool of 40 raw latent draws, 100 random subsets of 8 latents and 16 channels per d, averaged (our simulation of the construction). Left: the d = 0.15 example, each channel rescaled to mean 0 and unit spread for the overlay.";
  LSTEPS[5].foot = "Taga, Ildiz and Oymak (2025): corpus size, windows, model and training as reported in the paper; numbers transcribed from the text.";
  LSTEPS[6].foot = "Zero-shot MSE of TimePFN and of TimePFN trained on independent channels only, as reported in the paper (transcribed; check against the paper's table before citing).";
  LSTEPS[7].foot = "Left column: LMC_Synth.py. Right column: src/tsfmxai/generators (AST over Lag(variable, lag), Multiply, Tanh) and the recorded seeds of the experiments. The paper also mentions multiplicative Gaussian noise (σ = 0.1); the script has no noise term.";

  // ================= comparison with LMC-Synth: Studies A and B, zero-shot on Ibirapuera and Bahia =================
  const RED = "#dc2626";
  // relative MAE vs the 24-hour seasonal naive (geometric mean over a dataset's channels); null = pending
  const ARMS = [
    ["LMC-Synth", col.blue, [1.0175, 1.2390], [0.9584, 1.1800], "done"],
    ["Ours", col.ours, [null, null], [null, null], "next, p = 0.50"],
    ["Ours + equation head", col.ours, [null, null], [null, null], "needs the teacher"],
    ["Ours + curriculum", col.ours, ["n/a", "n/a"], [null, null], "deferred"],
    ["Ours + head + curriculum", col.ours, ["n/a", "n/a"], [null, null], "deferred"],
  ];
  function compare(ctx, p) {
    // top strip: preliminary choice -> two studies -> zero-shot real data
    const A = (a, b) => ease(seg(p, a, b));
    const strip = [
      [20, 250, "① Preliminary", "KernelSynth share p = 0.25 or 0.50", "done: p = 0.50 selected"],
      [290, 290, "② Study A · equal exposure", "445,433 updates, corpus on disk", "secondary"],
      [600, 290, "③ Study B · equal time", "10 h, generation inside the clock", "principal"],
      [910, 270, "Zero-shot, real data", "Ibirapuera (5 ch) · Bahia (4 ch), 2019", "96 → 96 hours"],
    ];
    strip.forEach((s, i) => {
      ctx.globalAlpha = A(i * 0.08, i * 0.08 + 0.2);
      box(ctx, s[0], 62, s[1], 78, { stroke: i === 3 ? "#6d5bd0" : col.teal, lw: 2.2 });
      text(ctx, s[2], s[0] + 14, 88, 16.5, i === 3 ? "#6d5bd0" : col.teal, { weight: 700 });
      text(ctx, s[3], s[0] + 14, 110, 14, col.ink);
      text(ctx, s[4], s[0] + 14, 130, 13, col.muted);
      if (i < 3) arrow(ctx, s[0] + s[1] + 2, 101, strip[i + 1][0] - 2, 101, col.muted, 2);
      ctx.globalAlpha = 1;
    });

    // table: relative MAE per study and dataset
    const x0 = 20, y0 = 160, rowH = 40, cw = [250, 110, 110, 110, 110, 150];
    ctx.globalAlpha = A(0.35, 0.5);
    text(ctx, "Study A", x0 + cw[0] + cw[1], y0 + 14, 14, col.teal, { align: "center", weight: 700 });
    text(ctx, "Study B", x0 + cw[0] + cw[1] * 2 + cw[3], y0 + 14, 14, col.teal, { align: "center", weight: 700 });
    ["Arm (same Chronos-2, random start)", "Ibirapuera", "Bahia", "Ibirapuera", "Bahia", "Status"].forEach((h, i) => {
      const x = x0 + cw.slice(0, i).reduce((s, v) => s + v, 0);
      text(ctx, h, i ? x + cw[i] / 2 : x + 8, y0 + 38, 14, col.ink, { align: i ? "center" : "left", weight: 700 });
    });
    ctx.strokeStyle = col.muted;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y0 + 48); ctx.lineTo(x0 + 840, y0 + 48); ctx.stroke();
    ctx.globalAlpha = 1;
    ARMS.forEach((r, k) => {
      const y = y0 + 52 + k * rowH;
      ctx.globalAlpha = A(0.45 + k * 0.06, 0.6 + k * 0.06);
      if (k % 2 === 0) { ctx.fillStyle = col.grey; ctx.fillRect(x0, y, 840, rowH); }
      text(ctx, r[0], x0 + 8, y + 26, 16, r[1], { weight: 700 });
      const cells = [...r[2], ...r[3]];
      let x = x0 + cw[0];
      cells.forEach((v, i) => {
        const s = v === null ? "–" : typeof v === "number" ? v.toFixed(4) : v;
        text(ctx, s, x + cw[i + 1] / 2, y + 26, v === null ? 20 : 16, v === null ? RED : typeof v === "number" ? col.ink : col.muted, { align: "center", weight: v === null ? 700 : 600 });
        x += cw[i + 1];
      });
      text(ctx, r[4], x + cw[5] / 2, y + 26, 14, r[4] === "done" ? col.teal : col.muted, { align: "center", weight: 600 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = A(0.75, 0.9);
    text(ctx, "MAE relative to the 24-hour seasonal naive, geometric mean over the dataset's channels; below 1 beats it. Red dash: not run yet.", x0, y0 + 52 + ARMS.length * rowH + 22, 13.5, col.muted);
    ctx.globalAlpha = 1;

    // right-hand notes
    ctx.globalAlpha = A(0.7, 1);
    box(ctx, 880, 160, 300, 304, { fill: col.osoft, stroke: col.ours });
    text(ctx, "What is compared", 898, 190, 18, col.ours, { weight: 700 });
    ["Only synthetic data in training; the real series are never fitted.",
      "Without a head: unmodified Chronos-2. With a head: plus a channel index.",
      "Principal: ours vs LMC, and head vs no head, in Study B, per dataset.",
      "2019 was inspected during development: exploratory, not confirmatory."].forEach((s, i) => wrap(ctx, s, 898, 222 + i * 60, 266, 19, 14.5, col.ink));
    ctx.globalAlpha = 1;
  }
  Z.mount({
    id: "evalplan", W, H, foot: "Design: report/english/chapters/experimental_design/foundation_prior_evaluation.tex. LMC rows: final checkpoints of the completed Study A and Study B runs, scripts/evaluate_lmc_checkpoints.py; 255 origins at Ibirapuera, 357 in Bahia.",
    steps: [{ narr: "Our generator against LMC-Synth", sub: "same model, same budgets, zero-shot on two real systems", dur: 7000 }],
    paint(ctx, step, p) { compare(ctx, p); },
  });
})();
