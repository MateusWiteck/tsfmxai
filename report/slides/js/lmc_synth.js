/* Final section of the deck: LMC-Synth (TimePFN's multivariate generator) and the planned comparison.
 *
 * Slide "lmc": how the generator works, step by step, with a real example. The construction follows
 * synthetic_data_generation/LMC_Synth.py of github.com/egetaga/TimePFN (read on 9 Nov 2026) and
 * Taga, Ildiz and Oymak (2025). Curves and matrices come from data/lmc_demo.js, written by
 * scripts/export_lmc_demo.py (illustrative sizes: 5 channels, 4 latents; official: 160 channels).
 * Corpus and training numbers, and the ablation bars, are transcribed from the paper.
 *
 * Slide "evalplan": the planned comparative evaluation. NOTHING in it has been run: it is a design,
 * and every cell that would hold a result is drawn as a hypothesis.
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

  // ---- step 4: what comes out: the correlation between channels ----
  function correlation(ctx, p) {
    [["low", 20], ["high", 300]].forEach(([k, x0], i) => {
      const a = ease(seg(p, i * 0.2, i * 0.2 + 0.4));
      ctx.globalAlpha = a;
      box(ctx, x0, 62, 262, 330);
      text(ctx, `d = ${D.examples[k].d}`, x0 + 18, 94, 19, k === "low" ? col.ours : col.teal, { weight: 700 });
      CORR[k].forEach((r, ri) => r.forEach((v, cj) => {
        const cellA = ease(seg(p, 0.1 + i * 0.2 + (ri + cj) * 0.025, 0.35 + i * 0.2 + (ri + cj) * 0.025));
        ctx.globalAlpha = a * cellA;
        const X = x0 + 18 + cj * 42, Y = 116 + ri * 42;
        ctx.fillStyle = heat(Math.abs(v));
        ctx.fillRect(X, Y, 40, 40);
        text(ctx, v.toFixed(2), X + 20, Y + 25, 13.5, Math.abs(v) > 0.55 ? "#fff" : col.ink, { align: "center", weight: 600 });
      }));
      ctx.globalAlpha = a;
      text(ctx, "correlation of the channels", x0 + 18, 376, 13.5, col.muted);
      text(ctx, `mean |corr| = ${D.examples[k].mean_abs_corr.toFixed(2)}`, x0 + 18, 352, 15, col.ink, { weight: 700 });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.4, 1));
    box(ctx, 580, 62, 600, 330);
    text(ctx, "Concentration against channel correlation", 600, 94, 19, col.ink, { weight: 700 });
    const bx = 640, by = 120, bw = 500, bh = 170, sw = D.sweep, lx = (d) => bx + ((Math.log(d) - Math.log(0.05)) / (Math.log(10) - Math.log(0.05))) * bw, Y = (v) => by + bh - v * bh;
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(bx, by + bh);
    ctx.lineTo(bx + bw, by + bh);
    ctx.moveTo(bx, by);
    ctx.lineTo(bx, by + bh);
    ctx.stroke();
    const q = seg(p, 0.45, 1) * (sw.length - 1), n0 = Math.floor(q), fr = q - n0;
    const pts = sw.slice(0, n0 + 1).map((s) => [lx(s.d), Y(s.mean_abs_corr)]);
    if (n0 < sw.length - 1) pts.push([lx(sw[n0].d) + (lx(sw[n0 + 1].d) - lx(sw[n0].d)) * fr, Y(sw[n0].mean_abs_corr) + (Y(sw[n0 + 1].mean_abs_corr) - Y(sw[n0].mean_abs_corr)) * fr]);
    ctx.strokeStyle = col.blue;
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    pts.forEach((pt, i) => (i ? ctx.lineTo(pt[0], pt[1]) : ctx.moveTo(pt[0], pt[1])));
    ctx.stroke();
    sw.forEach((s, i) => {
      text(ctx, String(s.d), lx(s.d), by + bh + 20, 12.5, col.muted, { align: "center" });
      if (i > q) return;
      ctx.beginPath();
      ctx.arc(lx(s.d), Y(s.mean_abs_corr), 4, 0, 7);
      ctx.fillStyle = col.blue;
      ctx.fill();
    });
    // a moving marker at the current d
    const last = pts[pts.length - 1];
    if (seg(p, 0.45, 1) < 1) {
      ctx.beginPath();
      ctx.arc(last[0], last[1], 8, 0, 7);
      ctx.strokeStyle = col.ours;
      ctx.lineWidth = 2.4;
      ctx.stroke();
    }
    [0, 0.5, 1].forEach((v) => text(ctx, v.toFixed(1), bx - 10, Y(v) + 5, 12.5, col.muted, { align: "right" }));
    text(ctx, "d (log scale)", bx + bw / 2, by + bh + 40, 13.5, col.muted, { align: "center" });
    text(ctx, "mean |corr|", bx + 6, by - 8, 13.5, col.muted);
    ctx.globalAlpha = ease(seg(p, 0.8, 1));
    wrap(ctx, "About a quarter of the corpus uses as many latents as channels with identity weights: independent channels (paper).", 600, 366, 560, 18, 14, col.muted);
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
    ["Dependence between channels", "shared latent functions", true, "any series, any lag", true],
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
    { narr: "④ What comes out: correlated channels", sub: "the concentration sets how correlated", dur: 6000 },
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
  LSTEPS[4].foot = LFOOT + " Right: pool of 40 raw latent draws, 100 random subsets of 8 latents and 16 channels per d; independent smooth series already show |corr| around 0.2 in one realisation.";
  LSTEPS[5].foot = "Taga, Ildiz and Oymak (2025): corpus size, windows, model and training as reported in the paper; numbers transcribed from the text.";
  LSTEPS[6].foot = "Zero-shot MSE of TimePFN and of TimePFN trained on independent channels only, as reported in the paper (transcribed; check against the paper's table before citing).";
  LSTEPS[7].foot = "Left column: LMC_Synth.py. Right column: src/tsfmxai/generators (AST over Lag(variable, lag), Multiply, Tanh) and the recorded seeds of the experiments. The paper also mentions multiplicative Gaussian noise (σ = 0.1); the script has no noise term.";

  // ================= planned comparative evaluation (design only) =================
  function chip(ctx, x, y, w, label, color) {
    Z.rrect(ctx, x, y, w, 28, 14);
    ctx.fillStyle = color;
    ctx.fill();
    text(ctx, label, x + w / 2, y + 20, 14.5, "#fff", { align: "center", weight: 700 });
  }
  function planWhat(ctx, p) {
    chip(ctx, 20, 62, 300, "PLANNED: no model has been trained yet", col.ours);
    const cols = [
      ["Generators", ["LMC-Synth (official code)", "This project's generator", "A 50/50 mixture of both"], col.blue],
      ["Models", ["Chronos-2 architecture", "Moirai-Small", "same size, same budget, same seeds"], col.teal],
      ["Benchmarks", ["the 9 datasets of TimePFN", "ETT×4, Weather, Solar, ECL,", "Exchange, Traffic; zero-shot"], "#6d5bd0"],
    ];
    cols.forEach((c, i) => {
      const a = ease(seg(p, i * 0.15, i * 0.15 + 0.3)), x = 20 + i * 392;
      ctx.globalAlpha = a;
      box(ctx, x, 110, 372, 168, { stroke: c[2], lw: 2.4 });
      text(ctx, c[0], x + 18, 142, 20, c[2], { weight: 700 });
      c[1].forEach((s, k) => text(ctx, s, x + 18, 180 + k * 32, 17, col.ink));
      ctx.globalAlpha = 1;
      if (i < 2) {
        ctx.globalAlpha = ease(seg(p, 0.25 + i * 0.15, 0.5 + i * 0.15));
        text(ctx, i ? "→" : "×", x + 372 + 10, 200, 26, col.muted, { align: "center", weight: 700 });
        ctx.globalAlpha = 1;
      }
    });
    ctx.globalAlpha = ease(seg(p, 0.5, 1));
    text(ctx, "What has to exist first", 20, 312, 19, col.ink, { weight: 700 });
    const runs = Math.round(18 * ease(seg(p, 0.5, 0.9)));
    box(ctx, 900, 316, 280, 108, { fill: col.osoft, stroke: col.ours, dash: [6, 5] });
    text(ctx, String(runs), 920, 372, 44, col.ours, { weight: 700 });
    text(ctx, "training runs", 990, 360, 17, col.ink, { weight: 700 });
    text(ctx, "3 corpora × 2 models × 3 seeds", 920, 402, 14.5, col.muted);
    [["Chronos-2 pretraining code is not public", "only fine-tuning code and weights are; here the architecture would be trained with our own pipeline"],
      ["A corpus-scale multivariate corpus from our generator", "multivariate stability certificates: Shao–Wu and Heaps VAR yes; the Jacobian one is scalar only (repository notes)"],
      ["Mix, Don't Pick models", "that paper trains Chronos-T5-Mini and Moirai-Small; Chronos-2 is a different architecture"]].forEach((r, i) => {
      text(ctx, r[0], 20, 346 + i * 44, 16.5, col.ours, { weight: 700 });
      text(ctx, r[1], 20, 366 + i * 44, 14.5, col.muted);
    });
    ctx.globalAlpha = 1;
  }
  function planProtocol(ctx, p) {
    const steps = [
      ["1  Match the corpora", "same number of windows, same length, same channel counts for every generator"],
      ["2  Train", "each model on each corpus: same steps, same optimiser, 3 seeds"],
      ["3  Evaluate zero-shot", "on the real multivariate benchmarks, never trained on"],
      ["4  Aggregate", "geometric mean of scores normalised by a seasonal-naive baseline"],
      ["5  Test", "paired bootstrap over tasks, confidence intervals, seeds reported"],
    ];
    steps.forEach((s, i) => {
      const a = ease(seg(p, i * 0.14, i * 0.14 + 0.3)), y = 72 + i * 76;
      ctx.globalAlpha = a;
      box(ctx, 20, y, 760, 64, { stroke: col.blue, lw: 2 });
      text(ctx, s[0], 38, y + 28, 19, col.blue, { weight: 700 });
      text(ctx, s[1], 38, y + 52, 15.5, col.ink);
      if (i < 4) arrow(ctx, 400, y + 66, 400, y + 74, col.muted, 2);
      ctx.globalAlpha = 1;
    });
    const tk = seg(p, 0.1, 0.9);
    if (tk > 0 && tk < 1) {
      const ty = 72 + 32 + tk * 4 * 76;
      ctx.beginPath();
      ctx.arc(796, ty, 8, 0, 7);
      ctx.fillStyle = col.ours;
      ctx.fill();
    }
    ctx.globalAlpha = ease(seg(p, 0.6, 1));
    box(ctx, 810, 72, 370, 364, { fill: col.osoft, stroke: col.ours });
    text(ctx, "Lessons taken from", 830, 106, 18, col.ours, { weight: 700 });
    text(ctx, "Mix, Don't Pick", 830, 130, 18, col.ours, { weight: 700 });
    ["Match the budget exactly across corpora.", "Include a mixture: single generators can be brittle.", "Report intervals, not single numbers.", "Univariate there: here the corpora are multivariate."].forEach((s, i) => wrap(ctx, s, 830, 166 + i * 58, 330, 21, 16, col.ink));
    ctx.globalAlpha = 1;
  }
  function planMetrics(ctx, p) {
    const groups = [
      ["Point error", ["MSE and MAE on standardised variables (TimePFN, DynLMC)", "MASE, scaled by the seasonal naive"], col.blue],
      ["Probabilistic", ["CRPS scaled by the seasonal naive (GIFT-Eval)", "scaled quantile loss, WQL (fev-bench, Chronos-2)"], col.teal],
      ["Dependence between series", ["energy score and variogram score", "CRPS-sum; error of the correlation matrix"], "#6d5bd0"],
      ["Across benchmarks", ["win rate, skill score, geometric mean", "paired bootstrap, several seeds"], col.ours],
    ];
    groups.forEach((g, i) => {
      const a = ease(seg(p, i * 0.14, i * 0.14 + 0.3)), x = 20 + (i % 2) * 590, y = 66 + Math.floor(i / 2) * 178;
      ctx.globalAlpha = a;
      box(ctx, x, y, 570, 164, { stroke: g[2], lw: 2.4 });
      text(ctx, g[0], x + 20, y + 36, 20, g[2], { weight: 700 });
      g[1].forEach((s, k) => wrap(ctx, "• " + s, x + 20, y + 76 + k * 44, 530, 21, 16.5, col.ink));
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "Energy and variogram scores need joint samples of all series; point metrics do not.", 20, 440, 16.5, col.ink, { weight: 600 });
    text(ctx, "Definitions to be confirmed against each original source before they go into the thesis.", 20, 466, 14.5, col.muted);
    ctx.globalAlpha = 1;
  }
  function planInteraction(ctx, p) {
    text(ctx, "The test where the two generators should differ: what is learned from each corpus, scored on tasks with a known dependency", 20, 66, 17.5, col.ink, { weight: 700 });
    const tasks = ["instantaneous, linear", "lagged between series", "product between series", "regime change"];
    const gx = 330, cw = 210;
    ["LMC-Synth", "This project"].forEach((g, i) => text(ctx, g, gx + i * cw + cw / 2, 108, 18, i ? col.ours : col.blue, { align: "center", weight: 700 }));
    const hyp = [["expected good", "expected good"], ["expected weak", "expected good"], ["expected weak", "expected good"], ["expected weak", "expected good"]];
    tasks.forEach((t, r) => {
      const a = ease(seg(p, r * 0.12, r * 0.12 + 0.3)), y = 120 + r * 66;
      ctx.globalAlpha = a;
      text(ctx, t, 20, y + 38, 17.5, col.ink, { weight: 700 });
      hyp[r].forEach((h, i) => {
        const good = h === "expected good";
        box(ctx, gx + i * cw + 6, y + 6, cw - 12, 52, { r: 10, fill: good ? col.tsoft : col.grey, stroke: good ? col.teal : col.muted, dash: [6, 5], lw: 1.8 });
        text(ctx, h, gx + i * cw + cw / 2, y + 38, 16, good ? col.teal : col.muted, { align: "center", weight: 700 });
      });
      ctx.globalAlpha = 1;
    });
    box(ctx, 780, 108, 400, 276, { fill: col.osoft, stroke: col.ours });
    ctx.globalAlpha = ease(seg(p, 0.5, 1));
    text(ctx, "Hypothesis, not a result", 800, 142, 19, col.ours, { weight: 700 });
    [["The first row is a control: the LMC-Synth mixture is built for it."],
      ["The other rows need dependencies LMC-Synth cannot write: lags and products between series."],
      ["If this project's generator is not better there, the hypothesis fails."]].forEach((s, i) => wrap(ctx, s[0], 800, 178 + i * 66, 360, 21, 16, col.ink));
    text(ctx, "Cells are dashed: nothing here has been measured.", 20, 410, 15, col.muted);
    wrap(ctx, "The tasks have a known generating equation, so an explanation of the forecast can also be scored against the true dependency.", 20, 436, 740, 20, 15.5, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  const ESTEPS = [
    { narr: "Comparative evaluation: what is compared", sub: "a plan, not results", dur: 5200,
      foot: "Design only. Models and benchmarks follow Taga et al. (2025) and Nagpal et al. (2026, Mix, Don't Pick). Chronos-2: github.com/amazon-science/chronos-forecasting (inference and fine-tuning code, no pretraining pipeline)." },
    { narr: "Protocol", sub: "matched budget, zero-shot, with intervals", dur: 5200,
      foot: "Protocol modelled on Nagpal et al. (2026): matched corpora and budgets, zero-shot evaluation, paired bootstrap over tasks, several seeds." },
    { narr: "Metrics", sub: "all established, none new", dur: 5200,
      foot: "Sources of use: TimePFN and DynLMC (MSE, MAE), GIFT-Eval and Mix, Don't Pick (scaled CRPS and MASE), fev-bench and Chronos-2 (scaled quantile loss, win rate, skill score). Multivariate scores: energy and variogram score, CRPS-sum." },
    { narr: "The test that matters: dependencies LMC-Synth cannot write", sub: "hypotheses, drawn dashed", dur: 6000,
      foot: "Hypothesis table: no measurement behind any cell. The tasks need a multivariate sampler of this project's generator with recorded equations." },
  ];
  Z.mount({
    id: "evalplan", W, H, steps: ESTEPS, foot: "",
    paint(ctx, step, p) {
      [planWhat, planProtocol, planMetrics, planInteraction][step](ctx, p);
    },
  });
})();
