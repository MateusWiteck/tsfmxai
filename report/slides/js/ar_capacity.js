/* Slide "What lagged recurrences can already do" (placed before the generator is explained).
 *
 * Evidence: experiments/autoregressive_process_coverage (KernelSynth bank + 20 seeded compositions,
 * TimePFN LMC latent series). Every curve and number comes from data/ar_coverage_demo.js, written
 * by scripts/export_ar_coverage_demo.py, which re-runs the experiment's own code. The metric is the
 * experiment's: one-step normalised MSE (NMSE) on held-out trajectories. Here each function is given
 * the SMALLEST grid order that reaches NMSE <= 0.001; the notebooks report the best-of-grid order.
 * The last step's trajectory is the analytic example of the vocabulary-coverage notebook.
 */
(function () {
  const D = window.AR_DATA, Z = window.ZC;
  if (!D || !Z) return;
  const { col, seg, ease, text, wrap, box } = Z;
  const MINUS = "−";
  const W = 1200, H = 500;
  const KS = D.kernelsynth, TP = D.timepfn;

  const NAMES = {
    periodic_01_p24: "periodic, period 24 steps",
    "rbf_scale0.1": "smooth RBF kernel, short length scale",
    composite_01: "composite kernel (sum and product)",
    composite_05: "composite kernel (sum and product)",
    composite_10: "composite kernel (sum and product)",
    white_level1: "white noise",
  };
  const sci = (v) => {
    const [m, e] = v.toExponential(1).split("e");
    return `${m}e${e.replace("+", "")}`;
  };
  const sgn = (v) => (v < 0 ? ` ${MINUS} ` : " + ") + Math.abs(v).toFixed(2);

  function equation(ex) {
    if (!ex.covered) return "irreducible: nothing in the past predicts it";
    if (ex.order <= 2) {
      const c0 = Math.abs(ex.intercept) < 0.005 ? "" : ex.intercept.toFixed(2);
      const terms = ex.coefficients.map((c, i) => `${sgn(c)}·x(t${MINUS}${i + 1})`).join("");
      return `x(t) = ${c0 ? c0 + terms : terms.replace(/^ [+] /, "").replace(/^ − /, MINUS)}`;
    }
    return `x(t) = c + Σ φₖ·x(t${MINUS}k),  k = 1 … ${ex.order}`;
  }

  function examplePanel(ctx, x, y, w, h, ex, p, k) {
    const bad = !ex.covered, c = bad ? col.ours : col.teal;
    box(ctx, x, y, w, h, { stroke: bad ? col.ours : col.line, lw: bad ? 2.2 : 1.6 });
    text(ctx, NAMES[ex.id] || ex.id, x + 14, y + 26, 15.5, col.ink, { weight: 700 });
    const tag = bad ? "no useful lag" : `${ex.order} lag${ex.order > 1 ? "s" : ""}`;
    text(ctx, tag, x + w - 14, y + 26, 15.5, c, { align: "right", weight: 700 });
    const px = x + 14, py = y + 40, pw = w - 28, ph = 92;
    const a = ex.actual, lo = Math.min(...a), hi = Math.max(...a), pad = (hi - lo) * 0.08 || 1;
    const X = (i) => px + (i / (a.length - 1)) * pw, Y = (v) => py + ph - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * ph;
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, py + ph);
    ctx.lineTo(px + pw, py + ph);
    ctx.stroke();
    const n1 = Math.max(2, Math.floor(a.length * seg(p, k * 0.06, k * 0.06 + 0.45)));
    const n2 = Math.max(0, Math.floor(a.length * seg(p, k * 0.06 + 0.3, k * 0.06 + 0.7)));
    const draw = (arr, n, color, lw, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), Y(arr[i])) : ctx.moveTo(X(i), Y(arr[i])));
      ctx.stroke();
      ctx.setLineDash([]);
    };
    draw(a, n1, col.blue, 2.6);
    draw(ex.prediction, n2, col.ours, 1.6, [4, 3]);
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, equation(ex), x + 14, y + h - 14, 13.5, col.muted, { family: Z.MATH });
    text(ctx, `NMSE ${sci(ex.nmse)}`, x + w - 14, y + h - 14, 13.5, c, { align: "right", weight: 700 });
    ctx.globalAlpha = 1;
  }

  function scene0(ctx, p) {
    const pw = 376, ph = 188, gx = 24;
    KS.examples.forEach((ex, i) => {
      const x = 20 + (i % 3) * (pw + gx), y = 60 + Math.floor(i / 3) * (ph + 12);
      examplePanel(ctx, x, y, pw, ph, ex, p, i);
    });
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    ctx.fillStyle = col.blue;
    ctx.fillRect(20, 462, 22, 5);
    text(ctx, "KernelSynth sample", 50, 468, 15, col.ink);
    ctx.strokeStyle = col.ours;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(210, 465);
    ctx.lineTo(232, 465);
    ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, "one-step prediction from the previous lags only", 240, 468, 15, col.ink);
    text(ctx, "Held-out trajectories. NMSE: squared error divided by the variance of the target.", 20, 490, 14, col.muted);
    ctx.globalAlpha = 1;
  }

  function chart(ctx, x, title, sub, d, p, delay) {
    box(ctx, x, 60, 570, 330);
    text(ctx, title, x + 20, 94, 20, col.ink, { weight: 700 });
    text(ctx, sub, x + 20, 116, 14, col.muted);
    const orders = d.cumulative, n = orders.length, bx = x + 30, bw = 510, base = 340, hmax = 170, slot = bw / n;
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, base);
    ctx.lineTo(bx + bw, base);
    ctx.stroke();
    // our generator allows lags up to 16: the line closes the bar of grid order 16
    const i16 = orders.findIndex((o) => o[0] === 16), lx = bx + (i16 + 1) * slot;
    ctx.strokeStyle = col.ours;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(lx, 128);
    ctx.lineTo(lx, base + 6);
    ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, "generator: up to 16 lags", lx - 8, 142, 13, col.ours, { align: "right", weight: 700 });
    orders.forEach(([o, c], i) => {
      const g = ease(seg(p, delay + (i * 0.5) / n, delay + (i * 0.5) / n + 0.3)), h = (c / d.total) * hmax * g;
      ctx.fillStyle = o <= 16 ? col.teal : "#a9b2c6";
      ctx.fillRect(bx + i * slot + 8, base - h, slot - 16, h);
      ctx.globalAlpha = g;
      text(ctx, String(c), bx + i * slot + slot / 2, base - h - 7, 14, col.ink, { align: "center", weight: 700 });
      ctx.globalAlpha = 1;
      text(ctx, String(o), bx + i * slot + slot / 2, base + 22, 13.5, col.muted, { align: "center" });
    });
    text(ctx, "number of lags allowed", bx + bw / 2, base + 44, 13.5, col.muted, { align: "center" });
  }

  function scene1(ctx, p) {
    chart(ctx, 20, `KernelSynth: ${KS.total} functions`, "official bank plus 20 seeded compositions; orders on the experiment's grid", KS, p, 0);
    chart(ctx, 610, `TimePFN: ${TP.total} latent series`, "11 mechanisms with 3 latent series each", TP, p, 0.15);
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    const at = (d, k) => d.cumulative.find((o) => o[0] === k)[1];
    const at16 = at(KS, 16), at256 = KS.cumulative[KS.cumulative.length - 1][1];
    text(ctx, `Cap of 16 lags: ${at16} of ${KS.total} KernelSynth functions (${Math.round((100 * at16) / KS.total)}%) and ${at(TP, 16)} of ${TP.total} TimePFN series; up to 256 lags adds only ${at256 - at16}.`, 20, 412, 17, col.teal, { weight: 700 });
    text(ctx, `Why 16 and not 12: ${at(KS, 12)} → ${at16} KernelSynth functions and ${at(TP, 12)} → ${at(TP, 16)} TimePFN series reach NMSE ≤ 0.001.`, 20, 438, 18, col.ink, { weight: 700 });
    // diagnosis of the 8 KernelSynth functions that no order reproduces (experiment AR/01, ids white_level0.1, white_level1,
    // composite_11 and _16 hold a WhiteKernel term; composite_08, _15, _19, _20 are products of periodic kernels)
    text(ctx, `Never reproduced (${KS.never}): 2 pure white noise + 2 composites with a white-noise term, unpredictable by construction; ${KS.never - 4} stacked periodic kernels. TimePFN: ${TP.never} series.`, 20, 460, 14.5, col.ink, { weight: 600 });
    text(ctx, `Order chosen on validation, scored on test. Original protocol (order picked on test): ${KS.best_of_grid} of ${KS.total} and ${TP.best_of_grid} of ${TP.total}. Exploratory: one step ahead, finite suites.`, 20, 480, 14.5, col.muted);
    ctx.globalAlpha = 1;
  }

  let tanhRun = null;
  function tanhPath() {
    if (tanhRun) return tanhRun;
    let s = 17;
    const r = () => {
      s |= 0; s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const x = [0];
    for (let t = 1; t < 200; t++) x.push(0.5 * Math.tanh(x[t - 1]) + (r() - 0.5));
    tanhRun = x;
    return x;
  }

  function scene2(ctx, p) {
    box(ctx, 20, 60, 560, 380);
    text(ctx, "From a sum of lags to the generator's words", 40, 96, 20, col.ink, { weight: 700 });
    const rows = [
      ["tested on the previous slide", "x(t) = c + Σ φₖ·x(t−k)", "a weighted sum of past values", col.blue],
      ["+ products", "x(t−1)·x(t−3)", "two past values multiplied", col.teal],
      ["+ squashing", "tanh x(t−2)", "keeps a factor in (−1, 1)", col.teal],
    ];
    rows.forEach((r, i) => {
      const a = ease(seg(p, i * 0.18, i * 0.18 + 0.3)), y = 120 + i * 82;
      ctx.globalAlpha = a;
      box(ctx, 40, y, 520, 70, { fill: i ? col.tsoft : col.soft, stroke: r[3], lw: 2.2 });
      text(ctx, r[0], 56, y + 26, 14, r[3], { weight: 700 });
      text(ctx, r[1], 56, y + 54, 22, col.ink, { family: Z.MATH });
      text(ctx, r[2], 540, y + 54, 14.5, col.muted, { align: "right" });
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.6, 1));
    wrap(ctx, "Terms are combined with + and ×: up to 6 operators, lags 1 to 16. The experiment above tested only the first row.", 40, 388, 520, 22, 16, col.ink);
    ctx.globalAlpha = 1;

    box(ctx, 600, 60, 580, 380);
    text(ctx, "Same words, a process outside the Gaussian family", 620, 96, 20, col.ink, { weight: 700 });
    text(ctx, "x(t) = 0.5·tanh(x(t−1)) + ε,   ε ∼ Uniform(−0.5, 0.5)", 620, 128, 18, col.ink, { family: Z.MATH });
    const xs = tanhPath(), bx = 620, bw = 540, top = 150, h = 170, Y = (v) => top + h / 2 - (v / 1.15) * (h / 2), X = (i) => bx + (i / 199) * bw;
    [-1, 1].forEach((b) => {
      ctx.strokeStyle = col.ours;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(bx, Y(b));
      ctx.lineTo(bx + bw, Y(b));
      ctx.stroke();
      ctx.setLineDash([]);
    });
    text(ctx, "+1", bx + bw + 2, Y(1) + 5, 12.5, col.ours, { weight: 700 });
    text(ctx, MINUS + "1", bx + bw + 2, Y(-1) + 5, 12.5, col.ours, { weight: 700 });
    const n = Math.max(2, Math.floor(200 * seg(p, 0.1, 0.8)));
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), Y(xs[i])) : ctx.moveTo(X(i), Y(xs[i])));
    ctx.stroke();
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    wrap(ctx, "|tanh| < 1 and the noise is bounded, so every value stays inside (−1, 1). A Gaussian process has unbounded range: this law is outside what KernelSynth draws.", 620, 352, 540, 22, 15.5, col.ink);
    wrap(ctx, "On Gaussian-process data the best one-step predictor is linear, so products and tanh add nothing there. They matter for processes like this one.", 620, 412, 540, 18, 13.5, col.muted);
    ctx.globalAlpha = 1;
  }

  const STEPS = [
    { narr: "Lagged recurrences reproduce complex series", sub: "each from a few previous values", dur: 5200,
      foot: "experiments/autoregressive_process_coverage: KernelSynth bank and seeded compositions, held-out trajectories of length 1,024. Each panel uses the order selected on validation (smallest that reaches NMSE ≤ 0.001); curves are test trajectory 0 of that function." },
    { narr: "How many lags are enough?", sub: "share of functions reproduced as the number of lags grows", dur: 5200,
      foot: "Counts per grid order: functions whose validation-selected order is at most k and whose pooled one-step NMSE on 16 test trajectories is ≤ 0.001 (KernelSynth, orders up to 256; TimePFN oracle latent series, orders up to 128). Corrected protocol of experiments/autoregressive_process_coverage/AR/01 and 02." },
    { narr: "Our generator writes richer recurrences", sub: "the same lags, plus products and tanh", dur: 6500,
      foot: "Left: grammar of the project's generator (src/tsfmxai/generators). Right: analytic example of experiments/autoregressive_process_coverage/vocabulary_coverage, simulated with a fixed seed." },
  ];

  Z.mount({
    id: "arcap", W, H, steps: STEPS,
    foot: "",
    paint(ctx, step, p) {
      [scene0, scene1, scene2][step](ctx, p);
    },
  });
})();
