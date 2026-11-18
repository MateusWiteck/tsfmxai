/* "Cosine start" button on the stability-method overlay (tab next to Shao–Wu, Jacobian,
 * Heaps VAR). It feeds the same starting history, one full cosine cycle, to the four
 * constructions and shows what each one does with it.
 *
 * Every curve is real output exported by scripts/export_cosine_start.py from the
 * condition "Cosseno completo; A=0.5" of
 * experiments/time_series_initialization/2_initial_magnitude.ipynb (same bank, seeds and
 * paired noise). Open directly with ?gen=cos&gstep=0 for screenshots.
 */
(function () {
  const G = window.GenDetail, Z = window.ZC, D = window.COSINE_START;
  if (!G || !Z || !D) return;
  const { col, seg, ease, text, wrap, box } = Z;
  const M = D.methods, STEPS = D.steps;
  const RUN = [col.blue, col.ours, col.teal];
  const LABEL = { "Original AST": "no protection", Jacobian: "Jacobian", "Shao-Wu": "Shao–Wu", Heaps: "Heaps VAR" };
  const ORDER = ["Original AST", "Jacobian", "Shao-Wu", "Heaps"];

  // symmetric log axis, as in the notebook: linear near 0, logarithmic for large values
  const SL = 0.1, sl = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v) / SL);

  /** One time-series panel: dashed history (t <= 0) then the runs (t >= 1), drawn up to p. */
  function series(ctx, b, hist, runs, p, o) {
    o = o || {};
    box(ctx, b.x, b.y, b.w, b.h, { fill: o.fill || col.panel, stroke: o.stroke || col.line });
    const pad = { l: 12, r: 12, t: o.title || o.note ? 30 : 12, b: 12 };
    const H0 = hist.length, N = STEPS, total = H0 + N;
    const all = hist.concat(...runs).map(sl);
    let lo = Math.min(...all), hi = Math.max(...all);
    const m = (hi - lo) * 0.08 || 0.1; lo -= m; hi += m;
    const X = (i) => b.x + pad.l + (i / (total - 1)) * (b.w - pad.l - pad.r);
    const Y = (v) => b.y + pad.t + (1 - (sl(v) - lo) / (hi - lo)) * (b.h - pad.t - pad.b);
    ctx.save();
    ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    ctx.strokeStyle = col.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(b.x + pad.l, Y(0)); ctx.lineTo(b.x + b.w - pad.r, Y(0)); ctx.stroke();
    ctx.fillStyle = "rgba(91,100,117,0.07)";
    ctx.fillRect(X(H0 - o.order), b.y + pad.t, X(H0 - 1) - X(H0 - o.order), b.h - pad.t - pad.b);
    ctx.setLineDash([3, 3]); ctx.strokeStyle = col.muted; ctx.lineWidth = 1.6;
    ctx.beginPath(); hist.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = col.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(H0 - 0.5), b.y + pad.t); ctx.lineTo(X(H0 - 0.5), b.y + b.h - pad.b); ctx.stroke();
    const n = Math.max(1, Math.floor(N * p));
    runs.forEach((r, k) => {
      ctx.strokeStyle = o.colors ? o.colors[k] : RUN[k % RUN.length];
      ctx.lineWidth = o.lw || 1.3;
      ctx.globalAlpha = o.alpha || 0.85;
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const x = X(H0 + i), y = Y(r[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
    ctx.restore();
    if (o.title) text(ctx, o.title, b.x + 12, b.y + 21, o.titleSize || 15, o.titleColor || col.ink, { weight: 700 });
    if (o.note) text(ctx, o.note, b.x + b.w - 12, b.y + 21, 12.5, col.muted, { align: "right" });
  }

  const ex = (method, i) => M[method].examples[i];
  const peak = (method, i) => Math.max(...ex(method, i).runs.flat().map(Math.abs));
  const fmtPeak = (v) => (v >= 1000 ? `|x| reaches ${v.toExponential(0).replace("e+", "×10^")}` : `|x| ≤ ${v.toFixed(2)}`);

  // step 0: one mechanism per method, same cosine start
  function sameStart(ctx, p) {
    const w = 292;
    ORDER.forEach((g, j) => {
      const e = ex(g, 1), a = seg(p, 0.05 + j * 0.06, 0.75 + j * 0.06);
      series(ctx, { x: 6 + j * (w + 6), y: 8, w, h: 300 }, e.history, e.runs, a, {
        order: e.order, title: LABEL[g], note: `memory ${e.order}`,
        stroke: g === "Heaps" ? col.ours : col.line, fill: g === "Heaps" ? "#fffaf6" : col.panel,
        titleColor: g === "Heaps" ? col.ours : col.ink,
      });
    });
    ctx.globalAlpha = ease(seg(p, 0.7, 1));
    text(ctx, "dashed: the starting history, one cosine cycle (period 96, amplitude 0.5)  ·  shaded: the lags the equation reads  ·  colours: 3 runs, different noise",
      10, 334, 14, col.muted);
    const peaks = ORDER.map((g) => `${LABEL[g]}: ${fmtPeak(peak(g, 1))}`).join("   ·   ");
    text(ctx, peaks, 10, 362, 14, col.ink, { weight: 600 });
    text(ctx, "vertical axis: symmetric log (linear near 0), so an explosion and a wave of size 0.5 fit in one picture", 10, 388, 13, col.muted);
    ctx.globalAlpha = 1;
  }

  // step 1: the wave belongs to the Heaps mechanism, not to the start
  function resonance(ctx, p) {
    const e = ex("Heaps", 1), root = M.Heaps.roots[1][0];
    series(ctx, { x: 6, y: 8, w: 640, h: 182 }, e.history, [e.runs[0]], seg(p, 0, 0.6), {
      order: e.order, title: "Cosine start", note: "Heaps mechanism 1, run 1", lw: 1.8, colors: [col.ours] });
    const zeros = new Array(e.history.length).fill(0);
    series(ctx, { x: 6, y: 202, w: 640, h: 182 }, zeros, [M.Heaps.zero_start], seg(p, 0.3, 0.9), {
      order: e.order, title: "Zero start, same noise", note: "same mechanism, same noise", lw: 1.8, colors: [col.blue] });
    ctx.globalAlpha = ease(seg(p, 0.45, 0.9));
    box(ctx, 662, 8, 530, 376);
    text(ctx, "An almost undamped slow mode", 684, 44, 21, col.ink, { weight: 700 });
    const half = Math.round(Math.log(0.5) / Math.log(root.modulus));
    let y = wrap(ctx, `This Heaps VAR has a pair of roots of modulus ${root.modulus} with period ${root.period} steps: just inside the unit circle and close to the period 96 of the start.`, 684, 76, 486, 23, 16, col.ink);
    y = wrap(ctx, `The cosine start puts its energy into that mode, which loses only ${((1 - root.modulus) * 100).toFixed(2)} % per step: half its size after about ${half} steps, far beyond the 384 shown (top).`, 684, y + 10, 486, 23, 16, col.ink);
    y = wrap(ctx, "From zero with the same noise (bottom) no slow wave appears in 384 steps: the noise feeds that mode far too slowly, and the fast roots dominate.", 684, y + 10, 486, 23, 16, col.ink);
    text(ctx, "So the wave is the start, remembered.", 684, y + 14, 17, col.ours, { weight: 700 });
    ctx.globalAlpha = 1;
  }

  // step 2: other Heaps mechanisms do not keep the wave
  function otherHeaps(ctx, p) {
    const w = 392;
    [0, 1, 2].forEach((i) => {
      const e = ex("Heaps", i), roots = M.Heaps.roots[i], a = seg(p, i * 0.12, 0.6 + i * 0.12);
      const slow = roots.filter((r) => r.period !== null).sort((u, v) => v.period - u.period)[0];
      const hit = slow && Math.abs(slow.period - D.period) < 30;
      series(ctx, { x: 6 + i * (w + 6), y: 8, w, h: 250 }, e.history, e.runs, a, {
        order: e.order, title: `Heaps mechanism ${i}`, note: `memory ${e.order}`,
        stroke: hit ? col.ours : col.line, titleColor: hit ? col.ours : col.ink });
      ctx.globalAlpha = ease(seg(p, 0.4 + i * 0.1, 0.8 + i * 0.1));
      text(ctx, "slowest roots near the unit circle:", 18 + i * (w + 6), 286, 14, col.muted, { weight: 600 });
      roots.forEach((r, k) => text(ctx, `|λ| = ${r.modulus.toFixed(4)},  period ${r.period === null ? "∞ (real root)" : r.period}`,
        18 + i * (w + 6), 310 + k * 22, 14.5, hit && r === slow ? col.ours : col.ink, { weight: hit && r === slow ? 700 : 400 }));
      ctx.globalAlpha = 1;
    });
    ctx.globalAlpha = ease(seg(p, 0.75, 1));
    text(ctx, "Only a mechanism with a slow root near period 96 picks the cosine up; the others turn it into their own fast oscillations.", 10, 392, 15, col.ink, { weight: 600 });
    ctx.globalAlpha = 1;
  }

  // step 3: the full 3 x 4 grid of the notebook
  function grid(ctx, p) {
    const w = 236, h = 118;
    ORDER.forEach((g, j) => {
      text(ctx, LABEL[g], 8 + j * (w + 6) + w / 2, 16, 15, g === "Heaps" ? col.ours : col.ink, { align: "center", weight: 700 });
      [0, 1, 2].forEach((i) => {
        const e = ex(g, i);
        series(ctx, { x: 8 + j * (w + 6), y: 24 + i * (h + 5), w, h }, e.history, e.runs, seg(p, 0.05 * j, 0.6 + 0.05 * j),
          { order: e.order, note: `mech. ${i}, memory ${e.order}`, lw: 1 });
      });
    });
    const x0 = 8 + 4 * (w + 6) + 6;
    ctx.globalAlpha = ease(seg(p, 0.5, 0.9));
    box(ctx, x0, 8, 1200 - x0 - 6, 384);
    text(ctx, `All ${D.mechanisms} mechanisms`, x0 + 16, 40, 17, col.ink, { weight: 700 });
    const rate = (g) => `${M[g].explosions} of ${M[g].trajectories}`;
    let y = wrap(ctx, `Explosions within ${STEPS} steps: no protection ${rate("Original AST")}; Jacobian, Shao–Wu and Heaps VAR none.`, x0 + 16, 68, 1200 - x0 - 38, 19, 13.5, col.ink);
    y = wrap(ctx, "Jacobian and Shao–Wu are contractions: two different starts end up at the same path, so the cosine start is forgotten.", x0 + 16, y + 8, 1200 - x0 - 38, 19, 13.5, col.ink);
    wrap(ctx, "Heaps VAR is linear with roots very close to the unit circle: what the start puts into a slow mode lasts, and it keeps oscillating at its own periods.", x0 + 16, y + 8, 1200 - x0 - 38, 19, 13.5, col.ink);
    ctx.globalAlpha = 1;
  }

  G.add("cos", "Cosine start", {
    title: "What each method does with a cosine start",
    foot: `Real runs: experiments/time_series_initialization/2_initial_magnitude.ipynb, condition "Cosseno completo; A=0.5" (${D.mechanisms} mechanisms per method, lags 12–24, Heaps order 24, noise σ = ${D.innovation_std}, 3 runs per mechanism, ${STEPS} steps); exported by report/slides/scripts/export_cosine_start.py. Roots: eigenvalues of the Heaps companion matrix. Explosion = non-finite or |x| > 100.`,
    steps: [
      { narr: "Same start, four methods", sub: "one full cosine cycle as the starting history", formula: "history: x(k) = 0.5 · cos(2πk / 96),&nbsp; k = 0 … 95", dur: 5200 },
      { narr: "Why does Heaps continue the wave?", sub: "compare with a zero start and the same noise", formula: "Heaps mechanism 1: roots |λ| = " + M.Heaps.roots[1][0].modulus + ",&nbsp; period " + M.Heaps.roots[1][0].period, dur: 6000 },
      { narr: "Not every Heaps mechanism does it", sub: "the first three Heaps mechanisms of the bank", formula: "", dur: 5200 },
      { narr: "All twelve example runs", sub: "the notebook figure: 3 mechanisms per method", formula: "", dur: 4200 },
    ],
    paint(ctx, step, p) {
      if (step === 0) sameStart(ctx, p);
      else if (step === 1) resonance(ctx, p);
      else if (step === 2) otherHeaps(ctx, p);
      else grid(ctx, p);
    },
  }, "stab");
})();
