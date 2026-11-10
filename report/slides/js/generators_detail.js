/* Click-through detail views for the AR, ETS and TSI base generators (slide 5).
 *
 * Provenance matters here, and the slide says so:
 *  - TSI: components and formulas follow Bahrpeyma et al. (2021), the work the
 *    Chronos-2 paper says its TSI generator is based on.
 *  - AR and ETS: the Chronos-2 paper only lists them as base generators. These
 *    animations use the textbook definitions with illustrative parameters.
 * Opened by clicking a node with data-gen; "?gen=ar&gstep=2" opens one directly
 * (used for screenshots, together with "?static").
 */
(function () {
  const T = window.TSX;
  if (!T) return;
  const W = 1200, H = 400;
  const STATIC = /[?&]static\b/.test(location.search);
  const INK = "#1f2430", MUTED = "#5b6475", BLUE = "#1d4ed8", TEAL = "#0f766e", PURPLE = "#6d5bd0", OURS = "#c2410c";

  // ---------- helpers ----------------------------------------------------
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
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const f2 = (v) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(2);

  function bounds(arrs, pad) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const d = hi - lo || 1;
    pad = pad === undefined ? 0.12 : pad;
    return [lo - d * pad, hi + d * pad];
  }
  function rr(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function panel(ctx, b, title) {
    ctx.fillStyle = "#fff";
    rr(ctx, b.x, b.y, b.w, b.h, 10);
    ctx.fill();
    ctx.strokeStyle = "#d9d6cc";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (title) {
      ctx.font = "600 14px Segoe UI, sans-serif";
      ctx.fillStyle = MUTED;
      ctx.textAlign = "left";
      ctx.fillText(title, b.x + 12, b.y + 19);
    }
  }
  const px = (b, N, t) => b.x + 10 + ((b.w - 20) * t) / (N - 1);
  const py = (b, lo, hi, v) => b.y + 28 + (b.h - 38) - ((v - lo) / (hi - lo)) * (b.h - 38);
  function line(ctx, b, ys, lo, hi, color, lw, upto) {
    T.drawLine(ctx, ys, b.x + 10, b.y + 28, b.w - 20, b.h - 38, lo, hi, color, lw, upto);
  }
  function text(ctx, s, x, y, font, color, align) {
    ctx.font = font || "600 15px Segoe UI, sans-serif";
    ctx.fillStyle = color || INK;
    ctx.textAlign = align || "left";
    ctx.fillText(s, x, y);
    ctx.textAlign = "left";
  }
  function wrap(ctx, s, x, y, w, lh, font, color) {
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    let line = "", yy = y;
    s.split(" ").forEach((word) => {
      const test = line ? line + " " + word : word;
      if (ctx.measureText(test).width > w && line) { ctx.fillText(line, x, yy); line = word; yy += lh; }
      else line = test;
    });
    if (line) ctx.fillText(line, x, yy);
  }
  function arrow(ctx, x1, y1, x2, y2, g, color) {
    if (g <= 0) return;
    const x = x1 + (x2 - x1) * Math.min(1, g), y = y1 + (y2 - y1) * Math.min(1, g);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x, y);
    ctx.stroke();
    if (g >= 1) {
      const a = Math.atan2(y2 - y1, x2 - x1);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 11 * Math.cos(a - 0.4), y2 - 11 * Math.sin(a - 0.4));
      ctx.lineTo(x2 - 11 * Math.cos(a + 0.4), y2 - 11 * Math.sin(a + 0.4));
      ctx.closePath();
      ctx.fill();
    }
  }
  function dot(ctx, x, y, r, fill, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 7);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2.4; ctx.stroke(); }
  }
  /** Stacked decomposition at a cursor: parts add up, from zero, to the total. */
  function stack(ctx, b, N, lo, hi, t, parts, colors) {
    const x = px(b, N, t);
    let acc = 0;
    parts.forEach((v, k) => {
      ctx.strokeStyle = colors[k];
      ctx.lineWidth = 7;
      ctx.lineCap = "butt";
      ctx.beginPath();
      ctx.moveTo(x, py(b, lo, hi, acc));
      ctx.lineTo(x, py(b, lo, hi, acc + v));
      ctx.stroke();
      acc += v;
    });
    dot(ctx, x, py(b, lo, hi, acc), 5, INK);
  }

  // ---------- "Fine print" tables -------------------------------------------
  // Row: [specification, value used in this animation, what the Chronos-2 paper says, status]
  // status: true = stated in the paper, false = not published, null = neutral note.
  const NP = "not published";
  const SCALE_ROW = ["Scale of the series", "arbitrary", "stated (Sec. 3.1): inputs are standardized, then sinh⁻¹, so scale does not matter", true];
  const AR_ROWS = [
    ["Order p (how far back it looks)", "p = 2: xₜ depends on xₜ₋₁ and xₜ₋₂ only", "not published (no maximum lag is stated)", false],
    ["Coefficients φ", "φ₁ ∼ U(−2, 2), φ₂ ∼ U(−1, 1), redrawn until stationary", NP, false],
    ["Stationarity test", "φ₁ + φ₂ < 1, φ₂ − φ₁ < 1, |φ₂| < 1 (general p: roots outside the unit circle)", "standard mathematics, not a Chronos-2 statement", null],
    ["Shocks εₜ", "independent 𝒩(0, 1)", NP, false],
    ["Noise scale σ", "σ = 1", NP, false],
    ["Start values and mean", "x₋₁ = x₋₂ = 0, no burn-in, zero mean", NP, false],
    ["Length", "100 steps shown", "not published for AR (the model reads contexts of up to 2048, then 8192 steps)", false],
    SCALE_ROW,
  ];
  const ETS_ROWS = [
    ["Variant", "ETS(A,A,A): additive error, trend and seasonality", "not published which variants are used", false],
    ["Seasonal lag m", "m = 12: sₜ looks back at sₜ₋₁₂", NP, false],
    ["Smoothing α, β, γ", "0.3, 0.05, 0.15 (usual region: 0 < α < 1, 0 < β < α, 0 < γ < 1 − α)", NP, false],
    ["Initial level and slope", "ℓ₀ = 0, b₀ = 0.12", NP, false],
    ["Initial seasonal pattern", "sinusoid with amplitude 2.2", NP, false],
    ["Shocks εₜ", "independent 𝒩(0, 0.6²)", NP, false],
    ["Length", "96 steps shown", NP, false],
    SCALE_ROW,
  ];
  const TSI_ROWS = [
    ["Components and formulas", "trend-cycle, 4 seasonal shapes, 3 irregularity types, 8 combination models", "stated: “based on Bahrpeyma et al. (2021)”", true],
    ["Trend coefficients a, b, c, d", "0.12, 0.7, 0.45, 2.5 over x ∈ [0, 10]", "ranges given neither by Chronos-2 nor by the original paper", false],
    ["Seasonality", "period 24; β₀ = 1, β₁ = 0.4, α₀ = 2π/24, α₁ = 2π/8", NP, false],
    ["Irregularity", "fractional Gaussian noise, Hurst exponent 0.75, standard deviation 0.4", NP, false],
    ["Combination model", "model 1: T + S + I", "stated: components are randomly constructed and combined; the choice rule is not", null],
    ["Lags", "none: every component is a function of the time t", "not applicable", null],
    ["Length", "240 steps shown", NP, false],
    SCALE_ROW,
  ];
  function specPaint(ctx, rows, p) {
    const b = { x: 30, y: 6, w: 1140, h: 388 };
    panel(ctx, b, null);
    const cols = [b.x + 20, b.x + 290, b.x + 770];
    const widths = [250, 460, 350];
    ["specification", "used in this animation", "what the Chronos-2 paper says"].forEach((h, i) =>
      text(ctx, h, cols[i], b.y + 30, "700 13px Segoe UI, sans-serif", MUTED));
    ctx.strokeStyle = "#d9d6cc";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(b.x + 14, b.y + 42);
    ctx.lineTo(b.x + b.w - 14, b.y + 42);
    ctx.stroke();
    const rh = 42;
    rows.forEach((r, i) => {
      const y = b.y + 46 + i * rh;
      ctx.globalAlpha = clamp01((p * (rows.length + 2) - i) / 1.5);
      const col = r[3] === true ? TEAL : r[3] === false ? OURS : MUTED;
      const mark = r[3] === true ? "✓" : r[3] === false ? "✗" : "•";
      text(ctx, r[0], cols[0], y + 24, "700 15px Segoe UI, sans-serif", INK);
      wrap(ctx, r[1], cols[1], y + 17, widths[1] - 20, 17, "15px Segoe UI, sans-serif", INK);
      text(ctx, mark, cols[2], y + 18, "700 17px Segoe UI, sans-serif", col);
      wrap(ctx, r[2], cols[2] + 24, y + 17, widths[2] - 30, 17, "14px Segoe UI, sans-serif", col);
      if (i < rows.length - 1) {
        ctx.strokeStyle = "#efece4";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(b.x + 14, y + rh - 3);
        ctx.lineTo(b.x + b.w - 14, y + rh - 3);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    });
  }

  // ---------- AR ---------------------------------------------------------
  const AR = (() => {
    const N = 100, TF = 12;
    const r = rng(11);
    const eps = Array.from({ length: N }, () => gauss(r));
    const sim = (phi) => {
      const x = new Array(N).fill(0);
      for (let t = 0; t < N; t++) {
        let s = eps[t];
        for (let i = 0; i < phi.length; i++) if (t - 1 - i >= 0) s += phi[i] * x[t - 1 - i];
        x[t] = s;
      }
      return x;
    };
    const PHI = [1.2, -0.6];
    const X = sim(PHI);
    const stationary = (a, b) => a + b < 1 && b - a < 1 && Math.abs(b) < 1;
    const PR = rng(21);
    const PTS = Array.from({ length: 260 }, () => {
      const a = -2 + 4 * PR(), b = -1 + 2 * PR();
      return [a, b, stationary(a, b)];
    });
    const REJ = [0.7, 0.4]; // outside the region: phi1 + phi2 = 1.1
    function paramPaint(ctx, step, p) {
      if (step === 5) {
        const b = { x: 30, y: 8, w: 600, h: 384 };
        panel(ctx, b, "parameter plane  (φ₁, φ₂)");
        const mx = (v) => b.x + 46 + (b.w - 76) * (v + 2.2) / 4.4;
        const my = (v) => b.y + 34 + (b.h - 64) * (1 - (v + 1.2) / 2.4);
        ctx.strokeStyle = "#ece9e0";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx(-2.2), my(0)); ctx.lineTo(mx(2.2), my(0));
        ctx.moveTo(mx(0), my(1.2)); ctx.lineTo(mx(0), my(-1.2));
        ctx.stroke();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = "#9aa6b8";
        ctx.lineWidth = 1.6;
        ctx.strokeRect(mx(-2), my(1), mx(2) - mx(-2), my(-1) - my(1));
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(mx(-2), my(-1)); ctx.lineTo(mx(2), my(-1)); ctx.lineTo(mx(0), my(1)); ctx.closePath();
        ctx.fillStyle = "rgba(15,118,110,0.09)";
        ctx.fill();
        ctx.strokeStyle = TEAL;
        ctx.lineWidth = 2.2;
        ctx.stroke();
        text(ctx, "stationary region", mx(0), my(-0.35), "700 15px Segoe UI, sans-serif", TEAL, "center");
        text(ctx, "sampling box", mx(-1.95), my(1) - 6, "13px Segoe UI, sans-serif", MUTED);
        [-2, 0, 2].forEach((v) => text(ctx, String(v), mx(v), my(-1.2) + 4, "12px Segoe UI, sans-serif", MUTED, "center"));
        [-1, 0, 1].forEach((v) => text(ctx, String(v), mx(-2.2) - 6, my(v) + 4, "12px Segoe UI, sans-serif", MUTED, "right"));
        text(ctx, "φ₁", mx(2.2), my(0) - 6, "600 14px Cambria Math, serif", MUTED, "right");
        text(ctx, "φ₂", mx(0) + 8, my(1.2) + 12, "600 14px Cambria Math, serif", MUTED);
        const n = Math.floor(p * PTS.length);
        let kept = 0;
        for (let i = 0; i < n; i++) {
          const [a, c, ok] = PTS[i];
          if (ok) kept++;
          dot(ctx, mx(a), my(c), 3.4, ok ? "rgba(15,118,110,0.85)" : "rgba(194,65,12,0.65)");
        }
        if (p > 0.92) {
          dot(ctx, mx(PHI[0]), my(PHI[1]), 9, null, INK);
          text(ctx, "our example (1.2, −0.6)", mx(PHI[0]) - 14, my(PHI[1]) + 28, "700 13px Segoe UI, sans-serif", INK, "right");
        }
        const c = { x: 660, y: 8, w: 510, h: 384 };
        panel(ctx, c, null);
        text(ctx, "How one parameter set is drawn", c.x + 22, c.y + 44, "700 22px Segoe UI, sans-serif", INK);
        const lines = [
          ["1", "draw φ₁ ∼ U(−2, 2) and φ₂ ∼ U(−1, 1)"],
          ["2", "keep the draw if the process is stationary"],
          ["3", "otherwise throw it away and draw again"],
        ];
        lines.forEach(([k, s], i) => {
          const show = clamp01(p * 3.2 - i * 0.5);
          ctx.globalAlpha = show;
          dot(ctx, c.x + 38, c.y + 100 + i * 52, 14, BLUE);
          text(ctx, k, c.x + 38, c.y + 105 + i * 52, "700 15px Segoe UI, sans-serif", "#fff", "center");
          text(ctx, s, c.x + 66, c.y + 106 + i * 52, "19px Segoe UI, sans-serif", INK);
          ctx.globalAlpha = 1;
        });
        text(ctx, `drawn: ${n}`, c.x + 22, c.y + 296, "600 18px Segoe UI, sans-serif", MUTED);
        text(ctx, `kept: ${kept}`, c.x + 170, c.y + 296, "700 18px Segoe UI, sans-serif", TEAL);
        text(ctx, `rejected: ${n - kept}`, c.x + 300, c.y + 296, "700 18px Segoe UI, sans-serif", OURS);
        text(ctx, "about half of the box is stationary (area 4 of 8)", c.x + 22, c.y + 340, "15px Segoe UI, sans-serif", MUTED);
        return;
      }
      if (step === 6) {
        const defs = [
          { phi: PHI, t: "kept: φ = (1.2, −0.6)", c: TEAL },
          { phi: REJ, t: "rejected: φ = (0.7, 0.4), φ₁ + φ₂ = 1.1", c: OURS },
        ];
        defs.forEach((d, i) => {
          const b = { x: 30 + i * 590, y: 20, w: 560, h: 340 };
          const ys = sim(d.phi).slice(0, 80);
          const [lo, hi] = bounds([ys]);
          panel(ctx, b, d.t);
          line(ctx, b, [lo + (hi - lo) / 2, lo + (hi - lo) / 2], lo, hi, "#ece9e0", 1, 1);
          line(ctx, b, ys, lo, hi, d.c, 2.4, p);
          const mxv = Math.max(...ys.map(Math.abs));
          text(ctx, `largest |x| = ${mxv < 100 ? mxv.toFixed(1) : Math.round(mxv)}`, b.x + b.w - 14, b.y + 19, "700 14px Segoe UI, sans-serif", d.c, "right");
        });
        text(ctx, "first 80 steps; the vertical scales are very different", 600, 388, "14px Segoe UI, sans-serif", MUTED, "center");
        return;
      }
      // step 7: recipe
      const cards = [
        ["① order p (lags)", "how many past values xₜ looks at: xₜ₋₁ … xₜ₋ₚ", "here p = 2; the maximum p used by Chronos-2 is not published"],
        ["② coefficients φ", "draw, then keep only stationary ones", "p > 2: all roots of 1 − φ₁z − … − φₚzᵖ outside the unit circle"],
        ["③ noise scale σ", "size of the fresh shock εₜ", "here σ = 1"],
        ["④ simulate", "xₜ = Σᵢ φᵢ·xₜ₋ᵢ + σ·εₜ", "start from zeros, one step after another"],
      ];
      cards.forEach((c, i) => {
        const b = { x: 30 + i * 292, y: 60, w: 276, h: 250 };
        const show = clamp01((p - i * 0.2) / 0.4);
        ctx.globalAlpha = show;
        panel(ctx, b, null);
        text(ctx, c[0], b.x + 18, b.y + 46, "700 21px Segoe UI, sans-serif", i === 1 ? OURS : BLUE);
        wrap(ctx, c[1], b.x + 18, b.y + 92, b.w - 36, 22, "16px Segoe UI, sans-serif", INK);
        wrap(ctx, c[2], b.x + 18, b.y + 170, b.w - 36, 22, "14px Segoe UI, sans-serif", MUTED);
        ctx.globalAlpha = 1;
      });
      text(ctx, "Illustrative recipe. The Chronos-2 paper does not publish orders, coefficient priors or noise scales.", 600, 370, "15px Segoe UI, sans-serif", MUTED, "center");
    }

    const NB = { x: 30, y: 8, w: 1140, h: 150 }, SB = { x: 30, y: 172, w: 1140, h: 220 };
    const FORM = "xₜ = φ₁·xₜ₋₁ + φ₂·xₜ₋₂ + εₜ,&nbsp;&nbsp; εₜ ∼ 𝒩(0, 1)&nbsp;&nbsp;&nbsp; <span class='mut'>(φ₁ = 1.2, φ₂ = −0.6)</span>";
    return {
      key: "ar",
      title: "AR: autoregressive models",
      steps: [
        { narr: "Each value is a weighted sum of its own past, plus fresh noise", sub: "here an AR(2): it looks back at most 2 steps", formula: FORM },
        { narr: "① Draw white noise", sub: "one independent shock εₜ per time step", formula: FORM, dur: 2200 },
        { narr: "② One step of the recursion", sub: `build x${TF} from the two previous values and the shock`, formula: FORM, dur: 4200 },
        { narr: "③ Repeat for every time step", sub: "each new value feeds the next ones", formula: FORM, dur: 3200 },
        { narr: "④ Other coefficients, other dynamics", sub: "same noise, different φ", formula: "xₜ = Σᵢ φᵢ·xₜ₋ᵢ + εₜ", dur: 3000 },
        { narr: "⑤ Where do the coefficients φ come from?", sub: "draw them at random, keep only draws that give a stable process", formula: "stationary AR(2):&nbsp; φ₁ + φ₂ &lt; 1,&nbsp;&nbsp; φ₂ − φ₁ &lt; 1,&nbsp;&nbsp; |φ₂| &lt; 1", dur: 5200 },
        { narr: "⑥ Why reject some draws?", sub: "outside the region the series explodes", formula: "same εₜ in both panels: only φ differs", dur: 3400 },
        { narr: "⑦ The sampling recipe", sub: "parameters first, then the series", formula: "xₜ = Σᵢ φᵢ·xₜ₋ᵢ + σ·εₜ,&nbsp;&nbsp; εₜ ∼ 𝒩(0, 1)&nbsp;&nbsp; <span class='mut'>(σ = 1 here)</span>", dur: 2600 },
        { narr: "⑧ Fine print: what this animation assumes", sub: "and what the Chronos-2 paper actually specifies", formula: "xₜ depends on at most the last p values:&nbsp; xₜ₋₁, xₜ₋₂, …, xₜ₋ₚ&nbsp;&nbsp; <span class='mut'>(here p = 2)</span>", dur: 3200 },
      ],
      foot: "Textbook AR(p) with illustrative coefficients. The parameter sampling shown (uniform draws, rejecting non-stationary ones) is one standard way to get stable processes, not the Chronos-2 prior: the paper lists AR models as a base generator but does not publish orders, coefficient priors or noise scales.",
      paint(ctx, step, p) {
        if (step === 8) return specPaint(ctx, AR_ROWS, p);
        if (step >= 5) return paramPaint(ctx, step, p);
        if (step === 4) {
          const defs = [
            { phi: [0.9], t: "φ = 0.9: persistent" },
            { phi: [-0.8], t: "φ = −0.8: alternating" },
            { phi: [1.2, -0.6], t: "φ = (1.2, −0.6): quasi-cyclical" },
          ];
          defs.forEach((d, i) => {
            const b = { x: 30 + i * 390, y: 40, w: 360, h: 330 };
            const ys = sim(d.phi);
            const [lo, hi] = bounds([ys]);
            panel(ctx, b, d.t);
            line(ctx, b, [lo + (hi - lo) / 2, lo + (hi - lo) / 2], lo, hi, "#ece9e0", 1, 1);
            line(ctx, b, ys, lo, hi, [BLUE, TEAL, PURPLE][i], 2.2, p);
          });
          text(ctx, "all three panels reuse the same noise sequence ε", 600, 392, "14px Segoe UI, sans-serif", MUTED, "center");
          return;
        }
        panel(ctx, NB, "noise  εₜ ∼ 𝒩(0, 1)");
        panel(ctx, SB, "series  xₜ");
        const [nlo, nhi] = bounds([eps]);
        const [slo, shi] = bounds([X]);
        const shown = step === 0 ? 0 : step === 1 ? Math.floor(p * N) : N;
        const bw = Math.max(3, ((NB.w - 20) / N) * 0.55);
        for (let t = 0; t < shown; t++) {
          const x = px(NB, N, t), y0 = py(NB, nlo, nhi, 0), y1 = py(NB, nlo, nhi, eps[t]);
          ctx.fillStyle = step >= 2 && t === TF ? OURS : "#9aa6b8";
          ctx.fillRect(x - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0));
        }
        if (step === 0) return;
        if (step === 2) {
          const up = (TF - 1) / (N - 1);
          line(ctx, SB, X, slo, shi, BLUE, 2.4, up);
          const gx = (t) => px(SB, N, t), gy = (v) => py(SB, slo, shi, v);
          const grow = clamp01((p - 0.78) / 0.22);
          const ring = clamp01(p / 0.2);
          const arr = clamp01((p - 0.25) / 0.45);
          const tx = gx(TF), ty = gy(X[TF]);
          if (grow > 0) {
            ctx.strokeStyle = BLUE;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.moveTo(gx(TF - 1), gy(X[TF - 1]));
            ctx.lineTo(gx(TF - 1) + (tx - gx(TF - 1)) * grow, gy(X[TF - 1]) + (ty - gy(X[TF - 1])) * grow);
            ctx.stroke();
          }
          if (ring > 0) {
            dot(ctx, gx(TF - 1), gy(X[TF - 1]), 7 * ring, null, OURS);
            dot(ctx, gx(TF - 2), gy(X[TF - 2]), 7 * ring, null, OURS);
          }
          const sx = (t) => gx(t);
          if (ring > 0) {
            ctx.strokeStyle = OURS;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx(TF) - bw / 2 - 2, py(NB, nlo, nhi, Math.max(eps[TF], 0)) - 2, bw + 4,
              Math.abs(py(NB, nlo, nhi, eps[TF]) - py(NB, nlo, nhi, 0)) + 4);
          }
          arrow(ctx, gx(TF - 1) + 4, gy(X[TF - 1]) - 4, tx - 6, ty - 6, arr, OURS);
          arrow(ctx, gx(TF - 2) + 4, gy(X[TF - 2]) - 4, tx - 8, ty - 4, arr, OURS);
          arrow(ctx, px(NB, N, TF), py(NB, nlo, nhi, eps[TF]) + (eps[TF] >= 0 ? 4 : -4), tx, ty - 8, arr, OURS);
          if (grow > 0) dot(ctx, tx, ty, 7 * grow, OURS);
          const a = PHI[0] * X[TF - 1], b2 = PHI[1] * X[TF - 2];
          ctx.globalAlpha = clamp01((p - 0.3) / 0.3);
          text(ctx, `x${TF} = φ₁·x${TF - 1} + φ₂·x${TF - 2} + ε${TF}`, 380, SB.y + 62, "600 20px Cambria Math, Segoe UI Symbol, serif", INK);
          text(ctx, `= 1.2 × ${f2(X[TF - 1])}  +  (−0.6) × ${f2(X[TF - 2])}  +  ${f2(eps[TF])}`, 380, SB.y + 92, "20px Cambria Math, Segoe UI Symbol, serif", MUTED);
          text(ctx, `= ${f2(a)}  +  ${f2(b2)}  +  ${f2(eps[TF])}  =  ${f2(X[TF])}`, 380, SB.y + 122, "700 20px Cambria Math, Segoe UI Symbol, serif", OURS);
          ctx.globalAlpha = 1;
          return;
        }
        const up = step === 3 ? (TF + p * (N - 1 - TF)) / (N - 1) : 1;
        line(ctx, SB, X, slo, shi, BLUE, 2.4, up);
        if (step === 3 && p < 1) {
          const t = Math.floor(up * (N - 1));
          dot(ctx, px(SB, N, t), py(SB, slo, shi, X[t]), 6, OURS);
        }
      },
    };
  })();

  // ---------- ETS --------------------------------------------------------
  const ETS = (() => {
    const N = 96, M = 12;
    const r = rng(5);
    const eps = Array.from({ length: N }, () => 0.6 * gauss(r));
    const S0 = Array.from({ length: M }, (_, k) => 2.2 * Math.sin((2 * Math.PI * k) / M));
    function sim(al, be, ga) {
      const lvl = [], slope = [], seas = [], y = [], part1 = [], part2 = [];
      let l = 0, b = 0.12;
      for (let t = 0; t < N; t++) {
        const sp = t >= M ? seas[t - M] : S0[t];
        part1.push(l + b);
        part2.push(sp);
        y.push(l + b + sp + eps[t]);
        l = l + b + al * eps[t];
        b = b + be * eps[t];
        lvl.push(l);
        slope.push(b);
        seas.push(sp + ga * eps[t]);
      }
      return { lvl, slope, seas, y, part1, part2 };
    }
    const A = sim(0.3, 0.05, 0.15);
    const FORM =
      "yₜ = ℓₜ₋₁ + bₜ₋₁ + sₜ₋ₘ + εₜ<br>ℓₜ = ℓₜ₋₁ + bₜ₋₁ + α·εₜ&nbsp;&nbsp;&nbsp; bₜ = bₜ₋₁ + β·εₜ&nbsp;&nbsp;&nbsp; sₜ = sₜ₋ₘ + γ·εₜ";
    const LEFT = [
      { y: 6, k: "eps", t: "noise  εₜ", c: "#9aa6b8" },
      { y: 100, k: "lvl", t: "level  ℓₜ", c: BLUE },
      { y: 194, k: "slope", t: "slope  bₜ", c: PURPLE },
      { y: 288, k: "seas", t: "seasonal  sₜ  (period 12)", c: TEAL },
    ];
    const data = (k) => (k === "eps" ? eps : A[k]);
    return {
      key: "ets",
      title: "ETS: exponential smoothing",
      steps: [
        { narr: "A series built from level, slope and seasonal states", sub: "all three are nudged by the same shock εₜ", formula: FORM },
        { narr: "① Start from initial states", sub: "level ℓ₀, slope b₀ and a seasonal pattern of 12 values", formula: FORM, dur: 2000 },
        { narr: "② The noise drives every state", sub: "α, β, γ set how strongly each one reacts", formula: FORM, dur: 3600 },
        { narr: "③ Add the parts to get the observation", sub: "yₜ = (ℓ + b) + s + ε", formula: FORM, dur: 5200 },
        { narr: "④ Smoothing parameters change the behaviour", sub: "same noise, different α, β, γ", formula: FORM, dur: 3000 },
        { narr: "⑤ Fine print: what this animation assumes", sub: "and what the Chronos-2 paper actually specifies", formula: "the only lag is the seasonal one:&nbsp; sₜ looks back m steps to sₜ₋ₘ&nbsp;&nbsp; <span class='mut'>(here m = 12)</span>", dur: 3200 },
      ],
      foot: "Textbook additive ETS(A,A,A) (Hyndman & Athanasopoulos, Forecasting: Principles and Practice) with illustrative parameters. Chronos-2 lists ETS models as a base generator but does not publish which variants or parameter priors it uses.",
      paint(ctx, step, p) {
        if (step === 5) return specPaint(ctx, ETS_ROWS, p);
        if (step === 4) {
          const defs = [
            { s: sim(0.05, 0.01, 0.02), t: "small α, β, γ: smooth drift", c: BLUE },
            { s: sim(0.8, 0.1, 0.1), t: "large α: reacts to every shock", c: OURS },
          ];
          defs.forEach((d, i) => {
            const b = { x: 30 + i * 590, y: 30, w: 560, h: 340 };
            const [lo, hi] = bounds([defs[0].s.y, defs[1].s.y]);
            panel(ctx, b, d.t);
            line(ctx, b, d.s.y, lo, hi, d.c, 2.2, p);
          });
          return;
        }
        const big = { x: 480, y: 6, w: 690, h: 380 };
        LEFT.forEach((l) => {
          const b = { x: 30, y: l.y, w: 420, h: 86 };
          l.b = b;
          panel(ctx, b, l.t);
        });
        panel(ctx, big, "observation  yₜ");
        if (step === 0) return;
        const upto = step === 1 ? (M * p) / (N - 1) : step === 2 ? p : 1;
        LEFT.forEach((l) => {
          const ys = data(l.k);
          const [lo, hi] = bounds([ys]);
          line(ctx, l.b, ys, lo, hi, l.c, 2, upto);
        });
        if (step < 3) return;
        const [lo, hi] = bounds([A.y, A.part1.map((v, i) => v + A.part2[i])], 0.15);
        const cur = p * (N - 1);
        line(ctx, big, A.y, lo, hi, "#c9d3ee", 1.6, 1);
        line(ctx, big, A.y, lo, hi, BLUE, 2.4, p);
        const t = Math.max(0, Math.min(N - 1, Math.floor(cur)));
        stack(ctx, big, N, lo, hi, t, [A.part1[t], A.part2[t], eps[t]], [PURPLE, TEAL, "#9aa6b8"]);
        text(ctx, "■ level + slope", big.x + 420, big.y + 19, "600 13px Segoe UI, sans-serif", PURPLE);
        text(ctx, "■ seasonal", big.x + 530, big.y + 19, "600 13px Segoe UI, sans-serif", TEAL);
        text(ctx, "■ noise", big.x + 612, big.y + 19, "600 13px Segoe UI, sans-serif", "#7b8798");
      },
    };
  })();

  // ---------- TSI --------------------------------------------------------
  let TSI_DATA = null;
  function tsiData() {
    if (TSI_DATA) return TSI_DATA;
    const N = 240;
    const r = rng(3);
    const a = 0.12, b = 0.7, c = 0.45, d = 2.5;
    const trend = Array.from({ length: N }, (_, k) => {
      const x = (10 * k) / (N - 1);
      return (a * x + b) * Math.sin(x) + c * x + d;
    });
    const P = 24;
    const sin = Array.from({ length: N }, (_, k) => Math.sin((2 * Math.PI * k) / P) + 0.4 * Math.sin((2 * Math.PI * k) / 8));
    const step = Array.from({ length: N }, (_, k) => (Math.floor(k / (P / 2)) % 2 === 0 ? 1 : -1));
    const tri = Array.from({ length: N }, (_, k) => 4 * Math.abs(((k / P) % 1) - 0.5) - 1);
    const imp0 = Array.from({ length: N }, (_, k) => (k % P === 0 ? 1 : 0));
    const imp = imp0.map((_, k) => {
      let s = 0, n = 0;
      for (let j = -2; j <= 2; j++) if (imp0[k + j] !== undefined) { s += imp0[k + j]; n++; }
      return s / n;
    });
    // fractional Gaussian noise (Hurst 0.75) by Cholesky of its covariance.
    const Hh = 0.75;
    const g = (k) => 0.5 * (Math.pow(Math.abs(k + 1), 2 * Hh) - 2 * Math.pow(Math.abs(k), 2 * Hh) + Math.pow(Math.abs(k - 1), 2 * Hh));
    const Lm = Array.from({ length: N }, () => new Float64Array(N));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        let s = g(i - j);
        for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
        Lm[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / Lm[j][j];
      }
    }
    const z = Array.from({ length: N }, () => gauss(r));
    const fgn = Array.from({ length: N }, (_, i) => {
      let s = 0;
      for (let k = 0; k <= i; k++) s += Lm[i][k] * z[k];
      return s;
    });
    let acc = 0;
    const fbm = fgn.map((v) => (acc += v));
    const sd = Math.sqrt(fgn.reduce((s, v) => s + v * v, 0) / N);
    const irr = fgn.map((v) => (0.4 * v) / sd);
    TSI_DATA = { N, trend, sin, step, tri, imp, fgn, fbm, irr, par: { a, b, c, d } };
    return TSI_DATA;
  }
  const TSI = (() => {
    const MODELS = ["T + S + I", "(T + S) × I", "(T + I) × S", "(S + I) × T", "T × S + I", "T × I + S", "S × I + T", "T × S × I"];
    return {
      key: "tsi",
      title: "TSI: trend, seasonality, irregularity",
      steps: [
        { narr: "A series built from three components", sub: "trend-cycle T, seasonality S and irregularity I", formula: "xₜ = combine( Tₜ , Sₜ , Iₜ )" },
        { narr: "① Trend-cycle", sub: "a linear drift plus a growing oscillation", formula: "Tₜ = (a·x + b)·sin x + c·x + d", dur: 2600 },
        { narr: "② Seasonality: one of four repeating shapes", sub: "all with a period p", formula: "sinusoidal:&nbsp; Sₜ = β₀·sin(α₀t) + β₁·sin(α₁t)", dur: 3600 },
        { narr: "③ Irregularity: a signal with its own memory", sub: "fractional Gaussian noise, fractional Brownian motion or multifractal Brownian motion", formula: "Iₜ ∈ { fGn , fBm , mBm }", dur: 3000 },
        { narr: "④ Combine the components", sub: "here the purely additive model", formula: "xₜ = Tₜ + Sₜ + Iₜ", dur: 5200 },
        { narr: "⑤ Eight ways to combine them", sub: "additive, multiplicative or mixed", formula: "Chronos-2: “randomly constructing and combining different trend, seasonality, and irregularity components”", dur: 1800 },
        { narr: "⑥ Fine print: what this animation assumes", sub: "and what the Chronos-2 paper actually specifies", formula: "no autoregression here: every component is a function of the time t", dur: 3200 },
      ],
      foot: "Components and formulas follow Bahrpeyma et al. (2021). Chronos-2 says its TSI generator is “based on” this work; the parameter distributions it samples from are not published. Values shown are illustrative.",
      paint(ctx, step, p) {
        if (step === 6) return specPaint(ctx, TSI_ROWS, p);
        const D = tsiData();
        const N = D.N;
        if (step === 0) {
          const cards = [
            ["Trend-cycle  T", "slow drift and long oscillations", BLUE],
            ["Seasonality  S", "repeating pattern with period p", TEAL],
            ["Irregularity  I", "noise with its own memory", PURPLE],
          ];
          cards.forEach((c, i) => {
            const b = { x: 40 + i * 390, y: 70, w: 360, h: 230 };
            panel(ctx, b, null);
            text(ctx, c[0], b.x + 24, b.y + 60, "700 26px Segoe UI, sans-serif", c[2]);
            text(ctx, c[1], b.x + 24, b.y + 100, "18px Segoe UI, sans-serif", MUTED);
          });
          text(ctx, "+  or  ×", 600, 345, "700 24px Segoe UI, sans-serif", OURS, "center");
          return;
        }
        if (step === 1) {
          const b = { x: 30, y: 8, w: 1140, h: 330 };
          panel(ctx, b, "trend-cycle  Tₜ");
          const [lo, hi] = bounds([D.trend]);
          const lin = D.trend.map((_, k) => D.par.c * ((10 * k) / (N - 1)) + D.par.d);
          if (p > 0.45) {
            ctx.setLineDash([6, 6]);
            line(ctx, b, lin, lo, hi, OURS, 2, clamp01((p - 0.45) / 0.4));
            ctx.setLineDash([]);
            ctx.globalAlpha = clamp01((p - 0.6) / 0.3);
            text(ctx, "c·x + d: linear drift", b.x + 640, b.y + 275, "600 17px Segoe UI, sans-serif", OURS);
            ctx.globalAlpha = 1;
          }
          line(ctx, b, D.trend, lo, hi, BLUE, 3, clamp01(p / 0.6));
          ctx.globalAlpha = clamp01((p - 0.6) / 0.3);
          text(ctx, "(a·x + b)·sin x: growing oscillation around it", b.x + 40, b.y + 62, "600 17px Segoe UI, sans-serif", BLUE);
          ctx.globalAlpha = 1;
          text(ctx, `illustrative values:  a = ${D.par.a},  b = ${D.par.b},  c = ${D.par.c},  d = ${D.par.d}`, 600, 372, "15px Segoe UI, sans-serif", MUTED, "center");
          return;
        }
        if (step === 2) {
          const forms = [
            ["sinusoidal", D.sin, "two sine waves"],
            ["step-wise", D.step, "switches between two values"],
            ["triangular", D.tri, "linear up and down"],
            ["impulsive", D.imp, "periodic pulses, smoothed"],
          ];
          forms.forEach((f, i) => {
            const b = { x: 30 + i * 290, y: 50, w: 275, h: 240 };
            const local = clamp01((p - i * 0.18) / 0.5);
            panel(ctx, b, f[0]);
            const ys = f[1].slice(0, 120);
            const [lo, hi] = bounds([ys]);
            line(ctx, b, ys, lo, hi, TEAL, 2.4, local);
            text(ctx, f[2], b.x + b.w / 2, b.y + b.h + 28, "15px Segoe UI, sans-serif", MUTED, "center");
          });
          text(ctx, "the first 120 steps of each shape, period p = 24", 600, 372, "14px Segoe UI, sans-serif", MUTED, "center");
          return;
        }
        if (step === 3) {
          const defs = [
            { b: { x: 30, y: 30, w: 380, h: 300 }, t: "fractional Gaussian noise: stationary", ys: D.fgn, c: PURPLE },
            { b: { x: 430, y: 30, w: 380, h: 300 }, t: "fractional Brownian motion: non-stationary", ys: D.fbm, c: BLUE },
          ];
          defs.forEach((d, i) => {
            const local = clamp01((p - i * 0.2) / 0.6);
            panel(ctx, d.b, d.t);
            const [lo, hi] = bounds([d.ys]);
            line(ctx, d.b, d.ys, lo, hi, d.c, 2, local);
          });
          const m = { x: 830, y: 30, w: 340, h: 300 };
          panel(ctx, m, "multifractal Brownian motion");
          ctx.globalAlpha = clamp01((p - 0.4) / 0.4);
          text(ctx, "the Hurst exponent H", m.x + 24, m.y + 100, "600 20px Segoe UI, sans-serif", INK);
          text(ctx, "itself changes with time", m.x + 24, m.y + 130, "600 20px Segoe UI, sans-serif", INK);
          text(ctx, "(not drawn here)", m.x + 24, m.y + 175, "16px Segoe UI, sans-serif", MUTED);
          ctx.globalAlpha = 1;
          text(ctx, "left and middle: same random numbers, Hurst exponent 0.75", 600, 372, "14px Segoe UI, sans-serif", MUTED, "center");
          return;
        }
        if (step === 4) {
          const comps = [
            { k: "trend", t: "trend-cycle  T", c: BLUE },
            { k: "sin", t: "seasonality  S", c: TEAL },
            { k: "irr", t: "irregularity  I", c: PURPLE },
          ];
          comps.forEach((c, i) => {
            const b = { x: 30, y: 6 + i * 130, w: 420, h: 120 };
            panel(ctx, b, c.t);
            const ys = D[c.k];
            const [lo, hi] = bounds([ys]);
            line(ctx, b, ys, lo, hi, c.c, 2, 1);
          });
          const tot = D.trend.map((v, i) => v + D.sin[i] + D.irr[i]);
          const big = { x: 480, y: 6, w: 690, h: 380 };
          panel(ctx, big, "observation  xₜ = T + S + I");
          const [lo, hi] = bounds([tot, D.sin, D.irr], 0.12);
          line(ctx, big, tot, lo, hi, "#c9d3ee", 1.6, 1);
          line(ctx, big, tot, lo, hi, BLUE, 2.4, p);
          const t = Math.max(0, Math.min(N - 1, Math.floor(p * (N - 1))));
          stack(ctx, big, N, lo, hi, t, [D.trend[t], D.sin[t], D.irr[t]], [BLUE, TEAL, PURPLE]);
          return;
        }
        // step 5: the eight combination models
        MODELS.forEach((m, i) => {
          const col = i < 4 ? 0 : 1, row = i % 4;
          const b = { x: 60 + col * 560, y: 20 + row * 82, w: 520, h: 66 };
          const show = clamp01((p - i * 0.07) / 0.3);
          ctx.globalAlpha = show;
          panel(ctx, b, null);
          text(ctx, `model ${i + 1}`, b.x + 22, b.y + 40, "600 16px Segoe UI, sans-serif", MUTED);
          text(ctx, m, b.x + 150, b.y + 42, "700 24px Cambria Math, Segoe UI Symbol, serif", i === 0 ? TEAL : i === 7 ? OURS : INK);
          if (i === 0) text(ctx, "purely additive", b.x + 360, b.y + 40, "600 15px Segoe UI, sans-serif", TEAL);
          if (i === 7) text(ctx, "purely multiplicative", b.x + 330, b.y + 40, "600 15px Segoe UI, sans-serif", OURS);
          ctx.globalAlpha = 1;
        });
        text(ctx, "T = trend-cycle, S = seasonality, I = irregularity", 600, 372, "15px Segoe UI, sans-serif", MUTED, "center");
      },
    };
  })();

  const GENS = { ar: AR, ets: ETS, tsi: TSI };
  const ORDER = ["ar", "ets", "tsi"];
  const TABS = { ar: "AR", ets: "ETS", tsi: "TSI" };

  // ---------- Overlay ----------------------------------------------------
  const ov = document.createElement("div");
  ov.className = "gen-overlay";
  ov.innerHTML =
    '<div class="gen-stage">' +
    '<div class="gen-top"><div class="gen-title"></div><div class="gen-tabs"></div>' +
    '<button class="gen-close" aria-label="Close">✕ close</button></div>' +
    '<div class="gen-narr"></div><div class="gen-formula"></div>' +
    '<canvas class="gen-canvas"></canvas>' +
    '<div class="gen-foot"></div><div class="gen-hint">→ next &nbsp;·&nbsp; ← back &nbsp;·&nbsp; Esc close</div></div>';
  document.body.appendChild(ov);
  const stage = ov.querySelector(".gen-stage");
  const elTitle = ov.querySelector(".gen-title");
  const elTabs = ov.querySelector(".gen-tabs");
  const elNarr = ov.querySelector(".gen-narr");
  const elForm = ov.querySelector(".gen-formula");
  const elFoot = ov.querySelector(".gen-foot");
  const canvas = ov.querySelector(".gen-canvas");
  const ctx = T.setupCanvas(canvas, W, H);

  // Tabs are grouped: the generators of slide 5 and the stability methods of the generator zoom
  // share this overlay but never share a tab row.
  const GROUP = { ar: "gen", ets: "gen", tsi: "gen" };
  function addTab(k) {
    const b = document.createElement("button");
    b.textContent = TABS[k];
    b.dataset.gen = k;
    b.addEventListener("click", (e) => { e.stopPropagation(); openGen(k, 0); });
    elTabs.appendChild(b);
  }
  ORDER.forEach(addTab);

  function fit() {
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    stage.style.transform = `translate(-50%, -50%) scale(${s})`;
  }
  window.addEventListener("resize", fit);
  fit();

  let cur = null, step = 0, p = 1, t0 = 0, raf = 0, isOpen = false;
  function paint() {
    ctx.clearRect(0, 0, W, H);
    cur.paint(ctx, step, ease(p));
  }
  function frame(now) {
    const dur = cur.steps[step].dur || 2000;
    p = Math.min(1, (now - t0) / dur);
    paint();
    raf = p < 1 ? requestAnimationFrame(frame) : 0;
  }
  function setStep(s, animate) {
    step = Math.max(0, Math.min(cur.steps.length - 1, s));
    const st = cur.steps[step];
    elNarr.innerHTML = `${st.narr}<small>${st.sub}</small>`;
    elForm.innerHTML = st.formula || "";
    cancelAnimationFrame(raf);
    if (animate && !STATIC) {
      p = 0;
      t0 = performance.now();
      raf = requestAnimationFrame(frame);
    } else {
      p = 1;
    }
    paint();
  }
  function openGen(key, s) {
    cur = GENS[key];
    isOpen = true;
    ov.classList.add("open");
    elTitle.textContent = cur.title;
    elFoot.textContent = cur.foot;
    elTabs.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("on", b.dataset.gen === key);
      b.style.display = GROUP[b.dataset.gen] === GROUP[key] ? "" : "none";
    });
    setStep(s || 0, false);
    if (window.Reveal) Reveal.configure({ controls: false });
  }
  function close() {
    isOpen = false;
    cancelAnimationFrame(raf);
    ov.classList.remove("open");
    if (window.Reveal) Reveal.configure({ controls: true });
  }

  function onKey(e) {
    if (!isOpen) return;
    const k = e.key;
    const next = ["ArrowRight", "ArrowDown", " ", "Enter", "PageDown"].includes(k);
    const prev = ["ArrowLeft", "ArrowUp", "PageUp", "Backspace"].includes(k);
    if (!(next || prev || k === "Escape")) return;
    e.stopPropagation();
    e.preventDefault();
    if (k === "Escape") close();
    else if (next) setStep(step + 1, true);
    else setStep(step - 1, false);
  }
  window.addEventListener("keydown", onKey, true);
  ov.querySelector(".gen-close").addEventListener("click", close);
  canvas.addEventListener("click", () => setStep(step + 1, true));

  // Entry points on slide 5.
  document.querySelectorAll("[data-gen]").forEach((n) => {
    if (!GENS[n.dataset.gen] || n.closest(".gen-overlay")) return;
    n.addEventListener("click", () => openGen(n.dataset.gen, 0));
  });
  // Other scripts add their own detail views (same overlay, own tab group).
  window.GenDetail = {
    add(key, tab, def, group) {
      GENS[key] = def; ORDER.push(key); TABS[key] = tab; GROUP[key] = group;
      addTab(key);
    },
    open: openGen,
  };
  window.addEventListener("load", () => {
    const q = location.search.match(/[?&]gen=(\w+)(?:&gstep=(\d+))?/);
    if (q && GENS[q[1]]) openGen(q[1], q[2] ? Number(q[2]) : 0);
  });
})();
