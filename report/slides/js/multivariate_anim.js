/* Step-by-step "multivariatizer" animation, driven by reveal.js fragments.
 *
 * Mental model shown on the slide: x1, x2, x3 are independent SOURCE series
 * (real KernelSynth draws). A multivariatizer builds the variates y1, y2, y3
 * FROM them, so every y is a function of the x it uses; nothing is thrown away.
 * The two constructions (cotemporaneous, sequential) are ILLUSTRATIVE: the
 * Chronos-2 paper names the classes but not their formulas. Every number on
 * screen, including the terms that add up to each y, is computed by
 * scripts/export_kernelsynth_demo.py from the series being shown.
 */
(function () {
  const D = window.MV_DATA;
  const T = window.TSX;
  const sec = document.getElementById("mv");
  if (!D || !T || !sec) return;
  const root = sec.querySelector(".mv");

  root.classList.add("noanim");
  if (/[?&]static\b/.test(location.search)) {
    const st = document.createElement("style");
    st.textContent = ".mv, .mv * { transition: none !important; }";
    document.head.appendChild(st);
  }

  const N = D.n, L = D.lag;
  const COLORS = ["#1d4ed8", "#0f766e", "#6d5bd0"];
  const OURS = "#c2410c";
  const LW = 720, LH = 450, PITCH = 148, PTOP = 28, PH = 112, PX0 = 8, PXW = LW - 16;
  const CUT = Math.floor(N * 0.72);
  const SWEEP_MS = 3400;
  const TEST_CALC = /[?&]calc/.test(location.search);

  const el = (tag, cls, parent, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  };
  const clamp01 = (v) => Math.max(0, Math.min(1, v));

  const narr = el("div", "narr", root);
  const lanes = el("canvas", "lanes", root);
  const ctx = T.setupCanvas(lanes, LW, LH);
  const note = el("div", "note", root);
  const wireBox = el("div", "wirebox hidden", root);
  const wire = el("canvas", "", wireBox);
  const WW = 450, WH = 200;
  const wctx = T.setupCanvas(wire, WW, WH);
  const form = el("div", "form", root);

  // Correlation matrix (DOM grid)
  const mat = el("div", "mat hidden", root);
  const matGrid = el("div", "grid", mat);
  const matCap = el("div", "cap", mat);
  const cells = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cells.push(el("div", "cell", matGrid));
  function setMatrix(m) {
    cells.forEach((cell, k) => {
      const v = m[Math.floor(k / 3)][k % 3];
      const a = Math.abs(v);
      cell.textContent = (Object.is(v, -0) ? 0 : v).toFixed(2);
      cell.style.background = T.heatColor(a);
      cell.style.color = a > 0.5 ? "#fff" : "#1f2430";
    });
  }

  const calc = el("div", "calc hidden", root);
  const xcBox = el("div", "xc hidden", root);
  el("div", "cap", xcBox, "cross-correlation of series 1 and 2, by lag (steps)");
  const xcCanvas = el("canvas", "", xcBox);
  const XW = 450, XH = 180;
  const xctx = T.setupCanvas(xcCanvas, XW, XH);
  const foot = el("div", "foot", root,
    "Illustrative constructions: the Chronos-2 paper names the two classes but not their formulas.");

  // ---- Series ---------------------------------------------------------------
  const arr = (a) => a.map((x) => Float32Array.from(x));
  const BASE = arr(D.base), COT = arr(D.cot), SEQ = arr(D.seq);
  const CP = arr(D.cot_parts), SP = arr(D.seq_parts);
  const lim = [0, 1, 2].map((i) => {
    const sets = [
      [BASE[i], COT[i], SEQ[i]],
      [BASE[i], COT[i], SEQ[i], CP[0], CP[1], SP[0], SP[1]],
      [BASE[i], COT[i], SEQ[i], CP[2], SP[2], SP[3]],
    ][i];
    let m = 0;
    for (const s of sets) for (const v of s) m = Math.max(m, Math.abs(v));
    return m * 1.12;
  });
  // Terms drawn inside each lane while a variate is being built: [array, source lane colour].
  const PARTS = {
    3: [[], [[CP[0], 0], [CP[1], 1]], [[CP[2], 2]]],
    5: [[], [[SP[0], 0], [SP[1], 1]], [[SP[2], 1], [SP[3], 2]]],
  };
  const LANE_NOTE = {
    3: ["y₁ = x₁ (unchanged)", "dashed: 0.8·x₁ and 0.6·x₂   solid: their sum", "dashed: x₃ + x₁   solid: tanh of it"],
    5: [
      "y₁ = x₁ (unchanged)",
      "dashed: 0.9·y₁ delayed by L, and 0.4·x₂   solid: their sum",
      "dashed: 0.9·y₂ delayed by L, and 0.4·x₃   solid: their sum",
    ],
  };

  let cur = BASE.map((a) => a.slice());
  let tw = null; // {from, to, t0, dur, sweep}
  let reveal = 0, revealAnim = null;
  let wireP = 1, wireAnim = null, wireMode = "cot";
  let step = 0, prev = -1, timer1 = null, timer2 = null;

  const X = (k) => PX0 + (PXW * k) / (N - 1);
  const Y = (i, v) => PTOP + i * PITCH + PH / 2 - (v / lim[i]) * (PH / 2);

  function shownSeries(i, now) {
    if (!tw) return cur[i];
    const p = Math.min(1, (now - tw.t0) / tw.dur);
    const out = new Float32Array(N);
    if (tw.sweep) {
      const edge = p * (N - 1);
      for (let k = 0; k < N; k++) out[k] = k <= edge ? tw.to[i][k] : tw.from[i][k];
    } else {
      const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      for (let k = 0; k < N; k++) out[k] = tw.from[i][k] + (tw.to[i][k] - tw.from[i][k]) * e;
    }
    return out;
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function paint(now) {
    ctx.clearRect(0, 0, LW, LH);
    const sweeping = !!(tw && tw.sweep);
    const p = tw ? Math.min(1, (now - tw.t0) / tw.dur) : 1;
    const edgeIdx = sweeping ? Math.floor(p * (N - 1)) : N - 1;
    const building = step === 3 || step === 5;
    for (let i = 0; i < 3; i++) {
      const top = PTOP + i * PITCH;
      const col = COLORS[i];
      const inTask = step >= 6;
      const isCov = step === 7 && i < 2;
      ctx.fillStyle = "#fff";
      roundRect(ctx, 0, top - 4, LW, PH + 8, 10);
      ctx.fill();
      ctx.strokeStyle = "#d9d6cc";
      ctx.lineWidth = 2;
      ctx.stroke();
      if (inTask) {
        ctx.fillStyle = isCov ? "rgba(29,78,216,0.07)" : "rgba(194,65,12,0.10)";
        ctx.fillRect(X(CUT), top - 2, LW - 3 - X(CUT), PH + 4);
      }
      ctx.strokeStyle = "#ece9e0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PX0, Y(i, 0));
      ctx.lineTo(PX0 + PXW, Y(i, 0));
      ctx.stroke();

      // labels
      const isY = step === 3 || step === 5 || step >= 6;
      ctx.font = "600 14px Segoe UI, sans-serif";
      ctx.fillStyle = col;
      ctx.textAlign = "left";
      ctx.fillText(isY ? `y${i + 1}` : `x${i + 1}`, 10, top - 9);
      if (!isY && step <= 4) {
        ctx.fillStyle = "#5b6475";
        ctx.font = "13px Segoe UI, sans-serif";
        ctx.fillText(step <= 1 ? `source, κ* = ${D.names[i]}` : "source", 38, top - 9);
      }
      if (building) {
        ctx.textAlign = "right";
        ctx.font = "13px Segoe UI, sans-serif";
        ctx.fillStyle = "#5b6475";
        ctx.fillText(LANE_NOTE[step][i], LW - 12, top - 9);
        ctx.textAlign = "left";
      }
      if (inTask) {
        ctx.textAlign = "right";
        ctx.font = "700 13px Segoe UI, sans-serif";
        ctx.fillStyle = isCov ? "#1d4ed8" : OURS;
        ctx.fillText(isCov ? "known covariate" : "target", LW - 12, top - 9);
        ctx.textAlign = "left";
      }
      if (step < 1) continue;

      // terms that add up to this variate (dashed), drawn before the variate itself
      if (building) {
        PARTS[step][i].forEach(([ys, srcLane]) => {
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = COLORS[srcLane];
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          for (let k = 0; k <= edgeIdx; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k), Y(i, ys[k]));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        });
      }

      const ys = shownSeries(i, now);
      const last = step === 1 ? Math.floor((N - 1) * reveal) : N - 1;
      ctx.lineJoin = "round";
      const draw = (from, to, color, lw) => {
        ctx.beginPath();
        for (let k = from; k <= to; k++) (k === from ? ctx.moveTo : ctx.lineTo).call(ctx, X(k), Y(i, ys[k]));
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.stroke();
      };
      if (inTask) {
        draw(0, CUT, col, 2.2);
        draw(CUT, N - 1, isCov ? col : OURS, 2.6);
      } else {
        draw(0, last, col, 2.4);
      }
    }

    if (step >= 6) {
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = OURS;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(X(CUT), PTOP - 6);
      ctx.lineTo(X(CUT), PTOP + 2 * PITCH + PH + 6);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = OURS;
      ctx.font = "600 12px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("forecast origin", X(CUT), PTOP + 3 * PITCH - 8);
      ctx.textAlign = "left";
    }

    // sweep line: the time step being processed, with the output value of each variate
    if (sweeping && p < 1) {
      const x = X(edgeIdx);
      ctx.strokeStyle = OURS;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, PTOP - 8);
      ctx.lineTo(x, PTOP + 2 * PITCH + PH + 8);
      ctx.stroke();
      for (let i = 0; i < 3; i++) {
        ctx.fillStyle = COLORS[i];
        ctx.beginPath();
        ctx.arc(x, Y(i, tw.to[i][edgeIdx]), 5.5, 0, 7);
        ctx.fill();
      }
      updateCalc(edgeIdx);
    }
  }

  // ---- Wiring diagram: which sources feed which variate ---------------------------
  const ROW = [52, 112, 172];
  const NX = 62, NY = 252;
  const WIRES = {
    cot: {
      edges: [[0, 0, "1"], [0, 1, "0.8"], [1, 1, "0.6"], [0, 2, "1"], [2, 2, "1"]],
      ydirect: [],
      notes: ["copy", "weighted sum", "sum, then tanh"],
    },
    seq: {
      edges: [[0, 0, "1"], [1, 1, "0.4"], [2, 2, "0.4"]],
      ydirect: [[0, 1, `0.9 · delay ${L}`], [1, 2, `0.9 · delay ${L}`]],
      notes: ["copy", "own source + delayed y₁", "own source + delayed y₂"],
    },
  };
  function paintWire() {
    wctx.clearRect(0, 0, WW, WH);
    const cfg = WIRES[wireMode];
    wctx.textAlign = "center";
    wctx.font = "600 12px Segoe UI, sans-serif";
    wctx.fillStyle = "#5b6475";
    wctx.fillText("sources (sampled)", NX, 16);
    wctx.fillText("variates (built)", NY, 16);
    const nEdges = cfg.edges.length + cfg.ydirect.length;
    const seg = (a, b, col, lw, g, dash) => {
      if (g <= 0) return;
      const gx = a[0] + (b[0] - a[0]) * Math.min(1, g), gy = a[1] + (b[1] - a[1]) * Math.min(1, g);
      wctx.setLineDash(dash || []);
      wctx.strokeStyle = col;
      wctx.lineWidth = lw;
      wctx.beginPath();
      wctx.moveTo(a[0], a[1]);
      wctx.lineTo(gx, gy);
      wctx.stroke();
      wctx.setLineDash([]);
    };
    const label = (txt, x, y, col) => {
      wctx.font = "700 12px Segoe UI, sans-serif";
      const w = wctx.measureText(txt).width + 8;
      wctx.fillStyle = "rgba(255,255,255,0.95)";
      wctx.fillRect(x - w / 2, y - 9, w, 17);
      wctx.fillStyle = col;
      wctx.fillText(txt, x, y + 4);
    };
    cfg.edges.forEach(([i, j, w], k) => {
      const g = clamp01(wireP * nEdges - k);
      const a = [NX + 15, ROW[i]], b = [NY - 15, ROW[j]];
      seg(a, b, COLORS[i], 2.4, g);
      if (g >= 1) {
        const f = i === j ? 0.5 : 0.64;
        label(w, a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, COLORS[i]);
      }
    });
    cfg.ydirect.forEach(([i, j, w], m) => {
      const g = clamp01(wireP * nEdges - cfg.edges.length - m);
      seg([NY, ROW[i] + 16], [NY, ROW[j] - 16], COLORS[i], 2.4, g, [5, 4]);
      if (g >= 1) {
        wctx.textAlign = "right";
        wctx.font = "700 12px Segoe UI, sans-serif";
        wctx.fillStyle = COLORS[i];
        wctx.fillText(w, NY - 10, (ROW[i] + ROW[j]) / 2 + 4);
        wctx.textAlign = "center";
      }
    });
    for (let i = 0; i < 3; i++) {
      [[NX, `x${i + 1}`], [NY, `y${i + 1}`]].forEach(([x, t]) => {
        wctx.beginPath();
        wctx.arc(x, ROW[i], 15, 0, 7);
        wctx.fillStyle = COLORS[i];
        wctx.fill();
        wctx.fillStyle = "#fff";
        wctx.font = "700 13px Segoe UI, sans-serif";
        wctx.fillText(t, x, ROW[i] + 4.5);
      });
      wctx.textAlign = "left";
      wctx.font = "13px Segoe UI, sans-serif";
      wctx.fillStyle = "#5b6475";
      wctx.fillText(cfg.notes[i], NY + 24, ROW[i] + 4);
      wctx.textAlign = "center";
    }
    wctx.textAlign = "left";
  }

  // ---- Live calculation for the time step under the sweep line --------------------
  const sg = (v) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
  function updateCalc(k) {
    if (step === 3) {
      const x1 = BASE[0][k], x2 = BASE[1][k], x3 = BASE[2][k];
      const y2 = 0.8 * x1 + 0.6 * x2, y3 = Math.tanh(x3 + x1);
      calc.innerHTML =
        `<div class='ch'>time step t = ${k}</div>` +
        `<div class='cl'>sources: x = (${sg(x1)}, ${sg(x2)}, ${sg(x3)})</div>` +
        `<div class='cl'>y₁ = x₁ = <b>${sg(x1)}</b></div>` +
        `<div class='cl'>y₂ = 0.8·(${sg(x1)}) + 0.6·(${sg(x2)}) = <b>${sg(y2)}</b></div>` +
        `<div class='cl'>y₃ = tanh(${sg(x3)} ${x1 < 0 ? "−" : "+"} ${Math.abs(x1).toFixed(2)}) = <b>${sg(y3)}</b></div>`;
    } else {
      const a2 = SP[0][k], b2 = SP[1][k], a3 = SP[2][k], b3 = SP[3][k];
      calc.innerHTML =
        `<div class='ch'>time step t = ${k}</div>` +
        `<div class='cl'>y₁ = x₁ = <b>${sg(BASE[0][k])}</b></div>` +
        `<div class='cl'>y₂ = 0.4·x₂ + 0.9·y₁(t−${L}) = ${sg(b2)} + ${sg(a2)} = <b>${sg(b2 + a2)}</b></div>` +
        `<div class='cl'>y₃ = 0.4·x₃ + 0.9·y₂(t−${L}) = ${sg(b3)} + ${sg(a3)} = <b>${sg(b3 + a3)}</b></div>`;
    }
  }

  function paintXc() {
    xctx.clearRect(0, 0, XW, XH);
    const lags = D.xc.lags;
    const x = (l) => 26 + ((XW - 52) * (l - lags[0])) / (lags[lags.length - 1] - lags[0]);
    const y = (v) => 12 + (XH - 40) * (1 - (v + 0.2) / 1.2);
    xctx.strokeStyle = "#ece9e0";
    xctx.lineWidth = 1;
    xctx.beginPath();
    xctx.moveTo(10, y(0));
    xctx.lineTo(XW - 10, y(0));
    xctx.stroke();
    const line = (vals, color, lw) => {
      xctx.beginPath();
      vals.forEach((v, k) => (k ? xctx.lineTo(x(lags[k]), y(v)) : xctx.moveTo(x(lags[k]), y(v))));
      xctx.strokeStyle = color;
      xctx.lineWidth = lw;
      xctx.stroke();
    };
    line(D.xc.base, "#c4c1b6", 2);
    line(D.xc.seq, "#0f766e", 2.6);
    let pk = 0;
    D.xc.seq.forEach((v, k) => { if (v > D.xc.seq[pk]) pk = k; });
    xctx.fillStyle = OURS;
    xctx.beginPath();
    xctx.arc(x(lags[pk]), y(D.xc.seq[pk]), 5, 0, 7);
    xctx.fill();
    xctx.font = "700 14px Segoe UI, sans-serif";
    xctx.setLineDash([3, 4]);
    xctx.strokeStyle = OURS;
    xctx.lineWidth = 1.4;
    xctx.beginPath();
    xctx.moveTo(x(L), y(-0.2));
    xctx.lineTo(x(L), 12);
    xctx.stroke();
    xctx.setLineDash([]);
    xctx.textAlign = "right";
    xctx.fillText(`peak at lag ${lags[pk]}  (L = ${L})`, x(lags[pk]) - 14, y(D.xc.seq[pk]) + 4);
    xctx.fillStyle = "#5b6475";
    xctx.font = "12px Segoe UI, sans-serif";
    xctx.textAlign = "center";
    [-200, -100, 0, 100, 200].forEach((l) => xctx.fillText(String(l), x(l), XH - 6));
    xctx.textAlign = "left";
    xctx.fillText("independent sources", x(lags[pk]) + 14, y(D.xc.base[pk]) + 4);
    xctx.fillStyle = "#0f766e";
    xctx.fillText("after the sequential construction", 30, 22);
  }
  paintXc();

  // ---- Animation loop ---------------------------------------------------------
  let running = false;
  function frame(now) {
    if (revealAnim) {
      reveal = Math.min(1, (now - revealAnim) / 2200);
      if (reveal >= 1) revealAnim = null;
    }
    if (wireAnim) {
      wireP = Math.min(1, (now - wireAnim) / 1800);
      if (wireP >= 1) wireAnim = null;
      paintWire();
    }
    if (tw && now - tw.t0 >= tw.dur) {
      cur = tw.to.map((a) => a.slice());
      tw = null;
    }
    paint(now);
    if (revealAnim || tw || wireAnim) requestAnimationFrame(frame);
    else running = false;
  }
  function kick() {
    if (!running) {
      running = true;
      requestAnimationFrame(frame);
    }
  }
  function goTo(to, dur, sweep, animate) {
    if (!animate) {
      tw = null;
      cur = to.map((a) => a.slice());
    } else {
      tw = { from: cur.map((a) => a.slice()), to, t0: performance.now(), dur, sweep };
    }
    kick();
  }

  // ---- Text per step ----------------------------------------------------------
  const NARR = [
    ["From univariate to multivariate", "a multivariatizer samples several series and imposes dependencies between them"],
    ["① Sample independent source series x₁, x₂, x₃", "from any base generator: AR, ETS, TSI or KernelSynth"],
    ["② Option A, cotemporaneous: y at time t is a function of x at time t", "linear or nonlinear"],
    ["② Option A: build y one time step at a time", "each y is made of the sources it uses"],
    ["③ Option B, sequential: y at time t also uses y from earlier times", "starting again from the same sources x"],
    ["③ Option B: build y one time step at a time", "a delayed copy of the previous variate, plus the series' own source"],
    ["④ Turn the series into a forecasting task: context and horizon", "multivariate task: every variate is a target"],
    ["④′ Or designate a random subset as known covariates", "covariate-informed task"],
  ];
  const NOTE = {
    0: "<b>A multivariate series is built from univariate ones.</b><br><br>① sample independent <b>source</b> series x<br>② build the <b>variates</b> y from them, imposing dependencies<br>③ cut the result into a forecasting task",
    1: "Here, three KernelSynth draws <b>x ∼ GP(0, κ*)</b>.<br><br>Think of them as <b>ingredients</b>. They are independent draws, so uncorrelated. They are <u>not</u> the final series: the next steps build the variates y <b>from</b> them.",
    2: "<b>Nothing is thrown away.</b> The sources x are the inputs: each y inherits the shapes of the x it uses, and that shared ingredient is what makes the variates correlated.",
    4: "Because y₃ uses y₂ and y₂ uses y₁, a bump in series 1 reaches series 2 after L steps and series 3 after 2L: a <b>chain of lead–lag effects</b>.",
    6: "<b>Multivariate task</b><br>All three variates must be predicted.<br><br><span class='sw' style='background:#c2410c'></span>horizon (to predict)",
    7: "<b>Covariate-informed task</b><br>Series 1 and 2 are <b>known covariates</b>: their future values are given.<br>Series 3 is the <b>target</b> to predict.<br><br>Chronos-2 sees many such tasks, with random roles.",
  };
  const FORM_COT =
    "<div class='rec'><b>y at time t is a function of x at time t</b></div>" +
    "<div class='fl'>y₁ = x₁</div><div class='fl'>y₂ = 0.8·x₁ + 0.6·x₂ <span class='tag'>linear</span></div>" +
    "<div class='fl'>y₃ = tanh(x₃ + x₁) <span class='tag'>nonlinear</span></div>";
  const FORM_SEQ =
    "<div class='rec'><b>Each variate = its own source + a delayed copy of the previous variate</b></div>" +
    "<div class='fl'>y₁(t) = x₁(t)</div><div class='fl'>y₂(t) = 0.4·x₂(t) + 0.9·y₁(t − L)</div>" +
    `<div class='fl'>y₃(t) = 0.4·x₃(t) + 0.9·y₂(t − L)&nbsp;&nbsp;<span class='mut'>L = ${L}</span></div>`;

  const setTop = (e, v) => (e.style.top = v + "px");
  const show = (e, on) => e.classList.toggle("hidden", !on);

  function render() {
    const animate = step === prev + 1;
    clearTimeout(timer1);
    clearTimeout(timer2);

    narr.innerHTML = `${NARR[step][0]}<small>${NARR[step][1]}</small>`;
    note.innerHTML = NOTE[step] || "";
    show(note, !!NOTE[step]);
    setTop(note, step === 2 ? 428 : step === 4 ? 432 : 62);
    note.classList.toggle("small", step === 1 || step === 2 || step === 4);
    foot.classList.toggle("hidden", step < 2);

    // wiring diagram
    const showWire = step >= 2 && step <= 5;
    show(wireBox, showWire);
    if (showWire) {
      wireMode = step <= 3 ? "cot" : "seq";
      if ((step === 2 || step === 4) && animate) {
        wireP = 0;
        wireAnim = performance.now();
      } else {
        wireP = 1;
        wireAnim = null;
      }
      paintWire();
    }
    form.innerHTML = step === 2 ? FORM_COT : step === 4 ? FORM_SEQ : "";
    show(form, step === 2 || step === 4);
    setTop(form, 270);

    // right-hand lower area: matrix / live calculation / cross-correlation
    show(calc, false);
    show(mat, false);
    show(xcBox, false);
    if (step === 1) {
      setTop(mat, 300);
      setMatrix(D.corr.base);
      matCap.textContent = "sample correlation of x₁, x₂, x₃: independent draws are uncorrelated";
      if (animate) timer1 = setTimeout(() => show(mat, true), 2300);
      else show(mat, true);
    } else if (step === 3 || step === 5) {
      setTop(mat, 276);
      setTop(calc, 276);
      setTop(xcBox, 276);
      setMatrix(D.corr.cot);
      matCap.textContent = "sample correlation of y₁, y₂, y₃: now coupled";
      const done = () => {
        show(calc, false);
        show(step === 3 ? mat : xcBox, true);
      };
      if (TEST_CALC) {
        show(calc, true);
        updateCalc(Math.floor(N * 0.4));
      } else if (animate) {
        show(calc, true);
        timer2 = setTimeout(done, SWEEP_MS + 150);
      } else {
        done();
      }
    }

    // series
    if (step === 0) {
      goTo(BASE, 0, false, false);
      reveal = 0;
      revealAnim = null;
    } else if (step === 1) {
      goTo(BASE, 0, false, false);
      if (animate) {
        reveal = 0;
        revealAnim = performance.now();
      } else {
        reveal = 1;
        revealAnim = null;
      }
    } else if (step === 2) {
      reveal = 1;
      goTo(BASE, 0, false, false);
    } else if (step === 3) {
      reveal = 1;
      goTo(COT, SWEEP_MS, true, animate);
    } else if (step === 4) {
      reveal = 1;
      goTo(BASE, 900, false, animate);
    } else if (step === 5) {
      reveal = 1;
      goTo(SEQ, SWEEP_MS, true, animate);
    } else {
      reveal = 1;
      goTo(SEQ, 0, false, false);
    }
    paint(performance.now());
    kick();
    prev = step;
  }

  function sync() {
    step = sec.querySelectorAll(".fragment.visible").length;
    render();
    if (root.classList.contains("noanim")) {
      requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("noanim")));
    }
  }
  const bind = () => {
    Reveal.on("fragmentshown", sync);
    Reveal.on("fragmenthidden", sync);
    Reveal.on("slidechanged", sync);
    sync();
  };
  if (window.Reveal && Reveal.isReady && Reveal.isReady()) bind();
  else Reveal.on("ready", bind);
})();
