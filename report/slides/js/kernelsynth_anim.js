/* Step-by-step KernelSynth animation, driven by reveal.js fragments.
 *
 * Every number shown comes from data/kernelsynth_demo.js, which is produced by
 * scripts/export_kernelsynth_demo.py with the official Chronos kernel bank and
 * composition rule. render(step) is idempotent, so stepping backwards works.
 */
(function () {
  const D = window.KS_DATA;
  const T = window.TSX;
  const sec = document.getElementById("ks");
  if (!D || !T || !sec) return;
  const root = sec.querySelector(".ks");
  // Initial state is applied without transitions; "?static" disables them (used for screenshots).
  root.classList.add("noanim");
  if (/[?&]static/.test(location.search)) {
    const st = document.createElement("style");
    st.textContent = ".ks, .ks * { transition: none !important; }";
    document.head.appendChild(st);
  }

  const R = D.demo.R;
  const PICKS = D.demo.picks;
  const OPS = D.demo.ops;
  const SUB = ["₁", "₂", "₃", "₄", "₅"];
  const LAST = 8;

  // Geometry (px inside the 1200 x 600 stage)
  const BW = 100, BH = 62, BGX = 8, BGY = 8, COLS = 11;
  const SW = 118, SH = 84, SGAP = 52, SY = 282, SX0 = 8;
  const bankPos = (i) => ({ x: (i % COLS) * (BW + BGX), y: Math.floor(i / COLS) * (BH + BGY) });
  const slotX = (j) => SX0 + j * (SW + SGAP);

  const el = (tag, cls, parent) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  };

  function drawThumb(canvas, ys, w, h, color) {
    const ctx = T.setupCanvas(canvas, w, h);
    const lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = (hi - lo || 1) * 0.12;
    T.drawLine(ctx, ys, 4, 4, w - 8, h - 8, lo - pad, hi + pad, color, 1.6);
  }

  function makeCard(parent, k, w, h, cls) {
    const card = el("div", "kcard " + (cls || ""), parent);
    const labelH = cls === "clone" ? 22 : 15;
    const cv = el("canvas", "", card);
    drawThumb(cv, k.thumb, w - 4, h - labelH - 4, T.FAMCOL[k.family]);
    const lbl = el("div", "lbl", card);
    lbl.textContent = k.label;
    lbl.style.color = T.FAMCOL[k.family];
    card.style.width = w + "px";
    card.style.height = h + "px";
    return card;
  }

  // ---- Kernel bank -------------------------------------------------------
  const bankEl = el("div", "bank", root);
  const bankCards = D.bank.map((k, i) => {
    const card = makeCard(bankEl, k, BW, BH);
    const p = bankPos(i);
    card.style.left = p.x + "px";
    card.style.top = p.y + "px";
    const badge = el("div", "badge", card);
    const nums = PICKS.map((p2, j) => (p2 === i ? j + 1 : 0)).filter(Boolean);
    badge.textContent = nums.join(",");
    return card;
  });

  // ---- Legend ------------------------------------------------------------
  const counts = {};
  D.bank.forEach((k) => (counts[k.family] = (counts[k.family] || 0) + 1));
  const legend = el("div", "legend", root);
  ["periodic", "linear", "rbf", "rq", "white", "constant"].forEach((f) => {
    const s = el("span", "", legend);
    s.innerHTML = `<i style="background:${T.FAMCOL[f]}"></i>${T.FAMNAME[f]} × ${counts[f]}`;
  });
  const total = el("span", "", legend);
  total.textContent = `= ${D.bank.length} kernels`;
  total.style.fontWeight = 650;

  // ---- Narration ---------------------------------------------------------
  const narr = el("div", "narr", root);
  const expr = el("div", "expr pipe", root);

  // ---- Pipeline: clones, operators, bracket ------------------------------
  const pipe = [];
  const clones = PICKS.map((id, j) => {
    const card = makeCard(root, D.bank[id], BW, BH, "clone");
    const p = bankPos(id);
    card.style.left = p.x + "px";
    card.style.top = p.y + "px";
    const b = el("div", "badge", card);
    b.textContent = "κ" + SUB[j];
    b.style.opacity = 1;
    b.style.fontSize = "15px";
    card.style.borderColor = T.FAMCOL[D.bank[id].family];
    pipe.push(card);
    return card;
  });
  const bracket = el("div", "bracket pipe", root);
  const ops = OPS.map((o, j) => {
    const e = el("div", "op pipe", root);
    e.textContent = o;
    e.style.left = slotX(j) + SW + 2 + "px";
    e.style.top = SY + "px";
    return e;
  });
  pipe.push(bracket, ...ops, expr);

  // ---- Right panel: morphing draw ---------------------------------------
  const rp = el("div", "rpanel pipe hidden", root);
  const rpTitle = el("h4", "", rp);
  const plot = el("canvas", "", rp);
  const PW = 530, PH = 290;
  const pctx = T.setupCanvas(plot, PW, PH);
  pipe.push(rp);

  const N = D.length;
  let cur = new Float32Array(N);
  let from = cur.slice();
  let to = cur.slice();
  let t0 = 0, morphing = false;
  let finalMode = false, strokeP = 1, strokeT0 = 0, stroking = false;

  function redraw() {
    pctx.clearRect(0, 0, PW, PH);
    let m = 0;
    for (let i = 0; i < N; i++) m = Math.max(m, Math.abs(cur[i]));
    const lim = Math.max(2.5, Math.min(6, m * 1.1));
    pctx.strokeStyle = "#e4e1d8";
    pctx.lineWidth = 1;
    pctx.beginPath();
    pctx.moveTo(0, PH / 2);
    pctx.lineTo(PW, PH / 2);
    pctx.stroke();
    if (finalMode) {
      T.drawLine(pctx, cur, 8, 10, PW - 16, PH - 20, -lim, lim, "#d4dcf3", 1.6);
      T.drawLine(pctx, cur, 8, 10, PW - 16, PH - 20, -lim, lim, "#1d4ed8", 2.4, strokeP);
    } else {
      T.drawLine(pctx, cur, 8, 10, PW - 16, PH - 20, -lim, lim, "#0f766e", 2);
    }
  }
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  function tick(now) {
    let go = false;
    if (morphing) {
      const p = Math.min(1, (now - t0) / 800);
      const e = ease(p);
      for (let i = 0; i < N; i++) cur[i] = from[i] + (to[i] - from[i]) * e;
      if (p >= 1) morphing = false;
      go = go || morphing;
    }
    if (stroking) {
      strokeP = Math.min(1, (now - strokeT0) / 2200);
      if (strokeP >= 1) stroking = false;
      go = go || stroking;
    }
    redraw();
    if (go) requestAnimationFrame(tick);
  }
  function setSeries(arr, animate) {
    to = Float32Array.from(arr);
    if (animate) {
      from = cur.slice();
      t0 = performance.now();
      morphing = true;
      requestAnimationFrame(tick);
    } else {
      morphing = false;
      cur = to.slice();
      redraw();
    }
  }

  // ---- Covariance heat-map ---------------------------------------------
  const heat = el("div", "heat pipe hidden", root);
  const hc = el("canvas", "", heat);
  T.drawHeat(T.setupCanvas(hc, 170, 170), D.demo.cov, 170);
  const heatCap = el("div", "heatcap pipe hidden", root);
  pipe.push(heat, heatCap);

  // ---- "Again": small multiples ---------------------------------------
  const more = el("div", "more hidden", root);
  D.more.forEach((m) => {
    const card = el("div", "mc", more);
    const cv = el("canvas", "", card);
    const ctx = T.setupCanvas(cv, 380, 100);
    const lim = Math.max(2.2, ...m.series.map(Math.abs));
    T.drawLine(ctx, m.series, 4, 4, 372, 92, -lim, lim, "#1d4ed8", 1.6);
    const cap = el("div", "", card);
    cap.textContent = foldNames(m.picks, m.ops);
  });
  function foldNames(picks, ops) {
    let s = T.FAMNAME[D.bank[picks[0]].family];
    for (let j = 1; j < picks.length; j++) {
      s = (j > 1 ? "(" + s + ")" : s) + " " + ops[j - 1] + " " + T.FAMNAME[D.bank[picks[j]].family];
    }
    return "κ* = " + s;
  }

  // ---- Text per step ----------------------------------------------------
  function exprHtml(upTo) {
    let s = "κ₁";
    for (let j = 1; j <= upTo; j++) s = (j > 1 ? "(" + s + ")" : s) + " " + OPS[j - 1] + " κ" + SUB[j];
    s = s.replace(/κ[₁-₅]/g, (m) => `<span class="k">${m}</span>`);
    s = s.replace(/ ([+×]) /g, ' <span class="o">$1</span> ');
    return (upTo === R - 1 ? "κ* = " : "") + s;
  }
  const NARR = [
    ["The kernel bank", "33 base kernels from the official Chronos script. White-noise kernels live in the bank: noise is not a separate step."],
    ["① Draw how many kernels to use:  R ∼ U{1, …, 5}", `→ R = ${R}`],
    ["② Draw R kernels from the bank, i.i.d.", "with replacement"],
    ["③ Combine them from left to right, each time with a random + or ×", "κ₁ ∘ κ₂"],
    ["③ Combine them from left to right, each time with a random + or ×", "(κ₁ ∘ κ₂) ∘ κ₃"],
    ["③ Combine them from left to right, each time with a random + or ×", "((κ₁ ∘ κ₂) ∘ κ₃) ∘ κ₄"],
    ["④ Evaluate the final kernel κ* on 1024 evenly spaced points of [0, 1]", "this gives the covariance matrix"],
    ["⑤ Draw one sample from the Gaussian process", "the sampled vector is the time series"],
    ["Repeat: a new random expression gives a new series", "each series is one training example for the forecaster"],
  ];
  const PANEL_TITLE = (step, reduce) => {
    if (step >= 7) return "x ∼ 𝒩(0, K*): the synthetic series (1024 steps)";
    if (reduce === 0) return "One draw from κ₁ alone (noise vector z fixed)";
    return "One draw from the partial kernel (same z): watch it change";
  };

  let prev = -1;
  function render(step) {
    const reduce = Math.max(0, Math.min(R - 1, step - 2));
    const animate = step === prev + 1;

    bankCards.forEach((c, i) => {
      const picked = PICKS.includes(i);
      c.classList.toggle("off", step >= 1 && !picked);
      c.classList.toggle("picked", step >= 1 && picked);
      c.classList.toggle("moved", step >= 2 && picked);
      c.style.borderColor = step >= 1 && picked ? T.FAMCOL[D.bank[i].family] : "";
    });

    clones.forEach((c, j) => {
      const on = step >= 2 && step < LAST + 1;
      const id = PICKS[j];
      c.classList.toggle("on", on);
      const p = bankPos(id);
      c.style.left = (on ? slotX(j) : p.x) + "px";
      c.style.top = (on ? SY : p.y) + "px";
      c.style.width = (on ? SW : BW) + "px";
      c.style.height = (on ? SH : BH) + "px";
    });
    ops.forEach((o, j) => o.classList.toggle("on", step >= 3 && j < reduce));

    if (step >= 3) {
      bracket.style.opacity = 1;
      bracket.style.left = slotX(0) - 9 + "px";
      bracket.style.top = SY - 9 + "px";
      bracket.style.width = slotX(reduce) + SW - slotX(0) + 14 + "px";
      bracket.style.height = SH + 14 + "px";
    } else {
      bracket.style.opacity = 0;
    }

    expr.innerHTML = step >= 2 ? exprHtml(reduce) : "";
    expr.style.opacity = step >= 2 ? 1 : 0;

    const hideAll = step >= LAST;
    const showRight = step >= 2 && !hideAll;
    [rp].forEach((e) => e.classList.toggle("hidden", !showRight));
    heat.classList.toggle("hidden", !(step >= 6 && !hideAll));
    heatCap.classList.toggle("hidden", !(step >= 6 && !hideAll));
    if (hideAll) {
      clones.forEach((c) => c.classList.remove("on"));
      ops.forEach((o) => o.classList.remove("on"));
      bracket.style.opacity = 0;
      expr.style.opacity = 0;
    }
    more.classList.toggle("hidden", !hideAll);

    // Narration
    narr.innerHTML = `${NARR[step][0]}<small>${NARR[step][1]}</small>`;
    rpTitle.textContent = PANEL_TITLE(step, reduce);
    heatCap.innerHTML =
      `<span class="m">K* = κ*(tᵢ, tⱼ)</span> for t₁…t₁₀₂₄ ∈ [0, 1]<br>` +
      `<span class="tiny">64×64 subsample of the 1024×1024 matrix</span>` +
      (step >= 7 ? `<br><span class="m">x = L z,&nbsp; z ∼ 𝒩(0, I)&nbsp; ⇒&nbsp; x ∼ 𝒩(0, K*)</span>` : "");

    // Series
    if (step >= 2) {
      const wasFinal = finalMode;
      finalMode = step >= 7;
      setSeries(D.demo.stages[reduce].series, animate && step <= 5);
      if (finalMode && !wasFinal && step === 7 && prev === 6) {
        strokeP = 0;
        strokeT0 = performance.now();
        stroking = true;
        requestAnimationFrame(tick);
      } else {
        strokeP = 1;
        stroking = false;
        redraw();
      }
    }
    prev = step;
  }

  function sync() {
    render(sec.querySelectorAll(".fragment.visible").length);
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
