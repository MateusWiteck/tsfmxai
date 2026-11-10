/* Shared canvas helpers plus the cover and Gaussian-process slides. */
(function () {
  const D = window.KS_DATA;
  const FAMCOL = {
    periodic: "#0f766e", linear: "#c2410c", rbf: "#6d5bd0",
    rq: "#be185d", white: "#5b6475", constant: "#8a6a1f",
  };
  const FAMNAME = {
    periodic: "Periodic", linear: "Linear", rbf: "RBF",
    rq: "RatQuad", white: "White", constant: "Constant",
  };
  const HEAT_STOPS = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"];

  function hex(h) {
    return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  function heatColor(t) {
    t = Math.max(0, Math.min(1, t));
    const x = t * (HEAT_STOPS.length - 1);
    const i = Math.min(HEAT_STOPS.length - 2, Math.floor(x));
    const a = hex(HEAT_STOPS[i]);
    const b = hex(HEAT_STOPS[i + 1]);
    const f = x - i;
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  }

  function setupCanvas(c, w, h) {
    const s = 2;
    c.width = w * s;
    c.height = h * s;
    c.style.width = w + "px";
    c.style.height = h + "px";
    const ctx = c.getContext("2d");
    ctx.scale(s, s);
    return ctx;
  }

  /** Polyline of ys inside the box (x0,y0,w,h); only the first `upto` fraction is drawn. */
  function drawLine(ctx, ys, x0, y0, w, h, ymin, ymax, color, lw, upto) {
    const n = ys.length;
    const last = Math.floor((n - 1) * (upto === undefined ? 1 : upto));
    ctx.beginPath();
    for (let i = 0; i <= last; i++) {
      const x = x0 + (w * i) / (n - 1);
      const y = y0 + h - ((ys[i] - ymin) / (ymax - ymin)) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function drawHeat(ctx, m, size) {
    const n = m.length;
    let lo = Infinity;
    for (const r of m) for (const v of r) lo = Math.min(lo, v);
    const cell = size / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = heatColor((m[i][j] - lo) / (1 - lo || 1));
        ctx.fillRect(j * cell, i * cell, cell + 0.6, cell + 0.6);
      }
    }
  }

  window.TSX = { FAMCOL, FAMNAME, heatColor, setupCanvas, drawLine, drawHeat };
  if (!D) return;

  /* ---------- Cover: slowly drifting kernel-synth curves ---------- */
  const cover = document.getElementById("cover-canvas");
  if (cover) {
    const W = 1200, H = 640;
    const ctx = setupCanvas(cover, W, H);
    const curves = D.more.slice(0, 3).map((m) => m.series);
    const cols = ["rgba(29,78,216,0.22)", "rgba(15,118,110,0.32)", "rgba(194,65,12,0.55)"];
    let off = 0;
    function paint() {
      ctx.clearRect(0, 0, W, H);
      curves.forEach((ys, k) => {
        const span = 520;
        const shift = Math.floor(off * (1 + k * 0.35)) % (ys.length - span);
        const seg = ys.slice(shift, shift + span);
        const lim = Math.max(2.2, Math.max(...seg.map(Math.abs)));
        drawLine(ctx, seg, 0, 455 + k * 8, W, 150, -lim, lim, cols[k], 2.4 - k * 0.3);
      });
    }
    function loop() {
      const cur = window.Reveal && Reveal.getCurrentSlide && Reveal.getCurrentSlide();
      if (!cur || cur.id === "cover") {
        off += 0.25;
        paint();
      }
      requestAnimationFrame(loop);
    }
    paint();
    loop();
  }

  /* ---------- Gaussian process primer ---------- */
  document.querySelectorAll("#gp .gpcol").forEach((col, i) => {
    const ex = D.gp[i];
    if (!ex) return;
    const heat = col.querySelector("canvas.heat");
    const draws = col.querySelector("canvas.draws");
    drawHeat(setupCanvas(heat, 150, 150), ex.cov, 150);
    const ctx = setupCanvas(draws, 372, 150);
    const lim = Math.max(2.2, ...ex.draws.flatMap((d) => d.map(Math.abs)));
    const colors = ["#1d4ed8", "#c2410c"];
    ex.draws.forEach((d, k) => drawLine(ctx, d, 6, 8, 360, 134, -lim, lim, colors[k], 2.2));
  });
})();
