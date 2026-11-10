/* Small figures: the "what the forecaster receives" slide and the explainability slide. */
(function () {
  const D = window.KS_DATA;
  const T = window.TSX;
  if (!D || !T) return;

  // Slide: what the pretraining sees -- one KernelSynth series split into context and future.
  const split = document.getElementById("split-canvas");
  if (split) {
    const W = 560, H = 300;
    const ctx = T.setupCanvas(split, W, H);
    const ys = D.demo.stages[D.demo.R - 1].series;
    const cut = 768;
    const lim = Math.max(2.5, ...ys.map(Math.abs)) * 1.05;
    const x = (i) => 10 + ((W - 20) * i) / (ys.length - 1);
    ctx.fillStyle = "rgba(194,65,12,0.08)";
    ctx.fillRect(x(cut), 6, W - 10 - x(cut), H - 12);
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "#c2410c";
    ctx.beginPath();
    ctx.moveTo(x(cut), 6);
    ctx.lineTo(x(cut), H - 6);
    ctx.stroke();
    ctx.setLineDash([]);
    T.drawLine(ctx, ys.slice(0, cut + 1), 10, 14, x(cut) - 10, H - 28, -lim, lim, "#1d4ed8", 2.2);
    const fut = ys.slice(cut);
    T.drawLine(ctx, fut, x(cut), 14, W - 10 - x(cut), H - 28, -lim, lim, "#c2410c", 2.2);
    ctx.font = "600 14px Segoe UI, sans-serif";
    ctx.fillStyle = "#1d4ed8";
    ctx.fillText("context", 16, 22);
    ctx.fillStyle = "#c2410c";
    ctx.fillText("future to predict", x(cut) + 8, 22);
  }

  // Slide: explainability -- a forecast with a question mark.
  const xai = document.getElementById("xai-canvas");
  if (xai) {
    const W = 560, H = 300;
    const ctx = T.setupCanvas(xai, W, H);
    const ys = D.more[2].series;
    const seg = ys.slice(100, 100 + 400);
    const cut = 300;
    const lim = Math.max(2.5, ...seg.map(Math.abs)) * 1.05;
    const x = (i) => 10 + ((W - 20) * i) / (seg.length - 1);
    T.drawLine(ctx, seg.slice(0, cut + 1), 10, 14, x(cut) - 10, H - 28, -lim, lim, "#1d4ed8", 2.2);
    // Forecast band around a smoothed continuation of the last window.
    const fut = seg.slice(cut);
    const px = (i, v) => [x(cut + i), 14 + (H - 28) - ((v + lim) / (2 * lim)) * (H - 28)];
    ctx.beginPath();
    fut.forEach((v, i) => { const [a, b] = px(i, v + 0.35 + i * 0.004); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    for (let i = fut.length - 1; i >= 0; i--) { const [a, b] = px(i, fut[i] - 0.35 - i * 0.004); ctx.lineTo(a, b); }
    ctx.closePath();
    ctx.fillStyle = "rgba(194,65,12,0.16)";
    ctx.fill();
    T.drawLine(ctx, fut, x(cut), 14, W - 10 - x(cut), H - 28, -lim, lim, "#c2410c", 2.4);
    ctx.font = "700 64px Segoe UI, sans-serif";
    ctx.fillStyle = "rgba(31,36,48,0.78)";
    ctx.textAlign = "center";
    ctx.fillText("?", x(cut) + (W - 10 - x(cut)) / 2, 150);
    ctx.font = "600 14px Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = "#1d4ed8";
    ctx.fillText("observed", 16, 22);
    ctx.fillStyle = "#c2410c";
    ctx.fillText("forecast", x(cut) + 8, 22);
  }
})();
