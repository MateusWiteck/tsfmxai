/* Shared helpers for the "zoom" slides (generator, equation head).
 *
 * Each zoom slide is a canvas painted by paint(ctx, step, p): `step` is the
 * number of visible fragments and p in [0, 1] the progress of the current
 * step. The painter is stateless, so going back or landing on a step through
 * the URL hash always draws the same picture. ?static draws every step at p=1.
 */
(function () {
  const C = (window.ZC = {});
  const col = {
    ink: "#1f2430", muted: "#5b6475", blue: "#1d4ed8", ours: "#c2410c", teal: "#0f766e",
    line: "#d9d6cc", soft: "#e3eafc", osoft: "#fdebdc", tsoft: "#dcf1ee", panel: "#ffffff", grey: "#eceae3",
  };
  C.col = col;
  C.clamp01 = (v) => Math.max(0, Math.min(1, v));
  C.seg = (p, a, b) => C.clamp01((p - a) / (b - a));
  C.ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  const SANS = "Segoe UI, system-ui, sans-serif";
  C.SANS = SANS;
  C.MATH = "Cambria Math, Segoe UI Symbol, serif";

  C.text = (ctx, s, x, y, size, color, o) => {
    o = o || {};
    ctx.font = `${o.weight || 400} ${size}px ${o.family || SANS}`;
    ctx.fillStyle = color || col.ink;
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.base || "alphabetic";
    ctx.fillText(s, x, y);
  };
  C.measure = (ctx, s, size, weight, family) => {
    ctx.font = `${weight || 400} ${size}px ${family || SANS}`;
    return ctx.measureText(s).width;
  };
  /** Word-wrapped text; returns the y after the last line. */
  C.wrap = (ctx, s, x, y, w, lh, size, color, o) => {
    o = o || {};
    ctx.font = `${o.weight || 400} ${size}px ${o.family || SANS}`;
    let line = "", yy = y;
    s.split(" ").forEach((word) => {
      const t = line ? line + " " + word : word;
      if (ctx.measureText(t).width > w && line) {
        C.text(ctx, line, x, yy, size, color, o);
        line = word;
        yy += lh;
      } else line = t;
    });
    if (line) C.text(ctx, line, x, yy, size, color, o);
    return yy + lh;
  };
  C.rrect = (ctx, x, y, w, h, r) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };
  C.box = (ctx, x, y, w, h, o) => {
    o = o || {};
    C.rrect(ctx, x, y, w, h, o.r === undefined ? 10 : o.r);
    ctx.fillStyle = o.fill || col.panel;
    ctx.fill();
    ctx.lineWidth = o.lw || 2;
    ctx.strokeStyle = o.stroke || col.line;
    ctx.setLineDash(o.dash || []);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  C.arrow = (ctx, x1, y1, x2, y2, color, lw, frac) => {
    frac = frac === undefined ? 1 : frac;
    const ex = x1 + (x2 - x1) * frac, ey = y1 + (y2 - y1) * frac;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw || 2.2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    if (frac > 0.98) {
      const a = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 11 * Math.cos(a - 0.4), y2 - 11 * Math.sin(a - 0.4));
      ctx.lineTo(x2 - 11 * Math.cos(a + 0.4), y2 - 11 * Math.sin(a + 0.4));
      ctx.closePath();
      ctx.fill();
    }
  };
  /** Row of numbered stage chips; active = index, or -1 (none), or n (all done). */
  C.chips = (ctx, labels, active, y) => {
    const gap = 12, h = 34, x0 = 20, w = (1160 - (labels.length - 1) * gap) / labels.length;
    labels.forEach((s, i) => {
      const x = x0 + i * (w + gap);
      const on = i === active, done = i < active || active >= labels.length;
      const fs = labels.length > 5 ? 16 : 18;
      C.box(ctx, x, y, w, h, {
        r: 17, lw: 2, fill: on ? col.blue : done ? col.tsoft : col.panel,
        stroke: on ? col.blue : done ? col.teal : col.line,
      });
      C.text(ctx, s, x + w / 2, y + 23, fs, on ? "#fff" : done ? col.teal : col.muted, { align: "center", weight: on ? 700 : 600 });
    });
  };

  const MINUS = "−";
  C.fmt = (v, d) => (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d === undefined ? 3 : d);
  C.lagName = (lag) => `x(t${MINUS}${lag})`;

  /** Pretty-print a tree (tanh nodes included) as a formula. */
  C.pretty = (n, d) => {
    if (n.t === "lag") return C.lagName(n.lag);
    if (n.t === "par") return C.fmt(n.v, d === undefined ? 3 : d);
    if (n.t === "tanh") return "tanh(" + C.pretty(n.k[0], d) + ")";
    const part = (c) => (n.t === "mul" && c.t === "add" ? "(" + C.pretty(c, d) + ")" : C.pretty(c, d));
    return n.t === "add" ? part(n.k[0]) + " + " + part(n.k[1]) : part(n.k[0]) + "·" + part(n.k[1]);
  };
  C.polyText = (terms, d) => terms.map((t) => {
    const atoms = t.atoms.map((a) => (a[2] === "tanh" ? "tanh " : "") + C.lagName(a[1]));
    const c = C.fmt(t.coef, d === undefined ? 3 : d);
    return atoms.length ? (Math.abs(t.coef - 1) < 1e-12 ? atoms.join("·") : c + "·" + atoms.join("·")) : c;
  });

  /** Drop tanh wrappers, remembering them as wrap = true on the lag leaf. */
  C.collapse = (n) => {
    if (n.t === "tanh") {
      const c = C.collapse(n.k[0]);
      c.wrap = true;
      return c;
    }
    const o = Object.assign({}, n);
    if (n.k) o.k = n.k.map(C.collapse);
    return o;
  };

  /** Top-down layout: leaves evenly spaced, parents centred over their children. */
  C.layout = (tree, box) => {
    const nodes = [];
    let leaves = 0, maxDepth = 0;
    const visit = (n, depth, parent) => {
      const o = { n, depth, parent, kids: [], id: nodes.length };
      nodes.push(o);
      maxDepth = Math.max(maxDepth, depth);
      if (n.k) o.kids = n.k.map((k) => visit(k, depth + 1, o));
      else o.leaf = leaves++;
      return o;
    };
    visit(tree, 0, null);
    const place = (o) => {
      o.kids.forEach(place);
      o.x = o.kids.length ? o.kids.reduce((s, k) => s + k.x, 0) / o.kids.length : box.x + (o.leaf + 0.5) * box.w / leaves;
      o.y = box.y + (maxDepth ? (o.depth * box.h) / maxDepth : 0);
    };
    place(nodes[0]);
    return { nodes, leaves, maxDepth };
  };

  /**
   * o.label(node) -> string; o.alpha(node) -> 0..1; o.style(node) -> {fill, stroke, color}.
   * Operators are circles, leaves are rounded boxes.
   */
  C.drawTree = (ctx, lay, o) => {
    const alpha = o.alpha || (() => 1);
    lay.nodes.forEach((nd) => {
      const a = alpha(nd);
      if (!nd.parent || a <= 0) return;
      ctx.globalAlpha = a;
      ctx.strokeStyle = col.line;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(nd.parent.x, nd.parent.y);
      ctx.lineTo(nd.x, nd.y);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
    lay.nodes.forEach((nd) => {
      const a = alpha(nd);
      if (a <= 0) return;
      ctx.globalAlpha = a;
      const st = o.style(nd) || {};
      const s = o.label(nd);
      if (nd.kids.length) {
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, 19, 0, Math.PI * 2);
        ctx.fillStyle = st.fill || col.panel;
        ctx.fill();
        ctx.lineWidth = st.lw || 2.4;
        ctx.strokeStyle = st.stroke || col.ink;
        ctx.stroke();
        C.text(ctx, s, nd.x, nd.y + 8, 24, st.color || col.ink, { align: "center", weight: 700 });
      } else {
        const size = o.size || 17;
        const w = C.measure(ctx, s, size, 600) + 20, h = 32;
        C.box(ctx, nd.x - w / 2, nd.y - h / 2, w, h, { r: 8, lw: st.lw || 2, fill: st.fill || col.panel, stroke: st.stroke || col.line });
        C.text(ctx, s, nd.x, nd.y + 6, size, st.color || col.ink, { align: "center", weight: 600 });
      }
      if (o.after) o.after(nd, a);
    });
    ctx.globalAlpha = 1;
  };

  /**
   * cfg: {id, W, H, kicker, steps: [{narr, sub, dur, foot}], foot, paint}
   * The section holds a .zoom div and one empty .fragment.ks-frag per step after the first.
   */
  C.mount = (cfg) => {
    const sec = document.getElementById(cfg.id);
    const T = window.TSX;
    if (!sec || !T) return;
    const root = sec.querySelector(".zoom");
    const staticMode = /[?&]static\b/.test(location.search);
    const mk = (tag, cls, html) => {
      const e = document.createElement(tag);
      e.className = cls;
      if (html !== undefined) e.innerHTML = html;
      root.appendChild(e);
      return e;
    };
    const narr = mk("div", "narr");
    const canvas = mk("canvas", "zc");
    const foot = mk("div", "foot");
    const ctx = T.setupCanvas(canvas, cfg.W, cfg.H);
    let step = -1, start = 0, raf = 0;

    const frame = (now) => {
      raf = 0;
      const dur = cfg.steps[step].dur || 3000;
      const p = staticMode ? 1 : C.clamp01((now - start) / dur);
      ctx.clearRect(0, 0, cfg.W, cfg.H);
      cfg.paint(ctx, step, p);
      if (p < 1) raf = requestAnimationFrame(frame);
    };
    const sync = () => {
      const s = Math.min(cfg.steps.length - 1, sec.querySelectorAll(".fragment.visible").length);
      if (s === step && raf) return;
      if (s !== step) {
        step = s;
        start = performance.now();
        const st = cfg.steps[step];
        narr.innerHTML = `${st.narr}<small>${st.sub || ""}</small>`;
        foot.textContent = st.foot || cfg.foot || "";
      }
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    };
    // Optional clickable regions: cfg.hits(step) -> [{x, y, w, h, key}] in canvas units.
    const hitAt = (e) => {
      if (!cfg.hits || step < 0) return null;
      const r = canvas.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * cfg.W, y = ((e.clientY - r.top) / r.height) * cfg.H;
      return cfg.hits(step).find((h) => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) || null;
    };
    if (cfg.hits) {
      canvas.addEventListener("mousemove", (e) => { canvas.style.cursor = hitAt(e) ? "pointer" : ""; });
      canvas.addEventListener("click", (e) => { const h = hitAt(e); if (h) cfg.onHit(h.key); });
    }
    const bind = () => {
      Reveal.on("fragmentshown", sync);
      Reveal.on("fragmenthidden", sync);
      Reveal.on("slidechanged", sync);
      sync();
    };
    if (window.Reveal && Reveal.isReady && Reveal.isReady()) bind();
    else Reveal.on("ready", bind);
  };
})();
