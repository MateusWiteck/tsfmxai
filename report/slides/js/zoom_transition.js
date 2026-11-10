/* Zoom-in animation: the generator / equation-head slide grows out of the node on the
 * proposal slide (slide 12) that links to it, and shrinks back into it when you return.
 * Runs on slide changes only; deep links and ?static draw the slide directly.
 */
(function () {
  if (!window.Reveal || /[?&]static\b/.test(location.search)) return;
  const EASE = "cubic-bezier(.25,.8,.25,1)";
  const rect = (el) => el.getBoundingClientRect();
  const nodeFor = (slide, id) => {
    const link = slide.querySelector(`a[href="#/${id}"]`);
    return link && link.closest(".node");
  };
  // Transform that maps `big` onto `small`, both viewport rects (uniform scale, centred).
  const into = (big, small) => {
    const s = Math.min(small.width / big.width, small.height / big.height);
    const dx = small.left + small.width / 2 - (big.left + big.width / 2);
    const dy = small.top + small.height / 2 - (big.top + big.height / 2);
    return `translate(${dx}px, ${dy}px) scale(${s})`;
  };

  function expand(from, to) {
    const node = nodeFor(from, to.id), content = to.querySelector(".canvas");
    if (!node || !content) return;
    const n = rect(node), c = rect(content);
    const ghost = document.createElement("div");
    Object.assign(ghost.style, {
      position: "fixed", left: n.left + "px", top: n.top + "px", width: n.width + "px", height: n.height + "px",
      border: "2px solid #c2410c", borderRadius: "12px", background: "#fffdf8", zIndex: 40, pointerEvents: "none", transformOrigin: "0 0",
    });
    document.body.appendChild(ghost);
    const sx = c.width / n.width, sy = c.height / n.height;
    ghost.animate(
      [{ transform: "none", opacity: 1 }, { transform: `translate(${c.left - n.left}px, ${c.top - n.top}px) scale(${sx}, ${sy})`, opacity: 0.0 }],
      { duration: 650, easing: EASE }
    ).onfinish = () => ghost.remove();
    content.animate(
      [{ transform: into(c, n), opacity: 0, offset: 0 }, { opacity: 0, offset: 0.25 }, { transform: "none", opacity: 1, offset: 1 }],
      { duration: 650, easing: EASE }
    );
  }

  function collapse(from, to) {
    const node = nodeFor(to, from.id), content = from.querySelector(".canvas");
    if (!node || !content) return;
    const n = rect(node), c = rect(content);
    content.animate(
      [{ transform: "none", opacity: 1, offset: 0 }, { opacity: 0, offset: 0.75 }, { transform: into(c, n), opacity: 0, offset: 1 }],
      { duration: 450, easing: EASE }
    );
  }

  Reveal.on("slidechanged", (e) => {
    const from = e.previousSlide, to = e.currentSlide;
    if (!from || !to || from === to) return;
    if (from.id === "proposal" && /^zoom-/.test(to.id)) expand(from, to);
    else if (to.id === "proposal" && /^zoom-/.test(from.id)) collapse(from, to);
  });
})();
