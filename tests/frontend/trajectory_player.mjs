// Run with: node tests/frontend/trajectory_player.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const elements = new Map();
const timers = new Map();
const coordinates = [];
let nextTimer = 1;
const drawing = new Proxy({}, {
  get: (_, name) => (...args) => {
    if (["moveTo", "lineTo", "arc", "fillRect"].includes(name)) {
      assert.ok(args.every(Number.isFinite), `${name}: non-finite canvas coordinate`);
      coordinates.push(args);
    }
  },
  set: () => true,
});
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id === "trajectory-speed" ? "1500" : "linear",
    textContent: "", disabled: false, open: false,
    addEventListener() {}, setAttribute() {},
    getBoundingClientRect: () => ({ width: 900, height: 530 }),
    getContext: () => drawing,
  });
  return elements.get(id);
}
const sandbox = vm.createContext({
  document: { getElementById: element, addEventListener() {} },
  window: {
    devicePixelRatio: 1, addEventListener() {},
    clearInterval: (id) => timers.delete(id),
    setInterval: (callback, delay) => { const id = nextTimer++; timers.set(id, { callback, delay }); return id; },
  },
});
const source = readFileSync(new URL("../../frontend/static/app.js", import.meta.url), "utf8");
vm.runInContext(source.replace("generateMechanism().then(generateTrajectories);", ""), sandbox);
const run = (code) => vm.runInContext(code, sandbox);
const data = {
  times: [0, 1, 2], generated_start: 1,
  // Deliberately no ensemble summary: the selected charts must use individual values.
  trajectories: [
    [[-1e308, 0, 1e308], [1, 0, -1]],
    [[0, 1e-300, 2e-300], [0, 1, 2]],
    [[4, 4, 4], [8, 8, 8]],
  ],
};
sandbox.input = data;
const snapshot = JSON.stringify(data);
run("state.trajectories = input; renderTrajectories(input)");
assert.equal(element("trajectory-position").textContent, "Trajetória 1 de 3");
assert.equal(element("trajectory-previous").disabled, true);
run("playTrajectories()");
assert.equal([...timers.values()][0].delay, 1500);
[...timers.values()][0].callback();
assert.equal(element("trajectory-position").textContent, "Trajetória 2 de 3");
run("pausePlayback()");
assert.equal(timers.size, 0);
run("playTrajectories()");
[...timers.values()][0].callback();
assert.equal(element("trajectory-position").textContent, "Trajetória 3 de 3");
assert.equal(timers.size, 0);
assert.equal(element("trajectory-next").disabled, true);
run("playTrajectories()");
assert.equal(element("trajectory-position").textContent, "Trajetória 1 de 3");
run("pausePlayback(); selectTrajectory(-5)");
assert.equal(run("state.activeTrajectory"), 0);
run("selectTrajectory(99)");
assert.equal(run("state.activeTrajectory"), 2);
element("trajectory-scale").value = "symlog";
run("selectTrajectory(0)");
assert.ok(coordinates.length > 0);
assert.equal(JSON.stringify(data), snapshot);
run("state.trajectories = { ...input, trajectories: [input.trajectories[0]] }; selectTrajectory(0); playTrajectories()");
assert.equal(timers.size, 0);
assert.equal(element("trajectory-play").disabled, true);
run("state.trajectories = null; clearCanvases()");
assert.equal(element("trajectory-select").disabled, true);
element("stability-method").value = "shao_wu";
element("innovation").value = "0";
run("updateStabilityControls()");
assert.equal(element("innovation").value, "0.05");
assert.equal(element("require-lag").disabled, true);
assert.equal(element("initial-min").disabled, true);
element("stability-method").value = "stationary_var1";
run("updateStabilityControls()");
assert.equal(element("max-lag").disabled, true);
assert.equal(element("max-operators").disabled, true);
element("stability-method").value = "none";
run("updateStabilityControls()");
assert.equal(element("initial-min").disabled, false);
assert.equal(element("max-lag").disabled, false);
assert.equal(element("require-lag").disabled, false);
console.log("PASS: individual charts, finite extreme-value coordinates, playback, pause, end/restart, bounds, single/empty batches, unchanged data.");
