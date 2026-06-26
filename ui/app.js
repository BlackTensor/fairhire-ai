/* FairHire AI — Phase 5 UI logic.
 *
 * Everything runs client-side: we load the exported weights (model.json) and
 * precomputed evaluation data (data.json), reproduce the encoder + hiring
 * predictor forward pass in plain JS (verified to match PyTorch in
 * src/export_ui.py), and drive the live demo, the signature latent animation,
 * and the four evidence charts.
 */

"use strict";

const TOKENS = {
  bg: "#13151C", surface: "#1C1F28", text: "#E8EAF0", muted: "#8A90A0",
  grid: "#2A2F3A", accentA: "#A855F7", accentB: "#F59E0B", coral: "#E8614A",
};
const REDUCED_MOTION =
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let MODEL = null;
let DATA = null;

// --------------------------------------------------------------------------- //
// Forward pass (mirrors src/export_ui.py::_np_score)
// --------------------------------------------------------------------------- //
function linear(x, l) {
  const W = l.W, b = l.b, out = new Array(b.length);
  for (let o = 0; o < b.length; o++) {
    let s = b[o];
    const row = W[o];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}
function batchnorm(x, bn) {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = (x[i] - bn.mean[i]) / Math.sqrt(bn.var[i] + bn.eps)
      * bn.weight[i] + bn.bias[i];
  }
  return out;
}
function relu(x) { return x.map((v) => (v > 0 ? v : 0)); }

function score(model, x) {
  const e = model.encoder;
  let h = relu(batchnorm(linear(x, e.l1), e.bn1));
  h = relu(batchnorm(linear(h, e.l2), e.bn2));
  const z = linear(h, e.l3);
  const p = model.predictor;
  const logits = linear(relu(linear(z, p.l1)), p.l2);
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m), e1 = Math.exp(logits[1] - m);
  return e1 / (e0 + e1); // P(income > 50K)
}

// Build the 83-dim feature vector from the form (standardize + one-hot). fnlwgt
// (census sampling weight, not user-meaningful) is held at its mean -> 0.
function buildFeatureVector(values) {
  const x = [];
  const cont = MODEL.continuous_features;
  for (let i = 0; i < cont.length; i++) {
    const name = cont[i];
    const raw = name === "fnlwgt" ? MODEL.scaler.mean[i] : values[name];
    x.push((raw - MODEL.scaler.mean[i]) / MODEL.scaler.scale[i]);
  }
  for (const col of MODEL.categorical_features) {
    const cats = MODEL.categories[col];
    // Fields the form omits (e.g. the gender-proxy 'relationship') fall back
    // to their data-derived mode so every candidate is scored identically.
    const sel = col in values ? values[col] : MODEL.silent_defaults[col];
    for (const c of cats) x.push(c === sel ? 1 : 0);
  }
  return x;
}

// --------------------------------------------------------------------------- //
// Live demo
// --------------------------------------------------------------------------- //
// Fields the candidate form exposes. 'relationship' (household role) is
// intentionally omitted — it is a direct gender proxy, so it is held at its
// data-derived mode (MODEL.silent_defaults) during inference instead.
const FIELD_IDS = ["education_num", "hours_per_week", "capital_gain",
  "capital_loss", "occupation", "workclass", "marital_status",
  "native_country"];

function readForm() {
  const v = {};
  for (const id of FIELD_IDS) {
    const el = document.getElementById(id);
    v[id] = el.type === "range" || el.type === "number"
      ? Number(el.value) : el.value;
  }
  return v;
}

function updateDemo() {
  const values = readForm();
  const x = buildFeatureVector(values);
  const w = MODEL.gender_axis.w, step = MODEL.gender_axis.step;
  const xMale = x.map((v, i) => v + w[i] * step);
  const xFemale = x.map((v, i) => v - w[i] * step);

  let swings = {};
  for (const key of ["baseline", "sanitized"]) {
    const m = MODEL.models[key];
    const p = score(m, x);
    const pct = (p * 100).toFixed(1) + "%";
    document.querySelector(`[data-val="${key}"]`).textContent = pct;
    document.querySelector(`[data-fill="${key}"]`).style.width =
      (p * 100) + "%";

    const swing = Math.abs(score(m, xMale) - score(m, xFemale)) * 100;
    swings[key] = swing;
    const el = document.querySelector(`[data-swing="${key}"]`);
    el.textContent = (swing >= 0 ? "" : "") + swing.toFixed(1) + " pts";
  }

  const caveat = document.getElementById("swing-caveat");
  if (swings.sanitized > swings.baseline + 0.05) {
    caveat.textContent = "As the paper reports honestly: the sanitized model "
      + "is the more gender-sensitive of the two here. Scrubbing the latent is "
      + "not a counterfactually invariant score.";
  } else {
    caveat.textContent = "For this particular candidate the swings are close; "
      + "across the full test set the sanitized model swings more on average.";
  }
}

function buildForm() {
  // Selects
  for (const col of MODEL.categorical_features) {
    const sel = document.getElementById(col);
    if (!sel) continue;
    for (const c of MODEL.categories[col]) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c.replace(/-/g, " ");
      sel.appendChild(opt);
    }
  }
  // Sensible defaults for a relatable demo candidate.
  setSelect("occupation", "Prof-specialty");
  setSelect("workclass", "Private");
  setSelect("marital_status", "Married-civ-spouse");
  setSelect("native_country", "United-States");

  // Range outputs
  for (const id of ["education_num", "hours_per_week"]) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "_out");
    const sync = () => { out.textContent = el.value; };
    el.addEventListener("input", sync);
    sync();
  }

  for (const id of FIELD_IDS) {
    document.getElementById(id).addEventListener("input", updateDemo);
  }
}
function setSelect(id, val) {
  const el = document.getElementById(id);
  if (el && [...el.options].some((o) => o.value === val)) el.value = val;
}

// --------------------------------------------------------------------------- //
// Hero stats
// --------------------------------------------------------------------------- //
function fillHeroStats() {
  const acc = DATA.accuracy;
  const accRetained = (acc.after / acc.before) * 100;
  const genRemoved =
    (1 - DATA.leakage.after[0] / DATA.leakage.before[0]) * 100;
  const tprClosed =
    (1 - DATA.fairness.after[1] / DATA.fairness.before[1]) * 100;
  set("acc", accRetained.toFixed(1) + "%");
  set("genleak", genRemoved.toFixed(0) + "%");
  set("tpr", tprClosed.toFixed(0) + "%");
  function set(k, v) {
    document.querySelector(`[data-stat="${k}"]`).textContent = v;
  }
}

// --------------------------------------------------------------------------- //
// Signature latent animation (custom canvas morph: before -> after)
// --------------------------------------------------------------------------- //
let latentT = 0;       // 0 = before, 1 = after
let latentRAF = null;
let latentCtx = null;
let latentPts = null;  // [{bx,by,ax,ay,g}]

function setupLatent() {
  const canvas = document.getElementById("latent-canvas");
  latentCtx = canvas.getContext("2d");
  const t = DATA.tsne;
  latentPts = t.before.map((b, i) => ({
    bx: b[0], by: b[1], ax: t.after[i][0], ay: t.after[i][1], g: t.gender[i],
  }));
  document.querySelector('[data-legend="0"]').textContent =
    t.gender_classes[0];
  document.querySelector('[data-legend="1"]').textContent =
    t.gender_classes[1];

  const slider = document.getElementById("latent-slider");
  slider.addEventListener("input", () => {
    stopLatent();
    latentT = Number(slider.value) / 1000;
    drawLatent();
    syncStateLabels();
  });
  document.getElementById("latent-play").addEventListener("click", playLatent);

  window.addEventListener("resize", () => { sizeCanvas(); drawLatent(); });
  sizeCanvas();

  if (REDUCED_MOTION) {
    latentT = 1; slider.value = 1000; drawLatent(); syncStateLabels();
  } else {
    drawLatent();
  }
}

function sizeCanvas() {
  const canvas = document.getElementById("latent-canvas");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  latentCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function drawLatent() {
  const canvas = document.getElementById("latent-canvas");
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const pad = 28;
  const sx = (W - 2 * pad) / 2, sy = (H - 2 * pad) / 2;
  const cx = W / 2, cy = H / 2;
  const e = easeInOut(latentT);

  latentCtx.clearRect(0, 0, W, H);
  for (const p of latentPts) {
    const x = p.bx * (1 - e) + p.ax * e;
    const y = p.by * (1 - e) + p.ay * e;
    const px = cx + x * sx, py = cy + y * sy;
    latentCtx.beginPath();
    latentCtx.arc(px, py, 2.6, 0, Math.PI * 2);
    latentCtx.fillStyle = p.g === 0 ? TOKENS.accentA : TOKENS.accentB;
    latentCtx.globalAlpha = 0.62;
    latentCtx.fill();
  }
  latentCtx.globalAlpha = 1;
}

function syncStateLabels() {
  const before = document.querySelector('[data-state="before"]');
  const after = document.querySelector('[data-state="after"]');
  before.classList.toggle("active", latentT < 0.5);
  after.classList.toggle("active", latentT >= 0.5);
}

function stopLatent() {
  if (latentRAF) { cancelAnimationFrame(latentRAF); latentRAF = null; }
}

function playLatent() {
  stopLatent();
  if (REDUCED_MOTION) {
    latentT = 1;
    document.getElementById("latent-slider").value = 1000;
    drawLatent(); syncStateLabels();
    return;
  }
  const forward = latentT < 0.5; // toggle direction each press
  const dur = 2600;
  const from = latentT, to = forward ? 1 : 0;
  const t0 = performance.now();
  const slider = document.getElementById("latent-slider");
  const tick = (now) => {
    const k = Math.min((now - t0) / dur, 1);
    latentT = from + (to - from) * k;
    slider.value = Math.round(latentT * 1000);
    drawLatent();
    syncStateLabels();
    if (k < 1) latentRAF = requestAnimationFrame(tick);
    else latentRAF = null;
  };
  latentRAF = requestAnimationFrame(tick);
}

// --------------------------------------------------------------------------- //
// Charts (Chart.js)
// --------------------------------------------------------------------------- //
function chartDefaults() {
  Chart.defaults.color = TOKENS.muted;
  Chart.defaults.borderColor = TOKENS.grid;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.animation = REDUCED_MOTION ? false : { duration: 700 };
}

const GRID_OPTS = {
  grid: { color: TOKENS.grid, drawTicks: false },
  ticks: { color: TOKENS.muted },
};

function beforeAfterBar(canvasId, labels, before, after, opts) {
  const fmt = opts.fmt || ((v) => v.toFixed(1));
  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Before", data: before, backgroundColor: opts.beforeColor,
          borderRadius: 5, categoryPercentage: 0.62, barPercentage: 0.82 },
        { label: "After", data: after, backgroundColor: opts.afterColor,
          borderRadius: 5, categoryPercentage: 0.62, barPercentage: 0.82 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ...GRID_OPTS, grid: { display: false } },
        y: { ...GRID_OPTS, beginAtZero: true, max: opts.max,
          ticks: { color: TOKENS.muted, callback: (v) => fmt(v) } },
      },
      plugins: {
        legend: { labels: { color: TOKENS.text, usePointStyle: true,
          pointStyle: "rectRounded" } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}` } },
      },
    },
  });
}

function buildCharts() {
  chartDefaults();

  beforeAfterBar("chart-accuracy", ["Hiring accuracy"],
    [DATA.accuracy.before * 100], [DATA.accuracy.after * 100],
    { beforeColor: TOKENS.muted, afterColor: TOKENS.accentA, max: 100,
      fmt: (v) => v.toFixed(1) + "%" });

  beforeAfterBar("chart-leakage",
    DATA.leakage.attrs.map((a) => a[0].toUpperCase() + a.slice(1)),
    DATA.leakage.before.map((v) => v * 100),
    DATA.leakage.after.map((v) => v * 100),
    { beforeColor: TOKENS.coral, afterColor: hexA(TOKENS.coral, 0.32),
      fmt: (v) => "+" + v.toFixed(1) });

  beforeAfterBar("chart-fairness", DATA.fairness.labels,
    DATA.fairness.before.map((v) => v * 100),
    DATA.fairness.after.map((v) => v * 100),
    { beforeColor: TOKENS.coral, afterColor: hexA(TOKENS.coral, 0.32),
      fmt: (v) => v.toFixed(1) });

  buildCounterfactualChart();
}

function buildCounterfactualChart() {
  const cf = DATA.counterfactual;
  const all = cf.before.concat(cf.after).map(Math.abs);
  let lim = percentile(all, 99); lim = Math.max(lim, 0.02);
  const nbins = 41;
  const edges = [];
  for (let i = 0; i <= nbins; i++) edges.push(-lim + (2 * lim) * i / nbins);
  const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  const hb = histogram(cf.before, edges);
  const ha = histogram(cf.after, edges);

  new Chart(document.getElementById("chart-counterfactual"), {
    type: "bar",
    data: {
      labels: centers.map((c) => (c * 100).toFixed(0)),
      datasets: [
        { label: "Before", data: hb, backgroundColor: hexA(TOKENS.accentA, 0.75),
          barPercentage: 1.0, categoryPercentage: 1.0 },
        { label: "After", data: ha, backgroundColor: hexA(TOKENS.coral, 0.7),
          barPercentage: 1.0, categoryPercentage: 1.0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ...GRID_OPTS, grid: { display: false }, stacked: true,
          title: { display: true, text: "Score swing along gender axis (pts)",
            color: TOKENS.muted },
          ticks: { color: TOKENS.muted, maxTicksLimit: 9, autoSkip: true } },
        y: { ...GRID_OPTS, stacked: true, beginAtZero: true,
          title: { display: true, text: "Candidates", color: TOKENS.muted } },
      },
      plugins: {
        legend: { labels: { color: TOKENS.text, usePointStyle: true,
          pointStyle: "rectRounded" } },
        tooltip: { enabled: true },
      },
    },
  });
}

// Small numeric helpers ----------------------------------------------------- //
function histogram(vals, edges) {
  const counts = new Array(edges.length - 1).fill(0);
  for (const v of vals) {
    if (v < edges[0] || v > edges[edges.length - 1]) continue;
    let lo = 0, hi = edges.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (v >= edges[mid]) lo = mid; else hi = mid;
    }
    counts[lo]++;
  }
  return counts;
}
function percentile(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
  return s[idx];
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //
async function boot() {
  [MODEL, DATA] = await Promise.all([
    fetch("model.json").then((r) => r.json()),
    fetch("data.json").then((r) => r.json()),
  ]);
  fillHeroStats();
  buildForm();
  updateDemo();
  setupLatent();
  buildCharts();
  if (!REDUCED_MOTION) setTimeout(playLatent, 600);
}

boot().catch((err) => {
  console.error(err);
  const r = document.querySelector(".results");
  if (r) r.insertAdjacentHTML("beforeend",
    '<p class="caveat coral">Could not load the model. Serve this folder over '
    + 'HTTP (for example: <code>python -m http.server</code>) rather than '
    + 'opening the file directly.</p>');
});
