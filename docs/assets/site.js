const SITE_DATA_PATH = "./data/site-data.json";
const CONTROL_DATA_PATH = "./data/control-data.json";

const metricLabels = {
  "test/loss": "MSS",
  "test/lsd": "LSD",
  "test/flux_onset": "Onset",
  "test/mss_sc": "MSS SC",
  "test/mss_log": "MSS Log",
};

const state = {
  data: null,
  control: null,
  auditionRunSlug: "",
  bucket: "all",
  controlVariantIndex: 2,
};

let audioContext = null;
const waveformCache = new Map();

function $(selector) {
  return document.querySelector(selector);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatMetric(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  if (Math.abs(value) >= 100) return value.toFixed(2);
  if (Math.abs(value) >= 10) return value.toFixed(3);
  return value.toFixed(digits);
}

function metricName(name) {
  return metricLabels[name] || name.replace("test/", "");
}

function shortRunName(name) {
  return String(name || "")
    .replace(/^05_all_parallel_/, "")
    .replace(/_/g, " ");
}

function pickOverallRun(runs) {
  return (
    runs.find((run) => run.name.endsWith("_all")) ||
    runs.find((run) => run.pack === "all") ||
    runs[0]
  );
}

function setActiveView(viewName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("is-active", view.id === `view-${viewName}`);
  });
}

function renderSummary(data) {
  const overall = pickOverallRun(data.runs);
  $("#summary-run").textContent = shortRunName(overall?.name || "NOISEDAC");
  $("#summary-items").textContent = String(overall?.num_items ?? "-");
  $("#summary-loss").textContent = formatMetric(overall?.metrics?.["test/loss"]);
  $("#summary-lsd").textContent = formatMetric(overall?.metrics?.["test/lsd"]);
  $("#summary-flux").textContent = formatMetric(overall?.metrics?.["test/flux_onset"], 2);
  $("#generated-meta").textContent =
    `${data.generated_run_count} bundles from ${data.generated_from || "local results"}`;
}

function renderRunSelect(runs) {
  const select = $("#audition-run");
  const audioRuns = runs.filter((run) => run.has_audio && run.samples?.length);
  const overall = pickOverallRun(audioRuns);
  select.replaceChildren(
    ...audioRuns.map((run) => {
      const option = el("option");
      option.value = run.slug;
      option.textContent = `${shortRunName(run.name)} (${run.samples.length})`;
      return option;
    }),
  );
  state.auditionRunSlug = overall?.slug || audioRuns[0]?.slug || "";
  select.value = state.auditionRunSlug;
}

function renderBucketFilter() {
  const buckets = [
    ["all", "All"],
    ["best", "Best"],
    ["median", "Median"],
    ["worst", "Worst"],
  ];
  $("#bucket-filter").replaceChildren(
    ...buckets.map(([value, label]) => {
      const button = el("button", value === state.bucket ? "is-active" : "", label);
      button.type = "button";
      button.dataset.bucket = value;
      button.addEventListener("click", () => {
        state.bucket = value;
        renderBucketFilter();
        renderAudition();
      });
      return button;
    }),
  );
}

function sampleMatches(sample, query) {
  const text = [
    sample.pack,
    sample.bucket,
    sample.source_filename,
    ...Object.keys(sample.metrics || {}),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return text.includes(query.trim().toLowerCase());
}

function renderMetricChips(sample) {
  const chips = el("div", "chips");
  Object.entries(sample.metrics || {})
    .filter(([name]) => ["test/loss", "test/lsd", "test/flux_onset"].includes(name))
    .forEach(([name, value]) => {
      chips.append(el("span", "chip", `${metricName(name)} ${formatMetric(value)}`));
    });
  return chips;
}

function audioTile(label, source, tone) {
  const tile = el("div", "audio-tile");
  tile.append(el("p", "audio-label", label));
  const canvas = el("canvas", "wave");
  canvas.width = 900;
  canvas.height = 144;
  canvas.dataset.src = source || "";
  canvas.dataset.tone = tone;
  const audio = el("audio");
  audio.controls = true;
  audio.preload = "none";
  if (source) audio.src = encodeURI(source);
  tile.append(canvas, audio);
  return tile;
}

function renderSampleCard(sample) {
  const card = el("article", "sample-card");
  const top = el("div", "sample-top");
  const meta = el("div");
  meta.append(el("p", "pack", sample.pack || "sample"));
  meta.append(el("p", "file-name", sample.source_filename));
  const quality = el("span", "quality", sample.bucket || "sample");
  quality.dataset.bucket = sample.bucket || "";
  top.append(meta, quality);

  const compare = el("div", "ab-grid");
  compare.append(audioTile("Target", sample.audio?.target, "target"));
  compare.append(audioTile("Reconstruction", sample.audio?.recon, "recon"));

  card.append(top, renderMetricChips(sample), compare);
  return card;
}

function renderAudition() {
  const run = state.data.runs.find((item) => item.slug === state.auditionRunSlug);
  const query = $("#audition-search").value || "";
  const grid = $("#audition-grid");

  if (!run) {
    grid.replaceChildren(el("div", "empty-state", "No audition audio has been exported yet."));
    return;
  }

  const samples = run.samples
    .filter((sample) => state.bucket === "all" || sample.bucket === state.bucket)
    .filter((sample) => !query || sampleMatches(sample, query));

  if (!samples.length) {
    const empty = el("div", "empty-state");
    empty.append(el("p", null, "No samples match the current filter."));
    grid.replaceChildren(empty);
    return;
  }

  grid.replaceChildren(...samples.map(renderSampleCard));
  queueWaveformDraw();
}

function renderMetrics(runs) {
  const ranked = [...runs].sort(
    (a, b) => (a.metrics?.["test/loss"] ?? Infinity) - (b.metrics?.["test/loss"] ?? Infinity),
  );
  const maxLoss = Math.max(...ranked.map((run) => run.metrics?.["test/loss"] || 0), 1);
  $("#pack-ranking").replaceChildren(
    ...ranked.slice(0, 8).map((run, index) => {
      const card = el("article", "ranking-card");
      card.append(el("h3", null, `${index + 1}. ${shortRunName(run.name)}`));
      const row = el("div", "bar-row");
      const bar = el("div", "bar");
      const fill = el("span");
      fill.style.setProperty("--bar-width", `${Math.max(4, ((run.metrics?.["test/loss"] || 0) / maxLoss) * 100)}%`);
      bar.append(fill);
      row.append(bar, el("strong", null, formatMetric(run.metrics?.["test/loss"])));
      card.append(row);
      return card;
    }),
  );

  const metricKeys = ["test/loss", "test/lsd", "test/flux_onset", "test/mss_sc", "test/mss_log"];
  const head = el("tr");
  ["Bundle", "Items", ...metricKeys.map(metricName)].forEach((name) => head.append(el("th", null, name)));
  $("#metric-table thead").replaceChildren(head);
  $("#metric-table tbody").replaceChildren(
    ...ranked.map((run) => {
      const row = el("tr");
      row.append(el("td", null, shortRunName(run.name)));
      row.append(el("td", null, String(run.num_items ?? "")));
      metricKeys.forEach((key) => row.append(el("td", null, formatMetric(run.metrics?.[key]))));
      return row;
    }),
  );
}

async function loadOptionalJson(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function renderControlSelect(control) {
  const select = $("#control-demo");
  const demos = control?.demos || [];
  select.replaceChildren(
    ...demos.map((demo, index) => {
      const option = el("option");
      option.value = String(index);
      option.textContent = demo.title || demo.source_filename || `Control demo ${index + 1}`;
      return option;
    }),
  );
  select.disabled = demos.length === 0;
}

function renderControl() {
  const panel = $("#control-panel");
  const demos = state.control?.demos || [];
  if (!demos.length) {
    const empty = el("div", "empty-state");
    empty.append(
      el("p", null, "Static control audio has not been exported yet. The page is wired for it; run the exporter in the training environment where torch and the checkpoint are available."),
    );
    empty.append(
      el(
        "code",
        "command",
        "python scripts/recon/control_demo.py ../results/run_NOISEDAC_20260412_231956 --output docs --methods both --sample-count 3 --num-knobs 2 --axis-samples 512",
      ),
    );
    panel.replaceChildren(empty);
    return;
  }

  const index = Number($("#control-demo").value || 0);
  const demo = demos[Math.max(0, Math.min(index, demos.length - 1))];
  const variants = demo.variants || [];
  const variantIndex = Math.max(0, Math.min(state.controlVariantIndex, variants.length - 1));
  const selectedVariant = variants[variantIndex];
  const card = el("article", "control-card");
  const meta = el("div", "control-meta");
  const title = el("div");
  title.append(el("h3", null, demo.title || demo.module || "Control sweep"));
  title.append(el("p", "control-path", demo.source_filename || ""));
  meta.append(title, el("span", "quality", demo.module || "control"));

  const knob = el("div", "knob-panel");
  const knobHead = el("div", "knob-head");
  knobHead.append(el("span", "audio-label", "Transient latent knob"));
  knobHead.append(el("strong", null, selectedVariant?.label || "no variant"));
  const range = el("input", "knob");
  range.type = "range";
  range.min = "0";
  range.max = String(Math.max(0, variants.length - 1));
  range.step = "1";
  range.value = String(variantIndex);
  range.disabled = variants.length === 0;
  range.addEventListener("input", (event) => {
    state.controlVariantIndex = Number(event.target.value);
    renderControl();
  });
  const ticks = el("div", "knob-ticks");
  variants.forEach((variant, tickIndex) => {
    const tick = el("button", tickIndex === variantIndex ? "is-active" : "", String(variant.sigma));
    tick.type = "button";
    tick.addEventListener("click", () => {
      state.controlVariantIndex = tickIndex;
      renderControl();
    });
    ticks.append(tick);
  });
  knob.append(knobHead, range, ticks);

  const strip = el("div", "control-strip");
  if (demo.target) {
    const target = el("div", "control-step");
    target.append(el("strong", null, "target"), audioTile("", demo.target, "target"));
    strip.append(target);
  }
  if (demo.baseline) {
    const base = el("div", "control-step");
    base.append(el("strong", null, "reconstruction"), audioTile("", demo.baseline, "recon"));
    strip.append(base);
  }
  if (selectedVariant) {
    const step = el("div", "control-step is-selected");
    step.append(el("strong", null, selectedVariant.label || `sigma ${selectedVariant.sigma}`));
    step.append(audioTile("", selectedVariant.audio, "control"));
    strip.append(step);
  }

  card.append(meta, knob, strip);
  panel.replaceChildren(card);
  queueWaveformDraw();
}

function pauseOtherAudio(event) {
  document.querySelectorAll("audio").forEach((audio) => {
    if (audio !== event.target) audio.pause();
  });
}

function queueWaveformDraw() {
  document.querySelectorAll("canvas.wave").forEach((canvas) => drawWaveform(canvas));
}

async function decodeAudio(source) {
  if (!source) return null;
  if (waveformCache.has(source)) return waveformCache.get(source);
  audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
  const response = await fetch(encodeURI(source));
  if (!response.ok) return null;
  const bytes = await response.arrayBuffer();
  const buffer = await audioContext.decodeAudioData(bytes.slice(0));
  const channel = buffer.getChannelData(0);
  waveformCache.set(source, channel);
  return channel;
}

async function drawWaveform(canvas) {
  const source = canvas.dataset.src;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbf8f1";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8cec0";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  let data = null;
  try {
    data = await decodeAudio(source);
  } catch {
    data = null;
  }
  if (!data) return;

  const step = Math.max(1, Math.floor(data.length / width));
  const amp = height * 0.43;
  const color = canvas.dataset.tone === "target" ? "#2f5f9f" : canvas.dataset.tone === "control" ? "#b6423a" : "#12766c";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i += 1) {
      const value = data[start + i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    ctx.moveTo(x, height / 2 + min * amp);
    ctx.lineTo(x, height / 2 + max * amp);
  }
  ctx.stroke();
}

async function main() {
  const response = await fetch(SITE_DATA_PATH, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${SITE_DATA_PATH}`);
  state.data = await response.json();
  state.control = await loadOptionalJson(CONTROL_DATA_PATH);

  document.title = state.data.site_title || "DrumBlender Demo";
  renderSummary(state.data);
  renderRunSelect(state.data.runs);
  renderBucketFilter();
  renderAudition();
  renderMetrics(state.data.runs);
  renderControlSelect(state.control);
  renderControl();

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => setActiveView(button.dataset.view));
  });
  $("#audition-run").addEventListener("change", (event) => {
    state.auditionRunSlug = event.target.value;
    renderAudition();
  });
  $("#audition-search").addEventListener("input", renderAudition);
  $("#control-demo").addEventListener("change", renderControl);
  document.body.addEventListener("play", pauseOtherAudio, true);
}

main().catch((error) => {
  const empty = el("div", "empty-state");
  empty.append(el("p", null, String(error)));
  document.querySelector("main").replaceChildren(empty);
});
