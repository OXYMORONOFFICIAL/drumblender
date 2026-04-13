const dataPath = "./data/site-data.json";

function formatMetric(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(4);
}

function setText(node, value) {
  node.textContent = value ?? "";
}

function buildHeroStats(data) {
  const target = document.getElementById("hero-stats");
  const stats = [
    `${data.generated_run_count} runs`,
    `${data.audio_run_count} runs with audio`,
    "Static GitHub Pages deployment",
  ];
  target.replaceChildren(
    ...stats.map((label) => {
      const pill = document.createElement("span");
      pill.className = "stat-pill";
      pill.textContent = label;
      return pill;
    }),
  );
}

function buildRunGrid(runs) {
  const target = document.getElementById("run-grid");
  const template = document.getElementById("run-card-template");

  target.replaceChildren(
    ...runs.map((run) => {
      const fragment = template.content.cloneNode(true);
      setText(fragment.querySelector(".run-name"), run.name);
      setText(fragment.querySelector(".run-config"), run.config || "No config recorded");
      setText(
        fragment.querySelector(".audio-pill"),
        run.has_audio ? `${run.samples.length} audition clips` : "summary only",
      );

      const metrics = fragment.querySelector(".run-metrics");
      const entries = Object.entries(run.metrics || {}).slice(0, 5);
      entries.forEach(([metricName, metricValue]) => {
        const wrap = document.createElement("div");
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = metricName.replace("test/", "");
        dd.textContent = formatMetric(metricValue);
        wrap.append(dt, dd);
        metrics.append(wrap);
      });
      return fragment;
    }),
  );
}

function buildMetricTable(runs) {
  const table = document.getElementById("metric-table");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  const metricNames = Array.from(
    new Set(runs.flatMap((run) => Object.keys(run.metrics || {}))),
  );

  const headRow = document.createElement("tr");
  ["run", ...metricNames].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label.replace("test/", "");
    headRow.append(th);
  });
  thead.replaceChildren(headRow);

  tbody.replaceChildren(
    ...runs.map((run) => {
      const row = document.createElement("tr");
      const nameCell = document.createElement("td");
      nameCell.textContent = run.name;
      row.append(nameCell);
      metricNames.forEach((metricName) => {
        const td = document.createElement("td");
        td.textContent = formatMetric(run.metrics?.[metricName]);
        row.append(td);
      });
      return row;
    }),
  );
}

function buildRunSelect(runs) {
  const select = document.getElementById("run-select");
  const audioRuns = runs.filter((run) => run.has_audio);
  select.replaceChildren(
    ...audioRuns.map((run) => {
      const option = document.createElement("option");
      option.value = run.slug;
      option.textContent = run.name;
      return option;
    }),
  );
  return audioRuns;
}

function renderSamples(run, query) {
  const grid = document.getElementById("sample-grid");
  const empty = document.getElementById("audition-empty");
  const template = document.getElementById("sample-card-template");
  const q = (query || "").trim().toLowerCase();

  if (!run) {
    empty.classList.remove("hidden");
    grid.replaceChildren();
    return;
  }

  empty.classList.add("hidden");

  const samples = run.samples.filter((sample) => {
    if (!q) {
      return true;
    }
    const haystack = [
      sample.pack,
      sample.source_filename,
      sample.bucket,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(q);
  });

  grid.replaceChildren(
    ...samples.map((sample) => {
      const fragment = template.content.cloneNode(true);
      setText(fragment.querySelector(".sample-pack"), sample.pack || "sample");
      setText(fragment.querySelector(".sample-file"), sample.source_filename);
      setText(fragment.querySelector(".bucket-pill"), sample.bucket);

      const chipRow = fragment.querySelector(".metric-chip-row");
      Object.entries(sample.metrics || {})
        .slice(0, 5)
        .forEach(([metricName, metricValue]) => {
          const chip = document.createElement("span");
          chip.className = "metric-chip";
          chip.textContent = `${metricName.replace("test/", "")}: ${formatMetric(metricValue)}`;
          chipRow.append(chip);
        });

      const targetAudio = fragment.querySelector(".target-audio");
      const reconAudio = fragment.querySelector(".recon-audio");

      if (sample.audio?.target) {
        targetAudio.src = encodeURI(sample.audio.target);
      } else {
        targetAudio.closest(".audio-box").innerHTML =
          "<p class='audio-label'>Target</p><p class='muted'>Target audio not copied.</p>";
      }
      if (sample.audio?.recon) {
        reconAudio.src = encodeURI(sample.audio.recon);
      }

      return fragment;
    }),
  );
}

async function main() {
  const response = await fetch(dataPath);
  if (!response.ok) {
    throw new Error(`Failed to load ${dataPath}: ${response.status}`);
  }
  const data = await response.json();

  document.title = data.site_title || document.title;
  buildHeroStats(data);
  setText(
    document.getElementById("generated-meta"),
    `Generated from ${data.generated_from} | ${data.generated_run_count} runs`,
  );

  const runs = [...data.runs];
  buildRunGrid(runs);
  buildMetricTable(runs);

  const audioRuns = buildRunSelect(runs);
  const select = document.getElementById("run-select");
  const search = document.getElementById("sample-search");

  function refreshSamples() {
    const run = audioRuns.find((item) => item.slug === select.value) || audioRuns[0];
    renderSamples(run, search.value);
  }

  if (audioRuns.length === 0) {
    renderSamples(null, "");
  } else {
    select.value = audioRuns[0].slug;
    refreshSamples();
  }

  select.addEventListener("change", refreshSamples);
  search.addEventListener("input", refreshSamples);
}

main().catch((error) => {
  const target = document.getElementById("run-grid");
  const pre = document.createElement("pre");
  pre.textContent = String(error);
  target.replaceChildren(pre);
});
