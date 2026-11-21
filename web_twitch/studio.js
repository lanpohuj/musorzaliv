"use strict";

const $ = (id) => document.getElementById(id);
const state = { token: "", id: null, job: null, draft: [], jobs: [], busy: false, generation: 0, names: {} };
const phases = {
  watching: "Наблюдение за эфиром", live_done: "Наблюдение завершено",
  created: "Подготовка", loading: "Загрузка исходника", source_ready: "Настройка задачи",
  analyzing: "Поиск моментов", review: "Отбор клипов", translating: "Адаптация",
  scripted: "Проверка дубляжа", preparing: "Сборка предпросмотра",
  preview: "Предпросмотр", exporting: "Экспорт", done: "Экспорт завершён", failed: "Этап прерван",
};
const active = new Set(["watching", "loading", "analyzing", "translating", "preparing", "exporting"]);
const editPhases = new Set(["review", "scripted", "preview"]);

function element(tag, properties = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, properties);
  node.append(...children);
  return node;
}

function button(text, handler, style = "secondary") {
  const node = element("button", { type: "button", className: "button " + style, textContent: text });
  node.addEventListener("click", handler);
  return node;
}

function errorMessage(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function stringifyError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((part) => part.msg).join(". ");
  return "Запрос не выполнен.";
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET") headers.set("x-twitch-token", state.token);
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch("/api" + path, { ...options, headers });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(stringifyError(data.detail));
  }
  return response;
}

const json = async (path, options) => (await api(path, options)).json();
const route = (suffix = "") => "/jobs/" + state.id + suffix;

function formatTime(value) {
  const seconds = Math.max(0, Math.floor(value || 0));
  const parts = [Math.floor(seconds / 60) % 60, seconds % 60].map((n) => String(n).padStart(2, "0"));
  if (seconds >= 3600) parts.unshift(String(Math.floor(seconds / 3600)));
  return parts.join(":");
}

async function catalogStatus() {
  const settings = await json("/status");
  state.token = settings.token;
  state.jobs = settings.jobs;
  const analysis = settings.analysis;
  $("services").replaceChildren(...[
    [analysis.provider, analysis.ready], ["Fish Audio", settings.fish],
  ].map(([name, ok]) => element("span", {
    className: "status-pill" + (ok ? " ready" : ""),
    textContent: name + (ok ? " · ключ задан" : " · нужен ключ"),
  })));
  renderJobs();
}

function renderJobs() {
  const items = state.jobs.map((job) => {
    const item = element("button", { type: "button", className: "job" }, [
      element("span", { textContent: job.label }),
      element("small", { textContent: phases[job.status] || job.status }),
    ]);
    item.setAttribute("aria-current", job.id === state.id ? "true" : "false");
    item.addEventListener("click", () => openJob(job.id));
    return item;
  });
  $("jobs").replaceChildren(...items);
}

async function openJob(id) {
  if (state.busy) return;
  const generation = ++state.generation;
  state.id = id;
  state.job = null;
  state.names = {};
  localStorage.setItem("twitch-job", id || "");
  errorMessage();
  if (!id) {
    $("start").hidden = false;
    $("project").hidden = true;
    renderJobs();
    return;
  }
  try {
    const job = await json("/jobs/" + id);
    if (generation === state.generation) apply(job);
  } catch (error) {
    if (generation === state.generation) errorMessage(error.message);
  }
}

function apply(job, keepDraft = false) {
  const same = state.job?.id === job.id;
  if (!same || (state.job.status !== job.status && job.status !== "source_ready")) {
    $("catalog").hidden = true;
    $("audio").pause();
    $("audio").hidden = true;
  }
  const voices = same && keepDraft ? getVoices() : job.voices || {};
  if (!same || !keepDraft) state.draft = structuredClone(job.clips || []);
  state.id = job.id;
  state.job = job;
  state.jobs = [{ id: job.id, label: job.label, status: job.status },
    ...state.jobs.filter((previous) => previous.id !== job.id)];
  localStorage.setItem("twitch-job", job.id);
  $("start").hidden = true;
  $("project").hidden = false;
  $("phase").textContent = phases[job.status] || job.status;
  $("title").textContent = job.label;
  const progress = job.progress;
  $("progress").hidden = !progress;
  if (progress) {
    $("progress-text").textContent = progress.message;
    const bar = $("progress").querySelector("progress");
    bar.max = Math.max(1, progress.total);
    if (job.status === "loading" || !progress.total) bar.removeAttribute("value");
    else bar.value = progress.done;
  }
  const note = $("state-note");
  note.textContent = job.error || job.live_summary || "";
  note.hidden = !note.textContent;
  const setup = $("setup");
  setup.hidden = !job.source_info || active.has(job.status) || job.status === "done" || job.status === "failed";
  if (!setup.hidden) {
    $("source-image").src = "/api" + route("/source.jpg");
    const info = job.source_info;
    $("source-details").textContent = formatTime(info.duration) + " · " + info.width + "×" + info.height
      + (info.audio ? "" : " · без исходного звука");
    const estimate = job.estimate;
    $("estimate").textContent = job.status === "source_ready"
      ? (estimate.usd == null
        ? estimate.model + " · оценка недоступна для этой модели."
        : estimate.windows + " окон · " + (estimate.provider || "Gemini") + " "
          + (estimate.usd < 0.01 ? "< $0.01" : "≈ $" + estimate.usd.toFixed(2)) + ". Fish Audio отдельно.")
      : "";
    for (const role of ["streamer", "guest"]) {
      $(role).value = voices[role] ?? "";
      $(role + "-name").textContent = state.names[$(role).value] || "";
    }
    $("ad-name").textContent = job.ad ? job.ad.label + " · " + job.ad.info.duration.toFixed(1) + " с" : "Без рекламы";
  }
  renderJobs();
  renderClips();
  renderActions();
  $("exports").hidden = job.status !== "done";
  if (job.status === "done") {
    $("folder").textContent = job.output;
    $("files").replaceChildren(...job.files.map((file) => element("li", { textContent: file })));
  }
}

function getVoices() {
  return { streamer: $("streamer").value.trim(), guest: $("guest").value.trim() };
}

function editedClips() {
  return state.draft.map((clip) => ({
    id: clip.id, selected: clip.selected, start: clip.start, end: clip.end,
    camera_mode: clip.camera_mode, camera_box: clip.camera_box, ad_enabled: clip.ad_enabled !== false,
    ad_at: Number.isFinite(clip.ad_at) ? clip.ad_at : null,
    turns: clip.scripted ? clip.turns.map(({ start, end, role, russian }) => ({ start, end, role, russian })) : null,
  }));
}

async function perform(operation) {
  if (state.busy) return;
  state.busy = true;
  errorMessage();
  const generation = state.generation;
  const buttons = [...document.querySelectorAll("button:not(:disabled)")];
  buttons.forEach((btn) => { btn.disabled = true; });
  try {
    await operation(generation);
  } catch (error) {
    if (generation === state.generation) errorMessage(error.message);
  } finally {
    buttons.forEach((btn) => { btn.disabled = false; });
    state.busy = false;
  }
}

function action(name) {
  perform(async () => {
    const job = await json(route("/action"), { method: "POST", json: {
      action: name,
      voices: ["analyze", "script", "preview"].includes(name) ? getVoices() : null,
      clips: ["script", "preview"].includes(name) ? editedClips() : null,
    } });
    apply(job);
  });
}

function renderActions() {
  const job = state.job;
  const panel = $("actions");
  state.actionSignature = actionSignature();
  panel.replaceChildren();
  const add = (label, name, style) => panel.append(button(label, () => action(name), style));
  if (job.status === "watching") panel.append(button("Завершить наблюдение", () =>
    perform(async () => apply(await json(route("/stop-live"), {method: "POST"})))));
  for (const [i, id] of (job.batches || []).entries())
    panel.append(button("Подборка " + (i+1), () => openJob(id)));
  if (job.status === "source_ready") add("Найти моменты", "analyze", "primary");
  if (job.status === "review" && job.clips.length) add("Подготовить перевод", "script", "primary");
  if (["scripted", "preview"].includes(job.status)) {
    const needsScript = state.draft.some((clip) => clip.selected && !clip.scripted);
    if (needsScript) add("Подготовить перевод", "script", "primary");
    else add("Собрать предпросмотр", "preview", job.status === "preview" ? "secondary" : "primary");
  }
  if (job.status === "preview" && !job.previewDirty) add("Экспортировать клипы", "export", "primary");
  if (["review", "scripted", "preview", "done"].includes(job.status))
    panel.append(button("Повторить поиск в новой задаче", () => perform(async () =>
      apply(await json(route("/rescan"), {method: "POST"}))), "secondary"));
  if (job.status === "created") {
    if (job.source) add("Проверить исходник", "load", "primary");
    else panel.append(button("Выбрать исходное видео", () => $("local").click(), "primary"));
  }
  if (job.status === "failed") {
    if (job.operation !== "export") add("Вернуться к настройкам", "back", "secondary");
    add("Повторить этап", "retry", "primary");
  }
  if (job.status === "done") {
    panel.append(button("Открыть папку", () => perform(() => json(route("/open"), { method: "POST" })), "primary"));
  }
}

function actionSignature() {
  return [state.job.status, state.job.batches?.length, !!state.job.previewDirty,
    state.draft.some((clip) => clip.selected && !clip.scripted)].join(":");
}

function dirty() {
  if (state.job?.status === "preview" && !state.job.previewDirty) state.job.previewDirty = true;
  if (state.job && state.actionSignature !== actionSignature()) renderActions();
}

function newFile(file) {
  perform(async () => {
    const job = state.job?.status === "created" && !state.job?.vod ? state.job
      : await json("/jobs", { method: "POST", json: { label: file.name } });
    apply(job);
    $("progress").hidden = false;
    $("progress").querySelector("progress").removeAttribute("value");
    $("progress-text").textContent = "Копирование видео на диск…";
    try {
      apply(await json(route("/asset/source?name=" + encodeURIComponent(file.name)), {
        method: "PUT", body: file, headers: { "Content-Type": "application/octet-stream" },
      }));
    } finally {
      if (state.job?.status === "created") $("progress").hidden = true;
    }
  });
}

function input(label, value, handler, options = {}) {
  const control = element("input", { type: "number", value: value ?? "", step: ".001", ...options });
  control.setAttribute("aria-label", label);
  control.addEventListener("change", () => {
    handler(control.value === "" ? null : Number(control.value));
    dirty();
  });
  return element("div", {}, [element("label", {}, [label, control])]);
}

function checkbox(label, checked, handler) {
  const control = element("input", { type: "checkbox", checked });
  control.addEventListener("change", () => { handler(control.checked); dirty(); });
  return element("label", { className: "checkbox" }, [control, label]);
}

function select(label, options, value, handler) {
  const control = element("select", {}, options.map(([id, name]) => element("option", { value: id, textContent: name })));
  control.setAttribute("aria-label", label);
  control.value = value;
  control.addEventListener("change", () => { handler(control.value); dirty(); });
  return element("div", {}, [element("label", {}, [label, control])]);
}

function roleSelect(turn) {
  return select("Голос", [["streamer", "Стример"], ["guest", "Собеседник"], ["unknown", "Не определён · голос стримера"]],
    turn.role, (role) => { turn.role = role; });
}

function referencePlayer(clip, panel) {
  const wrap = element("div", { className: "crop-wrap" });
  const player = element("video", { controls: true, preload: "metadata", playsInline: true });
  const layer = element("div", { className: "crop-overlay", hidden: true, ariaHidden: "true" });
  const rect = element("div", { className: "crop-rect", hidden: true });
  layer.append(rect);
  wrap.append(player, layer);
  player.addEventListener("loadedmetadata", () => { player.currentTime = Math.max(0, clip.start - clip.reference_start); });
  player.addEventListener("play", () => {
    const time = player.currentTime + clip.reference_start;
    if (time < clip.start || time >= clip.end - .05) player.currentTime = clip.start - clip.reference_start;
  });
  player.addEventListener("timeupdate", () => {
    if (!player.paused && player.currentTime + clip.reference_start >= clip.end) player.pause();
  });
  function load() {
    if (!player.src) player.src = "/api" + route("/media/" + clip.id + "/reference");
  }
  panel.addEventListener("toggle", () => {
    if (panel.open) load();
    else player.pause();
  });
  if (panel.open) load();

  let origin = null;
  let box = null;
  function point(event) {
    const bounds = layer.getBoundingClientRect();
    const size = (at, start, span) => Math.max(0, Math.min(1, (at - start) / span));
    return [size(event.clientX, bounds.left, bounds.width), size(event.clientY, bounds.top, bounds.height)];
  }
  function draw(value) {
    rect.hidden = !value;
    if (!value) return;
    [rect.style.left, rect.style.top, rect.style.width, rect.style.height] = value.map((n) => n * 100 + "%");
  }
  function finish() {
    origin = null;
    layer.hidden = true;
    player.controls = true;
  }
  layer.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    layer.setPointerCapture(event.pointerId);
    origin = point(event);
    box = null;
  });
  layer.addEventListener("pointermove", (event) => {
    if (!origin) return;
    const [x, y] = point(event);
    box = [Math.min(x, origin[0]), Math.min(y, origin[1]), Math.abs(x - origin[0]), Math.abs(y - origin[1])];
    draw(box);
  });
  const tools = element("div", { className: "controls-row" });
  const mode = select("Кадрирование", [["auto", "Вебка автоматически"], ["manual", "Ручная вебка"], ["off", "Только стрим"]],
    clip.camera_mode, (value) => {
      if (value === "manual" && !clip.camera_box) {
        mode.querySelector("select").value = clip.camera_mode;
        pick();
      } else {
        clip.camera_mode = value;
      }
    });
  function pick() {
    load();
    player.pause();
    player.controls = false;
    layer.hidden = false;
    draw(clip.camera_box || clip.cameras?.find((span) => span.end >= clip.start)?.box);
  }
  layer.addEventListener("pointerup", () => {
    if (origin && box && Math.min(box[2], box[3]) >= .03) {
      clip.camera_mode = "manual";
      clip.camera_box = box;
      mode.querySelector("select").value = "manual";
      dirty();
    }
    finish();
  });
  layer.addEventListener("pointercancel", finish);
  tools.append(mode, button("Выделить вебку мышью", pick));
  return { wrap, tools, player };
}

function renderClips() {
  const root = $("clips");
  root.hidden = !editPhases.has(state.job.status);
  if (root.hidden) return;
  if (!state.draft.length) {
    const {covered_windows, candidates, finalists} = state.job;
    const counts = Number.isInteger(finalists)
      ? `Первичный поиск дал ${candidates ?? 0} кандидатов из ${covered_windows ?? 0} окон. Подробно проверено ${finalists}.`
      : `Просмотрено окон записи: ${covered_windows ?? 0}.`;
    root.replaceChildren(element("div", { className: "panel muted", textContent:
      "Готовых клипов нет. " + counts + " Ни один эпизод не прошёл проверку начала и развязки." }));
    return;
  }
  root.replaceChildren(...state.draft.map((clip, index) => {
    const panel = element("details", { className: "panel clip", open: index === 0 });
    const original = state.job.clips.find((item) => item.id === clip.id);
    const summary = element("summary", {}, [
      element("strong", { textContent: clip.title }),
      element("span", { textContent: formatTime(clip.start) + "–" + formatTime(clip.end) + (state.job.contextual_review ? "" : " · " + clip.score + "/100") }),
    ]);
    const reference = referencePlayer(clip, panel);
    const preview = element("video", { controls: true, preload: "none", className: "portrait", playsInline: true });
    const hasPreview = state.job.status === "preview" && clip.selected && !!original.preview_key;
    const media = element("div");
    if (hasPreview) {
      preview.src = "/api" + route("/media/" + clip.id + "/preview") + "?v=" + original.preview_key;
      const source = element("details", {}, [
        element("summary", { textContent: "Исходный кадр и вебка" }), reference.wrap,
      ]);
      media.append(preview, source);
      source.addEventListener("toggle", () => { if (source.open) preview.pause(); });
      panel.addEventListener("toggle", () => { if (!panel.open) preview.pause(); });
    } else {
      media.append(reference.wrap);
    }
    const controls = element("div", { className: "clip-controls" }, [
      element("p", { textContent: clip.reason }),
      element("p", { className: "muted", textContent: clip.evidence }),
      checkbox("Включить в пачку", clip.selected, (value) => { clip.selected = value; }),
      element("div", { className: "controls-row" }, [
        input("Начало VOD, сек.", clip.start, (value) => { clip.start = value; clip.scripted = false; clip.ad_at = null; },
          { min: clip.reference_start, max: clip.peak }),
        input("Конец VOD, сек.", clip.end, (value) => { clip.end = value; clip.scripted = false; clip.ad_at = null; },
          { min: clip.peak, max: clip.reference_end }),
      ]),
      reference.tools,
    ]);
    if (Number.isFinite(clip.hook_time)) controls.prepend(element("small", {className: "muted",
      textContent: "Хук на " + (clip.hook_time - original.start).toFixed(1) + " с · развязка на "
        + (clip.peak - original.start).toFixed(1) + " с от начала предложенной нарезки"}));
    if (clip.audience_views) controls.prepend(element("small", {className: "muted",
      textContent: "Зрительский клип Twitch · " + clip.audience_views.toLocaleString("ru-RU") + " просмотров"}));
    if (state.job.ad && state.job.status !== "review") {
      controls.append(element("div", { className: "controls-row" }, [
        checkbox("Рекламная пауза", clip.ad_enabled !== false, (value) => { clip.ad_enabled = value; }),
        input("Пауза от начала клипа, сек.", clip.ad_at, (value) => { clip.ad_at = value; },
          { min: .034, max: clip.end - clip.start }),
      ]), button("Взять точку из исходного плеера", () => {
        const point = Number((reference.player.currentTime + clip.reference_start - clip.start).toFixed(3));
        if (point <= 0 || point >= clip.end - clip.start) {
          errorMessage("Перейдите к паузе между репликами внутри выбранного клипа.");
          return;
        }
        clip.ad_at = point;
        const fields = controls.querySelectorAll('input[type="number"]');
        fields[fields.length - 1].value = point;
        errorMessage();
        dirty();
      }));
    }
    panel.append(summary, element("div", { className: "clip-body" }, [media, controls]));
    if (clip.scripted && clip.turns.length) {
      const lines = element("details", { className: "turn-list" }, [
        element("summary", { textContent: "Реплики · " + clip.turns.length }),
      ]);
      clip.turns.forEach((turn, i) => {
        const text = element("textarea", { value: turn.russian, rows: 2, maxLength: 400 });
        text.addEventListener("input", () => { turn.russian = text.value; dirty(); });
        const spoken = original.spoken?.[i]?.russian;
        const row = element("div", { className: "turn" }, [
          element("p", { className: "turn-source", textContent: turn.english }),
          element("label", {}, ["Русская реплика", text]),
          element("div", { className: "controls-row" }, [
            roleSelect(turn),
            input("С, сек.", turn.start, (value) => { turn.start = value; }, { min: 0 }),
            input("До, сек.", turn.end, (value) => { turn.end = value; }, { min: 0 }),
          ]),
        ]);
        if (hasPreview && spoken && spoken !== turn.russian) {
          row.append(element("small", { textContent: "В предпросмотре: " + spoken }));
        }
        lines.append(row);
      });
      panel.append(lines);
    }
    return panel;
  }));
}

let catalogPage = 1;
let voiceQuery = "";
let sampleUrl;
async function playVoice(id, publicSample) {
  if (!/^[a-f0-9]{32}$/i.test(id)) throw new Error("Сначала выберите голос или введите ID.");
  const audio = $("audio");
  if (sampleUrl) URL.revokeObjectURL(sampleUrl);
  if (publicSample) {
    audio.src = publicSample;
  } else {
    const response = await api("/voice-sample/" + id, { method: "POST" });
    sampleUrl = URL.createObjectURL(await response.blob());
    audio.src = sampleUrl;
  }
  audio.hidden = false;
  await audio.play().catch(() => {});
}

async function loadCatalog(reset) {
  if (reset) {
    catalogPage = 1;
    voiceQuery = $("voice-query").value.trim();
    $("catalog-list").replaceChildren(element("p", { className: "muted", textContent: "Загрузка голосов…" }));
  }
  const data = await json("/voices?page=" + catalogPage + "&q=" + encodeURIComponent(voiceQuery));
  if (reset) $("catalog-list").replaceChildren();
  for (const voice of data.voices) {
    state.names[voice.id] = voice.title;
    const assign = (role) => {
      $(role).value = voice.id;
      $(role + "-name").textContent = voice.title;
      dirty();
    };
    $("catalog-list").append(element("div", { className: "voice-result" }, [
      element("span", { className: "voice-title", textContent: voice.title }),
      button("▶", () => perform(() => playVoice(voice.id, voice.sample))),
      button("Стример", () => assign("streamer")),
      button("Собеседник", () => assign("guest")),
    ]));
  }
  if (reset && !data.voices.length) $("catalog-list").textContent = "Голоса не найдены.";
  $("catalog-more").hidden = !data.more;
  catalogPage += 1;
}

$("new").addEventListener("click", () => openJob(null));
$("url-form").addEventListener("submit", (event) => {
  event.preventDefault();
  perform(async () => { apply(await json("/jobs", { method: "POST", json: { url: $("url").value } })); });
});
$("local").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (file) newFile(file);
});
$("ad-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  perform(async () => apply(await json(route("/asset/ad?name=" + encodeURIComponent(file.name)), {
    method: "PUT", body: file, headers: { "Content-Type": "application/octet-stream" },
  }), true));
});
$("catalog-toggle").addEventListener("click", () => {
  $("catalog").hidden = !$("catalog").hidden;
  if (!$("catalog").hidden) perform(() => loadCatalog(true));
});
$("voice-search").addEventListener("submit", (event) => { event.preventDefault(); perform(() => loadCatalog(true)); });
$("catalog-more").addEventListener("click", () => perform(() => loadCatalog(false)));
document.querySelectorAll(".voice-test").forEach((node) => {
  node.addEventListener("click", () => perform(() => playVoice($(node.dataset.role).value.trim())));
});
for (const role of ["streamer", "guest"]) {
  $(role).addEventListener("input", () => {
    $(role + "-name").textContent = state.names[$(role).value] || "";
    dirty();
  });
}

catalogStatus().then(() => {
  const previous = new URLSearchParams(location.search).get("job") || localStorage.getItem("twitch-job");
  if (previous && state.jobs.some((job) => job.id === previous)) openJob(previous);
}).catch((error) => errorMessage(error.message));

let polling = false;
setInterval(async () => {
  if (polling || state.busy || !state.job || !active.has(state.job.status)) return;
  const generation = state.generation;
  const id = state.id;
  polling = true;
  try {
    const job = await json("/jobs/" + id);
    if (generation === state.generation && id === state.id) {
      errorMessage();
      apply(job);
    }
  } catch (error) {
    if (generation === state.generation) errorMessage(error.message);
  } finally {
    polling = false;
  }
}, 1250);
