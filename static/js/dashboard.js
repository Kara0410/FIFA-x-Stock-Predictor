/* ============================================================
   FIFA 2026 dashboard frontend
   - renders the knockout bracket from /api/bracket
   - match & team detail modals
   - sponsor stock chart from /api/stock/<ticker> (Plotly, SVG fallback)
   ============================================================ */
"use strict";

const ROUND_LABELS = { R32: "Round of 32", R16: "Round of 16", QF: "Quarter-finals", SF: "Semi-finals", F: "Final" };

let BRACKET = null;            // cached /api/bracket payload
let MATCH_BY_ID = {};
let CURRENT_TICKER = null;
let BRACKET_RESIZE_OBSERVER = null;
let MODEL_DEFAULTS = null;
let LAST_CONFIG = null;
let ACTIVE_CONFIG = null;
let SPONSOR_LIST = [];
let PREDICTION_PROGRESS_TIMER = null;
let PREDICTION_PROGRESS_VALUE = 0;

/* ---------------- helpers ---------------- */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const shortDate = (iso) => new Date(iso + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function modelMeta(section, name) {
  return MODEL_DEFAULTS?.[section]?.[name] || null;
}

function defaultConfig() {
  if (!MODEL_DEFAULTS) return null;
  return {
    football: { model: "baseline_poisson_blend", params: deepClone(MODEL_DEFAULTS.football.baseline_poisson_blend.defaults) },
    stock: { model: "baseline_gbr", params: deepClone(MODEL_DEFAULTS.stock.baseline_gbr.defaults) },
    stock_ticker: "KO",
    persist: true,
  };
}

function mergeConfig(base, override) {
  const out = deepClone(base);
  if (!override) return out;
  if (override.football) {
    out.football.model = override.football.model || out.football.model;
    out.football.params = { ...out.football.params, ...(override.football.params || {}) };
  }
  if (override.stock) {
    out.stock.model = override.stock.model || out.stock.model;
    out.stock.params = { ...out.stock.params, ...(override.stock.params || {}) };
  }
  if (override.stock_ticker) out.stock_ticker = override.stock_ticker;
  if (typeof override.persist === "boolean") out.persist = override.persist;
  return out;
}

function setSelectOptions(el, items, valueKey = "value", labelKey = "label") {
  el.innerHTML = items.map((item) => `<option value="${esc(item[valueKey])}">${esc(item[labelKey])}</option>`).join("");
}

function formatValueForInput(control, value) {
  if (control.type === "toggle") return Boolean(value);
  if (control.type === "select" || control.type === "text") return value ?? "";
  if (control.type === "number" || control.type === "slider") return value ?? "";
  return value ?? "";
}

function parseControlValue(control, el) {
  if (control.type === "toggle") return el.checked;
  if (control.type === "number" || control.type === "slider") {
    if (el.value === "" && control.allow_null) return null;
    return Number(el.value);
  }
  if (control.type === "text") {
    const raw = el.value.trim();
    if (!raw) return "";
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }
  return el.value;
}

function renderModelParams(section, modelName, params) {
  const meta = modelMeta(section, modelName);
  const target = document.getElementById(section === "football" ? "football-params" : "stock-params");
  if (!meta || !target) return;
  const controls = meta.controls || {};
  const html = Object.entries(controls).map(([key, control]) => {
    const value = params?.[key] ?? meta.defaults?.[key];
    const normalized = formatValueForInput(control, value);
    const id = `${section}-${key}`;
    if (control.type === "slider") {
      return `<div class="param-field">
        <label for="${id}">${esc(control.label || key)}</label>
        <input id="${id}" data-section="${section}" data-key="${key}" data-control-type="${control.type}" type="range" min="${control.min}" max="${control.max}" step="${control.step}" value="${esc(normalized)}">
        <div class="meta"><span class="value" data-value-for="${id}">${esc(normalized)}</span> ${meta.pros ? `&middot; ${esc(meta.pros)}` : ""}</div>
      </div>`;
    }
    if (control.type === "number") {
      return `<div class="param-field">
        <label for="${id}">${esc(control.label || key)}</label>
        <input id="${id}" data-section="${section}" data-key="${key}" data-control-type="${control.type}" type="number" min="${control.min}" max="${control.max}" step="${control.step}" value="${normalized === null ? "" : esc(normalized)}">
        <div class="meta">${meta.cons ? esc(meta.cons) : ""}</div>
      </div>`;
    }
    if (control.type === "select") {
      const opts = (control.options || []).map((opt) => `<option value="${esc(opt)}"${opt === normalized ? " selected" : ""}>${esc(opt)}</option>`).join("");
      return `<div class="param-field">
        <label for="${id}">${esc(control.label || key)}</label>
        <select id="${id}" data-section="${section}" data-key="${key}" data-control-type="${control.type}">${opts}</select>
        <div class="meta">${meta.pros ? esc(meta.pros) : ""}</div>
      </div>`;
    }
    if (control.type === "toggle") {
      return `<div class="param-field">
        <label for="${id}">${esc(control.label || key)}</label>
        <div class="inline"><input id="${id}" data-section="${section}" data-key="${key}" data-control-type="${control.type}" type="checkbox"${normalized ? " checked" : ""}><span>${normalized ? "On" : "Off"}</span></div>
        <div class="meta">${meta.cons ? esc(meta.cons) : ""}</div>
      </div>`;
    }
    const placeholder = control.placeholder || "";
    return `<div class="param-field">
      <label for="${id}">${esc(control.label || key)}</label>
      <input id="${id}" data-section="${section}" data-key="${key}" data-control-type="${control.type}" type="text" value="${esc(normalized)}" placeholder="${esc(placeholder)}">
      <div class="meta">${meta.pros ? esc(meta.pros) : ""}</div>
    </div>`;
  }).join("");
  target.innerHTML = html;

  target.querySelectorAll("[data-control-type='slider']").forEach((input) => {
    input.addEventListener("input", () => {
      const valueEl = target.querySelector(`[data-value-for="${input.id}"]`);
      if (valueEl) valueEl.textContent = input.value;
    });
  });
}

function applyConfigToControls(config) {
  if (!MODEL_DEFAULTS || !config) return;
  const footballSelect = document.getElementById("football-model-select");
  const stockSelect = document.getElementById("stock-model-select");
  const stockTickerSelect = document.getElementById("stock-select");
  footballSelect.value = config.football.model;
  stockSelect.value = config.stock.model;
  if (stockTickerSelect) stockTickerSelect.value = config.stock_ticker;
  renderModelParams("football", config.football.model, config.football.params);
  renderModelParams("stock", config.stock.model, config.stock.params);
  updateControlDescriptions();
}

function updateControlDescriptions() {
  const footballMeta = modelMeta("football", document.getElementById("football-model-select")?.value);
  const stockMeta = modelMeta("stock", document.getElementById("stock-model-select")?.value);
  const footballEl = document.getElementById("football-model-description");
  const stockEl = document.getElementById("stock-model-description");
  if (footballEl && footballMeta) {
    footballEl.textContent = `${footballMeta.description} Pro: ${footballMeta.pros} Con: ${footballMeta.cons}`;
  }
  if (stockEl && stockMeta) {
    stockEl.textContent = `${stockMeta.description} Pro: ${stockMeta.pros} Con: ${stockMeta.cons}`;
  }
}

function collectConfigFromUI() {
  const footballModel = document.getElementById("football-model-select").value;
  const stockModel = document.getElementById("stock-model-select").value;
  const config = {
    football: { model: footballModel, params: deepClone(modelMeta("football", footballModel).defaults) },
    stock: { model: stockModel, params: deepClone(modelMeta("stock", stockModel).defaults) },
    stock_ticker: document.getElementById("stock-select").value,
    persist: true,
  };
  document.querySelectorAll("#football-params [data-control-type], #stock-params [data-control-type]").forEach((el) => {
    const section = el.dataset.section;
    const key = el.dataset.key;
    const control = modelMeta(section, `${document.getElementById(`${section}-model-select`).value}`)?.controls?.[key];
    if (!control) return;
    const value = parseControlValue(control, el);
    if (section === "football") config.football.params[key] = value;
    else config.stock.params[key] = value;
  });
  return config;
}

function renderActiveModelChips(payload) {
  const football = payload.football_model_info;
  const stock = payload.stock_model_info;
  document.getElementById("active-football-model").innerHTML = `Football model: <b>${esc(football.model_label)}</b> &middot; rows ${football.training_rows} &middot; features ${football.feature_count}`;
  document.getElementById("active-stock-model").innerHTML = `Stock model: <b>${esc(stock.model_label)}</b> &middot; rows ${stock.training_rows} &middot; features ${stock.feature_count}`;
}

async function postRecalculate(config) {
  const res = await fetch("/api/recalculate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`POST /api/recalculate -> HTTP ${res.status}`);
  return res.json();
}

function persistUiState(config) {
  try {
    localStorage.setItem("fifa_dashboard_model_config", JSON.stringify(config));
  } catch {
    /* ignore */
  }
}

function readUiState() {
  try {
    const raw = localStorage.getItem("fifa_dashboard_model_config");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function setPredictionProgress(value, label) {
  const panel = document.getElementById("prediction-progress");
  const fill = document.getElementById("prediction-progress-fill");
  const pct = document.getElementById("prediction-progress-pct");
  const text = document.getElementById("prediction-progress-label");
  if (!panel || !fill || !pct || !text) return;
  const clamped = Math.max(0, Math.min(100, value));
  PREDICTION_PROGRESS_VALUE = clamped;
  panel.classList.remove("hidden");
  panel.classList.add("active");
  fill.style.width = `${clamped}%`;
  pct.textContent = `${Math.round(clamped)}%`;
  if (label) text.textContent = label;
}

function startPredictionProgress() {
  stopPredictionProgress();
  setPredictionProgress(8, "Starting model selection…");
  const stages = [
    [18, "Loading or training football model…"],
    [42, "Running bracket simulation…"],
    [68, "Loading or training stock model…"],
    [86, "Preparing forecast output…"],
  ];
  let idx = 0;
  PREDICTION_PROGRESS_TIMER = setInterval(() => {
    const next = stages[idx];
    if (next && PREDICTION_PROGRESS_VALUE < next[0]) {
      setPredictionProgress(next[0], next[1]);
      idx += 1;
      return;
    }
    if (PREDICTION_PROGRESS_VALUE < 96) {
      setPredictionProgress(PREDICTION_PROGRESS_VALUE + 2, next ? next[1] : "Finalizing…");
    }
  }, 260);
}

function stopPredictionProgress(success = true) {
  if (PREDICTION_PROGRESS_TIMER) {
    clearInterval(PREDICTION_PROGRESS_TIMER);
    PREDICTION_PROGRESS_TIMER = null;
  }
  if (success) {
    setPredictionProgress(100, "Dashboard refreshed.");
    setTimeout(() => {
      const panel = document.getElementById("prediction-progress");
      if (panel) panel.classList.add("hidden");
      if (document.getElementById("prediction-progress-fill")) {
        document.getElementById("prediction-progress-fill").style.width = "12%";
      }
      PREDICTION_PROGRESS_VALUE = 0;
    }, 600);
  } else {
    const panel = document.getElementById("prediction-progress");
    if (panel) panel.classList.add("hidden");
    PREDICTION_PROGRESS_VALUE = 0;
  }
}

/* ============================================================
   BRACKET
   ============================================================ */
async function loadBracket() {
  BRACKET = await fetchJSON("/api/bracket");
  MATCH_BY_ID = {};
  BRACKET.matches.forEach((m) => (MATCH_BY_ID[m.id] = m));
  renderHeroChips();
  renderBracket();
  renderTitleTable();
}

function applyDashboardPayload(payload) {
  BRACKET = payload.bracket;
  MATCH_BY_ID = {};
  BRACKET.matches.forEach((m) => (MATCH_BY_ID[m.id] = m));
  ACTIVE_CONFIG = payload.current_config || ACTIVE_CONFIG;
  CURRENT_TICKER = ACTIVE_CONFIG?.stock_ticker || CURRENT_TICKER;
  renderHeroChips();
  renderBracket();
  renderTitleTable();
  renderActiveModelChips(payload);
  if (ACTIVE_CONFIG?.stock_ticker) {
    const select = document.getElementById("stock-select");
    if (select && select.value !== ACTIVE_CONFIG.stock_ticker) select.value = ACTIVE_CONFIG.stock_ticker;
  }
  renderStockFromPayload(payload.stock);
}

function renderHeroChips() {
  const done = BRACKET.matches.filter((m) => m.status === "completed").length;
  const total = BRACKET.matches.length;
  const champ = BRACKET.predicted_champion;
  const champFlag = champ ? flagOf(champ) : "";
  document.getElementById("hero-chips").innerHTML = `
    <span class="chip">Knockout matches: <b>${done}/${total}</b> played</span>
    <span class="chip">Simulations: <b>${BRACKET.n_simulations.toLocaleString()}</b></span>
    <span class="chip">Projected champion: <b><span class="flag">${champFlag}</span>${esc(champ || "TBD")}</b></span>
    <span class="chip">Verified completed results &middot; as of 2026-07-04</span>`;
}

function flagOf(teamName) {
  for (const m of BRACKET.matches) {
    if (m.home && m.home.name === teamName) return m.home.flag;
    if (m.away && m.away.name === teamName) return m.away.flag;
  }
  const r = BRACKET.title_ranking.find((t) => t.team === teamName);
  return r ? r.flag : "";
}

/* Subtree utilities: which matches / teams feed into a given match */
function subtreeMatches(id, acc) {
  const m = MATCH_BY_ID[id];
  if (!m) return acc;
  acc.push(m.id);
  if (m.home_from) subtreeMatches(m.home_from, acc);
  if (m.away_from) subtreeMatches(m.away_from, acc);
  return acc;
}
function teamsInSubtree(id) {
  const names = new Set();
  subtreeMatches(id, []).forEach((mid) => {
    const m = MATCH_BY_ID[mid];
    if (m.home) names.add(m.home.name);
    if (m.away) names.add(m.away.name);
  });
  return names;
}
/* Ordered list of match ids per level below a root (root level 0) */
function levelsBelow(rootId) {
  const levels = [[rootId]];
  for (;;) {
    const next = [];
    for (const id of levels[levels.length - 1]) {
      const m = MATCH_BY_ID[id];
      if (m.home_from) next.push(m.home_from);
      if (m.away_from) next.push(m.away_from);
    }
    if (!next.length) break;
    levels.push(next);
  }
  return levels;
}

function renderBracket() {
  const el = document.getElementById("bracket");
  const final = BRACKET.matches.find((m) => m.round === "F");
  const leftLevels = levelsBelow(final.home_from);   // [SF],[QF,QF],[R16 x4],[R32 x8]
  const rightLevels = levelsBelow(final.away_from);

  // A standard 32-team bracket has sixteen vertical leaf slots. Later-round
  // cards sit exactly halfway between their two feeder cards.
  const slotsFor = (count) => ({
    1: [8],
    2: [4, 12],
    4: [2, 6, 10, 14],
    8: [1, 3, 5, 7, 9, 11, 13, 15],
  }[count]);

  const column = (title, ids, gridColumn) => {
    const slots = slotsFor(ids.length);
    return `<div class="col-title" style="grid-column:${gridColumn}">${title}</div>` +
      ids.map((id, index) =>
        matchCardHTML(MATCH_BY_ID[id], gridColumn, slots[index])).join("");
  };

  el.innerHTML = `
    <svg class="bracket-connectors" aria-hidden="true"></svg>
    ${column("Round of 32", leftLevels[3], 1)}
    ${column("Round of 16", leftLevels[2], 2)}
    ${column("Quarter-finals", leftLevels[1], 3)}
    ${column("Semi-finals", leftLevels[0], 4)}
    <div class="col-title final-title" style="grid-column:5">Final &middot; ${shortDate(final.date)}</div>
    ${matchCardHTML(final, 5, 8)}
    ${championCardHTML(final)}
    ${column("Semi-finals", rightLevels[0], 6)}
    ${column("Quarter-finals", rightLevels[1], 7)}
    ${column("Round of 16", rightLevels[2], 8)}
    ${column("Round of 32", rightLevels[3], 9)}
  `;

  el.querySelectorAll(".match-card").forEach((card) => {
    card.addEventListener("click", () => openMatchModal(card.dataset.mid));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openMatchModal(card.dataset.mid);
      }
    });
  });

  requestAnimationFrame(drawBracketConnectors);
  if (BRACKET_RESIZE_OBSERVER) BRACKET_RESIZE_OBSERVER.disconnect();
  if (window.ResizeObserver) {
    BRACKET_RESIZE_OBSERVER = new ResizeObserver(drawBracketConnectors);
    BRACKET_RESIZE_OBSERVER.observe(el);
  }
}

function drawBracketConnectors() {
  const bracket = document.getElementById("bracket");
  const svg = bracket.querySelector(".bracket-connectors");
  if (!svg) return;

  const bracketRect = bracket.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${bracketRect.width} ${bracketRect.height}`);
  svg.replaceChildren();

  const addPath = (source, target, isChampionPath = false) => {
    if (!source || !target) return;
    const sourceRect = source.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const flowsRight = sourceRect.left < targetRect.left;
    const startX = (flowsRight ? sourceRect.right : sourceRect.left) - bracketRect.left;
    const endX = (flowsRight ? targetRect.left : targetRect.right) - bracketRect.left;
    const startY = sourceRect.top + sourceRect.height / 2 - bracketRect.top;
    const endY = targetRect.top + targetRect.height / 2 - bracketRect.top;
    const elbowX = (startX + endX) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${startX} ${startY} H ${elbowX} V ${endY} H ${endX}`);
    path.setAttribute("class", isChampionPath ? "connector on-path" : "connector");
    svg.appendChild(path);
  };

  BRACKET.matches.forEach((match) => {
    if (!match.home_from || !match.away_from) return;
    const target = bracket.querySelector(`[data-mid="${match.id}"]`);
    [match.home_from, match.away_from].forEach((sourceId) => {
      const source = bracket.querySelector(`[data-mid="${sourceId}"]`);
      const onPath = source?.classList.contains("on-path") &&
        target?.classList.contains("on-path");
      addPath(source, target, onPath);
    });
  });

  const final = BRACKET.matches.find((match) => match.round === "F");
  const finalCard = bracket.querySelector(`[data-mid="${final.id}"]`);
  const winnerCard = bracket.querySelector("#winner-card");
  if (finalCard && winnerCard) {
    const finalRect = finalCard.getBoundingClientRect();
    const winnerRect = winnerCard.getBoundingClientRect();
    const x = finalRect.left + finalRect.width / 2 - bracketRect.left;
    const startY = finalRect.bottom - bracketRect.top;
    const endY = winnerRect.top - bracketRect.top;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x} ${startY} V ${endY}`);
    path.setAttribute("class", "connector winner-connector on-path");
    svg.appendChild(path);
  }
}

function championCardHTML(final) {
  if (final.status === "completed" && final.winner) {
    return `<div class="champ-card" id="winner-card"><div class="crown">&#127942;</div>
      <div class="name">${flagOf(final.winner)} ${esc(final.winner)}</div>
      <div class="lbl">World Champion</div></div>`;
  }
  const champ = BRACKET.predicted_champion;
  const row = BRACKET.title_ranking.find((t) => t.team === champ);
  const p = row ? pct(row.p_champion, 1) : "&mdash;";
  return `<div class="champ-card" id="winner-card"><div class="crown">&#128081;</div>
    <div class="name">${champ ? flagOf(champ) : ""} ${esc(champ || "TBD")}</div>
    <div class="lbl">Projected champion</div><div class="pct">${p} title probability</div></div>`;
}

function teamRowHTML(team, right, cls) {
  return `<div class="team-row ${cls || ""}">
    <span class="flag">${team ? team.flag : ""}</span>
    <span class="code">${team ? esc(team.code) : "TBD"}</span>
    <span class="val">${right}</span></div>`;
}

function matchCardHTML(m, gridColumn, slot) {
  const path = m.on_predicted_path ? " on-path" : "";
  let body = "";

  if (m.status === "completed") {
    const homeWin = m.winner === (m.home && m.home.name);
    body += teamRowHTML(m.home, m.home_score, homeWin ? "winner" : "loser");
    body += teamRowHTML(m.away, m.away_score, homeWin ? "loser" : "winner");
    if (m.penalties) body += `<div class="pen-note">&#9917; Pens ${m.penalties.home}&ndash;${m.penalties.away}</div>`;
  } else if (m.prediction) {
    const ph = m.prediction.home_advance, pa = m.prediction.away_advance;
    body += teamRowHTML(m.home, `<span class="${ph >= 0.5 ? "prob-hi" : "prob-lo"}">${pct(ph)}</span>`);
    body += teamRowHTML(m.away, `<span class="${pa >= 0.5 ? "prob-hi" : "prob-lo"}">${pct(pa)}</span>`);
    body += `<div class="prob-bar"><div class="ph" style="width:${ph * 100}%"></div><div class="pa" style="width:${pa * 100}%"></div></div>`;
    body += `<div class="conf-note">confidence ${pct(m.prediction.confidence)}</div>`;
  } else {
    // TBD match: most likely candidate for each slot
    const homeSide = teamsInSubtree(m.home_from);
    const cands = m.slot_candidates || [];
    const homeTop = cands.find((c) => homeSide.has(c.team));
    const awayTop = cands.find((c) => !homeSide.has(c.team));
    const mk = (c) => c
      ? `<div class="team-row tbd"><span class="flag">${c.flag}</span><span class="code">${esc(c.code)}</span><span class="val">${pct(c.p_appear)}</span></div>`
      : `<div class="team-row tbd"><span class="code">TBD</span></div>`;
    body += mk(homeTop) + mk(awayTop);
    body += `<div class="conf-note">most likely participants</div>`;
  }

  return `<div class="match-card${path}" style="grid-column:${gridColumn};grid-row:${slot + 2}" data-mid="${m.id}" role="button" tabindex="0"
    aria-label="${ROUND_LABELS[m.round]} match ${esc(m.id)} details">
    <div class="meta"><span>${m.id} &middot; ${ROUND_LABELS[m.round]}</span><span>${shortDate(m.date)}</span></div>
    ${body}</div>`;
}

/* ---------------- title race table ---------------- */
function renderTitleTable() {
  document.getElementById("sims-label").textContent = BRACKET.n_simulations.toLocaleString();
  const rows = BRACKET.title_ranking.slice(0, 10);
  const maxP = rows.length ? rows[0].p_champion : 1;
  let html = `<div class="title-row title-head"><span>#</span><span></span><span>Win the title</span><span class="pcts">Title</span><span class="pcts">Final</span></div>`;
  rows.forEach((r, i) => {
    html += `<div class="title-row${i === 0 ? " top" : ""}">
      <span>${i + 1}</span>
      <span>${r.flag} <b>${esc(r.code)}</b></span>
      <span class="bar-track"><span class="bar-fill" style="display:block;width:${(r.p_champion / maxP) * 100}%"></span></span>
      <span class="pcts main">${pct(r.p_champion, 1)}</span>
      <span class="pcts">${pct(r.p_final, 1)}</span></div>`;
  });
  document.getElementById("title-table").innerHTML = html;
}

/* ============================================================
   MODALS
   ============================================================ */
const backdrop = document.getElementById("modal-backdrop");
const modalBody = document.getElementById("modal-body");
function openModal(html) { modalBody.innerHTML = html; backdrop.classList.remove("hidden"); }
function closeModal() { backdrop.classList.add("hidden"); }
document.getElementById("modal-close").addEventListener("click", closeModal);
backdrop.addEventListener("click", (e) => { if (e.target === backdrop) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

function teamLink(name) {
  return `<span class="team-link" onclick="openTeamModal('${esc(name)}')">${esc(name)}</span>`;
}

function openMatchModal(mid) {
  const m = MATCH_BY_ID[mid];
  const head = `<h3>${ROUND_LABELS[m.round]} &middot; ${m.id}</h3>
    <div class="sub">${shortDate(m.date)}, 2026 &middot; ${esc(m.venue)}</div>`;

  if (m.status === "completed") {
    const pens = m.penalties ? ` <small>(${m.penalties.home}&ndash;${m.penalties.away} pens)</small>` : "";
    openModal(head + `
      <div class="big-prob">
        <div class="box"><div class="pct">${m.home_score}</div><div class="who">${m.home.flag} ${teamLink(m.home.name)}</div></div>
        <div class="box"><div class="pct">${m.away_score}</div><div class="who">${m.away.flag} ${teamLink(m.away.name)}</div></div>
      </div>
      <p class="sub">Final score${pens}. Winner: <b>${esc(m.winner)}</b> &mdash; already folded into the simulation.</p>`);
    return;
  }

  if (m.prediction) {
    const p = m.prediction;
    const dh = p.drivers[m.home.name], da = p.drivers[m.away.name];
    const row = (label, key) => `<tr><td>${label}</td><td>${dh[key]}</td><td>${da[key]}</td></tr>`;
    openModal(head + `
      <div class="big-prob">
        <div class="box"><div class="pct" style="color:var(--accent)">${pct(p.home_advance, 1)}</div>
          <div class="who">${m.home.flag} ${teamLink(m.home.name)} advances</div></div>
        <div class="box"><div class="pct" style="color:var(--red)">${pct(p.away_advance, 1)}</div>
          <div class="who">${m.away.flag} ${teamLink(m.away.name)} advances</div></div>
      </div>
      <p class="sub">Expected goals: ${p.xg_home} &ndash; ${p.xg_away} &middot; 90-min draw probability ${pct(p.p_draw_90)} (resolved by extra time / penalties) &middot; model confidence ${pct(p.confidence)}</p>
      <table>
        <tr><th>Key drivers</th><th>${m.home.flag} ${esc(m.home.code)}</th><th>${m.away.flag} ${esc(m.away.code)}</th></tr>
        ${row("Attack score", "attack")}${row("Defense score", "defense")}${row("Form score", "form")}
        ${row("Discipline score", "discipline")}${row("Player impact", "player_impact")}${row("Overall strength", "overall")}
      </table>
      <p class="sub">Scores are 0&ndash;100, normalized across the 32 knockout teams. Click a team name for its full profile.</p>`);
    return;
  }

  // TBD match: candidates table
  const cands = (m.slot_candidates || []).slice(0, 6);
  const rows = cands.map((c) =>
    `<tr><td>${c.flag} ${teamLink(c.team)}</td><td>${pct(c.p_appear, 1)}</td><td>${pct(c.p_win_match, 1)}</td></tr>`).join("");
  openModal(head + `
    <p class="sub">Participants are decided by earlier rounds. Most likely contenders from ${BRACKET.n_simulations.toLocaleString()} simulations:</p>
    <table><tr><th>Team</th><th>Plays this match</th><th>Wins this match</th></tr>${rows}</table>`);
}

async function openTeamModal(name) {
  openModal(`<h3>${esc(name)}</h3><p class="sub">Loading&hellip;</p>`);
  const t = await fetchJSON(`/api/team/${encodeURIComponent(name)}`);
  const sc = t.scores, adv = t.advancement;
  const cell = (k, label) =>
    `<div class="score-cell"><div class="v">${sc[k]}</div><div class="k">${label} &middot; #${t.ranks[k]}</div></div>`;
  const players = t.key_players.map((p) =>
    `<tr><td>${esc(p.name)}</td><td>${p.position}</td><td>${p.goals}</td><td>${p.assists}</td><td>${p.appearances}</td><td>${p.minutes}&prime;</td><td>${p.rating}</td></tr>`).join("");
  openModal(`
    <h3>${t.flag} ${esc(t.team)} <small style="color:var(--text-dim)">(${esc(t.code)} &middot; Group ${t.group})</small></h3>
    <div class="sub">${t.stats.record} &middot; goals ${t.stats.goals} &middot; group points ${t.stats.group_points} &middot; form: ${t.stats.recent_results.join(" ")}</div>
    <div class="score-grid">
      ${cell("overall", "Overall")}${cell("attack", "Attack")}${cell("defense", "Defense")}
      ${cell("form", "Form")}${cell("player_impact", "Players")}${cell("discipline", "Discipline")}
    </div>
    <div class="big-prob">
      <div class="box"><div class="pct">${pct(adv.reach_sf, 1)}</div><div class="who">reach semi-final</div></div>
      <div class="box"><div class="pct">${pct(adv.reach_final, 1)}</div><div class="who">reach final</div></div>
      <div class="box"><div class="pct" style="color:var(--gold)">${pct(adv.win_title, 1)}</div><div class="who">win the title</div></div>
    </div>
    <table>
      <tr><td>Possession</td><td>${t.stats.possession}%</td><td>Passing accuracy</td><td>${t.stats.passing_accuracy}%</td></tr>
      <tr><td>Shots / match</td><td>${t.stats.shots_per_match}</td><td>On target / match</td><td>${t.stats.shots_on_target_per_match}</td></tr>
      <tr><td>Fouls / match</td><td>${t.stats.fouls_per_match}</td><td>Cards</td><td>${t.stats.cards}</td></tr>
      <tr><td>Clean sheets</td><td>${t.stats.clean_sheets}</td><td>Matches played</td><td>${t.stats.matches_played}</td></tr>
    </table>
    <h3 style="font-size:0.95rem;margin-top:14px">Key players</h3>
    <table><tr><th>Player</th><th>Pos</th><th>G</th><th>A</th><th>Apps</th><th>Min</th><th>Rating</th></tr>${players}</table>`);
}
window.openTeamModal = openTeamModal;

/* ============================================================
   STOCKS
   ============================================================ */
async function loadSponsors() {
  const data = await fetchJSON("/api/stocks");
  SPONSOR_LIST = data.sponsors;
  const sel = document.getElementById("stock-select");
  sel.innerHTML = data.sponsors
    .map((s) => `<option value="${esc(s.ticker)}">${esc(s.name)} (${esc(s.ticker)})</option>`)
    .join("");
  sel.addEventListener("change", () => loadStock(sel.value));
  const selected = ACTIVE_CONFIG?.stock_ticker || data.sponsors[0]?.ticker;
  if (selected) {
    sel.value = selected;
    if (!BRACKET) await loadStock(selected);
  }
}

async function loadStock(ticker) {
  CURRENT_TICKER = ticker;
  const chartEl = document.getElementById("stock-chart");
  chartEl.innerHTML = `<div class="loading">Loading ${esc(ticker)}&hellip;</div>`;
  document.getElementById("stock-chips").innerHTML = "";

  let d;
  try {
    d = await fetchJSON(`/api/stock/${encodeURIComponent(ticker)}`);
  } catch (err) {
    chartEl.innerHTML = `<div class="loading">Failed to load ${esc(ticker)}: ${esc(err.message)}</div>`;
    return;
  }
  if (CURRENT_TICKER !== ticker) return;   // user switched meanwhile

  renderStockFromPayload(d);
}

function renderStockFromPayload(d) {
  renderStockChips(d);
  renderStockChart(d);
  const model = d.model_info || {};
  const chip = document.getElementById("active-stock-model");
  if (chip && model.model_label) {
    chip.innerHTML = `Stock model: <b>${esc(model.model_label)}</b> &middot; rows ${model.training_rows} &middot; features ${model.feature_count}`;
  }
}

function renderStockChips(d) {
  const last = d.history.close[d.history.close.length - 1];
  const chg = d.forecast.expected_7d_change_pct;
  const srcChip = d.is_demo
    ? `<span class="chip amber">&#9888; DEMO DATA (synthetic &mdash; download unavailable)</span>`
    : `<span class="chip blue">&#128309; REAL DATA &middot; Yahoo Finance</span>`;
  document.getElementById("stock-chips").innerHTML = `
    ${srcChip}
    <span class="chip">Last close: <b>${last} ${esc(d.sponsor.currency)}</b></span>
    <span class="chip ${chg >= 0 ? "green" : "red"}">7-day model forecast: <b>${chg >= 0 ? "+" : ""}${chg}%</b></span>
    <span class="chip">Sponsor exposure: <b>${(d.exposure_score * 100).toFixed(0)}/100</b></span>
    <span class="chip">Model: <b>${esc(d.forecast.model)}</b></span>`;
}

function renderStockChart(d) {
  const el = document.getElementById("stock-chart");
  const h = d.history, f = d.forecast;
  // Connect the forecast to the last real point for a continuous line.
  const lastDate = h.dates[h.dates.length - 1];
  const lastClose = h.close[h.close.length - 1];
  const fDates = [lastDate, ...f.dates];
  const fPred = [lastClose, ...f.predicted];
  const fLow = [lastClose, ...f.lower];
  const fUp = [lastClose, ...f.upper];

  el.innerHTML = "";   // remove the loading placeholder before plotting
  if (!window.Plotly) { renderSVGFallback(el, h, fDates, fPred, fLow, fUp); return; }

  const traces = [
    { x: fDates, y: fUp, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: fDates, y: fLow, mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: "rgba(77,163,255,0.12)", name: "80% confidence band", hoverinfo: "skip" },
    { x: h.dates, y: h.close, mode: "lines", name: d.is_demo ? "Price (DEMO)" : "Price (real)",
      line: { color: "#4da3ff", width: 2.5 } },
    { x: fDates, y: fPred, mode: "lines+markers", name: "7-day prediction (experimental)",
      line: { color: "#fbbf24", width: 2, dash: "dash" }, marker: { size: 5 } },
  ];
  const layout = {
    title: { text: `${d.sponsor.name} (${d.sponsor.ticker}) — since World Cup kickoff 2026-06-11`,
             font: { size: 14, color: "#e8ecf6" } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9aa5c3", size: 11 },
    xaxis: { gridcolor: "rgba(255,255,255,0.07)", zeroline: false },
    yaxis: { gridcolor: "rgba(255,255,255,0.07)", zeroline: false,
             title: { text: d.sponsor.currency } },
    margin: { l: 55, r: 20, t: 45, b: 40 },
    legend: { orientation: "h", y: -0.18 },
    hovermode: "x unified",
  };
  Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
}

/* Minimal SVG line chart used when the Plotly CDN is unreachable (offline). */
function renderSVGFallback(el, h, fDates, fPred, fLow, fUp) {
  const W = el.clientWidth || 900, H = el.clientHeight || 420, P = 40;
  const all = [...h.close, ...fPred, ...fLow, ...fUp];
  const min = Math.min(...all) * 0.995, max = Math.max(...all) * 1.005;
  const n = h.close.length + fPred.length - 1;
  const X = (i) => P + (i / (n - 1)) * (W - 2 * P);
  const Y = (v) => H - P - ((v - min) / (max - min)) * (H - 2 * P);
  const pts = (arr, off) => arr.map((v, i) => `${X(i + off)},${Y(v)}`).join(" ");
  const histPts = pts(h.close, 0);
  const off = h.close.length - 1;
  const band = fUp.map((v, i) => `${X(i + off)},${Y(v)}`).join(" ") + " " +
    [...fLow].reverse().map((v, i) => `${X(fLow.length - 1 - i + off)},${Y(v)}`).join(" ");
  el.innerHTML = `
    <svg width="100%" height="100%" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <polygon points="${band}" fill="rgba(77,163,255,0.12)"/>
      <polyline points="${histPts}" fill="none" stroke="#4da3ff" stroke-width="2.5"/>
      <polyline points="${pts(fPred, off)}" fill="none" stroke="#fbbf24" stroke-width="2" stroke-dasharray="6 4"/>
      <text x="${P}" y="18" fill="#9aa5c3" font-size="12">Offline fallback chart — blue: history, dashed: prediction</text>
    </svg>`;
}

/* ============================================================
   RECALCULATE + INIT
   ============================================================ */
async function refreshDashboard(config, button) {
  const btn = button || document.getElementById("btn-predict");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Predicting…";
  }
  startPredictionProgress();
  try {
    const payload = await postRecalculate(config);
    applyDashboardPayload(payload);
    persistUiState(config);
    stopPredictionProgress(true);
    return payload;
  } catch (err) {
    stopPredictionProgress(false);
    throw err;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = btn.id === "btn-predict" ? "Predict" : "&#8635; Recalculate";
      if (btn.id === "btn-recalc") btn.innerHTML = "&#8635; Recalculate";
    }
  }
}

function wireControls() {
  document.getElementById("football-model-select").addEventListener("change", () => {
    const current = collectConfigFromUI();
    renderModelParams("football", current.football.model, current.football.params);
    updateControlDescriptions();
  });
  document.getElementById("stock-model-select").addEventListener("change", () => {
    const current = collectConfigFromUI();
    renderModelParams("stock", current.stock.model, current.stock.params);
    updateControlDescriptions();
  });
  document.getElementById("stock-select").addEventListener("change", (e) => {
    CURRENT_TICKER = e.currentTarget.value;
    if (ACTIVE_CONFIG) ACTIVE_CONFIG.stock_ticker = CURRENT_TICKER;
    loadStock(CURRENT_TICKER);
  });
  document.getElementById("btn-predict").addEventListener("click", async () => {
    await refreshDashboard(collectConfigFromUI(), document.getElementById("btn-predict"));
  });
  document.getElementById("btn-reset-defaults").addEventListener("click", async () => {
    const config = defaultConfig();
    ACTIVE_CONFIG = config;
    applyConfigToControls(config);
    await refreshDashboard(config, document.getElementById("btn-predict"));
  });
  document.getElementById("btn-recalc").addEventListener("click", async (e) => {
    await refreshDashboard(collectConfigFromUI(), e.currentTarget);
  });
}

async function initControlsAndDashboard() {
  const [defaults, lastConfig, sponsorsResp] = await Promise.all([
    fetchJSON("/api/model_defaults"),
    fetchJSON("/api/last_config"),
    fetchJSON("/api/stocks"),
  ]);
  MODEL_DEFAULTS = defaults;
  LAST_CONFIG = mergeConfig(defaultConfig(), lastConfig);
  ACTIVE_CONFIG = LAST_CONFIG || defaultConfig();
  setSelectOptions(document.getElementById("football-model-select"), Object.keys(MODEL_DEFAULTS.football).map((name) => ({ value: name, label: `${MODEL_DEFAULTS.football[name].label}` })));
  setSelectOptions(document.getElementById("stock-model-select"), Object.keys(MODEL_DEFAULTS.stock).map((name) => ({ value: name, label: `${MODEL_DEFAULTS.stock[name].label}` })));
  SPONSOR_LIST = sponsorsResp.sponsors || [];
  const sponsorSelect = document.getElementById("stock-select");
  sponsorSelect.innerHTML = SPONSOR_LIST.map((s) => `<option value="${esc(s.ticker)}">${esc(s.name)} (${esc(s.ticker)})</option>`).join("");
  sponsorSelect.value = ACTIVE_CONFIG.stock_ticker;
  applyConfigToControls(ACTIVE_CONFIG);
  wireControls();
  await refreshDashboard(ACTIVE_CONFIG, document.getElementById("btn-predict"));
}

(async function init() {
  try {
    await initControlsAndDashboard();
  } catch (err) {
    document.getElementById("bracket").innerHTML =
      `<p style="color:var(--red)">Failed to load dashboard: ${esc(err.message)}</p>`;
    document.getElementById("stock-chart").innerHTML =
      `<div class="loading">Failed to load dashboard: ${esc(err.message)}</div>`;
  }
})();
