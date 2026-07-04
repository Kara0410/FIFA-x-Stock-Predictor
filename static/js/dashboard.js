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

/* ---------------- helpers ---------------- */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const shortDate = (iso) => new Date(iso + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });

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

function renderHeroChips() {
  const done = BRACKET.matches.filter((m) => m.status === "completed").length;
  const total = BRACKET.matches.length;
  const champ = BRACKET.predicted_champion;
  const champFlag = champ ? flagOf(champ) : "";
  document.getElementById("hero-chips").innerHTML = `
    <span class="chip">Knockout matches: <b>${done}/${total}</b> played</span>
    <span class="chip">Simulations: <b>${BRACKET.n_simulations.toLocaleString()}</b></span>
    <span class="chip">Projected champion: <b><span class="flag">${champFlag}</span>${esc(champ || "TBD")}</b></span>
    <span class="chip">Sample tournament data &middot; as of 2026-07-04</span>`;
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

  const col = (title, ids, side) =>
    `<div class="bracket-col branch-${side}"><div class="col-title">${title}</div>` +
    ids.map((id) => matchCardHTML(MATCH_BY_ID[id])).join("") +
    `</div>`;

  const titles = ["Semi-final", "Quarter-finals", "Round of 16", "Round of 32"];
  const leftCols = [3, 2, 1, 0].map((i) => col(titles[i], leftLevels[i], "left"));
  const rightCols = [0, 1, 2, 3].map((i) => col(titles[i], rightLevels[i], "right"));

  const champHTML = championCardHTML(final);
  const finalCol =
    `<div class="bracket-col"><div class="col-title">Final &middot; Jul 19</div>` +
    matchCardHTML(final) + champHTML + `</div>`;

  el.innerHTML = leftCols.join("") + finalCol + rightCols.join("");

  el.querySelectorAll(".match-card").forEach((card) => {
    card.addEventListener("click", () => openMatchModal(card.dataset.mid));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openMatchModal(card.dataset.mid);
      }
    });
  });
}

function championCardHTML(final) {
  if (final.status === "completed" && final.winner) {
    return `<div class="champ-card"><div class="crown">&#127942;</div>
      <div class="name">${flagOf(final.winner)} ${esc(final.winner)}</div>
      <div class="lbl">World Champion</div></div>`;
  }
  const champ = BRACKET.predicted_champion;
  const row = BRACKET.title_ranking.find((t) => t.team === champ);
  const p = row ? pct(row.p_champion, 1) : "&mdash;";
  return `<div class="champ-card"><div class="crown">&#128081;</div>
    <div class="name">${champ ? flagOf(champ) : ""} ${esc(champ || "TBD")}</div>
    <div class="lbl">Projected champion</div><div class="pct">${p} title probability</div></div>`;
}

function teamRowHTML(team, right, cls) {
  return `<div class="team-row ${cls || ""}">
    <span class="flag">${team ? team.flag : ""}</span>
    <span class="code">${team ? esc(team.code) : "TBD"}</span>
    <span class="val">${right}</span></div>`;
}

function matchCardHTML(m) {
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

  return `<div class="match-card${path}" data-mid="${m.id}" role="button" tabindex="0"
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
  const sel = document.getElementById("stock-select");
  sel.innerHTML = data.sponsors
    .map((s) => `<option value="${esc(s.ticker)}">${esc(s.name)} (${esc(s.ticker)})</option>`)
    .join("");
  sel.addEventListener("change", () => loadStock(sel.value));
  if (data.sponsors.length) loadStock(data.sponsors[0].ticker);
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

  renderStockChips(d);
  renderStockChart(d);
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
document.getElementById("btn-recalc").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true; btn.textContent = "Recalculating…";
  try {
    await fetchJSON("/api/recalculate");
    await loadBracket();
    if (CURRENT_TICKER) await loadStock(CURRENT_TICKER);
  } finally {
    btn.disabled = false; btn.innerHTML = "&#8635; Recalculate";
  }
});

(async function init() {
  try {
    await loadBracket();
  } catch (err) {
    document.getElementById("bracket").innerHTML =
      `<p style="color:var(--red)">Failed to load bracket: ${esc(err.message)}</p>`;
  }
  try {
    await loadSponsors();
  } catch (err) {
    document.getElementById("stock-chart").innerHTML =
      `<div class="loading">Failed to load sponsors: ${esc(err.message)}</div>`;
  }
})();
