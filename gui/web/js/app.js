/* ENTIP case review: worklist, patient page, slide workbench.
 * Two views over the same data: Clinician (findings, candidate review, plain language) and
 * Researcher (cohort table, layers, thresholds, teaching, export). The view is a per-user setting.
 * Routes live in the hash: #worklist, #case-<study_id>, #slide-<scan_id>. */
(function () {
  "use strict";

  const cfg = window.ENTIP_CONFIG || { mode: "mock", base: "mock" };
  const api = window.ENTIPApi.createApi(cfg);
  const { walk, featureKey } = window.ENTIPApi;
  const isFeature = (a, b) => featureKey(a) === featureKey(b);
  const main = document.getElementById("main");
  const crumbs = document.getElementById("crumbs");
  let view = null; // current view's cleanup

  /* ---------- small helpers ---------- */
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content; };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  const ver = (v) => (typeof v === "number" ? `v${v}` : String(v));
  let toastTimer;
  function toast(msg) {
    const t = document.getElementById("toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
  }
  const statusPill = (s) => `<span class="pill ${s === "Reported" ? "done" : s === "In review" ? "review" : "new"}">${esc(s)}</span>`;
  const when = (iso) => { try { return new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); } catch (e) { return iso; } };
  const actorName = (a) => a ? (a.kind === "model" ? `${a.name}${a.version ? " " + a.version : ""}` : a.name) : "unknown";
  const sourceText = (s) => s ? (s.kind === "model" ? `${s.name} ${s.version || ""}`.trim() : s.name) : "";
  const shortId = (id) => String(id).split("-").pop();
  const KEY_MARKERS = ["IDH", "1p/19q", "MGMT", "CDKN2A/B", "CDKN2A"];
  const MITOSIS = "Mitotic figure";

  function setCrumbs(parts) {
    crumbs.innerHTML = parts.map((p, i) => i === parts.length - 1
      ? `<span class="here">${esc(p.label)}</span>`
      : `<a href="${p.href}">${esc(p.label)}</a><span class="sep" aria-hidden="true">/</span>`).join("");
  }

  function toCSV(rows) {
    const q = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    return rows.map((r) => r.map(q).join(",")).join("\n") + "\n";
  }
  async function copyText(text, what) {
    try { await navigator.clipboard.writeText(text); toast(`${what} copied.`); }
    catch (e) { toast("Copy was blocked here. Use Download instead."); }
  }
  function download(text, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  /* ---------- who is marking ---------- */
  const whoBtn = document.getElementById("who");
  let me = store.get("entip-gui-user", "Demo pathologist");
  function renderWho() { whoBtn.textContent = me; whoBtn.setAttribute("aria-label", `Marking as ${me}. Change name`); }
  whoBtn.addEventListener("click", () => {
    if (whoBtn.querySelector("input")) return;
    whoBtn.innerHTML = `<input id="who-input" aria-label="Your name" value="${esc(me)}">`;
    const inp = whoBtn.querySelector("input");
    inp.focus(); inp.select();
    const done = () => { me = inp.value.trim() || me; store.set("entip-gui-user", me); renderWho(); };
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); done(); } if (e.key === "Escape") renderWho(); e.stopPropagation(); });
    inp.addEventListener("blur", done);
  });
  renderWho();
  document.getElementById("mode").hidden = api.mode !== "mock";

  /* ---------- clinician / researcher view ---------- */
  let mode = store.get("entip-gui-view", "clinician");
  const tabs = document.querySelectorAll("#viewtabs [data-view]");
  function renderTabs() {
    tabs.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === mode)));
    document.documentElement.dataset.view = mode;
  }
  tabs.forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.view === mode) return;
    mode = b.dataset.view; store.set("entip-gui-view", mode); renderTabs(); route();
  }));
  renderTabs();

  /* ---------- router ---------- */
  async function route() {
    if (view) { view(); view = null; }
    const hash = location.hash.slice(1) || "worklist";
    main.innerHTML = "";
    try {
      if (hash.startsWith("case-")) await caseView(hash.slice(5));
      else if (hash.startsWith("slide-")) await slideView(hash.slice(6));
      else await worklist();
    } catch (e) {
      console.error(e);
      main.innerHTML = `<div class="page"><div class="error"><b>Could not load this page.</b><p class="muted">${esc(e.message)}</p><a class="btn" href="#worklist">Back to cases</a></div></div>`;
    }
    main.focus({ preventScroll: true });
  }
  window.addEventListener("hashchange", route);

  /* ---------- worklist ---------- */
  async function worklist() {
    setCrumbs([{ label: mode === "researcher" ? "Cohort" : "Cases" }]);
    document.title = "ENTIP Case Review";
    const rows = await api.listPatients();
    if (mode === "researcher") return cohort(rows);
    const filter = { q: store.get("entip-gui-q", ""), status: store.get("entip-gui-status", "All") };
    const statuses = ["All", "New", "In review", "Reported"];
    main.append(h(`
      <div class="page">
        <div class="listhead">
          <div class="sect"><h1>Cases</h1><span class="muted">${rows.length} patients. One row per patient, by pseudonymised study ID.</span></div>
          <div class="sect" style="align-items:flex-end">
            <input class="search" id="q" type="search" placeholder="Search study ID, diagnosis or marker" aria-label="Search cases" value="${esc(filter.q)}">
            <div class="chips" role="group" aria-label="Filter by review status">
              ${statuses.map((s) => `<button class="chip" type="button" data-s="${s}" aria-pressed="${s === filter.status}">${s}</button>`).join("")}
            </div>
          </div>
        </div>
        <div class="panel tablewrap">
          <table class="cases">
            <thead><tr><th></th><th>Study ID</th><th>Diagnosis</th><th>Key markers</th><th>Model findings</th><th>Status</th></tr></thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
        <p class="empty-note">Invented cases for building the interface. Study IDs are research IDs; the link to the clinical record is held in the NHS key store, not here.</p>
      </div>`));
    const tbody = document.getElementById("rows");
    function draw() {
      const q = filter.q.toLowerCase();
      const hits = rows.filter((r) => (filter.status === "All" || r.review_status === filter.status) &&
        (!q || [r.study_id, r.diagnosis, ...Object.entries(r.molecular || {}).flat()].join(" ").toLowerCase().includes(q)));
      tbody.innerHTML = hits.map((r) => {
        const mo = (r.model_outputs || [])[0];
        return `
        <tr tabindex="0" data-id="${esc(r.study_id)}">
          <td>${r.thumb ? `<img class="thumb" src="${esc(r.thumb)}" alt="">` : `<div class="thumb"></div>`}</td>
          <td><span class="mono">${esc(r.study_id)}</span><div class="sub">${esc(r.sex)} · ${esc(r.age_band)}</div></td>
          <td><div class="dx">${esc(r.diagnosis)}</div><div class="sub">CNS WHO grade ${esc(r.who_grade)}</div></td>
          <td><div class="chips">${KEY_MARKERS.filter((k) => r.molecular && r.molecular[k]).map((k) => `<span class="tag" title="${esc(k)}: ${esc(r.molecular[k])}">${esc(k)} ${esc(short(r.molecular[k]))}</span>`).join("")}</div></td>
          <td class="num">${mo ? `<b>${esc(mo.value)}</b> <span class="sub">mitoses/mm²</span>` : `<span class="sub">none yet</span>`}</td>
          <td>${statusPill(r.review_status)}</td>
        </tr>`;
      }).join("") || `<tr><td colspan="6" class="empty-note" style="padding:20px">No cases match. Clear the search or pick “All”.</td></tr>`;
    }
    draw();
    document.getElementById("q").addEventListener("input", (e) => { filter.q = e.target.value; store.set("entip-gui-q", filter.q); draw(); });
    main.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
      filter.status = b.dataset.s; store.set("entip-gui-status", filter.status);
      main.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", x === b)); draw();
    }));
    rowLinks(tbody);
  }

  function rowLinks(tbody) {
    const open = (tr) => { if (tr && tr.dataset.id) location.hash = `case-${tr.dataset.id}`; };
    tbody.addEventListener("click", (e) => open(e.target.closest("tr")));
    tbody.addEventListener("keydown", (e) => { if (e.key === "Enter") open(e.target.closest("tr")); });
  }

  /* Researcher: the cohort as a sortable table, one row per patient, with export. */
  function cohort(rows) {
    const markers = [...new Set(rows.flatMap((r) => Object.keys(r.molecular || {})))];
    const cols = [
      { k: "study_id", label: "Study ID", v: (r) => r.study_id, mono: true },
      { k: "diagnosis", label: "Diagnosis", v: (r) => r.diagnosis },
      { k: "who_grade", label: "Grade", v: (r) => r.who_grade, num: true },
      { k: "mitoses", label: "Mitoses/mm²", v: (r) => (r.model_outputs || [])[0]?.value ?? "", num: true },
      ...markers.map((m) => ({ k: `m:${m}`, label: m, v: (r) => (r.molecular || {})[m] || "" })),
      { k: "n_scans", label: "Scans", v: (r) => r.n_scans, num: true },
      { k: "n_assays", label: "Assays", v: (r) => r.n_assays, num: true },
      { k: "n_marks", label: "Marks", v: (r) => r.n_marks || 0, num: true },
      { k: "review_status", label: "Status", v: (r) => r.review_status },
    ];
    let sort = store.get("entip-gui-sort", { k: "study_id", dir: 1 });
    if (!cols.some((c) => c.k === sort.k)) sort = { k: "study_id", dir: 1 };
    main.append(h(`
      <div class="page">
        <div class="listhead">
          <div class="sect"><h1>Cohort</h1><span class="muted">${rows.length} patients, ${markers.length} molecular fields. Click a heading to sort.</span></div>
          <div class="row">
            <button class="btn" type="button" id="copy-csv">Copy CSV</button>
            <button class="btn primary" type="button" id="dl-csv">Download CSV</button>
          </div>
        </div>
        <div class="panel tablewrap"><table class="cases cohort"><thead><tr id="heads"></tr></thead><tbody id="rows"></tbody></table></div>
        <p class="empty-note">Model values carry their source on the patient page. Synthetic data.</p>
      </div>`));
    const heads = document.getElementById("heads"), tbody = document.getElementById("rows");
    const sorted = () => {
      const c = cols.find((x) => x.k === sort.k);
      return rows.slice().sort((a, b) => {
        const x = c.v(a), y = c.v(b);
        return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })) * sort.dir;
      });
    };
    function draw() {
      heads.innerHTML = cols.map((c) => `<th aria-sort="${c.k === sort.k ? (sort.dir > 0 ? "ascending" : "descending") : "none"}"><button class="sorter" type="button" data-k="${esc(c.k)}">${esc(c.label)}${c.k === sort.k ? (sort.dir > 0 ? " ▲" : " ▼") : ""}</button></th>`).join("");
      tbody.innerHTML = sorted().map((r) => `<tr tabindex="0" data-id="${esc(r.study_id)}">${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.mono ? `<span class="mono">${esc(c.v(r))}</span>` : esc(c.v(r))}</td>`).join("")}</tr>`).join("");
    }
    heads.addEventListener("click", (e) => {
      const b = e.target.closest("[data-k]"); if (!b) return;
      sort = sort.k === b.dataset.k ? { k: sort.k, dir: -sort.dir } : { k: b.dataset.k, dir: 1 };
      store.set("entip-gui-sort", sort); draw();
    });
    const csv = () => toCSV([cols.map((c) => c.label), ...sorted().map((r) => cols.map((c) => c.v(r)))]);
    document.getElementById("copy-csv").onclick = () => copyText(csv(), "Cohort table");
    document.getElementById("dl-csv").onclick = () => download(csv(), "entip_cohort.csv");
    draw();
    rowLinks(tbody);
  }

  function short(v) {
    const m = { wildtype: "wt", "IDH1 R132H": "R132H", methylated: "meth", unmethylated: "unmeth", "homozygous deletion": "del",
      codeleted: "codel", intact: "intact", retained: "retained", "not tested": "n/t" };
    return m[v] || v;
  }

  /* ---------- patient page ---------- */
  async function caseView(id) {
    const p = await api.getPatient(id);
    const seen = new Set(); // keep every person's action; for model updates only the latest per feature and scan
    const actions = (await api.listActions({ study_id: id })).filter((a) => {
      if (a.verb !== "train_head") return true;
      const k = `${a.feature}|${a.scan_id}`;
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
    setCrumbs([{ label: mode === "researcher" ? "Cohort" : "Cases", href: "#worklist" }, { label: p.study_id }]);
    document.title = `${p.study_id} · ENTIP`;
    const scans = [], assays = [];
    walk(p, (k, o, trail) => { if (k === "scan") scans.push({ ...o, trail }); if (k === "assay") assays.push({ ...o, trail }); });
    const firstHE = scans.find((s) => s.tile_source);

    const cards = scans.map((sc) => sc.tile_source ? `
      <a class="panel slidecard" href="#slide-${esc(sc.scan_id)}">
        <img src="${esc(sc.thumb)}" alt="Thumbnail of ${esc(sc.stain)} slide ${esc(sc.scan_id)}">
        <div class="meta"><b>${esc(sc.stain)} · ${esc(sc.objective)}</b><span class="sub mono">${esc(sc.scan_id)} · ${esc(sc.mpp)} µm/px</span><span class="sub">${mode === "researcher" ? "Open to explore and teach" : "Open to review findings"}</span></div>
      </a>` : `
      <div class="panel slidecard empty">
        <div class="ph">${esc(sc.note || "No image yet")}</div>
        <div class="meta"><b>${esc(sc.stain)}</b><span class="sub mono">${esc(sc.scan_id)}</span></div>
      </div>`).join("");

    const outputs = (p.model_outputs || []).map((o) => `
      <div class="output">
        <span><b>${esc(o.measure)}</b></span><span class="v">${esc(o.value)}</span>
        <span class="prov">${esc(o.detail)} · ${esc(sourceText(o.source))} on <span class="mono">${esc(o.object_id)}</span></span>
      </div>`).join("");
    const assayList = assays.filter((a) => a.result).map((a) => `
      <div class="output"><span><b>${esc(a.kind)}</b> <span class="muted">${esc(a.platform)}</span></span><span></span>
        <span class="prov">${esc(a.result)} · ${esc(sourceText(a.source))}</span></div>`).join("");
    const head = `
        <div class="casehead">
          <div class="sect">
            <div class="ids"><span class="mono">${esc(p.study_id)}</span><span>${esc(p.sex)} · ${esc(p.age_band)}</span>${statusPill(p.review_status)}</div>
            <h1>${esc(p.diagnosis)}, CNS WHO grade ${esc(p.who_grade)}</h1>
          </div>
          ${firstHE ? `<a class="btn primary" href="#slide-${esc(firstHE.scan_id)}">Open H&amp;E</a>` : ""}
        </div>`;
    const mol = `<dl class="kv">${Object.entries(p.molecular || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
    const log = (n) => actions.length ? `<ul class="log">${actions.slice(0, n).map(logItem).join("")}</ul>` : `<p class="empty-note">No activity yet. Marks made on a slide appear here, with the person's name and the time.</p>`;

    if (mode === "clinician") {
      // A layered summary in the order of a WHO CNS5 integrated report, then the evidence.
      const meth = assays.find((a) => a.kind === "DNA methylation array" && a.result);
      main.append(h(`
        <div class="page">
          ${head}
          <div class="grid2">
            <div class="sect" style="gap:20px">
              <section class="panel report" aria-labelledby="t-rep"><div class="box">
                <h2 id="t-rep">Summary</h2>
                <div class="layer"><span class="lk">Integrated diagnosis</span><span class="lv">${esc(p.diagnosis)}</span></div>
                <div class="layer"><span class="lk">CNS WHO grade</span><span class="lv">${esc(p.who_grade)}</span></div>
                <div class="layer"><span class="lk">Molecular information</span><span class="lv">${Object.entries(p.molecular || {}).map(([k, v]) => `${esc(k)} ${esc(v)}`).join("; ")}</span></div>
                ${meth ? `<div class="layer"><span class="lk">Methylation class</span><span class="lv">${esc(meth.result)}</span></div>` : ""}
                <p class="help">Built from the research record. Not a clinical report.</p>
              </div></section>
              <section class="sect" aria-labelledby="t-slides"><h2 id="t-slides">Slides</h2><div class="slides">${cards}</div></section>
            </div>
            <div class="sect" style="gap:20px">
              <section class="panel" aria-labelledby="t-out"><div class="box"><h2 id="t-out">Model findings</h2>
                <div class="outputs">${outputs || `<p class="empty-note">No model findings yet.</p>`}</div>
                ${firstHE && outputs ? `<a class="btn" href="#slide-${esc(firstHE.scan_id)}">Check the detections on the slide</a>` : ""}
                <p class="help">Each finding names the model and version that made it. Mock values.</p></div></section>
              <section class="panel" aria-labelledby="t-log"><div class="box"><h2 id="t-log">Recent activity</h2>${log(6)}</div></section>
            </div>
          </div>
        </div>`));
      return;
    }

    const tree = (p.specimens || []).map((sp) => `
      <li><div class="node"><span class="k">Specimen</span><span>${esc(sp.procedure)}, ${esc(sp.site)}</span><span class="mono muted">${esc(sp.specimen_id)}</span></div>
        <ul>${(sp.blocks || []).map((bl) => `
          <li><div class="node"><span class="k">Block</span><span>${esc(bl.label)}</span><span class="mono muted">${esc(bl.block_id)}</span></div>
            <ul>${(bl.sections || []).map((se) => `
              <li><div class="node"><span class="k">Section</span><span>L${esc(se.level)} ${esc(se.stain)}</span><span class="muted">${esc(se.thickness_um)} µm</span></div>
                <ul>
                  ${(se.scans || []).map((sc) => `<li><div class="node"><span class="k">Scan</span>${sc.tile_source ? `<a href="#slide-${esc(sc.scan_id)}" class="mono">${esc(sc.scan_id)}</a>` : `<span class="mono muted">${esc(sc.scan_id)}</span>`}<span class="tag">${esc(sc.copy)} copy</span></div></li>`).join("")}
                  ${(se.assays || []).map((as) => `<li><div class="node"><span class="k">Assay</span><span>${esc(as.kind)}</span><span class="tag ${as.status === "planned" ? "" : "alt"}">${esc(as.status)}</span></div></li>`).join("")}
                </ul></li>`).join("")}
            </ul></li>`).join("")}
        </ul></li>`).join("");

    main.append(h(`
      <div class="page">
        ${head}
        <div class="grid3">
          <section class="panel tree" aria-labelledby="t-tree"><div class="box"><h2 id="t-tree">Tissue</h2><ul>${tree}</ul></div></section>
          <div class="sect" style="gap:20px">
            <section class="sect" aria-labelledby="t-slides"><h2 id="t-slides">Slides</h2><div class="slides">${cards}</div></section>
            <section class="panel" aria-labelledby="t-out"><div class="box"><h2 id="t-out">Model outputs and assays</h2>
              <div class="outputs">${outputs || `<p class="empty-note">No model outputs yet.</p>`}${assayList}</div>
              <p class="help">Each result names the model and version that produced it. Mock values, not diagnoses.</p></div></section>
          </div>
          <div class="sect" style="gap:20px">
            <section class="panel" aria-labelledby="t-mol"><div class="box"><h2 id="t-mol">Molecular</h2>${mol}</div></section>
            <section class="panel" aria-labelledby="t-log"><div class="box"><h2 id="t-log">Activity</h2>${log(60)}</div></section>
          </div>
        </div>
      </div>`));
  }

  function logItem(a) {
    const who = esc(actorName(a.actor));
    let text;
    if (a.verb === "label") text = `<b>${who}</b> marked <span class="mono">${esc(shortId(a.object_id))}</span> as ${a.label ? "" : "not "}${esc(a.feature)}${a.context ? ` <span class="muted">(${esc(a.context)})</span>` : ""}`;
    else if (a.verb === "retract_label") text = `<b>${who}</b> removed the mark on <span class="mono">${esc(shortId(a.object_id))}</span> (${esc(a.feature)})`;
    else if (a.verb === "train_head") text = `<b>${who}</b> updated: ${esc(a.detail)}`;
    else if (a.verb === "define_feature") text = `<b>${who}</b> added feature “${esc(a.feature)}”`;
    else text = `<b>${who}</b> ${esc(a.verb)}`;
    const cls = a.verb === "label" ? (a.label ? "yes" : "no") : a.actor && a.actor.kind === "person" ? "person" : "";
    return `<li class="${cls}"><span>${text}<br><span class="muted">${esc(when(a.ts))}${a.scan_id ? ` · <span class="mono">${esc(a.scan_id)}</span>` : ""}</span></span></li>`;
  }

  /* ---------- slide page (shared viewer, panel by view) ---------- */
  async function slideView(scanId) {
    const scan = await api.getScan(scanId);
    if (!scan.tile_source) throw new Error(`Scan ${scanId} has no image in this mock.`);
    const [table, outputs] = await Promise.all([api.getCells(scanId), api.getOutputs(scanId)]);
    setCrumbs([{ label: mode === "researcher" ? "Cohort" : "Cases", href: "#worklist" }, { label: scan.study_id, href: `#case-${scan.study_id}` }, { label: `${scan.stain} ${scanId}` }]);
    document.title = `${scanId} · ENTIP`;

    main.append(h(`
      <div class="bench">
        <div class="stage" id="stage" aria-label="Slide viewer">
          <div class="zoomer">
            <button class="btn" type="button" id="z-in" aria-label="Zoom in">+</button>
            <button class="btn" type="button" id="z-out" aria-label="Zoom out">−</button>
            <button class="btn" type="button" id="z-home" aria-label="Whole slide">⤢</button>
          </div>
          <div class="scalebar" aria-hidden="true"><span id="sb-l">–</span><i id="sb"></i></div>
          <div class="readout" id="readout">scan:${esc(scanId)}</div>
        </div>
        <aside class="side" id="side" aria-label="${mode === "researcher" ? "Layers, teaching and export" : "Findings and review"}"></aside>
      </div>`));

    let onCell = () => {};
    const sv = new window.ENTIPViewer.SlideView(document.getElementById("stage"), {
      tileSource: scan.tile_source, mpp: table.mpp, cells: table.cells,
      onCellClick: (c) => onCell(c),
      onMove: (x, y) => { document.getElementById("readout").textContent = `x ${x.toFixed(1)} µm  y ${y.toFixed(1)} µm`; },
    });
    sv.setRegions(table.regions || []);
    const scalebar = () => {
      const ppu = sv.pxPerUm(); if (!ppu) return;
      const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
      const um = nice.find((n) => n * ppu >= 70) || 2000;
      document.getElementById("sb").style.width = `${um * ppu}px`;
      document.getElementById("sb-l").textContent = um >= 1000 ? `${um / 1000} mm` : `${um} µm`;
    };
    ["animation", "open", "resize"].forEach((ev) => sv.osd.addHandler(ev, scalebar));
    document.getElementById("z-in").onclick = () => sv.osd.viewport.zoomBy(1.6);
    document.getElementById("z-out").onclick = () => sv.osd.viewport.zoomBy(1 / 1.6);
    document.getElementById("z-home").onclick = () => sv.osd.viewport.goHome();
    const opened = new Promise((res) => { if (sv.osd.world.getItemAt(0)) res(); else sv.osd.addHandler("open", () => res()); });

    const ctx = { scan, scanId, table, outputs, sv, opened, side: document.getElementById("side"), setOnCell: (f) => { onCell = f; } };
    const stop = mode === "researcher" ? await researcherPanel(ctx) : await clinicianPanel(ctx);
    view = () => { if (stop) stop(); sv.destroy(); };
  }

  /* Labels and the live head for one feature on this scan, shared by both panels. */
  async function teacher(scanId, studyId, feature) {
    const t = { feature, head: null, labels: new Map(), status: null, queue: [] };
    t.start = async () => {
      t.head = (await api.createHead({ feature, level: "cell", scan_id: scanId })).head_id;
      t.labels = new Map((await api.getLabels(t.head)).map((x) => [x.cell_id, x.label]));
      t.status = await api.headStatus(t.head);
      t.queue = await api.getQueue(t.head, { scan_id: scanId, n: 20 });
      return t;
    };
    t.mark = async (cellId, label, context) => {
      const res = label === null
        ? await api.removeLabel(t.head, { scan_id: scanId, cell_id: cellId, actor: me, study_id: studyId })
        : await api.addLabel(t.head, { scan_id: scanId, cell_id: cellId, label, actor: me, study_id: studyId, context });
      if (label === null) t.labels.delete(cellId); else t.labels.set(cellId, label);
      t.last = res;
      t.status = await api.headStatus(t.head);
      t.queue = res.queue || await api.getQueue(t.head, { scan_id: scanId, n: 20 });
      return res;
    };
    t.predictions = async () => (t.status && t.status.n_pos + t.status.n_neg ? api.getPredictions(t.head, { scan_id: scanId }) : []);
    return t.start();
  }

  /* ----- Clinician: what the models found, and a quick agree / disagree on each detection. ----- */
  async function clinicianPanel({ scan, scanId, table, outputs, sv, opened, side, setOnCell }) {
    const out = outputs.find((o) => isFeature(o.feature, MITOSIS)) || outputs[0];
    const t = await teacher(scanId, scan.study_id, out ? out.feature : MITOSIS);
    sv.setShow("cells", false);
    sv.setShow("queue", false);
    sv.setLabels(t.labels);
    if (!out) {
      side.innerHTML = `<section class="sect"><h2>Findings</h2><p class="empty-note">No model has been run on this slide yet.</p></section>`;
      return null;
    }
    sv.setThreshold(out.threshold ?? 0.5);
    sv.setPredictions(out.predictions);
    const thr = out.threshold ?? 0.5;
    const called = out.predictions.filter((x) => x.p >= thr).sort((a, b) => b.p - a.p);
    const border = out.predictions.filter((x) => x.p >= 0.3 && x.p < thr).sort((a, b) => b.p - a.p);
    const area = scan.tissue_area_mm2 || table.tissue_area_mm2;
    let cur = 0;

    side.innerHTML = `
      <section class="sect" aria-labelledby="t-f">
        <h2 id="t-f">Findings</h2>
        <div class="finding">
          <div class="big num">${area ? (called.length / area).toFixed(1) : called.length}</div>
          <div><b>${esc(isFeature(out.feature, MITOSIS) ? "mitotic figures per mm²" : out.feature)}</b>
            <div class="prov">${called.length} detections in ${area ? area + " mm² of" : "the"} tissue · ${esc(out.model)} ${esc(out.version)}</div></div>
        </div>
        <div class="reviewbar" aria-live="polite"><div class="track"><i id="rv-bar"></i></div><span id="rv-text" class="num"></span></div>
      </section>
      <section class="sect" aria-labelledby="t-c">
        <div class="row" style="justify-content:space-between"><h2 id="t-c">Detections to check</h2><span class="help">Agree <kbd>1</kbd> · Disagree <kbd>2</kbd> · Next <kbd>J</kbd></span></div>
        <div class="gallery" id="gal"></div>
      </section>
      ${border.length ? `<details class="sect more"><summary>Borderline cells (${border.length})</summary><p class="help">Just under the model's cut-off. Worth a look if the count is near a grading threshold.</p><div class="gallery" id="gal2"></div></details>` : ""}
      <p class="help">Your answers are saved with your name and teach the model. Research use only.</p>`;

    const all = [...called, ...border];
    async function tile(x, i) {
      const el = document.createElement("div");
      el.className = "cand"; el.dataset.id = x.cell_id; el.dataset.i = i;
      el.innerHTML = `
        <button class="pic" type="button" aria-label="Show cell ${esc(shortId(x.cell_id))} on the slide"></button>
        <div class="cap"><span class="num">${Math.round(x.p * 100)}%</span>
          <span class="acts"><button class="ok" type="button" aria-label="Agree: mitotic figure">✓</button><button class="nope" type="button" aria-label="Disagree: not a mitotic figure">✕</button></span></div>`;
      opened.then(() => sv.crop(x.cell_id, 120, 30)).then((cv) => el.querySelector(".pic").appendChild(cv));
      el.querySelector(".pic").onclick = () => { cur = i; focusCur(true); };
      el.querySelector(".ok").onclick = () => { cur = i; answer(1); };
      el.querySelector(".nope").onclick = () => { cur = i; answer(0); };
      return el;
    }
    const gal = side.querySelector("#gal"), gal2 = side.querySelector("#gal2");
    for (let i = 0; i < called.length; i++) gal.appendChild(await tile(called[i], i));
    if (!called.length) gal.innerHTML = `<p class="empty-note">No detections above the cut-off on this slide.</p>`;
    if (gal2) for (let i = 0; i < border.length; i++) gal2.appendChild(await tile(border[i], called.length + i));

    function paint() {
      side.querySelectorAll(".cand").forEach((el) => {
        const lab = t.labels.get(el.dataset.id);
        el.classList.toggle("agreed", lab === 1); el.classList.toggle("rejected", lab === 0);
        el.classList.toggle("cur", +el.dataset.i === cur);
      });
      const done = called.filter((x) => t.labels.has(x.cell_id)).length;
      const agree = called.filter((x) => t.labels.get(x.cell_id) === 1).length;
      document.getElementById("rv-bar").style.width = `${called.length ? (done / called.length) * 100 : 0}%`;
      document.getElementById("rv-text").textContent = called.length ? `${done} of ${called.length} checked${done ? `, ${agree} agreed` : ""}` : "";
      sv.setLabels(t.labels);
      sv.setQueue(all[cur] ? [all[cur]] : [], all[cur] && all[cur].cell_id);
    }
    function focusCur(fly) {
      paint();
      const x = all[cur]; if (!x) return;
      if (fly) sv.goTo(x.cell_id, 90);
      const el = side.querySelector(`.cand[data-i="${cur}"]`);
      if (el) el.scrollIntoView({ block: "nearest" });
    }
    async function answer(label) {
      const x = all[cur]; if (!x) return;
      try {
        const same = t.labels.get(x.cell_id) === label;
        await t.mark(x.cell_id, same ? null : label, "checking model detections");
        const res = t.last || {};
        toast(res.warning || (same ? "Answer removed." : label ? "Agreed." : "Disagreed. The model will learn from this."));
        // move on to the next unchecked detection
        if (!same) { const nxt = all.findIndex((y, j) => j > cur && !t.labels.has(y.cell_id)); if (nxt >= 0) cur = nxt; }
        focusCur(true);
      } catch (e) { toast(e.message); }
    }
    setOnCell((c) => { const i = all.findIndex((x) => x.cell_id === c.cell_id); if (i >= 0) { cur = i; focusCur(false); } });
    paint();

    const keys = (e) => {
      if (e.target.closest("input, select, textarea")) return;
      const k = e.key.toLowerCase();
      if (k === "1") answer(1);
      else if (k === "2") answer(0);
      else if (k === "j" || k === "3") { cur = Math.min(all.length - 1, cur + 1); focusCur(true); }
      else if (k === "k") { cur = Math.max(0, cur - 1); focusCur(true); }
      else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", keys);
    return () => document.removeEventListener("keydown", keys);
  }

  /* ----- Researcher: layers, model source and threshold, teaching any feature, export. ----- */
  async function researcherPanel({ scan, scanId, table, outputs, sv, side, setOnCell }) {
    const features = await api.listFeatures();
    const st = {
      feature: store.get("entip-gui-feature", features[0].name),
      source: store.get("entip-gui-source", "live"), // live | <model name>
      threshold: 0.5, click: "pan", qi: 0, t: null, busy: false, preds: [],
    };
    if (!features.some((f) => f.name === st.feature)) st.feature = features[0].name;

    side.innerHTML = `
      <section class="sect" aria-labelledby="t-l">
        <h2 id="t-l">Layers</h2>
        <div class="toggles">
          <label><input type="checkbox" id="ly-model" checked> Model calls <span class="muted">(dashed)</span></label>
          <label><input type="checkbox" id="ly-marks" checked> Marks</label>
          <label><input type="checkbox" id="ly-cells" checked> Cell outlines <span class="muted">(when zoomed in)</span></label>
          <label><input type="checkbox" id="ly-regions" checked> Regions${(table.regions || []).length ? ` <span class="muted">(${table.regions.map((r) => r.kind).join(", ")})</span>` : ""}</label>
          <label><input type="checkbox" id="ly-queue" checked> Review queue</label>
        </div>
      </section>

      <section class="sect" aria-labelledby="t-s">
        <h2 id="t-s">Model and threshold</h2>
        <select id="src" aria-label="Which model's calls to show">
          <option value="live">Live head</option>
          ${outputs.map((o) => `<option value="${esc(o.model)}">${esc(o.model)} ${esc(o.version)}</option>`).join("")}
        </select>
        <svg class="hist" id="hist" viewBox="0 0 300 70" preserveAspectRatio="none" role="img" aria-label="Distribution of model probabilities"></svg>
        <div class="row"><input type="range" id="thr" min="0.05" max="0.95" step="0.01" value="0.5" aria-label="Threshold" style="flex:1"><span class="num" id="thr-v">0.50</span></div>
        <p class="help" id="thr-n"></p>
      </section>

      <section class="sect">
        <h2>Teach a feature</h2>
        <select id="feat" aria-label="Feature to teach">${features.map((f) => `<option ${f.name === st.feature ? "selected" : ""}>${esc(f.name)}</option>`).join("")}<option value="__new">Add a new feature…</option></select>
        <div class="newfeat" id="newfeat" hidden>
          <input type="text" id="newfeat-name" placeholder="e.g. Gemistocyte" aria-label="New feature name">
          <button class="btn primary" type="button" id="newfeat-add">Add</button>
        </div>
        <p class="help" id="feat-desc"></p>
        <div class="seg" role="group" aria-label="Click mode">
          <button type="button" class="pan" data-m="pan" aria-pressed="true">Move</button>
          <button type="button" class="yes" data-m="yes" aria-pressed="false">Mark yes</button>
          <button type="button" class="no" data-m="no" aria-pressed="false">Mark no</button>
        </div>
        <p class="help">Click a marked cell again to remove the mark. Keys: <b>M</b> move, <b>Y</b>/<b>N</b> mark modes.</p>
      </section>

      <section class="sect" aria-labelledby="t-q">
        <h2 id="t-q">Review queue</h2>
        <div class="current" id="current"></div>
        <ol class="queue" id="queue"></ol>
      </section>

      <section class="sect" aria-labelledby="t-m">
        <h2 id="t-m">Live head on this slide</h2>
        <div class="stats">
          <div class="stat yes"><b id="s-pos">0</b><span>marked yes</span></div>
          <div class="stat no"><b id="s-neg">0</b><span>marked no</span></div>
          <div class="stat"><b id="s-flag">–</b><span>called yes</span></div>
        </div>
        <svg class="spark" id="spark" viewBox="0 0 300 44" preserveAspectRatio="none" role="img" aria-label="Cells that changed call after each update"></svg>
        <p class="help" id="s-note"></p>
      </section>

      <section class="sect" aria-labelledby="t-x">
        <h2 id="t-x">Export</h2>
        <p class="help">One row per cell: position in µm (frame <span class="mono">scan:${esc(scanId)}</span>), the chosen model's probability, and your mark.</p>
        <div class="row"><button class="btn" type="button" id="x-copy">Copy CSV</button><button class="btn" type="button" id="x-dl">Download CSV</button></div>
      </section>

      <section class="sect">
        <h2>Scan</h2>
        <dl class="kv">
          <dt>Study ID</dt><dd><a class="mono" href="#case-${esc(scan.study_id)}">${esc(scan.study_id)}</a></dd>
          <dt>Diagnosis</dt><dd>${esc(scan.diagnosis)}</dd>
          <dt>Resolution</dt><dd class="num">${esc(scan.mpp)} µm/px · ${esc(scan.objective)}</dd>
          <dt>Cells</dt><dd class="num">${table.cells.length.toLocaleString()} <span class="muted">(${esc(sourceText(table.source))})</span></dd>
          <dt>Embedding</dt><dd>${esc(table.embedding ? `${table.embedding.name}, ${table.embedding.dim} numbers` : "not given")}</dd>
          <dt>Copy</dt><dd>${esc(scan.copy)}, clinical link ${esc(scan.linkage)}</dd>
        </dl>
      </section>`;
    const $ = (id) => side.querySelector(`#${id}`);
    if (!outputs.some((o) => o.model === st.source)) st.source = "live";
    $("src").value = st.source;

    // layers
    [["ly-model", "model"], ["ly-marks", "marks"], ["ly-cells", "cells"], ["ly-regions", "regions"], ["ly-queue", "queue"]]
      .forEach(([id, key]) => $(id).addEventListener("change", (e) => sv.setShow(key, e.target.checked)));

    // model source and threshold
    async function refreshPreds() {
      $("src").options[0].textContent = `Live head: ${st.feature}`;
      if (st.source === "live") st.preds = await st.t.predictions();
      else st.preds = (outputs.find((o) => o.model === st.source) || { predictions: [] }).predictions;
      sv.setPredictions(st.preds);
      drawHist();
    }
    function drawHist() {
      const bins = new Array(20).fill(0);
      for (const x of st.preds) bins[Math.min(19, Math.floor(x.p * 20))]++;
      const max = Math.max(1, ...bins.map((b) => Math.log10(b + 1)));
      const w = 300 / 20;
      const svg = $("hist");
      svg.innerHTML = bins.map((b, i) => {
        const hgt = (Math.log10(b + 1) / max) * 60;
        const on = (i + 0.5) / 20 >= st.threshold;
        return `<rect x="${(i * w + 1).toFixed(1)}" y="${(64 - hgt).toFixed(1)}" width="${(w - 2).toFixed(1)}" height="${hgt.toFixed(1)}" rx="1.5" fill="${on ? "var(--yes)" : "var(--line)"}"><title>p ${(i / 20).toFixed(2)}–${((i + 1) / 20).toFixed(2)}: ${b} cells</title></rect>`;
      }).join("") + `<line x1="${st.threshold * 300}" x2="${st.threshold * 300}" y1="0" y2="66" stroke="var(--ink)" stroke-width="1.5" vector-effect="non-scaling-stroke"></line><line x1="0" x2="300" y1="64.5" y2="64.5" stroke="var(--line)" vector-effect="non-scaling-stroke"></line>`;
      const n = st.preds.filter((x) => x.p >= st.threshold).length;
      const area = table.tissue_area_mm2;
      $("thr-n").textContent = st.preds.length
        ? `${n.toLocaleString()} cells at or above ${st.threshold.toFixed(2)}${area ? ` (${(n / area).toFixed(1)} per mm²)` : ""}. Bars use a log scale.`
        : "No calls yet. Mark a few cells with the live head, or pick another model.";
    }
    $("src").addEventListener("change", (e) => { st.source = e.target.value; store.set("entip-gui-source", st.source); refreshPreds(); });
    $("thr").addEventListener("input", (e) => { st.threshold = +e.target.value; $("thr-v").textContent = st.threshold.toFixed(2); sv.setThreshold(st.threshold); drawHist(); });

    // teaching
    async function startHead() {
      const f = features.find((x) => x.name === st.feature);
      $("feat-desc").textContent = f ? f.description : "";
      st.t = await teacher(scanId, scan.study_id, st.feature);
      st.qi = 0;
      sv.setLabels(st.t.labels);
      await refreshPreds();
      drawPanel(false);
    }
    async function mark(cellId, label) {
      if (st.busy) return;
      st.busy = true;
      try {
        const res = await st.t.mark(cellId, label);
        st.qi = 0;
        sv.setLabels(st.t.labels);
        if (st.source === "live") await refreshPreds();
        drawPanel();
        const verb = label === null ? "Removed mark" : label ? "Marked yes" : "Marked no";
        toast(res.recorded_only ? (res.warning || `${verb}. Saved; live training is not connected for this slide.`)
          : res.changed === undefined ? `${verb}. Model ${ver(res.model_version)}.` : `${verb}. Model ${ver(res.model_version)}: ${res.changed} cells changed call.`);
      } catch (e) { toast(e.message); } finally { st.busy = false; }
    }
    setOnCell((c) => {
      if (st.click === "pan") { const i = st.t.queue.findIndex((q) => q.cell_id === c.cell_id); if (i >= 0) { st.qi = i; drawPanel(); } return; }
      const want = st.click === "yes" ? 1 : 0;
      mark(c.cell_id, st.t.labels.get(c.cell_id) === want ? null : want);
    });

    function drawPanel(fly = true) {
      const q = st.t.queue, cur = q[st.qi];
      sv.setQueue(q, cur && cur.cell_id);
      const curEl = $("current");
      if (cur) {
        curEl.innerHTML = `
          <div class="why"><span>${cur.reason === "uncertain" ? "Model is unsure" : "Random check"} · <span class="mono">${esc(shortId(cur.cell_id))}</span></span><span class="num">p = ${cur.p.toFixed(2)}</span></div>
          <div class="pbar" title="Model probability"><i style="left:${(cur.p * 100).toFixed(1)}%"></i></div>
          <div class="row">
            <button class="btn yes big" type="button" id="q-yes">Yes <kbd>1</kbd></button>
            <button class="btn no big" type="button" id="q-no">No <kbd>2</kbd></button>
            <button class="btn" type="button" id="q-skip">Skip <kbd>3</kbd></button>
          </div>
          <p class="help">Is this a ${esc(st.feature.toLowerCase())}? <b>J</b>/<b>K</b> next and previous.</p>`;
        curEl.querySelector("#q-yes").onclick = () => answer(1);
        curEl.querySelector("#q-no").onclick = () => answer(0);
        curEl.querySelector("#q-skip").onclick = () => step(1);
      } else curEl.innerHTML = `<p class="help">Nothing left to review on this slide.</p>`;
      $("queue").innerHTML = q.slice(0, 10).map((x, i) => `
        <li><button type="button" data-i="${i}" aria-current="${i === st.qi}">
          <span class="n">${i + 1}</span><span class="mono">${esc(shortId(x.cell_id))} <span class="r">${x.reason === "uncertain" ? "unsure" : "random"}</span></span><span class="num">${x.p.toFixed(2)}</span>
        </button></li>`).join("");
      if (cur && fly) sv.goTo(cur.cell_id);

      const s = st.t.status || { n_pos: 0, n_neg: 0, metrics: {}, history: [] };
      $("s-pos").textContent = s.n_pos;
      $("s-neg").textContent = s.n_neg;
      $("s-flag").textContent = s.n_pos + s.n_neg ? (s.metrics.flagged ?? 0).toLocaleString() : "–";
      $("s-note").textContent = s.live === false ? "Marks are saved to the database. Live training is not connected to this slide yet, so the queue comes from the existing model."
        : !s.n_pos && !s.n_neg ? "Mark a few cells to start. The head retrains after every mark."
        : !s.n_neg ? `Version ${ver(s.model_version)}. Only yes marks so far, so random cells stand in as background. A few “no” marks will sharpen it.`
        : `Version ${ver(s.model_version)}. Agrees with ${Math.round((s.metrics.train_agreement || 0) * 100)}% of your marks on this slide (the marks it learned from, so optimistic).${s.metrics.cv_balanced_accuracy != null ? ` Cross-validated balanced accuracy ${Math.round(s.metrics.cv_balanced_accuracy * 100)}%.` : ""}`;
      spark(s.history || []);
    }
    function spark(hist) {
      const svg = $("spark");
      const vals = hist.filter((x) => x.n_pos + x.n_neg > 0).map((x) => x.changed);
      if (vals.length < 2) { svg.innerHTML = ""; return; }
      const max = Math.max(1, ...vals), stp = 300 / (vals.length - 1);
      const line = vals.map((v, i) => `${(i * stp).toFixed(1)},${(40 - (v / max) * 34).toFixed(1)}`).join(" ");
      svg.innerHTML = `<polyline points="0,40 ${line} 300,40" fill="color-mix(in srgb, var(--accent) 14%, transparent)" stroke="none"></polyline>
        <polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"></polyline>
        <title>Cells that changed call at each update; falling toward zero means the head is settling.</title>`;
    }
    const step = (d) => { const n = st.t.queue.length; if (!n) return; st.qi = (st.qi + d + n) % n; drawPanel(); };
    const answer = (label) => { const cur = st.t.queue[st.qi]; if (cur) mark(cur.cell_id, label); };
    function setClick(m) {
      st.click = m;
      side.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === m));
      sv.setMarking(m !== "pan");
    }
    side.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setClick(b.dataset.m)));
    $("queue").addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) { st.qi = +b.dataset.i; drawPanel(); } });

    const featSel = $("feat");
    featSel.addEventListener("change", () => {
      if (featSel.value === "__new") { $("newfeat").hidden = false; $("newfeat-name").focus(); return; }
      $("newfeat").hidden = true;
      st.feature = featSel.value; store.set("entip-gui-feature", st.feature); startHead();
    });
    $("newfeat-add").addEventListener("click", async () => {
      const name = $("newfeat-name").value.trim();
      if (!name) { toast("Type a name for the feature first."); return; }
      if (!features.some((f) => f.name.toLowerCase() === name.toLowerCase())) {
        await api.logAction({ verb: "define_feature", object_type: "feature", object_id: name, feature: name, level: "cell",
          study_id: scan.study_id, scan_id: scanId, actor: { kind: "person", name: me } });
        features.push({ name, level: "cell", description: "Added by a user. Mark a few examples to start.", source: "user" });
        featSel.insertBefore(new Option(name, name), featSel.lastElementChild);
      }
      featSel.value = name; $("newfeat").hidden = true;
      st.feature = name; store.set("entip-gui-feature", name); startHead();
      toast(`Teaching “${name}”.`);
    });

    // export
    const csv = () => {
      const p = new Map(st.preds.map((x) => [x.cell_id, x.p]));
      const src = st.source === "live" ? `live:${st.feature}` : st.source;
      return toCSV([["cell_id", "x_um", "y_um", "r_um", "frame", "model", "p", "mark_feature", "mark"],
        ...table.cells.map((c) => [c.cell_id, c.x_um, c.y_um, c.r_um, table.frame, src, p.has(c.cell_id) ? p.get(c.cell_id).toFixed(3) : "",
          st.feature, st.t.labels.has(c.cell_id) ? st.t.labels.get(c.cell_id) : ""])]);
    };
    $("x-copy").onclick = () => copyText(csv(), "Cell table");
    $("x-dl").onclick = () => download(csv(), `${scanId}_cells.csv`);

    const keys = (e) => {
      if (e.target.closest("input, select, textarea")) return;
      const k = e.key.toLowerCase();
      if (k === "1") answer(1);
      else if (k === "2") answer(0);
      else if (k === "3" || k === "j") step(1);
      else if (k === "k") step(-1);
      else if (k === "m") setClick("pan");
      else if (k === "y") setClick("yes");
      else if (k === "n") setClick("no");
      else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", keys);
    await startHead();
    return () => document.removeEventListener("keydown", keys);
  }

  route();
})();
