/* ENTIP case review: worklist, patient page, slide workbench with click-to-teach.
 * Routes live in the hash: #worklist, #case-<study_id>, #slide-<scan_id>. */
(function () {
  "use strict";

  const cfg = window.ENTIP_CONFIG || { mode: "mock", base: "mock" };
  const api = window.ENTIPApi.createApi(cfg);
  const { walk } = window.ENTIPApi;
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
  const KEY_MARKERS = ["IDH", "1p/19q", "MGMT", "CDKN2A/B"];

  function setCrumbs(parts) {
    crumbs.innerHTML = parts.map((p, i) => i === parts.length - 1
      ? `<span class="here">${esc(p.label)}</span>`
      : `<a href="${p.href}">${esc(p.label)}</a><span class="sep" aria-hidden="true">/</span>`).join("");
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
    setCrumbs([{ label: "Cases" }]);
    document.title = "ENTIP Case Review";
    const rows = await api.listPatients();
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
            <thead><tr><th></th><th>Study ID</th><th>Diagnosis</th><th>Key markers</th><th>Material</th><th>Marks</th><th>Status</th></tr></thead>
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
      tbody.innerHTML = hits.map((r) => `
        <tr tabindex="0" data-id="${esc(r.study_id)}">
          <td>${r.thumb ? `<img class="thumb" src="${esc(r.thumb)}" alt="">` : `<div class="thumb"></div>`}</td>
          <td><span class="mono">${esc(r.study_id)}</span><div class="sub">${esc(r.sex)} · ${esc(r.age_band)}</div></td>
          <td><div class="dx">${esc(r.diagnosis)}</div><div class="sub">CNS WHO grade ${esc(r.who_grade)}</div></td>
          <td><div class="chips">${KEY_MARKERS.filter((k) => r.molecular && r.molecular[k]).map((k) => `<span class="tag" title="${esc(k)}: ${esc(r.molecular[k])}">${esc(k)} ${esc(short(r.molecular[k]))}</span>`).join("")}</div></td>
          <td class="num">${r.n_scans} scans<div class="sub">${r.n_assays} assays</div></td>
          <td class="num">${r.n_marks || "–"}</td>
          <td>${statusPill(r.review_status)}</td>
        </tr>`).join("") || `<tr><td colspan="7" class="empty-note" style="padding:20px">No cases match. Clear the search or pick “All”.</td></tr>`;
    }
    draw();
    document.getElementById("q").addEventListener("input", (e) => { filter.q = e.target.value; store.set("entip-gui-q", filter.q); draw(); });
    main.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
      filter.status = b.dataset.s; store.set("entip-gui-status", filter.status);
      main.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", x === b)); draw();
    }));
    const open = (tr) => { if (tr && tr.dataset.id) location.hash = `case-${tr.dataset.id}`; };
    tbody.addEventListener("click", (e) => open(e.target.closest("tr")));
    tbody.addEventListener("keydown", (e) => { if (e.key === "Enter") open(e.target.closest("tr")); });
  }

  function short(v) {
    const m = { wildtype: "wt", "IDH1 R132H": "R132H", methylated: "meth", unmethylated: "unmeth", "homozygous deletion": "del",
      codeleted: "codel", intact: "intact", retained: "retained", "not tested": "n/t" };
    return m[v] || v;
  }

  /* ---------- patient page ---------- */
  async function caseView(id) {
    const p = await api.getPatient(id);
    // Keep every person's action; for model updates keep only the latest per feature and scan.
    const seen = new Set();
    const actions = (await api.listActions({ study_id: id })).filter((a) => {
      if (a.verb !== "train_head") return true;
      const k = `${a.feature}|${a.scan_id}`;
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
    setCrumbs([{ label: "Cases", href: "#worklist" }, { label: p.study_id }]);
    document.title = `${p.study_id} · ENTIP`;
    const scans = [], assays = [];
    walk(p, (k, o, trail) => { if (k === "scan") scans.push({ ...o, trail }); if (k === "assay") assays.push({ ...o, trail }); });

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

    const cards = scans.map((sc) => sc.tile_source ? `
      <a class="panel slidecard" href="#slide-${esc(sc.scan_id)}">
        <img src="${esc(sc.thumb)}" alt="Thumbnail of ${esc(sc.stain)} slide ${esc(sc.scan_id)}">
        <div class="meta"><b>${esc(sc.stain)} · ${esc(sc.objective)}</b><span class="sub mono">${esc(sc.scan_id)} · ${esc(sc.mpp)} µm/px</span><span class="sub">Open to review and teach</span></div>
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

    main.append(h(`
      <div class="page">
        <div class="casehead">
          <div class="sect">
            <div class="ids"><span class="mono">${esc(p.study_id)}</span><span>${esc(p.sex)} · ${esc(p.age_band)}</span>${statusPill(p.review_status)}</div>
            <h1>${esc(p.diagnosis)}, CNS WHO grade ${esc(p.who_grade)}</h1>
          </div>
          ${scans.find((s) => s.tile_source) ? `<a class="btn primary" href="#slide-${esc(scans.find((s) => s.tile_source).scan_id)}">Open H&amp;E</a>` : ""}
        </div>
        <div class="grid3">
          <section class="panel tree" aria-labelledby="t-tree"><div class="box"><h2 id="t-tree">Tissue</h2><ul>${tree}</ul></div></section>
          <div class="sect" style="gap:20px">
            <section class="sect" aria-labelledby="t-slides"><h2 id="t-slides">Slides</h2><div class="slides">${cards}</div></section>
            <section class="panel" aria-labelledby="t-out"><div class="box"><h2 id="t-out">Model outputs</h2>
              <div class="outputs">${outputs || `<p class="empty-note">No model outputs yet.</p>`}${assayList}</div>
              <p class="help">Each result names the model and version that produced it. Mock values, not diagnoses.</p></div></section>
          </div>
          <div class="sect" style="gap:20px">
            <section class="panel" aria-labelledby="t-mol"><div class="box"><h2 id="t-mol">Molecular</h2>
              <dl class="kv">${Object.entries(p.molecular || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></div></section>
            <section class="panel" aria-labelledby="t-log"><div class="box"><h2 id="t-log">Activity</h2>
              ${actions.length ? `<ul class="log">${actions.slice(0, 60).map(logItem).join("")}</ul>` : `<p class="empty-note">No activity yet. Marks you make on a slide appear here, with your name and the time.</p>`}
            </div></section>
          </div>
        </div>
      </div>`));
  }

  function logItem(a) {
    const who = esc(actorName(a.actor));
    let text;
    if (a.verb === "label") text = `<b>${who}</b> marked <span class="mono">${esc(a.object_id.split("-").pop())}</span> as ${a.label ? "" : "not "}${esc(a.feature)}`;
    else if (a.verb === "retract_label") text = `<b>${who}</b> removed the mark on <span class="mono">${esc(a.object_id.split("-").pop())}</span> (${esc(a.feature)})`;
    else if (a.verb === "train_head") text = `<b>${who}</b> updated: ${esc(a.detail)}`;
    else if (a.verb === "define_feature") text = `<b>${who}</b> added feature “${esc(a.feature)}”`;
    else text = `<b>${who}</b> ${esc(a.verb)}`;
    const cls = a.verb === "label" ? (a.label ? "yes" : "no") : a.actor && a.actor.kind === "person" ? "person" : "";
    return `<li class="${cls}"><span>${text}<br><span class="muted">${esc(when(a.ts))}${a.scan_id ? ` · <span class="mono">${esc(a.scan_id)}</span>` : ""}</span></span></li>`;
  }

  /* ---------- slide workbench ---------- */
  async function slideView(scanId) {
    const scan = await api.getScan(scanId);
    if (!scan.tile_source) throw new Error(`Scan ${scanId} has no image in this mock.`);
    const table = await api.getCells(scanId);
    const features = await api.listFeatures();
    setCrumbs([{ label: "Cases", href: "#worklist" }, { label: scan.study_id, href: `#case-${scan.study_id}` }, { label: `${scan.stain} ${scanId}` }]);
    document.title = `${scanId} · ENTIP`;

    const st = {
      feature: store.get("entip-gui-feature", features[0].name),
      mode: "pan", // pan | yes | no
      head: null, queue: [], qi: 0, labels: new Map(), status: null, busy: false,
    };
    if (!features.some((f) => f.name === st.feature)) st.feature = features[0].name;

    main.append(h(`
      <div class="bench">
        <div class="stage" id="stage" aria-label="Slide viewer">
          <div class="floatbar" role="toolbar" aria-label="Overlays">
            <button class="btn" type="button" id="tg-model" aria-pressed="true" title="Cells the model calls positive (dashed)">Model calls</button>
            <button class="btn" type="button" id="tg-cells" aria-pressed="true" title="Outline every detected cell when zoomed in">Cell outlines</button>
          </div>
          <div class="zoomer">
            <button class="btn" type="button" id="z-in" aria-label="Zoom in">+</button>
            <button class="btn" type="button" id="z-out" aria-label="Zoom out">−</button>
            <button class="btn" type="button" id="z-home" aria-label="Whole slide">⤢</button>
          </div>
          <div class="scalebar" aria-hidden="true"><span id="sb-l">–</span><i id="sb"></i></div>
          <div class="readout" id="readout">scan:${esc(scanId)}</div>
        </div>
        <aside class="side" aria-label="Teach and review">
          <section class="sect">
            <h2>Feature</h2>
            <select id="feat" aria-label="Feature to teach">${features.map((f) => `<option ${f.name === st.feature ? "selected" : ""}>${esc(f.name)}</option>`).join("")}<option value="__new">Add a new feature…</option></select>
            <div class="newfeat" id="newfeat" hidden>
              <input type="text" id="newfeat-name" placeholder="e.g. Gemistocyte" aria-label="New feature name">
              <button class="btn primary" type="button" id="newfeat-add">Add</button>
            </div>
            <p class="help" id="feat-desc"></p>
          </section>

          <section class="sect">
            <h2>Click on the slide to</h2>
            <div class="seg" role="group" aria-label="Click mode">
              <button type="button" class="pan" data-m="pan" aria-pressed="true">Move</button>
              <button type="button" class="yes" data-m="yes" aria-pressed="false">Mark yes</button>
              <button type="button" class="no" data-m="no" aria-pressed="false">Mark no</button>
            </div>
            <p class="help">Click a marked cell again to remove the mark. Keys: <b>M</b> move, <b>Y</b>/<b>N</b> mark modes.</p>
            <div class="legend"><span class="y"><i></i>yes</span><span class="n"><i></i>no</span><span class="q"><i></i>to review</span></div>
          </section>

          <section class="sect" aria-labelledby="t-q">
            <h2 id="t-q">Review queue</h2>
            <div class="current" id="current"></div>
            <ol class="queue" id="queue"></ol>
          </section>

          <section class="sect" aria-labelledby="t-m">
            <h2 id="t-m">Model on this slide</h2>
            <div class="stats">
              <div class="stat yes"><b id="s-pos">0</b><span>marked yes</span></div>
              <div class="stat no"><b id="s-neg">0</b><span>marked no</span></div>
              <div class="stat"><b id="s-flag">–</b><span>cells called yes</span></div>
            </div>
            <svg class="spark" id="spark" viewBox="0 0 300 44" preserveAspectRatio="none" role="img" aria-label="Cells called yes after each update"></svg>
            <p class="help" id="s-note">Mark a few cells to start. The model retrains after every mark.</p>
          </section>

          <section class="sect">
            <h2>Scan</h2>
            <dl class="kv">
              <dt>Study ID</dt><dd><a class="mono" href="#case-${esc(scan.study_id)}">${esc(scan.study_id)}</a></dd>
              <dt>Diagnosis</dt><dd>${esc(scan.diagnosis)}</dd>
              <dt>Resolution</dt><dd class="num">${esc(scan.mpp)} µm/px · ${esc(scan.objective)}</dd>
              <dt>Cells</dt><dd class="num">${table.cells.length.toLocaleString()} <span class="muted">(${esc(sourceText(table.source))})</span></dd>
              <dt>Copy</dt><dd>${esc(scan.copy)}, clinical link ${esc(scan.linkage)}</dd>
            </dl>
          </section>
        </aside>
      </div>`));

    const sv = new window.ENTIPViewer.SlideView(document.getElementById("stage"), {
      tileSource: scan.tile_source, mpp: table.mpp, cells: table.cells,
      onCellClick: (c) => onCell(c),
      onMove: (x, y) => { document.getElementById("readout").textContent = `x ${x.toFixed(1)} µm  y ${y.toFixed(1)} µm`; },
    });
    sv.osd.addHandler("animation", scalebar);
    sv.osd.addHandler("open", scalebar);
    sv.osd.addHandler("resize", scalebar);

    function scalebar() {
      const ppu = sv.pxPerUm(); if (!ppu) return;
      const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
      const um = nice.find((n) => n * ppu >= 70) || 2000;
      document.getElementById("sb").style.width = `${um * ppu}px`;
      document.getElementById("sb-l").textContent = um >= 1000 ? `${um / 1000} mm` : `${um} µm`;
    }

    /* teaching */
    async function startHead() {
      const f = features.find((x) => x.name === st.feature);
      document.getElementById("feat-desc").textContent = f ? f.description : "";
      st.head = (await api.createHead({ feature: st.feature, level: "cell", scan_id: scanId })).head_id;
      st.labels = new Map((await api.getLabels(st.head)).map((x) => [x.cell_id, x.label]));
      st.queue = await api.getQueue(st.head, { scan_id: scanId, n: 20 });
      st.qi = 0;
      sv.setLabels(st.labels);
      st.status = await api.headStatus(st.head);
      sv.setPredictions(st.status.n_pos + st.status.n_neg ? await api.getPredictions(st.head, { scan_id: scanId }) : []);
      drawPanel(false);
    }

    async function mark(cellId, label) {
      if (st.busy) return;
      st.busy = true;
      try {
        const res = label === null
          ? await api.removeLabel(st.head, { scan_id: scanId, cell_id: cellId, actor: me, study_id: scan.study_id })
          : await api.addLabel(st.head, { scan_id: scanId, cell_id: cellId, label, actor: me, study_id: scan.study_id });
        if (label === null) st.labels.delete(cellId); else st.labels.set(cellId, label);
        st.queue = res.queue || await api.getQueue(st.head, { scan_id: scanId, n: 20 });
        st.qi = 0;
        sv.setLabels(st.labels);
        sv.setPredictions(await api.getPredictions(st.head, { scan_id: scanId }));
        st.status = await api.headStatus(st.head);
        drawPanel();
        const verb = label === null ? "Removed mark" : label ? "Marked yes" : "Marked no";
        toast(res.changed === undefined ? `${verb}. Model v${res.model_version}.` : `${verb}. Model v${res.model_version}: ${res.changed} cells changed call.`);
      } catch (e) { toast(e.message); } finally { st.busy = false; }
    }

    function onCell(c) {
      if (st.mode === "pan") { // in Move mode a click selects the cell if it is in the queue
        const i = st.queue.findIndex((q) => q.cell_id === c.cell_id);
        if (i >= 0) { st.qi = i; drawPanel(); }
        return;
      }
      const want = st.mode === "yes" ? 1 : 0;
      mark(c.cell_id, st.labels.get(c.cell_id) === want ? null : want);
    }

    function drawPanel(fly = true) {
      const cur = st.queue[st.qi];
      sv.setQueue(st.queue, cur && cur.cell_id);
      const curEl = document.getElementById("current");
      if (cur) {
        curEl.innerHTML = `
          <div class="why"><span>${cur.reason === "uncertain" ? "Model is unsure" : "Random check"} · <span class="mono">${esc(cur.cell_id.split("-").pop())}</span></span><span class="num">p = ${cur.p.toFixed(2)}</span></div>
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
      } else {
        curEl.innerHTML = `<p class="help">Nothing left to review on this slide.</p>`;
      }
      document.getElementById("queue").innerHTML = st.queue.slice(0, 10).map((q, i) => `
        <li><button type="button" data-i="${i}" aria-current="${i === st.qi}">
          <span class="n">${i + 1}</span><span class="mono">${esc(q.cell_id.split("-").pop())} <span class="r">${q.reason === "uncertain" ? "unsure" : "random"}</span></span><span class="num">${q.p.toFixed(2)}</span>
        </button></li>`).join("");
      if (cur && fly) sv.goTo(cur.cell_id);

      const s = st.status || { n_pos: 0, n_neg: 0, metrics: {}, history: [] };
      document.getElementById("s-pos").textContent = s.n_pos;
      document.getElementById("s-neg").textContent = s.n_neg;
      document.getElementById("s-flag").textContent = s.n_pos + s.n_neg ? (s.metrics.flagged ?? 0).toLocaleString() : "–";
      const note = document.getElementById("s-note");
      if (!s.n_pos && !s.n_neg) note.textContent = "Mark a few cells to start. The model retrains after every mark.";
      else if (!s.n_neg) note.textContent = `Version ${s.model_version}. Only yes marks so far, so the model treats random cells as background. A few “no” marks will sharpen it.`;
      else note.textContent = `Version ${s.model_version}. Agrees with ${Math.round((s.metrics.train_agreement || 0) * 100)}% of your marks (marks it learned from, so this is optimistic).`;
      spark(s.history || []);
    }

    function spark(hist) {
      const svg = document.getElementById("spark");
      const pts = hist.filter((x) => x.n_pos + x.n_neg > 0);
      if (pts.length < 2) { svg.innerHTML = ""; return; }
      const n = table.cells.length;
      const vals = pts.map((x) => x.changed);
      const max = Math.max(1, ...vals);
      const step = 300 / (vals.length - 1);
      const line = vals.map((v, i) => `${(i * step).toFixed(1)},${(40 - (v / max) * 34).toFixed(1)}`).join(" ");
      svg.innerHTML = `
        <polyline points="0,40 ${line} 300,40" fill="color-mix(in srgb, var(--accent) 14%, transparent)" stroke="none"></polyline>
        <polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"></polyline>
        <title>Cells that changed call at each update (last ${vals.length}); settling toward zero means the model is stabilising. ${n} cells on slide.</title>`;
    }

    function step(d) {
      if (!st.queue.length) return;
      st.qi = (st.qi + d + st.queue.length) % st.queue.length;
      drawPanel();
    }
    function answer(label) { const cur = st.queue[st.qi]; if (cur) mark(cur.cell_id, label); }

    function setMode(m) {
      st.mode = m;
      main.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === m));
      sv.setMarking(m !== "pan");
    }
    main.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.m)));
    document.getElementById("queue").addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) { st.qi = +b.dataset.i; drawPanel(); } });

    const featSel = document.getElementById("feat");
    featSel.addEventListener("change", () => {
      if (featSel.value === "__new") { document.getElementById("newfeat").hidden = false; document.getElementById("newfeat-name").focus(); return; }
      document.getElementById("newfeat").hidden = true;
      st.feature = featSel.value; store.set("entip-gui-feature", st.feature); startHead();
    });
    document.getElementById("newfeat-add").addEventListener("click", async () => {
      const name = document.getElementById("newfeat-name").value.trim();
      if (!name) { toast("Type a name for the feature first."); return; }
      if (!features.some((f) => f.name.toLowerCase() === name.toLowerCase())) {
        await api.logAction({ verb: "define_feature", object_type: "feature", object_id: name, feature: name, level: "cell",
          study_id: scan.study_id, scan_id: scanId, actor: { kind: "person", name: me } });
        features.push({ name, level: "cell", description: "Added by a user. Mark a few examples to start.", source: "user" });
        featSel.insertBefore(new Option(name, name), featSel.lastElementChild);
      }
      featSel.value = name; document.getElementById("newfeat").hidden = true;
      st.feature = name; store.set("entip-gui-feature", name); startHead();
      toast(`Teaching “${name}”.`);
    });

    const tg = (id, key) => document.getElementById(id).addEventListener("click", (e) => {
      const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
      e.currentTarget.setAttribute("aria-pressed", on); sv.setShow(key, on);
    });
    tg("tg-model", "model"); tg("tg-cells", "cells");
    document.getElementById("z-in").onclick = () => sv.osd.viewport.zoomBy(1.6);
    document.getElementById("z-out").onclick = () => sv.osd.viewport.zoomBy(1 / 1.6);
    document.getElementById("z-home").onclick = () => sv.osd.viewport.goHome();

    const keys = (e) => {
      if (e.target.closest("input, select, textarea")) return;
      const k = e.key.toLowerCase();
      if (k === "1") answer(1);
      else if (k === "2") answer(0);
      else if (k === "3" || k === "j") step(1);
      else if (k === "k") step(-1);
      else if (k === "m") setMode("pan");
      else if (k === "y") setMode("yes");
      else if (k === "n") setMode("no");
      else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", keys);
    view = () => { document.removeEventListener("keydown", keys); sv.destroy(); };

    await startHead();
  }

  route();
})();
