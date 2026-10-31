/* Slide viewer: OpenSeadragon (BSD-3-Clause) for the image, one canvas on top for cells, marks,
 * model calls and the review queue. Cells come in µm; the viewer converts with the scan's µm/px. */
(function (root) {
  "use strict";

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  class SlideView {
    constructor(host, { tileSource, mpp, cells, onCellClick, onMove }) {
      this.host = host;
      this.mpp = mpp;
      this.cells = cells; // [{cell_id, x_um, y_um, r_um}]
      this.byId = new Map(cells.map((c) => [c.cell_id, c]));
      this.labels = new Map();
      this.p = new Map();
      this.queueIds = new Set();
      this.currentId = null;
      this.show = { cells: true, model: true, marks: true };
      this.onCellClick = onCellClick;
      this.onMove = onMove;

      const el = document.createElement("div");
      el.className = "osd";
      host.appendChild(el);
      this.el = el;
      this.osd = root.OpenSeadragon({
        element: el,
        tileSources: tileSource,
        showNavigationControl: false,
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorSizeRatio: 0.16,
        navigatorAutoFade: false,
        maxZoomPixelRatio: 3,
        visibilityRatio: 0.5,
        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
        gestureSettingsTouch: { clickToZoom: false, dblClickToZoom: true },
        crossOriginPolicy: false,
        animationTime: 0.6,
      });
      // Keep the navigator clear of the µm readout.
      this.osd.addHandler("open", () => { const nav = this.osd.navigator && this.osd.navigator.element; if (nav) nav.style.marginBottom = "36px"; });

      this.canvas = document.createElement("canvas");
      this.canvas.className = "overlay";
      el.appendChild(this.canvas);
      this.ctx = this.canvas.getContext("2d");

      const redraw = () => this.draw();
      this.osd.addHandler("animation", redraw);
      this.osd.addHandler("update-viewport", redraw);
      this.osd.addHandler("resize", redraw);
      this.osd.addHandler("open", redraw);
      this.osd.addHandler("canvas-click", (e) => {
        if (!e.quick) return;
        const c = this.cellAt(e.position);
        if (c && this.onCellClick) { e.preventDefaultAction = true; this.onCellClick(c, e.originalEvent); }
      });
      new root.OpenSeadragon.MouseTracker({
        element: el,
        moveHandler: (e) => {
          if (!this.onMove || !this.osd.world.getItemAt(0)) return;
          const ip = this.osd.viewport.viewerElementToImageCoordinates(e.position);
          this.onMove(ip.x * this.mpp, ip.y * this.mpp);
        },
      });
    }

    setMarking(on) { this.el.classList.toggle("marking", !!on); }
    setLabels(map) { this.labels = map; this.draw(); }
    setPredictions(list) { this.p = new Map(list.map((x) => [x.cell_id, x.p])); this.draw(); }
    setQueue(list, currentId) { this.queueIds = new Set(list.map((x) => x.cell_id)); this.currentId = currentId; this.draw(); }
    setShow(key, on) { this.show[key] = on; this.draw(); }

    // Screen pixels per image pixel, and the image origin on screen.
    _xf() {
      const vp = this.osd.viewport;
      const a = vp.imageToViewerElementCoordinates(new root.OpenSeadragon.Point(0, 0));
      const b = vp.imageToViewerElementCoordinates(new root.OpenSeadragon.Point(1000, 0));
      return { ox: a.x, oy: a.y, s: (b.x - a.x) / 1000 };
    }

    // Screen pixels per µm, for the scale bar.
    pxPerUm() { return this.osd.world.getItemAt(0) ? this._xf().s / this.mpp : 0; }

    cellAt(pos) {
      if (!this.osd.world.getItemAt(0)) return null;
      const { ox, oy, s } = this._xf();
      let best = null, bd = Infinity;
      for (const c of this.cells) {
        const x = ox + (c.x_um / this.mpp) * s, y = oy + (c.y_um / this.mpp) * s;
        const d = (x - pos.x) ** 2 + (y - pos.y) ** 2;
        const reach = Math.max(10, (c.r_um / this.mpp) * s * 1.4);
        if (d < reach * reach && d < bd) { bd = d; best = c; }
      }
      return best;
    }

    goTo(cellId, fieldUm = 120) {
      const c = this.byId.get(cellId);
      if (!c || !this.osd.world.getItemAt(0)) return;
      const vp = this.osd.viewport;
      const half = fieldUm / this.mpp / 2;
      const r = vp.imageToViewportRectangle(c.x_um / this.mpp - half, c.y_um / this.mpp - half, half * 2, half * 2);
      vp.fitBoundsWithConstraints(r);
      this.currentId = cellId;
      this.draw();
    }

    draw() {
      const cv = this.canvas, dpr = window.devicePixelRatio || 1;
      const w = this.el.clientWidth, h = this.el.clientHeight;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
        cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
        cv.style.width = w + "px"; cv.style.height = h + "px";
      }
      const g = this.ctx;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      if (!this.osd.world.getItemAt(0)) return;
      const { ox, oy, s } = this._xf();
      const yes = css("--yes") || "#c0265e", no = css("--no") || "#2c7a82", unsure = css("--unsure") || "#a86a12";
      const close = s * (1 / this.mpp) > 0.9; // more than ~0.9 screen px per µm: outlines are useful
      for (const c of this.cells) {
        const x = ox + (c.x_um / this.mpp) * s, y = oy + (c.y_um / this.mpp) * s;
        if (x < -40 || y < -40 || x > w + 40 || y > h + 40) continue;
        const r = Math.max(3, (c.r_um / this.mpp) * s * 0.95);
        const lab = this.labels.get(c.cell_id);
        const p = this.p.get(c.cell_id);
        if (this.show.marks && lab !== undefined) {
          g.lineWidth = close ? 3 : 2;
          g.strokeStyle = lab ? yes : no;
          g.fillStyle = lab ? hexA(yes, 0.18) : hexA(no, 0.18);
          circle(g, x, y, close ? r : 4); g.fill(); g.stroke();
          continue;
        }
        if (this.show.model && p !== undefined && p > 0.5) {
          g.lineWidth = close ? 2 : 1.5;
          g.strokeStyle = hexA(yes, 0.35 + 0.65 * (p - 0.5) * 2);
          g.setLineDash(close ? [5, 4] : []);
          circle(g, x, y, close ? r : 3); g.stroke();
          g.setLineDash([]);
          continue;
        }
        if (this.show.cells && close) {
          g.lineWidth = 1;
          g.strokeStyle = "rgba(255,255,255,.55)";
          circle(g, x, y, r); g.stroke();
        }
      }
      // Queue: amber rings, the current one bold.
      for (const id of this.queueIds) {
        const c = this.byId.get(id); if (!c) continue;
        const x = ox + (c.x_um / this.mpp) * s, y = oy + (c.y_um / this.mpp) * s;
        const r = Math.max(6, (c.r_um / this.mpp) * s * 1.5);
        g.lineWidth = id === this.currentId ? 3 : 1.5;
        g.strokeStyle = unsure;
        g.setLineDash(id === this.currentId ? [] : [3, 3]);
        circle(g, x, y, r); g.stroke();
        g.setLineDash([]);
      }
    }

    destroy() { try { this.osd.destroy(); } catch (e) { /* ignore */ } this.host.innerHTML = ""; }
  }

  function circle(g, x, y, r) { g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); }
  function hexA(color, a) {
    const m = /^#([0-9a-f]{6})$/i.exec(color);
    if (!m) return color;
    const n = parseInt(m[1], 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
  }

  root.ENTIPViewer = { SlideView };
})(typeof self !== "undefined" ? self : this);
