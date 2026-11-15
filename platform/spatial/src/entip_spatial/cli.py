"""entip-spatial command line.

  entip-spatial synth OUT                  synthetic H&E + Xenium + Visium
  entip-spatial register ...               assay -> H&E Transform, records for P5
  entip-spatial spot-table ...             H&E cells (e.g. MitoEd calls) per spot
  entip-spatial labels ...                 per-nucleus training labels for P2
  entip-spatial demo OUT                   all of the above on synthetic data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import labels as L
from . import records as R
from . import synthetic
from .loaders import load_assay, sha256
from .ontology import Assay, Provenance, assay_frame, new_id
from .register import register, to_he
from .spots import join_obs, spot_table


def _landmarks(path: str, um_per_unit: float):
    lm = pd.read_csv(path)
    if {"assay_x_um", "assay_y_um"} <= set(lm):
        a = lm[["assay_x_um", "assay_y_um"]].to_numpy(float)
    elif {"assay_x", "assay_y"} <= set(lm):  # native assay units (e.g. Visium full-res pixels)
        a = lm[["assay_x", "assay_y"]].to_numpy(float) * um_per_unit
    else:
        raise SystemExit("landmarks need assay_x_um,assay_y_um (or assay_x,assay_y) and he_x_um,he_y_um")
    return a, lm[["he_x_um", "he_y_um"]].to_numpy(float)


def _clean_uns(adata):
    adata.uns["entip"] = {k: v for k, v in adata.uns["entip"].items() if v is not None}
    return adata


def cmd_register(a) -> dict:
    kw = {"um_per_unit": a.um_per_unit} if a.kind in ("anndata", "slide_tags") else {}
    adata = load_assay(a.kind, a.assay, **kw)
    assay_id = a.assay_id or new_id("assay")
    src, dst = _landmarks(a.landmarks, adata.uns["entip"]["um_per_unit"])
    refine = {}
    if a.refine_cells:
        cells = pd.read_csv(a.refine_cells)
        refine = {"refine_assay_pts": adata.obsm["spatial_um"], "refine_he_pts": cells[["x_um", "y_um"]].to_numpy(float),
                  "max_dist_um": a.max_dist}
    tf = register(assay_id, a.scan_id, src, dst, model=a.model, **refine)
    to_he(adata, tf)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.save_transform(tf, out / "transform.json")
    adata.uns["entip"].update({"assay_id": assay_id, "section_id": a.section_id, "scan_id": a.scan_id})
    _clean_uns(adata).write_h5ad(out / "assay_registered.h5ad")
    sink = R.JsonlSink(out / "records")
    prov = Provenance.program("register")
    src_file = Path(a.assay)
    R.write_assay(sink, Assay(assay_id=assay_id, section_id=a.section_id, kind=a.kind, frame=assay_frame(assay_id),
                              n_obs=adata.n_obs, n_vars=adata.n_vars, source_path=str(src_file),
                              source_sha256=sha256(src_file), provenance=prov,
                              meta={k: v for k, v in adata.uns["entip"].items() if k != "source"}))
    R.write_transform(sink, tf)
    R.write_objects(sink, R.spatial_objects(adata, assay_id, tf), prov)
    loo = "n/a" if tf.loo_rmse_um is None else f"{tf.loo_rmse_um:.2f}"
    print(f"{tf.transform_id}: {tf.method}, landmark RMSE {tf.rmse_um:.2f} um, leave-one-out {loo} um, "
          f"worst {tf.max_err_um:.2f} um" + (f", ICP matched {tf.meta['icp_matched']}" if "icp_matched" in tf.meta else ""))
    return {"transform": tf, "adata": adata}


def cmd_spot_table(a) -> pd.DataFrame:
    adata = ad.read_h5ad(a.assay_h5ad)
    radius = a.radius or adata.uns["entip"].get("radius_um")
    if not radius:
        raise SystemExit("spot radius unknown for this assay; pass --radius (micrometres)")
    tab = spot_table(adata.obsm["spatial_he_um"], adata.obs_names, pd.read_csv(a.cells), radius_um=radius,
                     class_col=a.class_col)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out)
    join_obs(adata, tab, prefix=a.prefix).write_h5ad(out.with_suffix(".h5ad"))
    print(f"spot table: {len(tab)} spots, {int(tab['n_cells'].sum())} cell-in-spot hits -> {out}")
    return tab


def cmd_labels(a) -> pd.DataFrame:
    adata = ad.read_h5ad(a.assay_h5ad)
    tf = R.read_transform(a.transform)
    cells = pd.read_csv(a.cells)
    meta = adata.uns["entip"]
    scan_id, assay_id = meta.get("scan_id", "unknown"), meta.get("assay_id", "unknown")
    if meta["object_kind"] == "spot":
        if not a.fractions:
            raise SystemExit("spot assays need --fractions (CSV: barcode x label fractions)")
        fr = pd.read_csv(a.fractions, index_col=0)
        fr.index = fr.index.astype(str)
        tab = L.spot_labels(cells, adata, tf, fr, scan_id, assay_id, min_fraction=a.min_fraction)
    else:
        markers = json.loads(Path(a.markers).read_text()) if a.markers else None
        lab = L.assign_labels(adata, markers=markers, label_key=a.label_key, min_margin=a.min_margin)
        tab = L.single_cell_labels(cells, adata, tf, lab, scan_id, assay_id, max_dist_um=a.max_dist)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"labels: {len(tab)} of {len(cells)} H&E nuclei labelled -> {out}")
    print(tab["label"].value_counts().to_string())
    return tab


def _accuracy(tab, cells):
    truth = cells.set_index("cell_id")["true_type"]
    return float((tab["label"].to_numpy() == truth.reindex(tab["cell_id"]).to_numpy()).mean()) if len(tab) else float("nan")


def _true_error(tf, true_matrix, adata) -> float:
    """RMS distance (um) between the fitted and true maps over the assay's own points."""
    from .register import apply
    pts = adata.obsm["spatial_um"]
    return float(np.sqrt(np.mean(np.sum((apply(np.array(tf.matrix), pts) - apply(np.array(true_matrix), pts)) ** 2, 1))))


def cmd_demo(a) -> dict:
    out = Path(a.out)
    synthetic.make(out / "data", seed=a.seed)
    d = out / "data"
    ns = argparse.Namespace
    reg_x = cmd_register(ns(kind="xenium", assay=str(d / "xenium"), landmarks=str(d / "xenium_landmarks.csv"),
                            scan_id="scan_demo", section_id="section_demo", assay_id="assay_xenium_demo", model="affine",
                            refine_cells=str(d / "he_cells.csv"), max_dist=10.0, um_per_unit=1.0, out=str(out / "xenium")))
    reg_v = cmd_register(ns(kind="visium", assay=str(d / "visium"), landmarks=str(d / "visium_landmarks.csv"),
                            scan_id="scan_demo", section_id="section_demo", assay_id="assay_visium_demo", model="affine",
                            refine_cells=None, max_dist=10.0, um_per_unit=1.0, out=str(out / "visium")))
    cmd_spot_table(ns(assay_h5ad=str(out / "visium" / "assay_registered.h5ad"), cells=str(d / "he_cells.csv"), radius=None,
                      class_col="call", prefix="mitoed_", out=str(out / "visium" / "spot_table.csv")))
    lx = cmd_labels(ns(assay_h5ad=str(out / "xenium" / "assay_registered.h5ad"), transform=str(out / "xenium" / "transform.json"),
                       cells=str(d / "he_cells.csv"), markers=str(d / "markers.json"), label_key=None, min_margin=0.25,
                       max_dist=None, fractions=None, min_fraction=0.8, out=str(out / "xenium" / "training_labels.csv")))
    lv = cmd_labels(ns(assay_h5ad=str(out / "visium" / "assay_registered.h5ad"), transform=str(out / "visium" / "transform.json"),
                       cells=str(d / "he_cells.csv"), markers=None, label_key=None, min_margin=0.25, max_dist=None,
                       fractions=str(d / "visium_fractions.csv"), min_fraction=0.8,
                       out=str(out / "visium" / "training_labels.csv")))
    cells = pd.read_csv(d / "he_cells.csv")
    truth = json.loads((d / "truth.json").read_text())
    summary = {"xenium_true_error_um": _true_error(reg_x["transform"], truth["xenium_to_he"], reg_x["adata"]),
               "visium_true_error_um": _true_error(reg_v["transform"], truth["visium_to_he"], reg_v["adata"]),"xenium_label_accuracy": _accuracy(lx, cells), "xenium_labelled": len(lx),
               "visium_label_accuracy": _accuracy(lv, cells), "visium_labelled": len(lv), "n_he": len(cells),
               "xenium_loo_rmse_um": reg_x["transform"].loo_rmse_um, "visium_loo_rmse_um": reg_v["transform"].loo_rmse_um}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="entip-spatial", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write synthetic paired data")
    s.add_argument("out"); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=lambda a: print(json.dumps(synthetic.make(a.out, a.seed), indent=2)))

    s = sub.add_parser("register", help="fit assay -> H&E Transform and write records")
    s.add_argument("--kind", required=True, choices=["visium", "xenium", "slide_tags", "anndata"])
    s.add_argument("--assay", required=True, help="Space Ranger outs/, Xenium output folder, or .h5ad")
    s.add_argument("--landmarks", required=True, help="CSV: assay_x_um,assay_y_um,he_x_um,he_y_um")
    s.add_argument("--scan-id", required=True); s.add_argument("--section-id", required=True)
    s.add_argument("--assay-id")
    s.add_argument("--model", default="affine", choices=["affine", "similarity"])
    s.add_argument("--refine-cells", help="H&E nucleus CSV (x_um,y_um) for ICP refinement; single-cell assays only")
    s.add_argument("--max-dist", type=float, default=10.0, help="ICP match distance, micrometres")
    s.add_argument("--um-per-unit", type=float, default=1.0, help=".h5ad only: micrometres per obsm['spatial'] unit")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("spot-table", help="summarise H&E cells under each spot")
    s.add_argument("--assay-h5ad", required=True); s.add_argument("--cells", required=True)
    s.add_argument("--radius", type=float, help="micrometres (default: assay spot radius)")
    s.add_argument("--class-col", default="call"); s.add_argument("--prefix", default="mitoed_")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_spot_table)

    s = sub.add_parser("labels", help="per-nucleus training labels from the spatial assay")
    s.add_argument("--assay-h5ad", required=True); s.add_argument("--transform", required=True)
    s.add_argument("--cells", required=True, help="H&E nucleus CSV: cell_id,x_um,y_um")
    s.add_argument("--markers", help="JSON {label: [genes]} for single-cell assays")
    s.add_argument("--label-key", help="use labels already in obs[label_key]")
    s.add_argument("--min-margin", type=float, default=0.25)
    s.add_argument("--max-dist", type=float, help="match gate, micrometres (default from transform error)")
    s.add_argument("--fractions", help="spot assays: CSV barcode x label fractions")
    s.add_argument("--min-fraction", type=float, default=0.8)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_labels)

    s = sub.add_parser("demo", help="run every step on synthetic data")
    s.add_argument("out"); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
