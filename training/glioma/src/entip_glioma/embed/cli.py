"""``entip-train embed ...``: step 1, slides -> frozen foundation-model tile bags (extract, build-bags)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

INDEX = "index.jsonl"


def _record(out_dir: Path, info: dict) -> None:
    """Append one slide's provenance to ``index.jsonl``: which encoder, tile size and pixel size made the bag."""
    with (out_dir / INDEX).open("a") as f:
        f.write(json.dumps(info) + "\n")


def embed_slide(path: str, scan_id: str, enc, a, out_dir: Path) -> dict:
    from entip_train.slides import open_slide
    from ..wsi.bags import extract_bag
    r = open_slide(path, mpp=a.mpp)
    bag = extract_bag(r, scan_id, r.width_um, r.height_um, enc, tile_um=a.tile_um, min_tissue=a.min_tissue,
                      max_tiles=a.max_tiles)
    p = bag.save(out_dir / f"{scan_id}.npz")
    if len(bag) == 0:
        import sys
        print(f"warning: {scan_id}: no tissue tiles kept (blank, tiny or unreadable slide); "
              "triage training will skip it", file=sys.stderr)
    return {"scan_id": scan_id, "tiles": len(bag), "reader": type(r).__name__, "mpp": round(r.mpp, 4),
            "tile_um": a.tile_um, "encoder": bag.encoder, "out": str(p)}


def extract(a) -> int:
    from entip_train.encoders import get_encoder
    enc = get_encoder(a.encoder)
    if a.scan_id and len(a.slides) > 1:
        raise SystemExit("--scan-id names one slide; pass one slide, or leave it out to use file names")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in a.slides:
        p = Path(path)
        sid = a.scan_id or (p.name if p.is_dir() else p.stem)
        info = embed_slide(path, sid, enc, a, out)
        _record(out, info)
        print(json.dumps(info), flush=True)
    return 0


def build_bags(a) -> int:
    """Manifest -> bags, one slide at a time: download, embed, delete. Safe to stop and rerun."""
    import shutil
    import tempfile
    from entip_train.encoders import get_encoder
    from ..triage.manifest import read_manifest
    rows = read_manifest(a.manifest)
    if a.per_class:
        seen, pick = Counter(), []
        for r in sorted(rows, key=lambda r: float(r.get("series_size_MB") or 0)):
            if seen[(r["label"], r["group"])] < a.per_class:
                seen[(r["label"], r["group"])] += 1
                pick.append(r)
        rows = pick
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out / f"{r['SeriesInstanceUID']}.npz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    gb = sum(float(r.get("series_size_MB") or 0) for r in todo) / 1000
    print(json.dumps({"slides": len(rows), "already_done": len(rows) - len(todo) if not a.limit else None,
                      "to_do": len(todo), "download_GB": round(gb, 1)}), flush=True)
    if a.dry_run or not todo:
        return 0
    from idc_index import IDCClient
    client, enc = IDCClient(), get_encoder(a.encoder)
    failed = 0
    for r in todo:
        uid = r["SeriesInstanceUID"]
        tmp = Path(tempfile.mkdtemp(prefix="idc-", dir=a.tmp))
        try:
            client.download_from_selection(downloadDir=str(tmp), seriesInstanceUID=[uid], quiet=True,
                                           show_progress_bar=False)
            dcm = next(iter(tmp.rglob("*.dcm")), None)
            if dcm is None:
                raise RuntimeError("download returned no DICOM files")
            info = embed_slide(str(dcm.parent), uid, enc, a, out)
            info["label"] = r["label"]
            _record(out, info)
            print(json.dumps(info), flush=True)
        except Exception as e:                      # one bad slide must not stop a run of thousands
            failed += 1
            with (out / "failed.jsonl").open("a") as f:
                f.write(json.dumps({"SeriesInstanceUID": uid, "error": repr(e)[:500]}) + "\n")
            print(json.dumps({"scan_id": uid, "error": repr(e)[:200]}), flush=True)
        finally:
            if not a.keep_slides:
                shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"done": len(todo) - failed, "failed": failed}))
    return 0


def _embed_args(p) -> None:
    p.add_argument("--out", required=True, help="folder for <id>.npz bags and index.jsonl")
    p.add_argument("--encoder", default="h-optimus-0", help="frozen foundation model (handcrafted: tests only)")
    p.add_argument("--mpp", type=float, default=None, help="override the slide's pixel size (µm/px)")
    p.add_argument("--tile-um", type=float, default=112.0)
    p.add_argument("--min-tissue", type=float, default=0.5)
    p.add_argument("--max-tiles", type=int, default=None, help="random cap on tiles per slide")


def add_commands(sp, help_suffix: str = "") -> None:
    """Add ``extract`` and ``build-bags`` to a subparser group (used by ``embed`` and, as aliases, ``triage``)."""
    p = sp.add_parser("extract", help="local slides -> tile embedding bags (.npz)" + help_suffix)
    p.add_argument("slides", nargs="+", help="slide files or DICOM series folders")
    p.add_argument("--scan-id", default=None)
    _embed_args(p)
    p.set_defaults(fn=extract)
    p = sp.add_parser("build-bags", help="manifest -> bags: download each IDC slide, embed it, delete it"
                      + help_suffix)
    p.add_argument("--manifest", required=True)
    _embed_args(p)
    p.add_argument("--per-class", type=int, default=None,
                   help="trial run: only the N smallest slides of each class and source")
    p.add_argument("--limit", type=int, default=None, help="stop after this many slides")
    p.add_argument("--tmp", default=None, help="folder for the slide being processed")
    p.add_argument("--keep-slides", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="only say how many slides and GB are left")
    p.set_defaults(fn=build_bags)


def add_parser(sub) -> None:
    e = sub.add_parser("embed", help="step 1: pre-training, slides -> frozen H-optimus-0 tile bags")
    add_commands(e.add_subparsers(dest="embed_cmd", required=True))
