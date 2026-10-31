"""entip-fork command line.

    entip-fork synth OUT.svs [--seed N]
    entip-fork fork SLIDE --nhs-root DIR --research-root DIR --chi .. --accession .. --block .. --section .. --stain ..
    entip-fork batch MANIFEST.csv --nhs-root DIR --research-root DIR
    entip-fork reidentify CODE --nhs-root DIR --reason TEXT      (NHS side only, audited)
    entip-fork check FILE --identifier X [--identifier Y ...]
    entip-fork licences
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import sys
from pathlib import Path

from . import __version__, deid, synth
from .fork import IdentifierLeak, Stores, fork_slide

LICENCES = [
    ("numpy", "BSD-3-Clause", "required"),
    ("tifffile", "BSD-3-Clause", "required"),
    ("imagecodecs", "BSD-3-Clause", "optional: JPEG/JPEG2000 vendor slides"),
    ("sqlite3", "Python standard library (PSF-2.0); SQLite itself is public domain", "required"),
    ("pytest", "MIT", "tests only"),
]


def _stores(a) -> Stores:
    return Stores(a.nhs_root, a.research_root, allow_real_identifiers=a.allow_real_identifiers)


def _add_store_args(p, research=True):
    p.add_argument("--nhs-root", required=True, type=Path, help="NHS-side folder (key store + clinical copies)")
    if research:
        p.add_argument("--research-root", required=True, type=Path, help="research-side folder")
    p.add_argument("--actor", default=getpass.getuser(), help="person or service doing this (provenance)")
    p.add_argument("--allow-real-identifiers", action="store_true",
                   help="accept CHIs with a real date of birth (only once governance approval is in place)")


def _do_fork(stores, slide, row, actor, compression):
    r = fork_slide(
        stores, slide, chi=row["chi"], accession=row["accession"], block=row["block"],
        section=row["section"], stain=row["stain"], actor=actor,
        patient_name=row.get("patient_name") or None,
        extra_identifiers=[x for x in (row.get("extra_identifiers") or "").split(";") if x],
        compression=compression,
    )
    return {"slide": str(slide), "link_code": r.link_code, "study_id": r.study_id,
            "research_copy": str(r.research_path), "reused": r.reused, "dropped": r.dropped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entip-fork", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a synthetic SVS-like slide with fake identifiers")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("fork", help="fork one slide")
    p.add_argument("slide", type=Path)
    for f in ("chi", "accession", "block", "section", "stain"):
        p.add_argument(f"--{f}", required=True)
    p.add_argument("--patient-name")
    p.add_argument("--extra-identifier", action="append", default=[])
    p.add_argument("--compression", default="zlib", choices=["zlib", "jpeg", "none"])
    _add_store_args(p)

    p = sub.add_parser("batch", help="fork every slide in a CSV manifest")
    p.add_argument("manifest", type=Path, help="CSV: slide,chi,accession,block,section,stain[,patient_name,extra_identifiers]")
    p.add_argument("--compression", default="zlib", choices=["zlib", "jpeg", "none"])
    _add_store_args(p)

    p = sub.add_parser("reidentify", help="NHS side: show the clinical identifiers behind a research ID (audited)")
    p.add_argument("code")
    p.add_argument("--reason", required=True)
    _add_store_args(p, research=False)

    p = sub.add_parser("check", help="scan a file for identifiers")
    p.add_argument("file", type=Path)
    p.add_argument("--identifier", action="append", default=[])

    sub.add_parser("licences", help="list dependencies and their licences")

    a = ap.parse_args(argv)
    comp = getattr(a, "compression", "zlib")
    comp = None if comp == "none" else comp

    if a.cmd == "synth":
        c = synth.make_slide(a.out, seed=a.seed)
        print(json.dumps({"path": str(c.path), "chi": c.chi, "accession": c.accession, "block": c.block,
                          "section": c.section, "stain": c.stain, "patient_name": c.patient_name,
                          "note": "synthetic identifiers only"}, indent=2))
        return 0
    if a.cmd == "fork":
        row = {"chi": a.chi, "accession": a.accession, "block": a.block, "section": a.section,
               "stain": a.stain, "patient_name": a.patient_name, "extra_identifiers": ";".join(a.extra_identifier)}
        try:
            print(json.dumps(_do_fork(_stores(a), a.slide, row, a.actor, comp), indent=2))
        except IdentifierLeak as e:
            print(f"refused: research copy still held identifiers ({e})", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "batch":
        stores = _stores(a)
        failed = 0
        with open(a.manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                slide = (a.manifest.parent / row["slide"]).resolve()
                try:
                    print(json.dumps(_do_fork(stores, slide, row, a.actor, comp)))
                except Exception as e:  # keep going; report each failure without identifiers
                    failed += 1
                    print(json.dumps({"slide_row": row["slide"], "error": type(e).__name__}), file=sys.stderr)
        return 1 if failed else 0
    if a.cmd == "reidentify":
        from .keystore import KeyStore
        with KeyStore(a.nhs_root / "keystore.sqlite") as ks:
            print(json.dumps(ks.reidentify(a.code, actor=a.actor, reason=a.reason), indent=2))
        return 0
    if a.cmd == "check":
        findings = deid.scan_for_identifiers(a.file, a.identifier)
        print(json.dumps({"file": str(a.file), "clean": not findings, "findings": findings}, indent=2))
        return 0 if not findings else 2
    if a.cmd == "licences":
        for name, lic, use in LICENCES:
            print(f"{name:12s} {lic:60s} {use}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
