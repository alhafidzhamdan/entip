"""Pick the slides for a triage training run from a triage manifest, following a run plan (``plans/*.json``).

    python pick_slides.py manifest.csv trial.csv --plan plans/diverse.json --idh idh.tsv

A plan lists groups; each group takes up to ``n`` patients (``null``: all of them) of one class from the given
IDC collections, one slide each or, with ``"all_slides": true``, every eligible slide of each patient, optionally
only patients with a given IDH status. With several collections in a group the
patients are spread evenly across them. Slides must be between ``min_mb`` and ``max_mb``: the very smallest
IDC slides are small because they hold almost no stained tissue (an 18 MB TCGA-GBM slide was a faint, nearly
unstained smear and gave no tissue tiles), and a cap keeps the download in hand. ``order`` is ``random``
(seeded, for a representative set) or ``smallest`` (for the quickest pipeline check).

IDH status (``--idh``) is a table with ``PATIENT_ID`` and ``IDH_STATUS`` (Mutant / WT), as written by
``fetch_idh.py`` from cBioPortal's TCGA pan-glioma study. A group with ``"idh"`` keeps only patients whose
status is known and matches; the others are dropped and counted, never guessed.

A group with ``"categories"`` (for example ``["astro4"]``) keeps only gliomas in those step 3 slide categories
(``highgrade_common/common.py``: grade and type from the WHO CNS5 diagnosis below), for a run that only needs some
grades.

A group can override the plan's ``min_mb`` / ``max_mb`` (``"max_mb": null``: no upper limit).

Every chosen slide gets ``diagnosis`` (its WHO CNS5 term) and ``who_cns5_basis`` (what that rests on), from
``entip_glioma.triage.who_cns5``: IDH, 1p/19q and grade from the ``--idh`` table for TCGA and CPTAC gliomas, the
source diagnosis (``dx``) otherwise, and confirmed label corrections first.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

IDH_WORDS = {"Mutant": "mutant", "WT": "wildtype"}


def read_idh(path: str) -> dict:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            call = IDH_WORDS.get((r.get("IDH_STATUS") or "").strip())
            pid = (r.get("PATIENT_ID") or "").strip()
            if pid and call:
                out[pid] = call if out.get(pid, call) == call else "conflict"
    return out


def pick(rows: list, plan: dict, idh: dict | None = None, category=None) -> tuple[list, list]:
    """Rows chosen by ``plan`` and one note per group: how many were wanted, eligible and taken."""
    p = plan.get("pick", {})
    lo, hi = float(p.get("min_mb", 0)), float(p.get("max_mb", float("inf")))
    order, rng = p.get("order", "random"), random.Random(p.get("seed", 0))
    used, chosen, notes = set(), [], []
    for g in plan["groups"]:
        cols = g.get("collections")
        g_lo = float(g["min_mb"]) if "min_mb" in g else lo          # a group can set its own size limits;
        g_hi = (float("inf") if g["max_mb"] is None else float(g["max_mb"])) if "max_mb" in g else hi  # null: none
        dropped = Counter()
        by_col = defaultdict(list)
        for r in rows:
            if r["label"] != g["label"] or (cols and r["collection_id"] not in cols):
                continue
            mb = float(r.get("series_size_MB") or 0)
            if not g_lo <= mb <= g_hi:
                dropped["size"] += 1
                continue
            if g.get("idh"):
                status = (idh or {}).get(r["PatientID"])
                if status != g["idh"]:
                    dropped["idh_unknown" if status in (None, "conflict") else "idh_other"] += 1
                    continue
            if g.get("categories") and category(r) not in g["categories"]:
                dropped["category_other"] += 1
                continue
            by_col[r["collection_id"]].append(r)
        for c in by_col:
            if order == "smallest":
                by_col[c].sort(key=lambda r: float(r.get("series_size_MB") or 0))
            else:
                by_col[c].sort(key=lambda r: r["SeriesInstanceUID"])
                rng.shuffle(by_col[c])
        queues = [by_col[c] for c in sorted(by_col)]
        if order == "smallest" and len(queues) > 1:      # one size-sorted queue across collections
            queues = [sorted((r for q in queues for r in q), key=lambda r: float(r.get("series_size_MB") or 0))]
        # each queue becomes a list of patients, in the order their first slide comes up
        slides_of = defaultdict(list)
        pqueues = []
        for q in queues:
            seen, pq = set(), []
            for r in q:
                slides_of[r["PatientID"]].append(r)
                if r["PatientID"] not in seen:
                    seen.add(r["PatientID"])
                    pq.append(r["PatientID"])
            pqueues.append(pq)
        want = g.get("n")                                 # null: every eligible patient
        all_slides = bool(g.get("all_slides"))            # every eligible slide of a chosen patient, not just one
        taken, n_pat = [], 0
        while (want is None or n_pat < want) and any(pqueues):    # round-robin across collections
            for pq in pqueues:
                while pq and pq[0] in used:
                    pq.pop(0)
                if pq and (want is None or n_pat < want):
                    pid = pq.pop(0)
                    used.add(pid)
                    n_pat += 1
                    taken += slides_of[pid] if all_slides else slides_of[pid][:1]
        chosen += taken
        notes.append({"group": g.get("name", g["label"]), "label": g["label"],
                      "wanted": "all" if want is None else want, "taken": n_pat, "slides": len(taken),
                      "by_collection": dict(Counter(r["collection_id"] for r in taken)), "skipped": dict(dropped)})
    return chosen, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--plan", required=True, help="run plan JSON (plans/*.json)")
    ap.add_argument("--idh", help="IDH status table from fetch_idh.py (needed when a group filters on IDH)")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    with open(a.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    if any(g.get("idh") for g in plan["groups"]) and not a.idh:
        raise SystemExit("this plan filters on IDH status: pass --idh (see fetch_idh.py)")
    from entip_glioma.triage import who_cns5                  # WHO CNS5 diagnosis for every chosen slide
    from entip_glioma.triage.manifest import read_corrections
    category = None
    if any(g.get("categories") for g in plan["groups"]):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
        from common import category as step3_category
        if not a.idh:
            raise SystemExit("this plan filters on slide category: pass --idh (grade and 1p/19q come from it)")
        mol, corr = who_cns5.read_molecular(a.idh), read_corrections()
        cache = {}

        def category(r):
            if r["SeriesInstanceUID"] not in cache:
                cache[r["SeriesInstanceUID"]] = step3_category(who_cns5.annotate([dict(r)], mol, corr)[0])
            return cache[r["SeriesInstanceUID"]]
    chosen, notes = pick(rows, plan, read_idh(a.idh) if a.idh else None, category)
    chosen = who_cns5.annotate(chosen, who_cns5.read_molecular(a.idh) if a.idh else None, read_corrections())
    fields += [k for k in ("diagnosis", "who_cns5_basis") if k not in fields]
    short = [n for n in notes if n["wanted"] != "all" and n["taken"] < n["wanted"] and not plan_group(plan, n).get("up_to")]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(chosen)
    mb = sum(float(r["series_size_MB"]) for r in chosen)
    print(f"{len(chosen)} slides from {len({r['PatientID'] for r in chosen})} patients, {mb / 1000:.1f} GB to download: "
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['label'] for r in chosen).items())))
    for n in notes:
        print(f"  {n['group']:<24} {n['taken']:>5} of {str(n['wanted']):<4} patients, {n['slides']:>5} slides  "
              f"{n['by_collection']}  skipped {n['skipped']}")
    if short:
        raise SystemExit(f"not enough eligible slides for: {[n['group'] for n in short]}")


def plan_group(plan: dict, note: dict) -> dict:
    return next(g for g in plan["groups"] if g.get("name", g["label"]) == note["group"])


if __name__ == "__main__":
    main()
